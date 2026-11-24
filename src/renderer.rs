//! The whole GPU-side pipeline: one full-screen triangle, the cloud shader,
//! temporal accumulation, and hot-reloading of that shader.
//!
//! Graphics note: *temporal accumulation* spreads work over time. The cloud
//! march starts every ray at a random offset, which hides banding but leaves
//! grain. Rather than take more samples per frame, each frame is blended
//! with the result of the frames before it, so every pixel shows the average
//! of the last dozen or so — the same samples a slower renderer would take
//! in one frame, collected over a fifth of a second instead.
//!
//! That needs somewhere to keep the previous result, so a frame is two passes:
//!
//! 1. **Clouds**: render into a *history* texture, reading the other history
//!    texture (last frame's result) and blending with it.
//! 2. **Present**: copy the texture just written to the screen.
//!
//! The two history textures swap roles every frame ("ping-pong"), because a
//! pass may not read the texture it is writing.

use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::atmosphere;
use crate::baked::Baked;
use crate::camera::Uniforms;
use crate::defines::Defines;
use crate::gpu::Gpu;
use crate::noise;
use crate::params::Params;
use crate::shader;
use crate::terrain;

/// Bindings the renderer itself owns in `@group(0)`. Everything else in the
/// group is a feature's baked texture and carries its own number.
const BINDING_UNIFORMS: u32 = 0;
const BINDING_PARAMS: u32 = 1;
const BINDING_REPEAT_SAMPLER: u32 = 7;
const BINDING_CLAMP_SAMPLER: u32 = 8;
const BINDING_HISTORY: u32 = 9;

/// Half floats: the accumulated image is an average of many frames, and 8
/// bits would round each blend step and never quite converge.
const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The accumulated image, twice over. `bind_groups[i]` is the full set of
/// shader inputs with `views[i]` bound as the history to read.
struct History {
    size: (u32, u32),
    views: [wgpu::TextureView; 2],
    bind_groups: [wgpu::BindGroup; 2],
}

/// The last pipelines that compiled successfully. A failed hot-reload leaves
/// these untouched, which is what keeps us rendering instead of crashing.
struct Pipelines {
    clouds: wgpu::RenderPipeline,
    /// `None` when drawing straight to the target (the field dump).
    present: Option<wgpu::RenderPipeline>,
}

pub struct Renderer {
    /// GPU-side copy of [`Uniforms`], rewritten every frame.
    uniform_buffer: wgpu::Buffer,
    /// GPU-side copy of [`Params`], rewritten only when a dial moves.
    params_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: Pipelines,
    /// Every feature's baked textures, kept so the bind groups can be
    /// rebuilt when the history textures are resized.
    baked: Vec<Baked>,
    noise_sampler: wgpu::Sampler,
    table_sampler: wgpu::Sampler,
    history: History,
    /// Which history texture the next frame draws into.
    write: usize,
    /// Frames drawn since the history was last valid from scratch. The
    /// shader averages evenly until this is large, so a fresh history
    /// converges in a few frames instead of fading in.
    history_frames: u32,
    /// Counts every frame, to step the jitter sequences.
    frame: u32,
    /// Last frame's camera basis, for finding where a pixel was on screen.
    prev_basis: Option<[[f32; 3]; 3]>,
    /// Format of whatever we present into (swapchain, or offscreen RGBA).
    target_format: wgpu::TextureFormat,
    shader_dir: PathBuf,
    /// `--def` overrides, re-applied on every hot-reload so a sweep's
    /// constants survive editing the file while it runs.
    defines: Defines,
    /// Which fragment entry point draws the image.
    entry: &'static str,
}

impl Renderer {
    pub async fn new(
        gpu: &Gpu,
        target_format: wgpu::TextureFormat,
        defines: &Defines,
    ) -> Result<Self> {
        Self::with_entry(gpu, target_format, defines, "fs_main").await
    }

    /// As [`Renderer::new`], but drawing a different fragment entry point.
    /// `--dump-field` uses this to run `fs_field` through the same bindings
    /// and noise textures as the real renderer. Anything but `fs_main` draws
    /// straight to the target with no accumulation: a field dump is data.
    pub async fn with_entry(
        gpu: &Gpu,
        target_format: wgpu::TextureFormat,
        defines: &Defines,
        entry: &'static str,
    ) -> Result<Self> {
        let device = &gpu.device;

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            // COPY_DST so `queue.write_buffer` can update it each frame.
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Each feature bakes what its shader file samples and says which
        // binding it goes on. See the feature's module for what each holds.
        let mut baked = noise::bake(device, &gpu.queue);
        baked.extend(atmosphere::bake_tables(device, &gpu.queue));
        baked.extend(terrain::bake(device, &gpu.queue));

        // Repeat addressing is what makes one cube of noise cover the whole
        // sky: sample coordinates outside 0..1 wrap back around.
        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("noise sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        // The atmosphere tables are functions, not patterns: reading past
        // the edge must clamp to the last value, not wrap to the other side.
        // The history texture and the terrain heightmap, which does not
        // tile, are read through this one too.
        let table_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("table sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // A bind group *layout* describes the shape of the shader's inputs;
        // the bind group is the actual set of resources matching that shape.
        //
        // The vertex stage reads the tables too: it works out the handful of
        // per-frame colours (ambient sky light, exposure) once per vertex
        // instead of once per pixel.
        let both = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX;
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: both,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture_entry =
            |binding: u32, view_dimension: wgpu::TextureViewDimension| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: both,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension,
                    multisampled: false,
                },
                count: None,
            };
        let sampler_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: both,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };

        let mut layout_entries = vec![
            uniform_entry(BINDING_UNIFORMS),
            uniform_entry(BINDING_PARAMS),
            sampler_entry(BINDING_REPEAT_SAMPLER),
            sampler_entry(BINDING_CLAMP_SAMPLER),
            texture_entry(BINDING_HISTORY, wgpu::TextureViewDimension::D2),
        ];
        layout_entries.extend(baked.iter().map(|b| texture_entry(b.binding, b.dimension)));
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud layout"),
            entries: &layout_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            // "Immediate" data is wgpu 30's push-constant equivalent; unused here.
            immediate_size: 0,
        });

        let shader_dir = shader::resolve_dir();
        let source = shader::load(&shader_dir)?;
        let text = defines.apply(&source.text)?;
        if !defines.is_empty() {
            log::info!("shader overrides: {}", defines.summary());
        }

        // On first load a broken shader is fatal — there is no "last good"
        // version to fall back to yet.
        let pipelines = build_pipelines(gpu, &pipeline_layout, target_format, &text, entry)
            .await
            .map_err(|e| {
                anyhow::anyhow!("shader failed to compile:\n{}", source.locate_errors(&e))
            })?;

        // A placeholder history. The first `update` replaces it with one
        // the size of the frame.
        let history = make_history(
            device,
            &bind_group_layout,
            &uniform_buffer,
            &params_buffer,
            &baked,
            &noise_sampler,
            &table_sampler,
            (1, 1),
        );

        Ok(Self {
            uniform_buffer,
            params_buffer,
            bind_group_layout,
            pipeline_layout,
            pipelines,
            baked,
            noise_sampler,
            table_sampler,
            history,
            write: 0,
            history_frames: 0,
            frame: 0,
            prev_basis: None,
            target_format,
            shader_dir,
            defines: defines.clone(),
            entry,
        })
    }

    /// Re-read the shader from disk and swap in new pipelines.
    ///
    /// Returns `Ok(())` if the new shader compiled. On failure the error is
    /// returned and the previously working pipelines stay in place.
    ///
    /// Native only: the web build has no disk to re-read.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn reload_shader(&mut self, gpu: &Gpu) -> Result<()> {
        let source = shader::load(&self.shader_dir)?;
        let text = self.defines.apply(&source.text)?;

        match pollster::block_on(build_pipelines(
            gpu,
            &self.pipeline_layout,
            self.target_format,
            &text,
            self.entry,
        )) {
            Ok(pipelines) => {
                self.pipelines = pipelines;
                // The old history was drawn by the old shader. Start again
                // so an edit shows at once instead of fading in.
                self.history_frames = 0;
                log::info!("shader reloaded OK");
                Ok(())
            }
            Err(e) => Err(anyhow::anyhow!("{}", source.locate_errors(&e))),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn shader_dir(&self) -> &Path {
        &self.shader_dir
    }

    /// Upload this frame's camera uniforms. Call once per frame, before
    /// [`Renderer::draw`].
    ///
    /// The camera fills in where it is looking; this adds everything to do
    /// with *time* — last frame's view, this frame's random offsets, how
    /// much history there is — so no caller has to track it.
    pub fn update(&mut self, gpu: &Gpu, uniforms: &Uniforms) {
        let size = (
            (uniforms.resolution[0] as u32).max(1),
            (uniforms.resolution[1] as u32).max(1),
        );
        if size != self.history.size {
            log::info!("history textures: {}x{}", size.0, size.1);
            self.history = make_history(
                &gpu.device,
                &self.bind_group_layout,
                &self.uniform_buffer,
                &self.params_buffer,
                &self.baked,
                &self.noise_sampler,
                &self.table_sampler,
                size,
            );
            self.history_frames = 0;
        }

        let mut u = *uniforms;
        let basis = [u.cam_forward, u.cam_right, u.cam_up];
        let prev = self.prev_basis.unwrap_or(basis);
        u.prev_forward = prev[0];
        u.prev_right = prev[1];
        u.prev_up = prev[2];
        self.prev_basis = Some(basis);

        // Low-discrepancy sequences: each new value lands in the biggest gap
        // left by the ones before, so a short run of frames covers 0..1
        // evenly. Random numbers would clump and converge more slowly.
        // Computed in f64 here because a large frame count times an
        // irrational loses its fraction in a shader's 32-bit floats.
        let n = self.frame as f64;
        u.jitter = (n * 0.618_033_988_749_894_9).fract() as f32;
        u.subpixel_x = ((0.5 + n * 0.754_877_666_246_692_7).fract() - 0.5) as f32;
        u.subpixel_y = ((0.5 + n * 0.569_840_290_998_053_2).fract() - 0.5) as f32;
        u.history_frames = self.history_frames as f32;
        self.frame = self.frame.wrapping_add(1);

        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&u));
    }

    /// Upload the cloud dials. Cheap enough (16 bytes) to send every frame
    /// rather than tracking whether a slider actually moved.
    pub fn update_params(&self, gpu: &Gpu, params: &Params) {
        gpu.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Record one frame into `encoder`, ending with the image in `view`.
    pub fn draw(&mut self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        self.draw_timed(encoder, view, None);
    }

    /// As [`Renderer::draw`], but optionally asks the GPU to write its own
    /// clock at the start and end of the frame. `query` is the query set and
    /// the index of the first of two slots. That is how `--bench` measures
    /// the frame by itself, with no vsync or compositor in the number.
    pub fn draw_timed(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        query: Option<(&wgpu::QuerySet, u32)>,
    ) {
        let start = query.map(|(query_set, index)| wgpu::RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(index),
            end_of_pass_write_index: None,
        });
        let end = query.map(|(query_set, index)| wgpu::RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: None,
            end_of_pass_write_index: Some(index + 1),
        });

        let Some(present) = &self.pipelines.present else {
            // No accumulation: one pass, straight to the target.
            let both = query.map(|(query_set, index)| wgpu::RenderPassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(index),
                end_of_pass_write_index: Some(index + 1),
            });
            fullscreen_pass(
                encoder,
                "field pass",
                view,
                &self.pipelines.clouds,
                &self.history.bind_groups[0],
                both,
            );
            return;
        };

        let read = 1 - self.write;
        // Clouds: read last frame's result, write this frame's.
        fullscreen_pass(
            encoder,
            "cloud pass",
            &self.history.views[self.write],
            &self.pipelines.clouds,
            &self.history.bind_groups[read],
            start,
        );
        // Present: read what was just written, write the screen.
        fullscreen_pass(
            encoder,
            "present pass",
            view,
            present,
            &self.history.bind_groups[self.write],
            end,
        );

        self.write = read;
        self.history_frames = (self.history_frames + 1).min(1 << 20);
    }
}

/// One render pass drawing the full-screen triangle.
///
/// A render pass is a scope that targets one or more textures. Three
/// vertices, no vertex buffer: the vertex shader generates a triangle big
/// enough to cover the screen from the vertex index alone.
fn fullscreen_pass(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                // The clear colour never shows; the shader covers every pixel.
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

/// Create the two history textures and a bind group reading each.
#[allow(clippy::too_many_arguments)]
fn make_history(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    params_buffer: &wgpu::Buffer,
    baked: &[Baked],
    noise_sampler: &wgpu::Sampler,
    table_sampler: &wgpu::Sampler,
    size: (u32, u32),
) -> History {
    let make_view = |label: &str| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HISTORY_FORMAT,
                // Drawn into on one frame, sampled on the next.
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    };
    let views = [make_view("history a"), make_view("history b")];

    let make_group = |history: &wgpu::TextureView| {
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: BINDING_UNIFORMS,
                resource: uniform_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: BINDING_PARAMS,
                resource: params_buffer.as_entire_binding(),
            },
        ];
        for b in baked {
            entries.push(wgpu::BindGroupEntry {
                binding: b.binding,
                resource: wgpu::BindingResource::TextureView(&b.view),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: BINDING_REPEAT_SAMPLER,
            resource: wgpu::BindingResource::Sampler(noise_sampler),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: BINDING_CLAMP_SAMPLER,
            resource: wgpu::BindingResource::Sampler(table_sampler),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: BINDING_HISTORY,
            resource: wgpu::BindingResource::TextureView(history),
        });
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud bind group"),
            layout,
            entries: &entries,
        })
    };
    let bind_groups = [make_group(&views[0]), make_group(&views[1])];

    History { size, views, bind_groups }
}

/// Compile WGSL and build the render pipelines, capturing validation errors
/// instead of letting them panic the process.
async fn build_pipelines(
    gpu: &Gpu,
    layout: &wgpu::PipelineLayout,
    target_format: wgpu::TextureFormat,
    source: &str,
    entry: &str,
) -> Result<Pipelines, String> {
    let device = &gpu.device;

    // An *error scope* catches validation errors produced inside it, so a typo
    // in the shader becomes a `Some(error)` we can print rather than a crash.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(shader::MODULE_LABEL),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let pipeline = |label: &str, fragment: &str, format: wgpu::TextureFormat| {
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(fragment),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            // No vertex buffers and no depth testing: one flat triangle.
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    };

    // The real image accumulates in a history texture and is then presented.
    // Anything else (the field dump) draws straight to the target.
    let pipelines = if entry == "fs_main" {
        Pipelines {
            clouds: pipeline("cloud pipeline", entry, HISTORY_FORMAT),
            present: Some(pipeline("present pipeline", "fs_present", target_format)),
        }
    } else {
        Pipelines { clouds: pipeline("field pipeline", entry, target_format), present: None }
    };

    // wgpu's Display already includes the full WGSL diagnostic with line
    // numbers and a caret pointing at the offending token.
    match scope.pop().await {
        Some(err) => Err(err.to_string()),
        None => Ok(pipelines),
    }
}
