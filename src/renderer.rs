//! The whole GPU-side pipeline: one full-screen triangle, one uniform buffer,
//! and the cloud shader — plus hot-reloading of that shader.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::camera::Uniforms;
use crate::defines::Defines;
use crate::atmosphere;
use crate::noise;
use crate::gpu::Gpu;
use crate::params::Params;

/// Where the shader lives, relative to the crate root.
pub const SHADER_PATH: &str = "shaders/clouds.wgsl";

pub struct Renderer {
    /// GPU-side copy of [`Uniforms`], rewritten every frame.
    uniform_buffer: wgpu::Buffer,
    /// GPU-side copy of [`Params`], rewritten only when a dial moves.
    params_buffer: wgpu::Buffer,
    /// Binds both buffers to `@group(0)` bindings 0 and 1 in the shader.
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    /// The last pipeline that compiled successfully. A failed hot-reload leaves
    /// this untouched, which is what keeps us rendering instead of crashing.
    pipeline: wgpu::RenderPipeline,
    /// Tiling 3D noise, baked once at startup. Keeps the textures alive for
    /// as long as the bind group that references them.
    #[allow(dead_code)]
    textures: [wgpu::Texture; 5],
    /// Format of whatever we draw into (swapchain texture, or offscreen RGBA).
    target_format: wgpu::TextureFormat,
    shader_path: PathBuf,
    /// `--def` overrides, re-applied on every hot-reload so a sweep's
    /// constants survive editing the file while it runs.
    defines: Defines,
    /// Which fragment entry point this pipeline draws.
    entry: &'static str,
}

impl Renderer {
    pub fn new(
        gpu: &Gpu,
        target_format: wgpu::TextureFormat,
        defines: &Defines,
    ) -> Result<Self> {
        Self::with_entry(gpu, target_format, defines, "fs_main")
    }

    /// As [`Renderer::new`], but drawing a different fragment entry point.
    /// `--dump-field` uses this to run `fs_field` through the same
    /// pipeline, bindings and noise textures as the real renderer.
    pub fn with_entry(
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

        // A bind group *layout* describes the shape of the shader's inputs; the
        // bind group is the actual set of resources matching that shape. Both
        // uniform blocks are the same shape, so the entries only differ by
        // binding number.
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        // Bake everything the shader samples. See `noise.rs` and
        // `atmosphere.rs` for what each texture holds and why.
        let noise_shape = noise::bake_shape(device, &gpu.queue, 128);
        let noise_detail = noise::bake_detail(device, &gpu.queue, 64);
        let weather = noise::bake_weather(device, &gpu.queue, 512);
        let (transmittance, multiscatter) = atmosphere::bake(device, &gpu.queue);
        let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let views = [
            view(&noise_shape),
            view(&noise_detail),
            view(&weather),
            view(&transmittance),
            view(&multiscatter),
        ];

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
        let table_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("table sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // The vertex stage reads the tables too: it works out the handful of
        // per-frame colours (ambient sky light, exposure) once per vertex
        // instead of once per pixel.
        let both = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX;
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

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cloud layout"),
            entries: &[
                uniform_entry(0),
                uniform_entry(1),
                texture_entry(2, wgpu::TextureViewDimension::D3),
                texture_entry(3, wgpu::TextureViewDimension::D3),
                texture_entry(4, wgpu::TextureViewDimension::D2),
                texture_entry(5, wgpu::TextureViewDimension::D2),
                texture_entry(6, wgpu::TextureViewDimension::D2),
                sampler_entry(7),
                sampler_entry(8),
            ],
        });

        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: params_buffer.as_entire_binding() },
        ];
        for (i, v) in views.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 2 + i as u32,
                resource: wgpu::BindingResource::TextureView(v),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 7,
            resource: wgpu::BindingResource::Sampler(&noise_sampler),
        });
        entries.push(wgpu::BindGroupEntry {
            binding: 8,
            resource: wgpu::BindingResource::Sampler(&table_sampler),
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cloud bind group"),
            layout: &bind_group_layout,
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            // "Immediate" data is wgpu 30's push-constant equivalent; unused here.
            immediate_size: 0,
        });

        let shader_path = resolve_shader_path();
        log::info!("loading shader from {}", shader_path.display());
        let source = std::fs::read_to_string(&shader_path)
            .with_context(|| format!("could not read {}", shader_path.display()))?;
        let source = defines.apply(&source)?;
        if !defines.is_empty() {
            log::info!("shader overrides: {}", defines.summary());
        }

        // On first load a broken shader is fatal — there is no "last good"
        // version to fall back to yet.
        let pipeline = build_pipeline(gpu, &pipeline_layout, target_format, &source, entry)
            .map_err(|e| anyhow::anyhow!("shader failed to compile:\n{e}"))?;

        Ok(Self {
            uniform_buffer,
            params_buffer,
            bind_group,
            bind_group_layout,
            pipeline_layout,
            pipeline,
            textures: [noise_shape, noise_detail, weather, transmittance, multiscatter],
            target_format,
            shader_path,
            defines: defines.clone(),
            entry,
        })
    }

    /// Re-read the shader from disk and swap in a new pipeline.
    ///
    /// Returns `Ok(())` if the new shader compiled. On failure the error is
    /// returned and the previously working pipeline stays in place.
    pub fn reload_shader(&mut self, gpu: &Gpu) -> Result<()> {
        let source = std::fs::read_to_string(&self.shader_path)
            .with_context(|| format!("could not read {}", self.shader_path.display()))?;
        let source = self.defines.apply(&source)?;

        match build_pipeline(
            gpu,
            &self.pipeline_layout,
            self.target_format,
            &source,
            self.entry,
        ) {
            Ok(pipeline) => {
                self.pipeline = pipeline;
                log::info!("shader reloaded OK");
                Ok(())
            }
            Err(e) => Err(anyhow::anyhow!("{e}")),
        }
    }

    pub fn shader_path(&self) -> &Path {
        &self.shader_path
    }

    /// Upload this frame's camera uniforms.
    pub fn update(&self, gpu: &Gpu, uniforms: &Uniforms) {
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Upload the cloud dials. Cheap enough (32 bytes) to send every frame
    /// rather than tracking whether a slider actually moved.
    pub fn update_params(&self, gpu: &Gpu, params: &Params) {
        gpu.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));
    }

    /// Record the single full-screen draw into `encoder`.
    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        self.draw_timed(encoder, view, None);
    }

    /// As [`Renderer::draw`], but optionally asks the GPU to write its own
    /// clock at the start and end of the pass. That is how `--bench` measures
    /// the cloud pass by itself, with no vsync or compositor in the number.
    pub fn draw_timed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        // A render pass is a scope that targets one or more textures. Ours
        // clears the target and then draws three vertices over it.
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("cloud pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
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

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        // Three vertices, no vertex buffer: the vertex shader generates a
        // triangle big enough to cover the screen from the vertex index alone.
        pass.draw(0..3, 0..1);
    }

    /// Unused for now, but keeps the layout alive and documents the binding.
    #[allow(dead_code)]
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}

/// Compile WGSL and build a render pipeline, capturing validation errors
/// instead of letting them panic the process.
fn build_pipeline(
    gpu: &Gpu,
    layout: &wgpu::PipelineLayout,
    target_format: wgpu::TextureFormat,
    source: &str,
    entry: &str,
) -> Result<wgpu::RenderPipeline, String> {
    let device = &gpu.device;

    // An *error scope* catches validation errors produced inside it, so a typo
    // in the shader becomes a `Some(error)` we can print rather than a crash.
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("clouds.wgsl"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("cloud pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: target_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        // No vertex buffers and no depth testing: we draw one flat triangle.
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    // wgpu's Display already includes the full WGSL diagnostic with line
    // numbers and a caret pointing at the offending token.
    match pollster::block_on(scope.pop()) {
        Some(err) => Err(err.to_string()),
        None => Ok(pipeline),
    }
}

/// Find `shaders/clouds.wgsl` whether we were launched from the crate root or
/// from somewhere else (e.g. a different cwd under `cargo run`).
fn resolve_shader_path() -> PathBuf {
    let direct = PathBuf::from(SHADER_PATH);
    if direct.exists() {
        return direct;
    }
    let from_manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join(SHADER_PATH);
    if from_manifest.exists() {
        return from_manifest;
    }
    direct
}
