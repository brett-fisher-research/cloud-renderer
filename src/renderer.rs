//! The whole GPU-side pipeline: one full-screen triangle, one uniform buffer,
//! and the cloud shader — plus hot-reloading of that shader.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::camera::Uniforms;
use crate::defines::Defines;
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
    noise_textures: (wgpu::Texture, wgpu::Texture),
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
        // Bake the noise field into two tiling 3D textures: a big one for the
        // overall cloud shape and a small one for edge detail. A filtered
        // texture fetch replaces dozens of hash-and-interpolate instructions,
        // and the GPU's texture units do the trilinear blend in fixed-function
        // hardware, in parallel with the shader's arithmetic.
        // The base tile is four times the feature size, so the loudest octave
        // is the third. See `bake_fbm` for why the stack peaks there instead
        // of falling off from the start.
        let noise_base = noise::bake_fbm(
            device,
            &gpu.queue,
            "noise base",
            256,
            &[0.25, 0.4, 0.5, 0.25, 0.125, 0.0625, 0.03125],
        );
        let noise_detail =
            noise::bake_fbm(device, &gpu.queue, "noise detail", 64, &[0.5, 0.25, 0.125, 0.0625]);
        let base_view = noise_base.create_view(&wgpu::TextureViewDescriptor::default());
        let detail_view = noise_detail.create_view(&wgpu::TextureViewDescriptor::default());

        // Repeat addressing is what makes one 128-voxel cube cover the whole
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

        let texture_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D3,
                multisampled: false,
            },
            count: None,
        };

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform layout"),
            entries: &[
                uniform_entry(0),
                uniform_entry(1),
                texture_entry(2),
                texture_entry(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&base_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&detail_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&noise_sampler),
                },
            ],
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
            noise_textures: (noise_base, noise_detail),
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

    /// Upload the cloud dials. Cheap enough (16 bytes) to send every frame
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
