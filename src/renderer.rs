//! The whole GPU-side pipeline: one full-screen triangle, one uniform buffer,
//! and the cloud shader — plus hot-reloading of that shader.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::camera::Uniforms;
use crate::gpu::Gpu;

/// Where the shader lives, relative to the crate root.
pub const SHADER_PATH: &str = "shaders/clouds.wgsl";

pub struct Renderer {
    /// GPU-side copy of [`Uniforms`], rewritten every frame.
    uniform_buffer: wgpu::Buffer,
    /// Binds that buffer to `@group(0) @binding(0)` in the shader.
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    /// The last pipeline that compiled successfully. A failed hot-reload leaves
    /// this untouched, which is what keeps us rendering instead of crashing.
    pipeline: wgpu::RenderPipeline,
    /// Format of whatever we draw into (swapchain texture, or offscreen RGBA).
    target_format: wgpu::TextureFormat,
    shader_path: PathBuf,
}

impl Renderer {
    pub fn new(gpu: &Gpu, target_format: wgpu::TextureFormat) -> Result<Self> {
        let device = &gpu.device;

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            // COPY_DST so `queue.write_buffer` can update it each frame.
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // A bind group *layout* describes the shape of the shader's inputs; the
        // bind group is the actual set of resources matching that shape.
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("uniform layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("uniform bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
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

        // On first load a broken shader is fatal — there is no "last good"
        // version to fall back to yet.
        let pipeline = build_pipeline(gpu, &pipeline_layout, target_format, &source)
            .map_err(|e| anyhow::anyhow!("shader failed to compile:\n{e}"))?;

        Ok(Self {
            uniform_buffer,
            bind_group,
            bind_group_layout,
            pipeline_layout,
            pipeline,
            target_format,
            shader_path,
        })
    }

    /// Re-read the shader from disk and swap in a new pipeline.
    ///
    /// Returns `Ok(())` if the new shader compiled. On failure the error is
    /// returned and the previously working pipeline stays in place.
    pub fn reload_shader(&mut self, gpu: &Gpu) -> Result<()> {
        let source = std::fs::read_to_string(&self.shader_path)
            .with_context(|| format!("could not read {}", self.shader_path.display()))?;

        match build_pipeline(gpu, &self.pipeline_layout, self.target_format, &source) {
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

    /// Upload this frame's uniforms.
    pub fn update(&self, gpu: &Gpu, uniforms: &Uniforms) {
        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Record the single full-screen draw into `encoder`.
    pub fn draw(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
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
            timestamp_writes: None,
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
            entry_point: Some("fs_main"),
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
