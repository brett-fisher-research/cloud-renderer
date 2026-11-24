//! Headless mode: render to an offscreen texture and write it out as a PNG.
//! No window, no event loop — this is how the renderer can check its own
//! output.

use anyhow::{Context, Result};

use crate::camera::Camera;
use crate::defines::Defines;
use crate::gpu::{create_instance, Gpu};
use crate::params::Params;
use crate::renderer::Renderer;

/// Same colour space as the window swapchain, so the PNG matches what the
/// window shows: the GPU applies linear -> sRGB encoding on write for us.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub struct ShotOptions {
    pub width: u32,
    pub height: u32,
    pub time: f32,
    pub sun_elevation: Option<f32>,
    pub yaw: Option<f32>,
    pub pitch: Option<f32>,
    pub params: Params,
    /// `--def` shader constant overrides for this run.
    pub defines: Defines,
    /// Frames to accumulate, the last of which is the one saved.
    pub frames: u32,
    /// Camera turn rate over the lead-up frames, (yaw, pitch) in radians per
    /// second, ending at the requested view. Zero for a still camera. This
    /// is how reprojection is tested: a pan that ends on a view should give
    /// the same picture as holding that view.
    pub pan: (f32, f32),
    /// Dump the raw density field instead of rendering the sky. Uses the
    /// `fs_field` entry point and a linear (non-sRGB) target, because the
    /// result is an input to a spectrum, not something to look at.
    pub field: bool,

}

pub fn render_to_png(path: &str, opts: &ShotOptions) -> Result<()> {
    let (width, height) = (opts.width, opts.height);
    log::info!(
        "headless {} {width}x{height} -> {path}",
        if opts.field { "field dump" } else { "render" }
    );

    let gpu = pollster::block_on(Gpu::new(create_instance(), None))?;

    // A field dump is data, not a picture: sRGB encoding on write would bend
    // every value through a gamma curve before anything could measure it.
    let format = if opts.field {
        wgpu::TextureFormat::Rgba8Unorm
    } else {
        FORMAT
    };
    let mut renderer = if opts.field {
        pollster::block_on(Renderer::with_entry(&gpu, format, &opts.defines, "fs_field"))?
    } else {
        pollster::block_on(Renderer::new(&gpu, format, &opts.defines))?
    };

    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screenshot target"),
        size: wgpu::Extent3d {
            width: width,
            height: height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        // RENDER_ATTACHMENT to draw into it, COPY_SRC to read it back.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut camera = Camera::default();
    if let Some(elev) = opts.sun_elevation {
        camera.sun_elevation = elev;
    }
    if let Some(yaw) = opts.yaw {
        camera.yaw = yaw;
    }
    if let Some(pitch) = opts.pitch {
        camera.pitch = pitch;
    }
    log::info!(
        "camera yaw {:.2} pitch {:.2}, sun elevation {:.2}, t = {:.1}s",
        camera.yaw,
        camera.pitch,
        camera.sun_elevation,
        opts.time
    );
    log::info!(
        "dials: coverage {:.2}, feature size {:.0} m, shadow {:.2}",
        opts.params.coverage,
        opts.params.feature_size,
        opts.params.shadow_strength
    );
    renderer.update_params(&gpu, &opts.params);

    // The window shows the average of the last several frames, so one frame
    // alone is not what anyone would see. Run the lead-up frames first, with
    // the clock advancing as it would at 60 fps so that anything the
    // accumulation smears (drifting cloud) is smeared in the still too.
    let frames = if opts.field { 1 } else { opts.frames.max(1) };
    for i in 0..frames - 1 {
        // Seconds before the final frame.
        let back = (frames - 1 - i) as f32 / 60.0;
        let moving = Camera {
            yaw: camera.yaw - opts.pan.0 * back,
            pitch: camera.pitch - opts.pan.1 * back,
            sun_elevation: camera.sun_elevation,
            sun_azimuth: camera.sun_azimuth,
        };
        renderer.update(&gpu, &moving.uniforms(width, height, opts.time - back));
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("screenshot lead-up"),
            });
        renderer.draw(&mut encoder, &view);
        // One submit per frame: the uniform upload above takes effect for
        // the commands submitted after it.
        gpu.queue.submit(Some(encoder.finish()));
    }
    renderer.update(&gpu, &camera.uniforms(width, height, opts.time));

    // GPU texture rows must start on a 256-byte boundary when copied to a
    // buffer, so the readback buffer may be wider than the image itself.
    let unpadded_bytes_per_row = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;

    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot readback"),
        size: (padded_bytes_per_row * height) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("screenshot encoder"),
        });

    renderer.draw(&mut encoder, &view);

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width: width,
            height: height,
            depth_or_array_layers: 1,
        },
    );

    gpu.queue.submit(Some(encoder.finish()));

    // Mapping is asynchronous: ask for it, then block until the GPU is done.
    let (tx, rx) = std::sync::mpsc::channel();
    readback.slice(..).map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| anyhow::anyhow!("device poll failed: {e:?}"))?;
    rx.recv()
        .context("readback channel closed")?
        .context("failed to map readback buffer")?;

    // Strip the row padding back out into a tight RGBA image.
    let mut pixels = Vec::with_capacity((unpadded_bytes_per_row * height) as usize);
    {
        let mapped = readback
            .slice(..)
            .get_mapped_range()
            .map_err(|e| anyhow::anyhow!("could not read mapped range: {e}"))?;
        for row in 0..height {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + unpadded_bytes_per_row as usize;
            pixels.extend_from_slice(&mapped[start..end]);
        }
    }
    readback.unmap();

    image::save_buffer(path, &pixels, width, height, image::ColorType::Rgba8)
        .with_context(|| format!("failed to write {path}"))?;

    log::info!("wrote {path}");
    Ok(())
}
