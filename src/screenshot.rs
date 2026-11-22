//! Headless mode: render exactly one frame to an offscreen texture and write
//! it out as a PNG. No window, no event loop — this is how the renderer can
//! check its own output.

use anyhow::{Context, Result};

use crate::camera::Camera;
use crate::gpu::{create_instance, Gpu};
use crate::renderer::Renderer;

pub const WIDTH: u32 = 1280;
pub const HEIGHT: u32 = 720;

/// Same colour space as the window swapchain, so the PNG matches what the
/// window shows: the GPU applies linear -> sRGB encoding on write for us.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub fn render_to_png(path: &str, time: f32, sun_elevation: Option<f32>) -> Result<()> {
    log::info!("headless render {WIDTH}x{HEIGHT} -> {path}");

    let gpu = Gpu::new(create_instance(), None)?;
    let renderer = Renderer::new(&gpu, FORMAT)?;

    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screenshot target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        // RENDER_ATTACHMENT to draw into it, COPY_SRC to read it back.
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut camera = Camera::default();
    if let Some(elev) = sun_elevation {
        camera.sun_elevation = elev;
    }
    renderer.update(&gpu, &camera.uniforms(WIDTH, HEIGHT, time));

    // GPU texture rows must start on a 256-byte boundary when copied to a
    // buffer, so the readback buffer may be wider than the image itself.
    let unpadded_bytes_per_row = WIDTH * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;

    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot readback"),
        size: (padded_bytes_per_row * HEIGHT) as u64,
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
                rows_per_image: Some(HEIGHT),
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
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
    let mut pixels = Vec::with_capacity((unpadded_bytes_per_row * HEIGHT) as usize);
    {
        let mapped = readback
            .slice(..)
            .get_mapped_range()
            .map_err(|e| anyhow::anyhow!("could not read mapped range: {e}"))?;
        for row in 0..HEIGHT {
            let start = (row * padded_bytes_per_row) as usize;
            let end = start + unpadded_bytes_per_row as usize;
            pixels.extend_from_slice(&mapped[start..end]);
        }
    }
    readback.unmap();

    image::save_buffer(path, &pixels, WIDTH, HEIGHT, image::ColorType::Rgba8)
        .with_context(|| format!("failed to write {path}"))?;

    log::info!("wrote {path}");
    Ok(())
}
