//! Headless benchmark: render the cloud pass over and over into an offscreen
//! texture and time it on the GPU itself.
//!
//! Graphics note: wall-clock frame time in a window measures the whole
//! pipeline — vsync, the compositor, egui, the present queue. A *timestamp
//! query* instead asks the GPU to write its own clock into a buffer at two
//! points in the command stream, so the difference is the time the GPU spent
//! on exactly the work between them. That is the number to optimise, because
//! it is the only one that is ours.

use anyhow::{Context, Result};

use crate::camera::{Camera, Uniforms};
use crate::defines::Defines;
use crate::gpu::{create_instance, Gpu};
use crate::params::Params;
use crate::renderer::Renderer;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Passes recorded per submit. Keeps any single GPU job well under the
/// driver's watchdog timeout even when one frame is slow.
const BATCH: u32 = 8;

pub struct BenchOptions {
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub time: f32,
    pub sun_elevation: Option<f32>,
    pub yaw: Option<f32>,
    pub pitch: Option<f32>,
    pub params: Params,
    /// Printed alongside the result so a sweep's output is self-describing.
    pub label: String,    /// `--def` shader constant overrides for this run.
    pub defines: Defines,

}

pub fn run(opts: &BenchOptions) -> Result<()> {
    let (width, height) = (opts.width, opts.height);

    let gpu = Gpu::new(create_instance(), None)?;
    if !gpu
        .device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
    {
        anyhow::bail!("this adapter has no TIMESTAMP_QUERY support, cannot benchmark");
    }
    let mut renderer = Renderer::new(&gpu, FORMAT, &opts.defines)?;

    let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench target"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut camera = Camera::default();
    if let Some(e) = opts.sun_elevation {
        camera.sun_elevation = e;
    }
    if let Some(y) = opts.yaw {
        camera.yaw = y;
    }
    if let Some(p) = opts.pitch {
        camera.pitch = p;
    }
    let uniforms = camera.uniforms(width, height, opts.time);
    renderer.update_params(&gpu, &opts.params);

    // Two timestamps (start, end) per pass.
    let query_set = gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("bench timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: BATCH * 2,
    });
    let resolve = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench query resolve"),
        size: (BATCH * 2 * 8) as u64,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bench query readback"),
        size: (BATCH * 2 * 8) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // Ticks to nanoseconds.
    let period = gpu.queue.get_timestamp_period() as f64;

    // Warm up: first submit pays for shader compilation, memory allocation and
    // clocking the GPU up out of its idle power state.
    for _ in 0..2 {
        let batch = time_batch(&gpu, &mut renderer, &uniforms, &view, &query_set, &resolve, &readback)?;
        let _ = batch;
    }

    // Apple GPUs are tile-based deferred renderers: they do not run passes in
    // the order recorded, so a pass's start and end timestamps do not bracket
    // its work. Measured on an M5 Pro, the timestamps said 30 ms a frame (with
    // a minimum of 0.00) where the batch took 8.5 ms a frame on the wall
    // clock. On Metal, time each batch on the wall clock instead: submit,
    // wait for the readback, divide by the frames in it. Nothing else is
    // queued and nothing is presented, so that is GPU time plus a small fixed
    // overhead. Each sample is then a batch mean, so min and p95 are coarser.
    let wall_clock = gpu.adapter.get_info().backend == wgpu::Backend::Metal;
    if wall_clock {
        log::warn!("Metal: GPU timestamps are unreliable here, timing batches on the wall clock");
    }

    let mut samples: Vec<f64> = Vec::with_capacity(opts.frames as usize);
    let wall = std::time::Instant::now();
    while samples.len() < opts.frames as usize {
        let started = std::time::Instant::now();
        let batch = time_batch(&gpu, &mut renderer, &uniforms, &view, &query_set, &resolve, &readback)?;
        let batch_ms = started.elapsed().as_secs_f64() * 1000.0 / BATCH as f64;
        for ticks in batch {
            samples.push(if wall_clock {
                batch_ms
            } else {
                ticks * period / 1.0e6 // ns -> ms
            });
        }
    }
    let wall = wall.elapsed().as_secs_f64();
    samples.truncate(opts.frames as usize);
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = samples.len();
    let median = samples[n / 2];
    let min = samples[0];
    let p95 = samples[(n * 95 / 100).min(n - 1)];
    let mean = samples.iter().sum::<f64>() / n as f64;

    println!(
        "{label:<28} {w}x{h}  cov {cov:.2}  feat {feat:<5.0}  \
         gpu median {median:6.2} ms ({fps:5.1} fps)  min {min:6.2}  p95 {p95:6.2}  \
         mean {mean:6.2}  wall {wall:.2}s{defs}",
        label = opts.label,
        defs = if opts.defines.is_empty() {
            String::new()
        } else {
            format!("  [{}]", opts.defines.summary())
        },
        w = width,
        h = height,
        cov = opts.params.coverage,
        feat = opts.params.feature_size,
        median = median,
        fps = 1000.0 / median,
        min = min,
        p95 = p95,
        mean = mean,
        wall = wall,
    );
    Ok(())
}

/// Record BATCH passes, each bracketed by a timestamp pair, submit them, and
/// read the elapsed ticks back.
fn time_batch(
    gpu: &Gpu,
    renderer: &mut Renderer,
    uniforms: &Uniforms,
    view: &wgpu::TextureView,
    query_set: &wgpu::QuerySet,
    resolve: &wgpu::Buffer,
    readback: &wgpu::Buffer,
) -> Result<Vec<f64>> {
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bench encoder"),
        });

    // One uniform upload per batch. Every frame in the batch therefore shares
    // its random offsets, which changes the picture but not the work: after
    // the first batch each frame reads a full history, as in the window.
    renderer.update(gpu, uniforms);
    for i in 0..BATCH {
        // The timestamps bracket the whole frame: the cloud pass and the
        // present pass after it.
        renderer.draw_timed(&mut encoder, view, Some((query_set, i * 2)));
    }
    encoder.resolve_query_set(query_set, 0..BATCH * 2, resolve, 0);
    encoder.copy_buffer_to_buffer(resolve, 0, readback, 0, (BATCH * 2 * 8) as u64);
    gpu.queue.submit(Some(encoder.finish()));

    let (tx, rx) = std::sync::mpsc::channel();
    readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| anyhow::anyhow!("device poll failed: {e:?}"))?;
    rx.recv()
        .context("query readback channel closed")?
        .context("failed to map query readback")?;

    let mut out = Vec::with_capacity(BATCH as usize);
    {
        let mapped = readback
            .slice(..)
            .get_mapped_range()
            .map_err(|e| anyhow::anyhow!("could not read mapped range: {e}"))?;
        let ticks: &[u64] = bytemuck::cast_slice(&mapped);
        for i in 0..BATCH as usize {
            out.push((ticks[i * 2 + 1].saturating_sub(ticks[i * 2])) as f64);
        }
    }
    readback.unmap();
    Ok(out)
}
