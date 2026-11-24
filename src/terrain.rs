//! The terrain heightmap, baked into one texture at startup.
//!
//! Graphics note: a *heightfield* stores the ground as one number per map
//! position, its height. It cannot describe caves or overhangs, but a ray can
//! be tested against it with one texture fetch ("is the ray below the ground
//! here?"), which is what lets the fragment shader march it with no geometry.
//!
//! The shape is *ridged multifractal* noise (Musgrave). Ordinary layered
//! noise, fBm, gives rounded hills. Folding each layer at zero (`1 - |n|`)
//! turns its zero crossings into sharp creases, which read as ridgelines, and
//! weighting each finer layer by the one before keeps valleys smooth while
//! the crests stay rough. It has no erosion yet: no gullies or drainage.
//!
//! The texture has three jobs, one per channel group:
//!
//! - **R + G: height in metres.** A half float holds about 11 bits, so 3 km
//!   of relief would land on steps of over a metre, and the lighting, which
//!   depends on height *differences*, would show them as terraces. R holds
//!   the height rounded to 8 m (exact in a half float) and G the remainder
//!   (small, so precise to millimetres). Filtering is linear, so the sum of
//!   the two filtered channels is the filtered height.
//! - **B: a conservative upper bound**, for skipping empty air. At mip level
//!   L it is at least the height of everything within one level-L texel of
//!   wherever it is sampled. A ray higher than that can stride a whole texel
//!   without testing anything. See `terrain_surface` in the shader.
//!
//! The map does not tile. It covers a fixed square centred on the camera:
//! a flat plain in the middle, a ring of mountains around it.

use web_time::Instant;

use crate::atmosphere::half;
use crate::baked::Baked;
use crate::noise::{fill, hash_u32};

/// Binding of the heightmap in `@group(0)`. Matches `shaders/terrain.wgsl`.
pub const BINDING: u32 = 10;

/// Metres across the whole map. Must match `TERRAIN_EXTENT` in the shader.
pub const EXTENT: f32 = 96_000.0;

/// Texels across. The web build bakes on one thread, so it takes a quarter
/// of the texels.
const SIZE: usize = if cfg!(target_arch = "wasm32") { 1024 } else { 2048 };

/// Height of the tallest summit, metres. With the cloud base dial below this, the
/// higher peaks stand in the cloud.
const PEAK_HEIGHT: f32 = 2800.0;
/// Width of the broadest ridges, metres.
const FEATURE_SIZE: f32 = 9_000.0;
/// The plain is flat out to here; the range reaches full height by
/// `RANGE_FULL` and dies away again before the map's edge.
const PLAIN_RADIUS: f32 = 5_000.0;
const RANGE_FULL: f32 = 16_000.0;
/// How far noise pushes the range's inner edge in and out, metres, so the
/// plain is not a perfect circle.
const EDGE_WANDER: f32 = 4_000.0;
/// Ridged multifractal: how strongly a layer's height gates the next one.
const RIDGE_GAIN: f32 = 2.0;
/// The level below which the ridged field is cut off to make valley floors.
const VALLEY_CUT: f32 = 0.35;

/// The step heights are rounded to in the R channel. Multiples of 8 are
/// exact in a half float up to 16 km.
const COARSE_STEP: f32 = 8.0;
/// How many texels the bound reaches in each direction. Three makes a
/// bilinear read of it valid for everything within one texel of the sample.
const BOUND_REACH: usize = 3;

type Gradients = [[f32; 2]; 32];

/// 2D gradient (Perlin) noise, not tiling. Roughly -1..1.
fn gradient_noise(p: [f32; 2], seed: u32, gradients: &Gradients) -> f32 {
    let cell = [p[0].floor(), p[1].floor()];
    let t = [p[0] - cell[0], p[1] - cell[1]];
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let corner = |dx: f32, dy: f32| -> f32 {
        // Negative cells wrap to large unsigned ones, which hash as well as
        // any other.
        let h = hash_u32((cell[0] + dx) as i32 as u32, (cell[1] + dy) as i32 as u32, 0, seed);
        let g = gradients[(h >> 27) as usize];
        g[0] * (t[0] - dx) + g[1] * (t[1] - dy)
    };
    let mix = |a: f32, b: f32, k: f32| a + (b - a) * k;
    let (wx, wy) = (fade(t[0]), fade(t[1]));
    mix(mix(corner(0.0, 0.0), corner(1.0, 0.0), wx), mix(corner(0.0, 1.0), corner(1.0, 1.0), wx), wy)
        * 1.4
}

/// Ridged multifractal noise, roughly 0..2. `p` is in units of the broadest
/// ridge.
fn ridged(mut p: [f32; 2], octaves: u32, gradients: &Gradients) -> f32 {
    let (mut sum, mut amplitude, mut weight) = (0.0, 1.0, 1.0);
    for octave in 0..octaves {
        let n = gradient_noise(p, 900 + octave, gradients);
        // Fold at zero and square: a sharp crest where the noise crossed
        // zero, a broad floor where it was far from it.
        let mut signal = 1.0 - n.abs();
        signal *= signal * weight;
        // The next, finer layer only shows where this one is high: rough
        // crests, smooth valleys.
        weight = (signal * RIDGE_GAIN).clamp(0.0, 1.0);
        sum += signal * amplitude;
        amplitude *= 0.5;
        // Double the frequency, and turn 37 degrees so the layers' grids do
        // not line up into visible streaks.
        p = [
            2.0 * (0.798_635_5 * p[0] - 0.601_815 * p[1]) + 17.3,
            2.0 * (0.601_815 * p[0] + 0.798_635_5 * p[1]) - 5.1,
        ];
    }
    sum
}

fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Un-normalised height at `p`, metres from the map's centre.
fn shape(p: [f32; 2], octaves: u32, gradients: &Gradients) -> f32 {
    let distance = (p[0] * p[0] + p[1] * p[1]).sqrt();
    // Where the range is: a ring, its inner edge wandering in and out.
    let wander = gradient_noise([p[0] / 21_000.0, p[1] / 21_000.0], 800, gradients);
    let rise = smoothstep(PLAIN_RADIUS, RANGE_FULL, distance + EDGE_WANDER * wander);
    let fall = 1.0 - smoothstep(0.38 * EXTENT, 0.48 * EXTENT, distance);
    let range = rise * fall;
    if range <= 0.0 {
        return 0.0;
    }
    // Broad swells, so some stretches are high massif and others low passes.
    let swell = 0.7 + 0.3 * gradient_noise([p[0] / 30_000.0, p[1] / 30_000.0], 801, gradients);
    let ridges = ridged([p[0] / FEATURE_SIZE, p[1] / FEATURE_SIZE], octaves, gradients);
    range * swell * (ridges - VALLEY_CUT * (1.0 - 0.6 * range)).max(0.0)
}

/// Bake the heightmap and upload it with its full mip chain.
pub fn bake(device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<Baked> {
    let started = Instant::now();
    let n = SIZE;
    let texel = EXTENT / n as f32;

    let gradients: Gradients = std::array::from_fn(|i| {
        let angle = (i as f32 + 0.5) / 32.0 * std::f32::consts::TAU;
        [angle.cos(), angle.sin()]
    });
    // Stop adding layers once a wave would span fewer than three texels;
    // finer ones cannot be stored and would only alias.
    let mut octaves = 1;
    while FEATURE_SIZE / (1 << octaves) as f32 > 3.0 * texel {
        octaves += 1;
    }

    let mut height = fill(n, n, 1, |x, y, _| {
        shape([(x - 0.5) * EXTENT, (y - 0.5) * EXTENT], octaves, &gradients)
    });
    let peak = height.iter().copied().fold(0.0f32, f32::max).max(1.0e-6);
    for h in &mut height {
        *h *= PEAK_HEIGHT / peak;
    }

    // Mip chains: heights average, the bound takes the maximum.
    let mut heights = vec![height.clone()];
    let mut maxima = vec![height];
    let mut extent = n;
    while extent > 1 {
        let half_extent = extent / 2;
        let (prev_h, prev_m) = (heights.last().unwrap(), maxima.last().unwrap());
        let mut level_h = vec![0.0f32; half_extent * half_extent];
        let mut level_m = vec![0.0f32; half_extent * half_extent];
        for y in 0..half_extent {
            for x in 0..half_extent {
                let at = |dx: usize, dy: usize| (y * 2 + dy) * extent + x * 2 + dx;
                let four = [at(0, 0), at(1, 0), at(0, 1), at(1, 1)];
                level_h[y * half_extent + x] = four.iter().map(|&i| prev_h[i]).sum::<f32>() / 4.0;
                level_m[y * half_extent + x] =
                    four.iter().map(|&i| prev_m[i]).fold(0.0f32, f32::max);
            }
        }
        heights.push(level_h);
        maxima.push(level_m);
        extent = half_extent;
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("terrain heightmap"),
        size: wgpu::Extent3d { width: n as u32, height: n as u32, depth_or_array_layers: 1 },
        mip_level_count: heights.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, (level_h, level_m)) in heights.iter().zip(&maxima).enumerate() {
        let extent = (n >> level).max(1);
        let bound = spread_max(level_m, extent, BOUND_REACH);
        let texels: Vec<u16> = level_h
            .iter()
            .zip(&bound)
            .flat_map(|(&h, &b)| {
                let coarse = (h / COARSE_STEP).round() * COARSE_STEP;
                // Rounded up, so the stored bound is never below the true one.
                let bound = (b / COARSE_STEP).ceil() * COARSE_STEP;
                [half(coarse), half(h - coarse), half(bound), 0]
            })
            .collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&texels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(extent as u32 * 8),
                rows_per_image: Some(extent as u32),
            },
            wgpu::Extent3d { width: extent as u32, height: extent as u32, depth_or_array_layers: 1 },
        );
    }
    log::info!(
        "baked {n}^2 terrain ({octaves} octaves, {texel:.0} m per texel) in {:.0} ms",
        started.elapsed().as_secs_f32() * 1000.0
    );
    vec![Baked {
        binding: BINDING,
        dimension: wgpu::TextureViewDimension::D2,
        view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
    }]
}

/// Replace every value with the maximum over the square reaching `reach`
/// texels each way, edges clamped. Done as two one-dimensional passes.
fn spread_max(field: &[f32], extent: usize, reach: usize) -> Vec<f32> {
    let pass = |src: &[f32], stride_major: usize, stride_minor: usize| -> Vec<f32> {
        let mut out = vec![0.0f32; src.len()];
        for major in 0..extent {
            for minor in 0..extent {
                let lo = minor.saturating_sub(reach);
                let hi = (minor + reach).min(extent - 1);
                let mut best = 0.0f32;
                for k in lo..=hi {
                    best = best.max(src[major * stride_major + k * stride_minor]);
                }
                out[major * stride_major + minor * stride_minor] = best;
            }
        }
        out
    };
    let rows = pass(field, extent, 1);
    pass(&rows, 1, extent)
}
