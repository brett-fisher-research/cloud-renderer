//! Tiling noise, baked into textures once at startup.
//!
//! Graphics note: procedural noise costs dozens of hash-and-interpolate
//! instructions per sample, and the raymarcher takes hundreds of samples per
//! pixel. Baking the field into a texture turns each sample into one filtered
//! fetch, done by the GPU's texture units in fixed-function hardware.
//!
//! Three textures, the standard set for this style of cloud (it comes from
//! Guerrilla's "Nubis" system for Horizon Zero Dawn):
//!
//! - **shape** (3D, 128^3): *Perlin-Worley* noise. Perlin noise is smooth and
//!   wispy; Worley (cellular) noise is a field of round bumps. Remapping one
//!   into the other gives a connected field with cauliflower bulges, which is
//!   what a cumulus cloud looks like.
//! - **detail** (3D, 64^3): three octaves of Worley only, used to erode the
//!   edges of the shape with small billows.
//! - **weather** (2D, 512^2): where clouds are (red) and how tall they grow
//!   (green). Read top-down; it places the clouds, the 3D fields sculpt them.
//!
//! Every field *tiles*: lattice coordinates are hashed modulo the frequency,
//! so the far edge of the cube matches the near edge and one cube can repeat
//! to cover the sky.

use std::time::Instant;

/// Scramble a lattice cell to 32 well-mixed bits.
fn hash_u32(x: u32, y: u32, z: u32, seed: u32) -> u32 {
    let mut h = x.wrapping_mul(0x8da6_b343)
        ^ y.wrapping_mul(0xd816_3841)
        ^ z.wrapping_mul(0xcb1a_b31f)
        ^ seed.wrapping_mul(0x9e37_79b9);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x2970_5aa3);
    h ^= h >> 15;
    h
}

fn unit(h: u32) -> f32 {
    (h >> 8) as f32 / 16_777_216.0
}

fn wrap(i: i32, freq: u32) -> u32 {
    i.rem_euclid(freq as i32) as u32
}

/// Tiling Perlin (gradient) noise, `freq` cells per tile, `p` in tile units.
/// Returns roughly -1..1.
fn perlin(p: [f32; 3], freq: u32, seed: u32) -> f32 {
    let f = freq as f32;
    let x = [p[0] * f, p[1] * f, p[2] * f];
    let i = [x[0].floor() as i32, x[1].floor() as i32, x[2].floor() as i32];
    let t = [x[0] - i[0] as f32, x[1] - i[1] as f32, x[2] - i[2] as f32];
    // Quintic fade: zero first and second derivative at the lattice points,
    // so no creases show where cells meet.
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let w = [fade(t[0]), fade(t[1]), fade(t[2])];

    // Each lattice corner owns a pseudo-random gradient; the noise is the
    // blend of each corner's gradient dotted with the offset to the point.
    let corner = |dx: i32, dy: i32, dz: i32| -> f32 {
        let h = hash_u32(
            wrap(i[0] + dx, freq),
            wrap(i[1] + dy, freq),
            wrap(i[2] + dz, freq),
            seed,
        );
        // A random direction on the sphere from two hashed numbers.
        let z = unit(h) * 2.0 - 1.0;
        let a = unit(hash_u32(h, 1, 2, 3)) * std::f32::consts::TAU;
        let r = (1.0 - z * z).max(0.0).sqrt();
        let g = [r * a.cos(), r * a.sin(), z];
        g[0] * (t[0] - dx as f32) + g[1] * (t[1] - dy as f32) + g[2] * (t[2] - dz as f32)
    };

    let mix = |a: f32, b: f32, k: f32| a + (b - a) * k;
    let x00 = mix(corner(0, 0, 0), corner(1, 0, 0), w[0]);
    let x10 = mix(corner(0, 1, 0), corner(1, 1, 0), w[0]);
    let x01 = mix(corner(0, 0, 1), corner(1, 0, 1), w[0]);
    let x11 = mix(corner(0, 1, 1), corner(1, 1, 1), w[0]);
    mix(mix(x00, x10, w[1]), mix(x01, x11, w[1]), w[2]) * 1.5
}

/// Perlin octaves: each doubles the frequency and halves the amplitude.
fn perlin_fbm(p: [f32; 3], freq: u32, octaves: u32, seed: u32) -> f32 {
    let (mut sum, mut amp, mut norm, mut f) = (0.0, 1.0, 0.0, freq);
    for o in 0..octaves {
        sum += amp * perlin(p, f, seed + o);
        norm += amp;
        amp *= 0.5;
        f *= 2;
    }
    sum / norm
}

/// Tiling Worley noise, inverted: 1 at a feature point, falling to 0 a cell
/// away. One random point per cell; the value is the distance to the nearest.
fn worley(p: [f32; 3], cells: u32, seed: u32) -> f32 {
    let f = cells as f32;
    let x = [p[0] * f, p[1] * f, p[2] * f];
    let i = [x[0].floor() as i32, x[1].floor() as i32, x[2].floor() as i32];
    let mut best = f32::MAX;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let c = [i[0] + dx, i[1] + dy, i[2] + dz];
                let h = hash_u32(wrap(c[0], cells), wrap(c[1], cells), wrap(c[2], cells), seed);
                let px = c[0] as f32 + unit(h);
                let py = c[1] as f32 + unit(hash_u32(h, 7, 0, 0));
                let pz = c[2] as f32 + unit(hash_u32(h, 0, 13, 0));
                let d = (px - x[0]).powi(2) + (py - x[1]).powi(2) + (pz - x[2]).powi(2);
                best = best.min(d);
            }
        }
    }
    1.0 - best.sqrt().min(1.0)
}

/// Three Worley octaves, weighted towards the coarsest.
fn worley_fbm(p: [f32; 3], cells: u32, seed: u32) -> f32 {
    worley(p, cells, seed) * 0.625
        + worley(p, cells * 2, seed + 1) * 0.25
        + worley(p, cells * 4, seed + 2) * 0.125
}

fn remap(v: f32, lo: f32, hi: f32, new_lo: f32, new_hi: f32) -> f32 {
    new_lo + (v - lo) / (hi - lo) * (new_hi - new_lo)
}

/// Evaluate `f(x, y, z)` (in tile units) over a `w*h*d` grid on all cores.
fn fill(w: usize, h: usize, d: usize, f: impl Fn(f32, f32, f32) -> f32 + Sync) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h * d];
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    let rows_per_chunk = (h * d).div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (chunk_index, chunk) in out.chunks_mut(rows_per_chunk * w).enumerate() {
            let f = &f;
            scope.spawn(move || {
                for (i, v) in chunk.iter_mut().enumerate() {
                    let flat = chunk_index * rows_per_chunk * w + i;
                    let (x, y, z) = (flat % w, (flat / w) % h, flat / (w * h));
                    *v = f(
                        (x as f32 + 0.5) / w as f32,
                        (y as f32 + 0.5) / h as f32,
                        (z as f32 + 0.5) / d as f32,
                    );
                }
            });
        }
    });
    out
}

/// Stretch a field so its 0.5th..99.5th percentiles span 0..255. Keeps the
/// field's shape (unlike equalising) but uses the whole 8-bit range.
fn stretch(field: &[f32]) -> Vec<u8> {
    let mut sorted: Vec<f32> = field.iter().copied().step_by(7).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let lo = sorted[sorted.len() / 200];
    let hi = sorted[sorted.len() - 1 - sorted.len() / 200];
    field
        .iter()
        .map(|v| (((v - lo) / (hi - lo)).clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

/// Remap a field through its own CDF so its values are spread evenly over
/// 0..255. A threshold at `1 - c` then passes exactly the fraction `c` of the
/// field — which is what makes the coverage dial mean what it says.
fn equalise(field: &[f32]) -> Vec<u8> {
    let mut order: Vec<u32> = (0..field.len() as u32).collect();
    order.sort_by(|&a, &b| field[a as usize].partial_cmp(&field[b as usize]).unwrap());
    let mut out = vec![0u8; field.len()];
    let n = field.len() as f32;
    for (rank, &i) in order.iter().enumerate() {
        out[i as usize] = ((rank as f32 + 0.5) / n * 255.0).round() as u8;
    }
    out
}

/// The cloud shape: Perlin-Worley, with higher Worley octaves remapped in.
pub fn bake_shape(device: &wgpu::Device, queue: &wgpu::Queue, size: u32) -> wgpu::Texture {
    let started = Instant::now();
    let n = size as usize;
    let field = fill(n, n, n, |x, y, z| {
        let p = [x, y, z];
        let perlin = (perlin_fbm(p, 4, 5, 100) * 0.5 + 0.5).clamp(0.0, 1.0);
        // Worley at four cells per tile sets the size of a cloud: tile / 4.
        let cells = worley_fbm(p, 4, 200);
        // Perlin-Worley: the cellular field becomes the *floor* of the Perlin
        // one, so the result keeps Perlin's connectedness but swells into a
        // round bump around every Worley feature point.
        let perlin_worley = remap(perlin, 0.0, 1.0, cells, 1.0);
        // A finer Worley stack carves the mid-size billows into it.
        let billows = worley_fbm(p, 8, 300);
        remap(perlin_worley, -(1.0 - billows), 1.0, 0.0, 1.0)
    });
    let texture = upload_3d(device, queue, "noise shape", size, &mip_chain(stretch(&field), n));
    log::info!("baked {size}^3 shape noise in {:.0} ms", started.elapsed().as_secs_f32() * 1000.0);
    texture
}

/// Edge detail: Worley only, three octaves.
pub fn bake_detail(device: &wgpu::Device, queue: &wgpu::Queue, size: u32) -> wgpu::Texture {
    let started = Instant::now();
    let n = size as usize;
    let field = fill(n, n, n, |x, y, z| worley_fbm([x, y, z], 3, 400));
    let texture = upload_3d(device, queue, "noise detail", size, &mip_chain(stretch(&field), n));
    log::info!("baked {size}^3 detail noise in {:.0} ms", started.elapsed().as_secs_f32() * 1000.0);
    texture
}

/// The weather map. Red: where cloud is. Green: how tall it grows there.
/// Both equalised, so thresholds on them are literal fractions of the sky.
pub fn bake_weather(device: &wgpu::Device, queue: &wgpu::Queue, size: u32) -> wgpu::Texture {
    let started = Instant::now();
    let n = size as usize;
    let coverage = equalise(&fill(n, n, 1, |x, y, _| {
        let p = [x, y, 0.37];
        // Perlin for the broad pattern, with a little Worley so that cloud
        // gathers into rounded cells rather than stringy bands.
        perlin_fbm(p, 6, 5, 500) * 0.5 + 0.5 + 0.35 * worley([x, y, 0.5], 8, 600)
    }));
    let height = equalise(&fill(n, n, 1, |x, y, _| perlin_fbm([x, y, 0.71], 3, 4, 700)));
    let texels: Vec<u8> = coverage.iter().zip(&height).flat_map(|(&c, &h)| [c, h]).collect();

    // 2D mip chain, two channels.
    let mut mips = vec![texels];
    let mut extent = n;
    while extent > 1 {
        let prev = mips.last().unwrap();
        let half = extent / 2;
        let mut level = vec![0u8; half * half * 2];
        for y in 0..half {
            for x in 0..half {
                for c in 0..2 {
                    let at = |sx: usize, sy: usize| prev[(sy * extent + sx) * 2 + c] as u32;
                    let sum = at(x * 2, y * 2)
                        + at(x * 2 + 1, y * 2)
                        + at(x * 2, y * 2 + 1)
                        + at(x * 2 + 1, y * 2 + 1);
                    level[(y * half + x) * 2 + c] = (sum / 4) as u8;
                }
            }
        }
        mips.push(level);
        extent = half;
    }

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("weather map"),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        mip_level_count: mips.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rg8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, texels) in mips.iter().enumerate() {
        let extent = (size >> level).max(1);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(extent * 2),
                rows_per_image: Some(extent),
            },
            wgpu::Extent3d { width: extent, height: extent, depth_or_array_layers: 1 },
        );
    }
    log::info!("baked {size}^2 weather map in {:.0} ms", started.elapsed().as_secs_f32() * 1000.0);
    texture
}

/// Create a 3D texture and upload every mip level of a baked field into it.
fn upload_3d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    size: u32,
    mips: &[Vec<u8>],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: size, height: size, depth_or_array_layers: size },
        mip_level_count: mips.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });

    for (level, voxels) in mips.iter().enumerate() {
        let extent = (size >> level).max(1);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            voxels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(extent),
                rows_per_image: Some(extent),
            },
            wgpu::Extent3d { width: extent, height: extent, depth_or_array_layers: extent },
        );
    }
    texture
}

/// Build the mip chain: each level is the previous one halved in every
/// direction, every voxel the average of the eight it replaces. Sampling a
/// coarser mip for longer ray steps is what stops distant cloud from fizzing.
fn mip_chain(base: Vec<u8>, size: usize) -> Vec<Vec<u8>> {
    let mut levels = vec![base];
    let mut extent = size;
    while extent > 1 {
        let previous = levels.last().unwrap();
        let half = extent / 2;
        let mut level = vec![0u8; half * half * half];
        for z in 0..half {
            for y in 0..half {
                for x in 0..half {
                    let mut sum = 0u32;
                    for dz in 0..2 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let at = ((z * 2 + dz) * extent + y * 2 + dy) * extent + x * 2 + dx;
                                sum += previous[at] as u32;
                            }
                        }
                    }
                    level[(z * half + y) * half + x] = (sum / 8) as u8;
                }
            }
        }
        levels.push(level);
        extent = half;
    }
    levels
}
