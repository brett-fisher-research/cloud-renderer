//! Tiling 3D noise, baked into textures once at startup.
//!
//! Graphics note: the shader's fBm costs five octaves of eight hashes and a
//! trilinear blend *per sample*, and the raymarcher takes hundreds of samples
//! per pixel. Baking that field into a 3D texture turns the whole thing into
//! one filtered fetch, which the texture units do in fixed-function hardware
//! while the shader's ALUs get on with something else.
//!
//! The field has to *tile* for this to work: one cube of noise repeats to
//! cover the sky. That falls out of hashing each octave's lattice coordinates
//! modulo that octave's frequency, so the cell at the far edge is the same
//! cell as the one at the near edge.

/// Hash a wrapped 3D lattice cell to a value in [0,1).
/// Integer mixing rather than float tricks: it has to agree with nothing, it
/// only has to be well scrambled and deterministic.
fn hash(x: u32, y: u32, z: u32) -> f32 {
    let mut h = x
        .wrapping_mul(0x8da6_b343)
        ^ y.wrapping_mul(0xd8163_841u32)
        ^ z.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    h = h.wrapping_mul(0x2970_5aa3);
    h ^= h >> 15;
    (h >> 8) as f32 / 16_777_216.0
}

/// Smooth value noise on a lattice of `freq` cells per tile, at a point given
/// in tile units (0..1 spans the whole tile).
fn value_noise(p: [f32; 3], freq: u32) -> f32 {
    let f = freq as f32;
    let x = [p[0] * f, p[1] * f, p[2] * f];
    let i = [x[0].floor(), x[1].floor(), x[2].floor()];
    let t = [x[0] - i[0], x[1] - i[1], x[2] - i[2]];
    // Same smoothstep weights the shader used, so the baked field has the
    // same character as the procedural one.
    let w = [
        t[0] * t[0] * (3.0 - 2.0 * t[0]),
        t[1] * t[1] * (3.0 - 2.0 * t[1]),
        t[2] * t[2] * (3.0 - 2.0 * t[2]),
    ];

    let cell = |dx: u32, dy: u32, dz: u32| -> f32 {
        // rem_euclid keeps the wrap correct for negative coordinates.
        let ix = (i[0] as i64 + dx as i64).rem_euclid(freq as i64) as u32;
        let iy = (i[1] as i64 + dy as i64).rem_euclid(freq as i64) as u32;
        let iz = (i[2] as i64 + dz as i64).rem_euclid(freq as i64) as u32;
        hash(ix, iy, iz)
    };

    let mix = |a: f32, b: f32, k: f32| a + (b - a) * k;
    let x00 = mix(cell(0, 0, 0), cell(1, 0, 0), w[0]);
    let x10 = mix(cell(0, 1, 0), cell(1, 1, 0), w[0]);
    let x01 = mix(cell(0, 0, 1), cell(1, 0, 1), w[0]);
    let x11 = mix(cell(0, 1, 1), cell(1, 1, 1), w[0]);
    mix(mix(x00, x10, w[1]), mix(x01, x11, w[1]), w[2])
}

/// Bake a tiling octave stack into a `size`-cubed single-channel texture.
///
/// Frequency doubles with each octave, the first spanning the whole tile, and
/// `amplitudes` says how loud each one is. Plain fBm halves the amplitude each
/// time; the base texture does not, and the reason is worth knowing.
///
/// The tile has to be several times larger than the clouds, or its repeat is
/// obvious — a sky of identical puffs on a grid. But if the *loudest* octave
/// is the one spanning that whole big tile, the clouds come out as wide flat
/// sheets rather than the compact puffy ones a 1500 m layer should have.
///
/// So the stack is peaked rather than falling: it is ordinary fBm from the
/// feature size downwards, with two quieter octaves added *above* it. Those
/// two do the job of a weather map — deciding where cloud gathers and where
/// the sky stays clear — while the feature-size octave stays the loudest and
/// keeps the clouds the right shape.
pub fn bake_fbm(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    size: u32,
    amplitudes: &[f32],
) -> wgpu::Texture {
    let started = std::time::Instant::now();
    let n = size as usize;
    let mut data = vec![0u8; n * n * n];

    // One thread per slab of z-slices. Each voxel is independent, so this is
    // the easiest possible parallelism and it takes the bake from seconds to
    // a fraction of one.
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(4)
        .min(n);
    let slice = n * n;
    std::thread::scope(|scope| {
        for (t, chunk) in data.chunks_mut(slice * n.div_ceil(threads)).enumerate() {
            let z0 = t * n.div_ceil(threads);
            scope.spawn(move || {
                bake_slab(chunk, z0, size, amplitudes);
            });
        }
    });

    // Flatten the field's histogram.
    //
    // Summed octaves pile up around the middle: the field is roughly a bell
    // curve, not a flat spread, so most of its values sit near 0.5 and the
    // ends are nearly empty. `cloud_density` then slices it at
    // `1 - coverage`, which means the dial's effect is wildly uneven — a
    // little past the middle wipes the sky clear.
    //
    // Remapping every value through the field's own cumulative distribution
    // spreads it flat across 0..1. The shape of the clouds is untouched,
    // because the remap is monotonic: it only relabels the levels, never
    // reorders them. What changes is that the dial becomes honest —
    // `coverage` 0.45 now means 45% of the volume is above the threshold.
    equalise(&mut data);

    // Mip levels: the same field at half resolution, then a quarter, and so
    // on down to a single voxel, each one the average of eight voxels above
    // it.
    //
    // This is what stops the clouds fizzing in the distance. Near the horizon
    // the raymarch takes steps hundreds of metres long through a field whose
    // finest detail is tens of metres across — it samples far too sparsely to
    // see that detail honestly, and what it gets instead is aliasing. Picking
    // a mip level to match the step length means the fine detail has already
    // been averaged away before it can alias, which is the same bargain an
    // ordinary 2D texture strikes when it recedes into the distance.
    let mips = mip_chain(data, size);

    let texture = upload_3d(device, queue, label, size, &mips);

    log::info!(
        "baked {label}: {size}^3, {} octaves, {} mips, {:.0} KiB, {:.0} ms on {threads} threads",
        amplitudes.len(),
        mips.len(),
        mips.iter().map(Vec::len).sum::<usize>() as f32 / 1024.0,
        started.elapsed().as_secs_f32() * 1000.0
    );
    texture
}

/// Create a 3D texture and upload every mip level of a baked field into it.
///
/// Shared by both bakes: the only thing that differs between a value-noise
/// field and a Worley one is how the voxels were computed.
fn upload_3d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    size: u32,
    mips: &[Vec<u8>],
) -> wgpu::Texture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: size,
        },
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
            wgpu::Extent3d {
                width: extent,
                height: extent,
                depth_or_array_layers: extent,
            },
        );
    }
    texture
}

/// Build the mip chain: each level is the previous one halved in every
/// direction, every voxel the average of the eight it replaces.
fn mip_chain(base: Vec<u8>, size: u32) -> Vec<Vec<u8>> {
    let mut levels = vec![base];
    let mut extent = size as usize;

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
                                let sx = x * 2 + dx;
                                let sy = y * 2 + dy;
                                let sz = z * 2 + dz;
                                sum += previous[(sz * extent + sy) * extent + sx] as u32;
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

/// Remap a baked field through its own CDF, so its values are spread evenly
/// over 0..255 instead of bunched in the middle.
fn equalise(data: &mut [u8]) {
    let mut histogram = [0u64; 256];
    for &v in data.iter() {
        histogram[v as usize] += 1;
    }

    // Running total, scaled to 0..255: the fraction of the field at or below
    // each level becomes that level's new value.
    let total = data.len() as f64;
    let mut lookup = [0u8; 256];
    let mut running = 0u64;
    for (level, &count) in histogram.iter().enumerate() {
        running += count;
        // Centre of this level's slice of the distribution, so the brightest
        // level maps near 255 rather than exactly onto it.
        let cumulative = (running as f64 - count as f64 * 0.5) / total;
        lookup[level] = (cumulative * 255.0).round().clamp(0.0, 255.0) as u8;
    }

    for v in data.iter_mut() {
        *v = lookup[*v as usize];
    }
}

/// Fill one slab of z-slices, starting at `z0`.
fn bake_slab(out: &mut [u8], z0: usize, size: u32, amplitudes: &[f32]) {
    let n = size as usize;
    for (i, voxel) in out.iter_mut().enumerate() {
        let z = z0 + i / (n * n);
        let y = (i / n) % n;
        let x = i % n;
        {
            {
                // Voxel centres, so the field is symmetric across the wrap.
                let p = [
                    (x as f32 + 0.5) / size as f32,
                    (y as f32 + 0.5) / size as f32,
                    (z as f32 + 0.5) / size as f32,
                ];
                let mut total = 0.0;
                let mut norm = 0.0;
                let mut freq = 1u32;
                for &amplitude in amplitudes {
                    total += amplitude * value_noise(p, freq);
                    norm += amplitude;
                    freq *= 2;
                }
                *voxel = ((total / norm).clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }
}

// ---------------------------------------------------------------------------
//  Worley (cellular) noise
//
//  Value noise answers "what is the smooth random number here". Worley asks a
//  different question: scatter one feature point per cell of a lattice, then
//  return the distance to the nearest one. Near a point the field is 0 and it
//  rises away from it, so *inverting* it gives a field of rounded bumps — and
//  a stack of those at several scales is the cauliflower surface that reads as
//  a cumulus cloud. Value-noise fBm cannot make that shape: its level sets are
//  smooth blobs, not bulges.
//
//  It tiles for the same reason the value noise does: cell indices wrap modulo
//  the octave's cell count, so the far face of the cube meets the near one.
// ---------------------------------------------------------------------------

/// One feature point per cell, laid out so cell (x,y,z) is at index
/// `(z * cells + y) * cells + x`. Positions are in tile units, 0..1.
fn feature_points(cells: u32, salt: u32) -> Vec<[f32; 3]> {
    let n = cells as usize;
    let mut points = Vec::with_capacity(n * n * n);
    for z in 0..cells {
        for y in 0..cells {
            for x in 0..cells {
                // Three decorrelated hashes of the same cell, one per axis.
                let jx = hash(x.wrapping_add(salt), y, z);
                let jy = hash(x, y.wrapping_add(salt ^ 0x9e37_79b9), z);
                let jz = hash(x, y, z.wrapping_add(salt ^ 0x85eb_ca6b));
                points.push([
                    (x as f32 + jx) / cells as f32,
                    (y as f32 + jy) / cells as f32,
                    (z as f32 + jz) / cells as f32,
                ]);
            }
        }
    }
    points
}

/// Distance from `p` (tile units) to the nearest feature point, scaled so 1.0
/// is roughly a whole cell. Searches the 27 cells around `p`, which is enough:
/// no point can be closer than one that lies outside them.
fn worley(p: [f32; 3], cells: u32, points: &[[f32; 3]]) -> f32 {
    let c = cells as i32;
    let base = [
        (p[0] * cells as f32).floor() as i32,
        (p[1] * cells as f32).floor() as i32,
        (p[2] * cells as f32).floor() as i32,
    ];

    let mut best = f32::MAX;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let ix = (base[0] + dx).rem_euclid(c);
                let iy = (base[1] + dy).rem_euclid(c);
                let iz = (base[2] + dz).rem_euclid(c);
                let q = points[((iz * c + iy) * c + ix) as usize];
                // The neighbour may have wrapped to the far side of the tile;
                // take the shortest way round on each axis.
                let mut d2 = 0.0;
                for axis in 0..3 {
                    let mut d = q[axis] - p[axis];
                    if d > 0.5 {
                        d -= 1.0;
                    } else if d < -0.5 {
                        d += 1.0;
                    }
                    d2 += d * d;
                }
                best = best.min(d2);
            }
        }
    }
    (best.sqrt() * cells as f32).min(1.0)
}

/// Bake a tiling stack of Worley octaves, cell counts doubling from `cells`.
///
/// Stored as the *distance* field, 0 on a feature point and 1 far from one,
/// because that is the form the shader's Perlin-Worley remap wants. The same
/// histogram flattening as [`bake_fbm`] is applied, so a threshold against it
/// means roughly what it says.
pub fn bake_worley(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    size: u32,
    cells: u32,
    amplitudes: &[f32],
) -> wgpu::Texture {
    let started = std::time::Instant::now();
    let n = size as usize;
    let mut data = vec![0u8; n * n * n];

    // Hashing a cell is the same work for every voxel that falls in it, so the
    // feature points are laid out once per octave rather than per sample.
    let octaves: Vec<(u32, Vec<[f32; 3]>, f32)> = amplitudes
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            let c = cells << i;
            (c, feature_points(c, 0x51_u32.wrapping_mul(i as u32 + 1)), a)
        })
        .collect();
    let norm: f32 = amplitudes.iter().sum();

    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(4)
        .min(n);
    let slice = n * n;
    let octaves = &octaves;
    std::thread::scope(|scope| {
        for (t, chunk) in data.chunks_mut(slice * n.div_ceil(threads)).enumerate() {
            let z0 = t * n.div_ceil(threads);
            scope.spawn(move || {
                for (i, voxel) in chunk.iter_mut().enumerate() {
                    let z = z0 + i / (n * n);
                    let y = (i / n) % n;
                    let x = i % n;
                    let p = [
                        (x as f32 + 0.5) / size as f32,
                        (y as f32 + 0.5) / size as f32,
                        (z as f32 + 0.5) / size as f32,
                    ];
                    let mut total = 0.0;
                    for (c, points, amplitude) in octaves.iter() {
                        total += amplitude * worley(p, *c, points);
                    }
                    *voxel = ((total / norm).clamp(0.0, 1.0) * 255.0) as u8;
                }
            });
        }
    });

    equalise(&mut data);
    let mips = mip_chain(data, size);
    let texture = upload_3d(device, queue, label, size, &mips);

    log::info!(
        "baked {label}: {size}^3, {} worley octaves from {cells} cells, {} mips, {:.0} ms",
        amplitudes.len(),
        mips.len(),
        started.elapsed().as_secs_f32() * 1000.0
    );
    texture
}
