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

/// Bake `octaves` of tiling fBm into a `size`-cubed single-channel texture.
///
/// Frequencies double per octave and amplitudes halve, exactly as the shader's
/// fBm did. The lowest octave is one cell across the whole tile.
pub fn bake_fbm(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    size: u32,
    octaves: u32,
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
                bake_slab(chunk, z0, size, octaves);
            });
        }
    });

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: size,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::R8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &data,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size),
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: size,
        },
    );
    log::info!(
        "baked {label}: {size}^3, {octaves} octaves, {:.0} KiB, {:.0} ms on {threads} threads",
        (n * n * n) as f32 / 1024.0,
        started.elapsed().as_secs_f32() * 1000.0
    );
    texture
}

/// Fill one slab of z-slices, starting at `z0`.
fn bake_slab(out: &mut [u8], z0: usize, size: u32, octaves: u32) {
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
                let mut amplitude = 0.5;
                let mut freq = 1u32;
                for _ in 0..octaves {
                    total += amplitude * value_noise(p, freq);
                    norm += amplitude;
                    freq *= 2;
                    amplitude *= 0.5;
                }
                *voxel = ((total / norm).clamp(0.0, 1.0) * 255.0) as u8;
            }
        }
    }
}
