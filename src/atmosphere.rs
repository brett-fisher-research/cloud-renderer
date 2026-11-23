//! Precomputed atmosphere tables for the sky.
//!
//! Graphics note: the sky is blue because air molecules scatter short
//! wavelengths much more than long ones (*Rayleigh* scattering), hazy near
//! the horizon because of larger particles that scatter all colours mostly
//! forwards (*Mie* scattering), and red at sunset because by then the blue has
//! been scattered *out* of the long path the sunlight took to reach you.
//! Rendering that means integrating along each view ray, and at every point
//! on it asking two questions whose answers do not depend on the camera:
//!
//! 1. **How much sunlight survives to reach this point?** That depends only
//!    on the point's altitude and the angle to the sun — the *transmittance*
//!    table, 256x64.
//! 2. **How much light arrives here after bouncing more than once?** The
//!    same two inputs — the *multiple scattering* table, 32x32. Without it
//!    the sky is too dark and too saturated, most visibly at twilight.
//!
//! Both are baked here on the CPU in a few milliseconds and never change. The
//! method is Sébastien Hillaire's "A Scalable and Production Ready Sky and
//! Atmosphere Rendering Technique" (2020). The physical constants below are
//! duplicated at the top of `clouds.wgsl` and must match.
//!
//! Units in this file are kilometres.

use std::time::Instant;

const R_GROUND: f64 = 6360.0;
const R_TOP: f64 = 6460.0;
/// Scattering coefficients per km at sea level, for red, green and blue.
const RAYLEIGH: [f64; 3] = [5.802e-3, 13.558e-3, 33.1e-3];
/// Rayleigh density falls off exponentially with this scale height.
const RAYLEIGH_H: f64 = 8.0;
const MIE_SCATTER: f64 = 3.996e-3;
const MIE_EXTINCTION: f64 = 4.40e-3;
const MIE_H: f64 = 1.2;
/// Ozone only absorbs. It sits in a layer around 25 km and is what keeps the
/// zenith blue at sunset, when Rayleigh alone would turn it grey-yellow.
const OZONE: [f64; 3] = [0.650e-3, 1.881e-3, 0.085e-3];
const GROUND_ALBEDO: f64 = 0.3;

pub const TRANSMITTANCE_SIZE: (u32, u32) = (256, 64);
pub const MULTISCATTER_SIZE: (u32, u32) = (32, 32);

type V3 = [f64; 3];

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// (scattering, extinction) per km at altitude `h` km.
fn medium(h: f64) -> (V3, V3) {
    let rayleigh = (-h / RAYLEIGH_H).exp();
    let mie = (-h / MIE_H).exp();
    let ozone = (1.0 - (h - 25.0).abs() / 15.0).max(0.0);
    let mut scatter = [0.0; 3];
    let mut extinction = [0.0; 3];
    for c in 0..3 {
        scatter[c] = RAYLEIGH[c] * rayleigh + MIE_SCATTER * mie;
        extinction[c] = RAYLEIGH[c] * rayleigh + MIE_EXTINCTION * mie + OZONE[c] * ozone;
    }
    (scatter, extinction)
}

/// Distance along a ray to a sphere centred on the planet, or None.
/// `r` is the ray origin's distance from the centre, `mu` the cosine of the
/// angle between the ray and straight up. Returns the far root.
fn exit_distance(r: f64, mu: f64, radius: f64) -> Option<f64> {
    let disc = r * r * (mu * mu - 1.0) + radius * radius;
    (disc >= 0.0).then(|| -r * mu + disc.sqrt())
}

/// Near intersection with the ground, if the ray hits it.
fn ground_distance(r: f64, mu: f64) -> Option<f64> {
    let disc = r * r * (mu * mu - 1.0) + R_GROUND * R_GROUND;
    (disc >= 0.0 && mu < 0.0).then(|| (-r * mu - disc.sqrt()).max(0.0))
}

/// Table coordinates to (radius, cos zenith). The mapping (Bruneton's) puts
/// most of the table's resolution near the horizon, where transmittance
/// changes fastest. The inverse is `transmittance_uv` in the shader.
fn transmittance_params(u: f64, v: f64) -> (f64, f64) {
    let h = (R_TOP * R_TOP - R_GROUND * R_GROUND).sqrt();
    let rho = h * v;
    let r = (rho * rho + R_GROUND * R_GROUND).sqrt();
    let d_min = R_TOP - r;
    let d_max = rho + h;
    let d = d_min + u * (d_max - d_min);
    let mu = if d == 0.0 { 1.0 } else { (h * h - rho * rho - d * d) / (2.0 * r * d) };
    (r, mu.clamp(-1.0, 1.0))
}

fn transmittance_uv(r: f64, mu: f64) -> (f64, f64) {
    let h = (R_TOP * R_TOP - R_GROUND * R_GROUND).sqrt();
    let rho = (r * r - R_GROUND * R_GROUND).max(0.0).sqrt();
    let disc = r * r * (mu * mu - 1.0) + R_TOP * R_TOP;
    let d = (-r * mu + disc.max(0.0).sqrt()).max(0.0);
    let d_min = R_TOP - r;
    let d_max = rho + h;
    ((d - d_min) / (d_max - d_min), rho / h)
}

struct Table {
    width: usize,
    height: usize,
    data: Vec<[f32; 3]>,
}

impl Table {
    /// Bilinear lookup with clamped edges, matching the GPU sampler.
    fn sample(&self, u: f64, v: f64) -> V3 {
        let x = (u * self.width as f64 - 0.5).clamp(0.0, self.width as f64 - 1.0);
        let y = (v * self.height as f64 - 0.5).clamp(0.0, self.height as f64 - 1.0);
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (fx, fy) = (x - x0 as f64, y - y0 as f64);
        let mut out = [0.0; 3];
        for c in 0..3 {
            let at = |x: usize, y: usize| self.data[y * self.width + x][c] as f64;
            out[c] = (at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx) * (1.0 - fy)
                + (at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx) * fy;
        }
        out
    }
}

fn bake_transmittance() -> Table {
    const STEPS: usize = 40;
    let (width, height) = (TRANSMITTANCE_SIZE.0 as usize, TRANSMITTANCE_SIZE.1 as usize);
    let mut data = Vec::with_capacity(width * height);
    for j in 0..height {
        for i in 0..width {
            let (r, mu) = transmittance_params(
                (i as f64 + 0.5) / width as f64,
                (j as f64 + 0.5) / height as f64,
            );
            let length = exit_distance(r, mu, R_TOP).unwrap_or(0.0);
            let dt = length / STEPS as f64;
            let mut depth = [0.0; 3];
            for s in 0..STEPS {
                let t = (s as f64 + 0.5) * dt;
                let altitude = (r * r + t * t + 2.0 * r * mu * t).sqrt() - R_GROUND;
                let (_, extinction) = medium(altitude);
                for c in 0..3 {
                    depth[c] += extinction[c] * dt;
                }
            }
            data.push([(-depth[0]).exp() as f32, (-depth[1]).exp() as f32, (-depth[2]).exp() as f32]);
        }
    }
    Table { width, height, data }
}

fn bake_multiscatter(transmittance: &Table) -> Table {
    const SQRT_DIRECTIONS: usize = 8;
    const STEPS: usize = 20;
    let (width, height) = (MULTISCATTER_SIZE.0 as usize, MULTISCATTER_SIZE.1 as usize);
    let mut data = Vec::with_capacity(width * height);

    for j in 0..height {
        for i in 0..width {
            let sun_mu = (i as f64 + 0.5) / width as f64 * 2.0 - 1.0;
            let r = R_GROUND
                + ((j as f64 + 0.5) / height as f64 * (R_TOP - R_GROUND)).clamp(0.001, R_TOP - R_GROUND - 0.001);
            let origin: V3 = [0.0, r, 0.0];
            let sun: V3 = [(1.0 - sun_mu * sun_mu).max(0.0).sqrt(), sun_mu, 0.0];

            // Second-order light arriving here from every direction, and the
            // fraction of light that scatters again rather than escaping.
            let mut second_order = [0.0; 3];
            let mut rescatter = [0.0; 3];

            for a in 0..SQRT_DIRECTIONS {
                for b in 0..SQRT_DIRECTIONS {
                    let theta = std::f64::consts::TAU * (a as f64 + 0.5) / SQRT_DIRECTIONS as f64;
                    let cos_phi = 1.0 - 2.0 * (b as f64 + 0.5) / SQRT_DIRECTIONS as f64;
                    let sin_phi = (1.0 - cos_phi * cos_phi).max(0.0).sqrt();
                    let dir: V3 = [sin_phi * theta.cos(), cos_phi, sin_phi * theta.sin()];

                    let ground = ground_distance(r, dir[1]);
                    let length = ground.or_else(|| exit_distance(r, dir[1], R_TOP)).unwrap_or(0.0);
                    let dt = length / STEPS as f64;
                    let mut through = [1.0; 3];

                    let sun_light_at = |p: V3| -> V3 {
                        let pr = dot(p, p).sqrt();
                        let mu = dot(p, sun) / pr;
                        if ground_distance(pr, mu).is_some() {
                            return [0.0; 3]; // in the planet's shadow
                        }
                        let (u, v) = transmittance_uv(pr, mu);
                        transmittance.sample(u, v)
                    };

                    for s in 0..STEPS {
                        let t = (s as f64 + 0.5) * dt;
                        let p = [origin[0] + dir[0] * t, origin[1] + dir[1] * t, origin[2] + dir[2] * t];
                        let (scatter, extinction) = medium(dot(p, p).sqrt() - R_GROUND);
                        let sun_light = sun_light_at(p);
                        for c in 0..3 {
                            let step_through = (-extinction[c] * dt).exp();
                            // Light gathered over this step, integrated
                            // exactly for a constant medium within the step.
                            let gathered = through[c] * (1.0 - step_through) / extinction[c];
                            // Isotropic phase function: 1 / 4 pi.
                            second_order[c] += gathered * scatter[c] * sun_light[c]
                                / (4.0 * std::f64::consts::PI);
                            rescatter[c] += gathered * scatter[c];
                            through[c] *= step_through;
                        }
                    }

                    if ground.is_some() {
                        let p = [
                            origin[0] + dir[0] * length,
                            origin[1] + dir[1] * length,
                            origin[2] + dir[2] * length,
                        ];
                        let pr = dot(p, p).sqrt();
                        let n_dot_l = (dot(p, sun) / pr).max(0.0);
                        // Nudge off the surface so the shadow test passes.
                        let lifted = [p[0] * 1.0001, p[1] * 1.0001, p[2] * 1.0001];
                        let sun_light = sun_light_at(lifted);
                        for c in 0..3 {
                            second_order[c] += through[c] * sun_light[c] * n_dot_l * GROUND_ALBEDO
                                / std::f64::consts::PI;
                        }
                    }
                }
            }

            let n = (SQRT_DIRECTIONS * SQRT_DIRECTIONS) as f64;
            let mut texel = [0.0f32; 3];
            for c in 0..3 {
                // Averaging over the sphere and multiplying by its 4 pi solid
                // angle cancels the isotropic phase function's 1 / 4 pi.
                let l2 = second_order[c] / n * 4.0 * std::f64::consts::PI;
                let f = rescatter[c] / n;
                // Each further bounce is the previous one times `f`: a
                // geometric series, summed in closed form.
                texel[c] = (l2 / (1.0 - f)) as f32;
            }
            data.push(texel);
        }
    }
    Table { width, height, data }
}

/// IEEE half-float bits for a small non-negative number.
pub fn half(v: f32) -> u16 {
    let v = v.clamp(0.0, 65000.0);
    if v < 6.2e-5 {
        // Subnormal halves: plenty for a transmittance near zero.
        return (v / 5.960_464_5e-8).round() as u16;
    }
    let bits = v.to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = (bits >> 13) & 0x3ff;
    ((exponent as u16) << 10) | mantissa as u16
}

fn upload(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, table: &Table) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: table.width as u32,
        height: table.height as u32,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        // Half floats: transmittance spans many orders of magnitude at the
        // horizon, which 8 bits cannot hold.
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let texels: Vec<u16> = table
        .data
        .iter()
        .flat_map(|t| [half(t[0]), half(t[1]), half(t[2]), half(1.0)])
        .collect();
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&texels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size.width * 8),
            rows_per_image: Some(size.height),
        },
        size,
    );
    texture
}

/// Bake and upload both tables: (transmittance, multiple scattering).
pub fn bake(device: &wgpu::Device, queue: &wgpu::Queue) -> (wgpu::Texture, wgpu::Texture) {
    let started = Instant::now();
    let transmittance = bake_transmittance();
    let multiscatter = bake_multiscatter(&transmittance);
    let out = (
        upload(device, queue, "atmosphere transmittance", &transmittance),
        upload(device, queue, "atmosphere multiscatter", &multiscatter),
    );
    log::info!("baked atmosphere tables in {:.0} ms", started.elapsed().as_secs_f32() * 1000.0);
    out
}
