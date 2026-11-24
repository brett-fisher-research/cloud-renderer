// The sky: Hillaire's 2020 atmosphere model.
//
// Answers one question: how much sunlight does the air scatter towards the
// eye along a stretch of ray, and how much of what lies behind survives?
// Two tables baked by src/atmosphere.rs do the expensive part.

// --- The atmosphere (kilometres) — must match src/atmosphere.rs -------------
const R_GROUND: f32 = 6360.0;
const R_TOP: f32 = 6460.0;
const RAYLEIGH: vec3<f32> = vec3<f32>(5.802e-3, 13.558e-3, 33.1e-3);
const RAYLEIGH_H: f32 = 8.0;
const MIE_SCATTER: f32 = 3.996e-3;
const MIE_EXTINCTION: f32 = 4.40e-3;
const MIE_H: f32 = 1.2;
const MIE_G: f32 = 0.80;
const OZONE: vec3<f32> = vec3<f32>(0.650e-3, 1.881e-3, 0.085e-3);


@group(0) @binding(5) var transmittance_table: texture_2d<f32>;
@group(0) @binding(6) var multiscatter_table: texture_2d<f32>;

// Where in the transmittance table to look, for a point at distance `r` from
// the planet's centre and a direction whose cosine to straight-up is `mu`.
fn transmittance_uv(r: f32, mu: f32) -> vec2<f32> {
    let h = sqrt(R_TOP * R_TOP - R_GROUND * R_GROUND);
    let rho = sqrt(max((r - R_GROUND) * (r + R_GROUND), 0.0));
    let disc = r * r * (mu * mu - 1.0) + R_TOP * R_TOP;
    let d = max(-r * mu + sqrt(max(disc, 0.0)), 0.0);
    let d_min = R_TOP - r;
    let d_max = rho + h;
    return vec2<f32>((d - d_min) / (d_max - d_min), rho / h);
}

// The fraction of sunlight, per colour, that reaches a point through the
// atmosphere — zero once the planet itself is in the way.
fn sun_transmittance(r: f32, mu_sun: f32) -> vec3<f32> {
    // Cosine of the angle down to the horizon as seen from this altitude.
    let horizon = -sqrt(max(1.0 - (R_GROUND / r) * (R_GROUND / r), 0.0));
    // Fade over the sun's own angular radius: a sunset, not a switch.
    let visible = smoothstep(horizon - 0.0047, horizon + 0.0047, mu_sun);
    let t = textureSampleLevel(transmittance_table, table_sampler, transmittance_uv(r, mu_sun), 0.0).rgb;
    return t * visible;
}

struct Scatter {
    light: vec3<f32>,    // sunlight scattered towards the eye along the ray
    through: vec3<f32>,  // fraction of what lies behind that survives
};

// March the atmosphere along a view ray from distance t0 to t1 (km).
//
// The ray is described without vectors: it starts at radius `r0`, `mu` is
// the cosine between it and straight-up there, `nu` the cosine between it
// and the sun, and `mu_sun0` the sun's cosine to straight-up at the origin.
// Those four numbers give the altitude and sun angle at any distance along
// it, which is all the two tables need.
//
// `lit` is how much of the sun reaches the air at t0 and at t1, 0..1, blended
// in between: 1 for open air, less where something solid (a mountain range)
// shades it. Shaded air still glows with light scattered in from elsewhere,
// only not with direct sun.
fn atmosphere(r0: f32, mu: f32, nu: f32, mu_sun0: f32, t0: f32, t1: f32, steps: i32, lit: vec2<f32>) -> Scatter {
    var out: Scatter;
    out.light = vec3<f32>(0.0);
    out.through = vec3<f32>(1.0);

    // Rayleigh scatters equally forwards and backwards; Mie mostly forwards,
    // which is the bright haze around the sun.
    let phase_rayleigh = 3.0 / (16.0 * PI) * (1.0 + nu * nu);
    let g = MIE_G;
    let phase_mie = 3.0 / (8.0 * PI) * ((1.0 - g * g) * (1.0 + nu * nu))
        / ((2.0 + g * g) * pow(1.0 + g * g - 2.0 * g * nu, 1.5));

    let n = f32(steps);
    for (var i = 0; i < steps; i++) {
        // Quadratic spacing: short steps near the start, where the air is
        // thickest and the result changes fastest.
        let a = f32(i) / n;
        let b = f32(i + 1) / n;
        let ta = t0 + (t1 - t0) * a * a;
        let tb = t0 + (t1 - t0) * b * b;
        let t = 0.5 * (ta + tb);
        let dt = tb - ta;

        let r = sqrt(r0 * r0 + t * t + 2.0 * r0 * mu * t);
        let mu_sun = clamp((r0 * mu_sun0 + t * nu) / r, -1.0, 1.0);
        let altitude = max(r - R_GROUND, 0.0);

        let rayleigh = RAYLEIGH * exp(-altitude / RAYLEIGH_H);
        let mie = exp(-altitude / MIE_H);
        let ozone = OZONE * max(1.0 - abs(altitude - 25.0) / 15.0, 0.0);
        let scatter = rayleigh + MIE_SCATTER * mie;
        let extinction = rayleigh + MIE_EXTINCTION * mie + ozone;

        let direct = sun_transmittance(r, mu_sun) * mix(lit.x, lit.y, (f32(i) + 0.5) / n);
        let bounced = textureSampleLevel(
            multiscatter_table, table_sampler,
            vec2<f32>(mu_sun * 0.5 + 0.5, altitude / (R_TOP - R_GROUND)), 0.0).rgb;

        let source = direct * (rayleigh * phase_rayleigh + MIE_SCATTER * mie * phase_mie)
            + bounced * scatter;
        // Integrate the step exactly, assuming the air is uniform within it.
        // More accurate than `source * dt` when a step is optically thick.
        let step_through = exp(-extinction * dt);
        out.light += out.through * (source - source * step_through) / extinction;
        out.through *= step_through;
    }
    return out;
}
