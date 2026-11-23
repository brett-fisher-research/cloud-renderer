// ===========================================================================
//  Volumetric clouds — single fragment shader.
//  Step 5: lit, animated clouds with a Henyey-Greenstein phase function.
// ===========================================================================

// ---------------------------------------------------------------------------
//  TUNABLES — change these and save; the renderer hot-reloads the file.
// ---------------------------------------------------------------------------

// --- Sky ---
// These are *linear* radiance values, not sRGB pixel values: everything is
// tone mapped and sRGB-encoded at the very end, so values can exceed 1.0 and
// the numbers look darker than the final image.
const SKY_ZENITH:  vec3<f32> = vec3<f32>(0.05, 0.17, 0.62); // straight up
const SKY_HORIZON: vec3<f32> = vec3<f32>(0.70, 0.90, 1.25); // at eye level
const GROUND:      vec3<f32> = vec3<f32>(0.055, 0.052, 0.042);
const HORIZON_FALLOFF: f32 = 2.2;   // bigger = thinner haze band

// --- Sun ---
const SUN_COLOR: vec3<f32> = vec3<f32>(1.0, 0.95, 0.85);
const SUN_GLOW:  f32 = 0.40;   // strength of the halo around the sun

// --- Cloud layer geometry (metres above the camera) ---
const CLOUD_BOTTOM: f32 = 1500.0;
const CLOUD_TOP:    f32 = 3000.0;
// Stop marching past this distance. Without it, a ray a hair above the
// horizon would need to march hundreds of kilometres.
const MAX_DISTANCE: f32 = 50000.0;

// --- Raymarching ---
const PRIMARY_STEPS: i32 = 64;      // samples along the view ray
// Give up once the cloud has blocked ~99% of the light behind it.
const MIN_TRANSMITTANCE: f32 = 0.01;
// Offset each pixel's first sample by up to this fraction of a step. With a
// fixed step count, neighbouring pixels sample at almost the same depths and
// the error lines up into visible arcs; nudging each pixel breaks those bands
// into fine noise, which the eye forgives far more readily. 0 disables it.
const JITTER: f32 = 1.0;

// --- Density ---
const DENSITY: f32 = 1.0;           // overall multiplier on cloud thickness
const EXTINCTION: f32 = 0.004;      // per metre: how fast cloud blocks light

// --- Noise shaping ---
// The size of the largest noise features is a live dial: `p.feature_size`.
const FBM_OCTAVES: i32 = 5;         // layers of detail; cost scales with this
const FBM_GAIN: f32 = 0.5;          // how much quieter each octave is
const FBM_LACUNARITY: f32 = 2.02;
const TILE_SCALE: f32 = 4.0;   // how much finer each octave is

// How much of the sky is cloud is a live dial: `p.coverage`.
// Where in the slab the cloud sits: fractions of the layer height.
const CLOUD_BASE_SOFTNESS: f32 = 0.22;  // fade-in above the bottom
const CLOUD_TOP_SOFTNESS:  f32 = 0.55;  // fade-out below the top

// --- Aerial perspective ---
// The air between you and a cloud both absorbs its light and scatters
// sunlight towards you in its place, so distant clouds trend towards the
// colour of the sky around them and finally vanish into it. Without this,
// cloud 40 km away stays as crisp and dark as cloud overhead, and all the
// perspective compression near the horizon piles up as a hard grey stripe.
// This is the distance at which ~63% of a cloud's colour has been replaced.
const AERIAL_DISTANCE: f32 = 35000.0;

// --- Wind ---
const WIND: vec3<f32> = vec3<f32>(-9.0, 0.0, -3.0);  // metres per second

// --- Lighting ---
// Steps taken towards the sun from each view sample to find the shadow.
const LIGHT_STEPS: i32 = 6;
const LIGHT_STEP_SIZE: f32 = 110.0;   // metres, length of the first step
const LIGHT_STEP_GROWTH: f32 = 1.55;  // each step is longer than the last
const SUN_LIGHT: vec3<f32> = vec3<f32>(1.0, 0.93, 0.80);
const SUN_LIGHT_INTENSITY: f32 = 3.2;

// Skylight landing on the cloud from the whole dome. Keeps shadowed sides
// blue-grey rather than black.
const AMBIENT_COLOR: vec3<f32> = vec3<f32>(0.42, 0.55, 0.78);
const AMBIENT_INTENSITY: f32 = 0.55;
// Undersides see less sky than tops, so dim ambient towards the cloud base.
const AMBIENT_BASE_DARKENING: f32 = 0.45;

// --- Phase function ---
// Water droplets scatter light mostly *forwards*, which is why a cloud with
// the sun behind it has a blazing silver lining. g is how directional that
// is: 0 scatters evenly, towards 1 is a tight forward beam, negative is
// backwards. Real droplets do both, so we mix a strong forward lobe with a
// weak backward one.
const PHASE_G_FORWARD:  f32 = 0.72;
const PHASE_G_BACKWARD: f32 = -0.28;
const PHASE_BLEND:      f32 = 0.62;   // weight of the forward lobe

// ---------------------------------------------------------------------------
//  Uniforms — must match `struct Uniforms` in src/camera.rs exactly.
// ---------------------------------------------------------------------------

struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    fov_tan: f32,       // tan(fov_y / 2)
    cam_forward: vec3<f32>,
    cam_right: vec3<f32>,
    cam_up: vec3<f32>,
    sun_dir: vec3<f32>, // points *towards* the sun
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// ---------------------------------------------------------------------------
//  Live dials — must match `struct Params` in src/params.rs exactly.
//
//  These are the four values the on-screen panel drives. They sit in their own
//  uniform buffer so moving a slider costs a 16-byte upload rather than a
//  shader recompile. Everything else in the block at the top of this file is
//  still a `const`, and still hot-reloads when you save the file.
// ---------------------------------------------------------------------------

struct Params {
    coverage: f32,         // 0 = clear sky, 1 = solid overcast
    feature_size: f32,     // metres: size of the largest noise features
    shadow_strength: f32,  // scales self-shadowing; low fakes multiple scattering
    exposure: f32,         // overall brightness, applied before tone mapping
};

@group(0) @binding(1) var<uniform> dials: Params;

// Eye height above the ground, in metres.
const CAMERA_HEIGHT: f32 = 2.0;

// ---------------------------------------------------------------------------
//  Vertex stage: a full-screen triangle.
//
//  Rather than upload geometry, we generate three clip-space points forming a
//  triangle twice the size of the screen. Everything outside the screen is
//  clipped away, so the fragment shader runs once per pixel with no vertex
//  buffer, no index buffer and no CPU-side geometry at all.
// ---------------------------------------------------------------------------

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,   // pixel position in [-1,1]
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let p = corners[idx];

    var out: VsOut;
    out.clip_pos = vec4<f32>(p, 0.0, 1.0);
    out.ndc = p;
    return out;
}

// ---------------------------------------------------------------------------
//  Sky
// ---------------------------------------------------------------------------

fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    // Angle between this ray and the sun, used for the halo below. Working in
    // angle space rather than the raw dot product keeps the falloff even:
    // near zero, a small change in angle is a tiny change in cosine.
    let cos_sun = clamp(dot(dir, u.sun_dir), -1.0, 1.0);
    let sun_angle = acos(cos_sun);

    // How low the sun is, for warm tinting.
    let dusk = 1.0 - clamp(u.sun_dir.y * 3.0, 0.0, 1.0);

    var col: vec3<f32>;
    if (dir.y < 0.0) {
        let haze = exp(dir.y * 6.0);       // 1 at the horizon -> 0 underfoot
        col = mix(GROUND, SKY_HORIZON * 0.8, haze * 0.65);
    } else {
        let t = pow(1.0 - dir.y, HORIZON_FALLOFF);
        col = mix(SKY_ZENITH, SKY_HORIZON, t);
        col = mix(col, col * vec3<f32>(1.35, 1.05, 0.80), t * dusk);
    }

    // Broad halo around the sun (forward-scattered light in the atmosphere).
    // There is no sun disc: the halo alone says where the sun is, and the
    // phase function does the rest once there is cloud in front of it.
    let glow = exp(-sun_angle * 9.0) * 0.8 + exp(-sun_angle * 1.6) * 0.2;
    col += SUN_COLOR * SUN_GLOW * glow * (0.5 + 1.2 * dusk);

    return col;
}

// ---------------------------------------------------------------------------
//  Cloud slab
// ---------------------------------------------------------------------------

// Where a ray enters and leaves the horizontal slab between CLOUD_BOTTOM and
// CLOUD_TOP. Returns (t_enter, t_exit); t_exit <= t_enter means "missed".
//
// Because the slab is horizontal, this is just two divisions: the ray reaches
// height h at t = (h - origin.y) / dir.y.
fn slab_intersect(origin: vec3<f32>, dir: vec3<f32>) -> vec2<f32> {
    let miss = vec2<f32>(0.0, -1.0);

    // Looking level or downwards from below the layer: never reaches it.
    if (dir.y <= 0.0001) {
        return miss;
    }

    let t_bottom = (CLOUD_BOTTOM - origin.y) / dir.y;
    let t_top    = (CLOUD_TOP    - origin.y) / dir.y;

    // Camera is below the layer, so bottom is always the near plane, but
    // min/max keeps this correct if the camera ever moves into or above it.
    var t_enter = max(min(t_bottom, t_top), 0.0);
    var t_exit  = min(max(t_bottom, t_top), MAX_DISTANCE);

    if (t_exit <= t_enter) {
        return miss;
    }
    return vec2<f32>(t_enter, t_exit);
}


// ---------------------------------------------------------------------------
//  Baked noise: octaves of the field precomputed into tiling 3D textures.
//  One filtered fetch in fixed-function hardware replaces several octaves of
//  hash-and-interpolate arithmetic.
// ---------------------------------------------------------------------------
@group(0) @binding(2) var noise_base: texture_3d<f32>;
@group(0) @binding(3) var noise_detail: texture_3d<f32>;
@group(0) @binding(4) var noise_sampler: sampler;

// The detail texture stands in for octaves 3-5, so its tile is a quarter of
// the feature size: the frequency octave 3 had.
const DETAIL_SCALE: f32 = 4.0;
const COARSE_WEIGHT: f32 = 0.75;    // summed amplitudes of octaves 1-2
const DETAIL_WEIGHT: f32 = 0.21875; // summed amplitudes of octaves 3-5

// Used only by the all-baked variants.
const BASE_DETAIL_SCALE: f32 = 7.0;
const DETAIL_STRENGTH: f32 = 0.35;

fn baked_noise(q: vec3<f32>, detail: bool) -> f32 {
    // textureSampleLevel, not textureSample: the raymarch loop has breaks in
    // it, so it is not uniform control flow, and implicit-derivative sampling
    // is illegal there. There are no mips anyway, so level 0 is all there is.
    var n = textureSampleLevel(noise_base, noise_sampler, q, 0.0).r;
    if (detail) {
        let d = textureSampleLevel(noise_detail, noise_sampler, q * BASE_DETAIL_SCALE, 0.0).r;
        n = n - (1.0 - n) * d * DETAIL_STRENGTH;
    }
    return n;
}

// ---------------------------------------------------------------------------
//  Noise
//
//  Clouds are shaped by *fractal noise*: a smooth random field summed at
//  several scales. One big smooth layer gives the overall blobs; each finer
//  layer adds wispier detail at half the amplitude. The sum is called fBm
//  (fractional Brownian motion).
//
//  This is all computed per sample, which is simple but expensive. The
//  roadmap replaces it with noise baked into a 3D texture by a compute shader.
// ---------------------------------------------------------------------------

// "Interleaved gradient noise": a cheap screen-space dither that looks evenly
// scattered rather than streaky. It depends only on the pixel coordinate, so
// it is stable from frame to frame — a time-varying jitter would make the
// image fizz instead of banding, which is worse without temporal filtering.
fn dither(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))));
}

// Hash a 3D grid cell to a pseudo-random number in [0,1).
// This is the standard "hash without sine" trick: multiply into a fractional
// range, mix the components together, and keep the fractional part.
fn hash13(p_in: vec3<f32>) -> f32 {
    var p = fract(p_in * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

// Smooth value noise: random values on an integer lattice, interpolated.
fn value_noise(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // Smoothstep the interpolation weights so the result has no visible
    // creases where it crosses a cell boundary.
    let w = f * f * (3.0 - 2.0 * f);

    let n000 = hash13(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash13(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash13(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash13(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash13(i + vec3<f32>(1.0, 1.0, 1.0));

    // Trilinear interpolation between the eight corners of the cell.
    let x00 = mix(n000, n100, w.x);
    let x10 = mix(n010, n110, w.x);
    let x01 = mix(n001, n101, w.x);
    let x11 = mix(n011, n111, w.x);
    return mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}

// Sum several octaves of value noise. Result is normalised to roughly [0,1].
fn fbm(p_in: vec3<f32>) -> f32 {
    var p = p_in;
    var amplitude = 0.5;
    var total = 0.0;
    var norm = 0.0;

    for (var i = 0; i < FBM_OCTAVES; i = i + 1) {
        total += amplitude * value_noise(p);
        norm += amplitude;
        p *= FBM_LACUNARITY;
        amplitude *= FBM_GAIN;
        // Offset each octave so the lattices do not line up and produce a
        // visible grid.
        p += vec3<f32>(17.3, 9.1, 23.7);
    }

    return total / norm;
}

// Density of the cloud medium at a world-space point.
fn cloud_density(p: vec3<f32>) -> f32 {
    // Height through the layer, 0 at the bottom and 1 at the top.
    let h = (p.y - CLOUD_BOTTOM) / (CLOUD_TOP - CLOUD_BOTTOM);
    if (h < 0.0 || h > 1.0) {
        return 0.0;
    }

    // Vertical profile: clouds have a flattish base and a soft, rounded top,
    // so fade density in quickly above the base and out gradually below the top.
    let profile = smoothstep(0.0, CLOUD_BASE_SOFTNESS, h)
                * (1.0 - smoothstep(CLOUD_TOP_SOFTNESS, 1.0, h));

    // Scroll the noise field to make the clouds drift. Moving the *sample
    // point* against the wind is the same as moving the clouds with it.
    let drift = WIND * u.time;
    let q = (p + drift) / (dials.feature_size * TILE_SCALE);

    let n = textureSampleLevel(noise_base, noise_sampler, q, 0.0).r;

    // Coverage: everything below the threshold becomes clear sky, and what
    // remains is rescaled back to [0,1] so the clouds keep their full density
    // instead of thinning out as coverage drops.
    let threshold = 1.0 - dials.coverage;
    let shaped = clamp((n - threshold) / max(dials.coverage, 0.001), 0.0, 1.0);

    return shaped * profile * DENSITY;
}

// ---------------------------------------------------------------------------
//  Lighting
// ---------------------------------------------------------------------------

// Henyey-Greenstein phase function: given the angle between the direction
// light came from and the direction we are looking, how much of it is
// scattered our way. Normalised so that g = 0 (even scattering) returns 1.0.
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (1.0 - g2) / pow(max(denom, 0.0001), 1.5);
}

// Two-lobe mix: a strong forward lobe plus a weak backward one.
fn cloud_phase(cos_theta: f32) -> f32 {
    return mix(
        henyey_greenstein(cos_theta, PHASE_G_BACKWARD),
        henyey_greenstein(cos_theta, PHASE_G_FORWARD),
        PHASE_BLEND,
    );
}

// How much sunlight reaches a point inside the cloud.
//
// We walk from the sample towards the sun, adding up how much cloud is in the
// way (the "optical depth"), then apply Beer's law once: exp(-depth). Steps
// get longer as they go, because distant blockers matter less than nearby
// ones and this buys depth cheaply. Six steps is enough to read as shadowing.
fn light_march(p: vec3<f32>) -> f32 {
    var optical_depth = 0.0;
    var t = 0.0;
    var step = LIGHT_STEP_SIZE;

    for (var i = 0; i < LIGHT_STEPS; i = i + 1) {
        // Sample at the middle of the step.
        let sp = p + u.sun_dir * (t + step * 0.5);
        optical_depth += cloud_density(sp) * step;
        t += step;
        step *= LIGHT_STEP_GROWTH;
    }

    return exp(-optical_depth * EXTINCTION * dials.shadow_strength);
}

// Result of marching the cloud layer: how much light got through, and how
// much the cloud itself added.
struct CloudResult {
    transmittance: f32,  // 1 = clear sky, 0 = fully opaque
    scattered: vec3<f32>,
};

// March the view ray through the slab, accumulating opacity front to back.
//
// Beer's law: a medium of extinction `s` over a distance `dt` lets through
// exp(-s * dt) of the light. Multiplying that factor over every step gives
// the total transmittance. The light the cloud *adds* over a step is
// (1 - exp(-s*dt)) — the fraction it absorbed from behind, re-emitted as
// scattered light — weighted by how much of it survives back to the eye.
fn march_clouds(
    origin: vec3<f32>,
    dir: vec3<f32>,
    pixel: vec2<f32>,
    air: vec3<f32>,   // sky radiance along this ray: what haze fades towards
) -> CloudResult {
    var result: CloudResult;
    result.transmittance = 1.0;
    result.scattered = vec3<f32>(0.0);

    let hit = slab_intersect(origin, dir);
    if (hit.y <= hit.x) {
        return result;
    }

    // Note: near the horizon this step is ~390 m, far coarser than the
    // finest noise octave. Capping it was measured and changed nothing: the
    // JITTER offset already scatters the error into noise, and a ray that
    // shallow crosses so much cloud that MIN_TRANSMITTANCE breaks the loop
    // after a handful of samples.
    let dt = (hit.y - hit.x) / f32(PRIMARY_STEPS);

    // The view-to-sun angle is the same for every sample along this ray, so
    // the phase function only needs evaluating once per pixel.
    let phase = cloud_phase(dot(dir, u.sun_dir));

    // Where in the first step this pixel starts sampling.
    let offset = mix(0.5, dither(pixel), JITTER);

    for (var i = 0; i < PRIMARY_STEPS; i = i + 1) {
        // Sample partway into each step: 0.5 would be the midpoint, which
        // integrates more accurately than the near edge, and the dither
        // spreads that offset out across neighbouring pixels.
        let t = hit.x + (f32(i) + offset) * dt;
        let p = origin + dir * t;

        let density = cloud_density(p);
        if (density <= 0.0) {
            continue;
        }

        let sigma = density * EXTINCTION;
        let step_transmittance = exp(-sigma * dt);

        // How much sun reaches this point through the cloud above it.
        let sun_visibility = light_march(p);
        let sunlight = SUN_LIGHT * SUN_LIGHT_INTENSITY * sun_visibility * phase;

        // Skylight, dimmed towards the base where less of the dome is visible.
        let h = clamp((p.y - CLOUD_BOTTOM) / (CLOUD_TOP - CLOUD_BOTTOM), 0.0, 1.0);
        let ambient = AMBIENT_COLOR * AMBIENT_INTENSITY
                    * mix(AMBIENT_BASE_DARKENING, 1.0, h);

        // Aerial perspective. Beer's law again, but for the clear air along
        // the way: exp(-t / AERIAL_DISTANCE) of the cloud's own light reaches
        // the eye and airlight makes up the rest. Fading towards `air` -- the
        // sky colour for this exact ray -- means a fully hazed cloud emits
        // precisely what the sky behind it would have, so it disappears
        // rather than turning into a grey silhouette.
        let haze = 1.0 - exp(-t / AERIAL_DISTANCE);
        let luminance = mix(sunlight + ambient, air, haze);

        result.scattered += luminance * (1.0 - step_transmittance) * result.transmittance;
        result.transmittance *= step_transmittance;

        // Everything further away is hidden; stop paying for it.
        if (result.transmittance < MIN_TRANSMITTANCE) {
            break;
        }
    }

    return result;
}

// ---------------------------------------------------------------------------
//  Fragment stage
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Build this pixel's view ray. `ndc` is [-1,1] across the screen; scaling
    // x by the aspect ratio keeps pixels square.
    let aspect = u.resolution.x / u.resolution.y;
    let dir = normalize(
        u.cam_forward
        + u.cam_right * in.ndc.x * aspect * u.fov_tan
        + u.cam_up    * in.ndc.y * u.fov_tan
    );
    let origin = vec3<f32>(0.0, CAMERA_HEIGHT, 0.0);

    let sky = sky_color(dir);
    let clouds = march_clouds(origin, dir, in.clip_pos.xy, sky);

    // Composite: whatever of the sky survives through the cloud, plus what the
    // cloud itself scattered towards us.
    var col = sky * clouds.transmittance + clouds.scattered;

    // Exposure, then tone mapping. Exposure scales the linear radiance like a
    // camera's shutter speed; the tone map then compresses the high end, since
    // sunlit cloud and the halo both run well past 1.0 and would otherwise
    // clip to flat white. Order matters: exposing *after* the tone map would
    // just wash out an already-compressed image.
    col = col * dials.exposure;
    col = col / (1.0 + col);
    // The target format is sRGB, so the GPU applies the final encoding for us.
    return vec4<f32>(col, 1.0);
}
