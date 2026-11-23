// Volumetric clouds under a physically based sky.
//
// One full-screen triangle; this fragment shader does everything visual.
// Per pixel it:
//
//   1. builds a view ray,
//   2. raymarches a spherical shell of cloud around a round planet,
//   3. raymarches the atmosphere in front of and behind that cloud,
//   4. composites, exposes and tone maps.
//
// Units: metres for the clouds, KILOMETRES for the atmosphere (so that the
// planet's radius fits comfortably in a 32-bit float). Colours are linear
// radiance and go far above 1.0; tone mapping happens at the very end.
//
// Every `const` in this file can be overridden headlessly with
// `--def NAME=VALUE`, and the file hot-reloads on save.

// ===========================================================================
//  Tunables
// ===========================================================================

// Eye height above the ground, in metres.
const CAMERA_HEIGHT: f32 = 150.0;

// --- The cloud layer --------------------------------------------------------
// Altitudes in metres. Cumulus bases sit at the height where rising air
// cools enough to condense, which is why real ones share a flat floor.
const CLOUD_BOTTOM: f32 = 1500.0;
const CLOUD_TOP: f32 = 4200.0;

// Sizes of the three noise textures, as multiples of the feature-size dial.
// The shape texture has four Worley cells across it, so 4.0 makes one cell —
// one cloud — one feature size wide. The other two are deliberately not
// simple ratios of it, so the three patterns never line up twice.
const SHAPE_TILE: f32 = 4.0;
const DETAIL_TILE: f32 = 0.93;
const WEATHER_TILE: f32 = 9.3;

// The weather map is read a second time, zoomed out and rotated, to vary the
// coverage from region to region: clouds gather in groups with clear sky
// between, and no two copies of the weather tile look the same.
const REGION_SCALE: f32 = 0.2360679;
const REGION_AMOUNT: f32 = 0.55;
const REGION_ROT: vec2<f32> = vec2<f32>(0.7986355, 0.6018150);  // cos, sin of 37 degrees

// How far inside a cloud region (in weather-map units) the cloud reaches
// full strength. Small = hard-edged slabs, large = thin scattered scraps.
const COVER_RAMP: f32 = 0.30;
// The shortest clouds reach this fraction of the layer; the tallest fill it.
const TOP_MIN: f32 = 0.30;
// Fraction of the layer over which the flat base fades in.
const BASE_SOFTNESS: f32 = 0.07;
// How much the height of the cloud floor wanders, as a fraction of the layer.
const BASE_VARIATION: f32 = 0.10;
// Shape of the dome: how fast a cloud narrows towards its top.
const TOP_ROUNDING: f32 = 0.60;
// How strongly the detail texture eats into the cloud's edges.
const DETAIL_STRENGTH: f32 = 0.32;
// Near the camera the detail texture is read again, this much finer.
const FINE_DETAIL_SCALE: f32 = 4.3;
const FINE_DETAIL_AMOUNT: f32 = 0.5;
const FINE_DETAIL_LOD: f32 = 2.5;   // detail mip beyond which it is skipped
// How far past the threshold the noise must rise for full density. Smaller =
// crisper edges.
const EDGE_WIDTH: f32 = 0.05;
// The shadow march sees edges this many times softer.
const SHADOW_SOFTNESS: f32 = 4.0;
// Density at the base relative to the top. Real clouds hold more water
// higher up, which is why their bases look soft and their tops look solid.
const BASE_DENSITY: f32 = 0.45;

// Extinction per metre at full density: how fast cloud blocks light. Real
// cumulus is around 0.05; lower reads as softer and more translucent.
const EXTINCTION: f32 = 0.022;

// Wind, metres per second. The whole sky drifts with WIND; the 3D shape
// drifts a little extra, so clouds slowly change form as they travel.
const WIND: vec3<f32> = vec3<f32>(-9.0, 0.0, -3.0);
const SHAPE_DRIFT: vec3<f32> = vec3<f32>(-2.5, 0.6, 1.0);
// Tops lean downwind by this many metres across the layer.
const WIND_SHEAR: f32 = 500.0;

// --- The view march ---------------------------------------------------------
// Each step is this fraction of the distance already travelled, but never
// shorter than STEP_MIN. Growing steps are what make a 200 km view affordable:
// a cloud twice as far away is half the size on screen and needs half the
// samples.
const STEP_MIN: f32 = 20.0;
const STEP_GROWTH: f32 = 0.003;
// Strides through clear air are this many times longer than steps in cloud.
const STEP_COARSE: f32 = 2.0;
// Consecutive empty samples before giving up on short steps.
const FINE_PATIENCE: i32 = 3;
const MAX_STEPS: i32 = 280;
// Where the weather map says the whole column is clear, steps are this much
// longer.
const EMPTY_STEP_SCALE: f32 = 2.0;
// Stop marching when this little of the background still shows through.
const MIN_TRANSMITTANCE: f32 = 0.02;
// Texture mip bias for the march. 0 = mip matched to the step length.
// Negative is sharper and noisier.
const LOD_BIAS: f32 = -1.0;
// Randomly offset each ray's start to turn banding into fine grain.
const JITTER: f32 = 1.0;
// Change the grain pattern every frame, so that averaging frames removes it.
const JITTER_ANIMATE: bool = true;

// --- Temporal accumulation --------------------------------------------------
// Each frame is blended with the accumulated result of the frames before it,
// so the grain above averages away over time instead of being paid for with
// more samples per frame.
//
// This is the share of each new frame in the blend. 0.1 means a pixel is
// roughly the average of the last ten to twenty frames: lower is smoother
// but slower to follow a change (the sun moving, a dial, drifting cloud);
// 1.0 turns accumulation off.
const TEMPORAL_BLEND: f32 = 0.1;
// Follow the camera when it turns, by looking up where each pixel was last
// frame. Off, a turning camera drags a smear of old frames behind it.
const REPROJECT: bool = true;
// Read the history with a sharpening (Catmull-Rom) filter instead of plain
// bilinear. Off, the picture goes soft whenever the camera turns.
const HISTORY_SHARP: bool = true;
// Shift each frame's rays by a different fraction of a pixel, so the average
// also smooths hard edges (anti-aliasing). 0 = always through pixel centres.
const SUBPIXEL_JITTER: f32 = 1.0;

// --- Lighting ---------------------------------------------------------------
// Samples marched towards the sun from each cloud sample, to find how much
// cloud the sunlight crossed to get there.
const LIGHT_STEPS: i32 = 8;
const LIGHT_STEP_SIZE: f32 = 55.0;   // metres, length of the first step
const LIGHT_STEP_GROWTH: f32 = 1.6;  // each step is this much longer
// How far each shadow sample is randomly shifted within its step, 0..1.
// Trades terraces in the shading (0) for fine grain (1).
const LIGHT_JITTER: f32 = 0.0;

// Multiple scattering, approximated. Light in a cloud bounces many times;
// simulating one bounce only gives clouds that are black inside. Each
// "octave" here stands for light that has bounced more: it is dimmer, it
// penetrates further, and it has forgotten more of its original direction.
const MS_OCTAVES: i32 = 3;
const MS_ATTENUATION: f32 = 0.32;   // optical depth multiplier per octave
const MS_CONTRIBUTION: f32 = 0.55;  // brightness multiplier per octave
const MS_ECCENTRICITY: f32 = 0.55;  // directionality multiplier per octave
// Overall gain on scattered sunlight, standing in for the bounces the octaves
// above still miss.
const SUN_GAIN: f32 = 2.0;
// Light that has bounced so often it is a slow glow diffusing through the
// whole cloud. It fades far more gently with depth than the octaves do,
// which is why the underside of a thick cloud is grey, not black, and
// darker where the cloud above is deeper.
const DIFFUSE_GAIN: f32 = 0.7;
const DIFFUSE_FALLOFF: f32 = 0.4;
// How much the glow still cares about the path to the sun, 0..1.
const DIFFUSE_DIRECTIONAL: f32 = 0.15;

// Phase function: which way a droplet sends light. Mostly forwards (the
// silver lining around a cloud in front of the sun), a little backwards.
const PHASE_G_FORWARD: f32 = 0.80;
const PHASE_G_BACKWARD: f32 = -0.25;
const PHASE_BLEND: f32 = 0.55;       // weight of the forward lobe

// Skylight and ground-bounce reaching the inside of a cloud.
const AMBIENT_GAIN: f32 = 0.9;
// Average brightness of the cloud field itself, as a fraction of the light
// falling on it. Clouds light each other and the ground beneath them.
const CLOUD_GLOW: f32 = 0.12;
// How much darker the base is than the top under skylight alone.
const AMBIENT_BASE: f32 = 0.3;
// How quickly skylight dies away inside dense cloud.
const AMBIENT_OCCLUSION: f32 = 0.6;

// --- Sky, ground and camera -------------------------------------------------
const SKY_STEPS: i32 = 14;           // atmosphere samples on a clear ray
const SKY_STEPS_NEAR: i32 = 6;       // ...in front of a cloud
const SKY_STEPS_FAR: i32 = 8;        // ...behind it
const GROUND_COLOR: vec3<f32> = vec3<f32>(0.16, 0.17, 0.11);
const GROUND_SHADOW_STEPS: i32 = 12; // cloud samples for shadows on the ground
const SUN_DISC: f32 = 1.0;           // 0 hides the sun's disc
// Lens glare around the sun, as a fraction of the disc's own brightness.
const SUN_GLARE: f32 = 0.004;
// Auto exposure. The sky at dusk is thousands of times darker than at noon;
// a camera compensates. 1.0 would compensate fully and make dusk look like
// day; 0 not at all.
const EXPOSURE_BASE: f32 = 5.5;
const EXPOSURE_ADAPTATION: f32 = 0.65;

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

const PI: f32 = 3.14159265;

// ===========================================================================
//  Inputs
// ===========================================================================

// Must match `struct Uniforms` in src/camera.rs exactly.
struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    fov_tan: f32,       // tan(fov_y / 2)
    // A vec3 occupies a 16-byte row but uses 12, so one float fits after
    // each. Those slots carry the per-frame temporal values.
    cam_forward: vec3<f32>,
    jitter: f32,            // this frame's ray-march offset, 0..1
    cam_right: vec3<f32>,
    history_frames: f32,    // frames of valid history to blend with; 0 = none
    cam_up: vec3<f32>,
    subpixel_x: f32,        // this frame's ray offset within the pixel,
    sun_dir: vec3<f32>,     // points *towards* the sun
    subpixel_y: f32,        // ...in pixels, -0.5..0.5
    // Last frame's camera basis.
    prev_forward: vec3<f32>,
    prev_right: vec3<f32>,
    prev_up: vec3<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

// The four live dials. Must match `struct Params` in src/params.rs exactly.
struct Params {
    coverage: f32,         // 0 = clear sky, 1 = solid overcast
    feature_size: f32,     // metres: width of a typical cloud
    shadow_strength: f32,  // scales self-shadowing
    exposure: f32,         // overall brightness, applied before tone mapping
};
@group(0) @binding(1) var<uniform> dials: Params;

@group(0) @binding(2) var noise_shape: texture_3d<f32>;
@group(0) @binding(3) var noise_detail: texture_3d<f32>;
@group(0) @binding(4) var weather_map: texture_2d<f32>;
@group(0) @binding(5) var transmittance_table: texture_2d<f32>;
@group(0) @binding(6) var multiscatter_table: texture_2d<f32>;
@group(0) @binding(7) var noise_sampler: sampler;   // repeats
@group(0) @binding(8) var table_sampler: sampler;   // clamps
// The accumulated image: last frame's in `fs_main`, this frame's in
// `fs_present`. Tone-mapped linear colour.
@group(0) @binding(9) var history: texture_2d<f32>;

// ===========================================================================
//  Atmosphere
// ===========================================================================

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
fn atmosphere(r0: f32, mu: f32, nu: f32, mu_sun0: f32, t0: f32, t1: f32, steps: i32) -> Scatter {
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

        let direct = sun_transmittance(r, mu_sun);
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

// Distance to where a ray leaves a sphere it starts inside. `b` is the ray
// origin's radius times the cosine to straight-up, `c` is (origin radius)^2
// minus (sphere radius)^2, negative. Written to avoid subtracting two large
// nearly equal numbers, which is what the textbook formula does.
fn sphere_exit(b: f32, c: f32) -> f32 {
    let s = sqrt(b * b - c);
    if (b > 0.0) {
        return -c / (b + s);
    }
    return s - b;
}

// ===========================================================================
//  Vertex stage: a full-screen triangle, plus once-per-frame lighting.
//
//  Rather than upload geometry, three clip-space points form a triangle
//  twice the size of the screen; what falls outside is clipped away, so the
//  fragment shader runs once per pixel.
//
//  A few colours are the same for every pixel — the skylight falling on the
//  clouds, the exposure — and are expensive to work out. The vertex shader
//  runs three times per frame rather than four million, so they are computed
//  here and handed down as `flat` (not interpolated) values.
// ===========================================================================

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,                               // pixel position in [-1,1]
    @location(1) @interpolate(flat) ambient_sky: vec3<f32>,    // mean sky radiance
    @location(2) @interpolate(flat) ambient_ground: vec3<f32>, // mean radiance of the ground
    @location(3) @interpolate(flat) exposure: f32,
    @location(4) @interpolate(flat) sun_glare: vec3<f32>,      // lens glare colour
};

// Average sky radiance seen from inside the cloud layer.
fn mean_sky() -> vec3<f32> {
    let r = R_GROUND + 0.0005 * (CLOUD_BOTTOM + CLOUD_TOP);
    let mu_sun = u.sun_dir.y;
    let sin_sun = sqrt(max(1.0 - mu_sun * mu_sun, 0.0));
    var sum = vec3<f32>(0.0);
    // Straight up, then a ring of four directions 60 degrees from vertical:
    // towards the sun, away from it, and to each side.
    for (var i = 0; i < 5; i++) {
        var mu = 1.0;
        var nu = mu_sun;
        if (i > 0) {
            mu = 0.5;
            let azimuth = array<f32, 4>(1.0, -1.0, 0.0, 0.0)[i - 1];
            nu = mu * mu_sun + 0.8660254 * sin_sun * azimuth;
        }
        let far = sphere_exit(r * mu, (r - R_TOP) * (r + R_TOP));
        sum += atmosphere(r, mu, nu, mu_sun, 0.0, far, 8).light;
    }
    return sum / 5.0;
}

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

    let luma = vec3<f32>(0.2126, 0.7152, 0.0722);
    let c = dials.coverage;
    let clear_sky = mean_sky();
    let sun_on_ground = sun_transmittance(R_GROUND + 0.001, u.sun_dir.y) * max(u.sun_dir.y, 0.0);
    let sun_on_cloud = sun_transmittance(R_GROUND + 0.0005 * (CLOUD_BOTTOM + CLOUD_TOP), u.sun_dir.y);

    // What the rest of the sky looks like from any one spot: blue between
    // the clouds, and the clouds themselves, which are white and bright.
    // The more cloud, the greyer and more even the light. This is why a
    // cloud's shaded side is grey rather than sky blue.
    let cloud_glow = vec3<f32>(CLOUD_GLOW * dot(sun_on_cloud, luma) + 0.6 * dot(clear_sky, luma));
    let sky = mix(clear_sky, cloud_glow, min(0.4 + 0.6 * c, 1.0));

    // A matte surface under light of irradiance E has radiance albedo*E/pi.
    // Clouds shade some of the ground from the direct sun.
    let ground = GROUND_COLOR * (sun_on_ground / PI * (1.0 - 0.7 * c) + sky);
    out.ambient_sky = sky;
    out.ambient_ground = ground;

    // Meter the scene the way a camera would: its brightness is mostly the
    // sky plus sunlit surfaces.
    let key = dot(clear_sky, luma) + 0.15 * dot(sun_on_ground, luma);
    // 0.055 is the key of a clear midday sky, so EXPOSURE_BASE means what it
    // says at noon and adaptation only acts as the light fails.
    out.exposure = EXPOSURE_BASE * pow(0.055 / max(key, 1.0e-5), EXPOSURE_ADAPTATION);

    // How much of the sun the camera can see: its colour after the air,
    // dimmed by any cloud in the way.
    let cam_r = R_GROUND + 0.001 * CAMERA_HEIGHT;
    out.sun_glare = SUN_GLARE / (PI * 0.00465 * 0.00465)
        * sun_transmittance(cam_r, u.sun_dir.y) * camera_sun_visibility();
    return out;
}

// ===========================================================================
//  Cloud density
// ===========================================================================

// Interleaved gradient noise: a cheap per-pixel pseudo-random number whose
// errors are spread so evenly over neighbouring pixels that it reads as fine
// grain, not as blotches.
fn pixel_noise(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))));
}

// What the weather is doing above a point on the ground.
//   x: cover — 0 where the column is clear, rising to 1 well inside a cloud.
//   y: top   — how far up the layer clouds here can grow, 0..1.
fn weather(xz: vec2<f32>, lod: f32) -> vec3<f32> {
    let q = (xz + WIND.xz * u.time) / (dials.feature_size * WEATHER_TILE);
    let local = textureSampleLevel(weather_map, noise_sampler, q, lod).rg;

    // The same map again, zoomed out and rotated, as regional weather.
    let q2 = vec2<f32>(
        q.x * REGION_ROT.x - q.y * REGION_ROT.y,
        q.x * REGION_ROT.y + q.y * REGION_ROT.x) * REGION_SCALE + vec2<f32>(0.31, 0.67);
    let region = textureSampleLevel(weather_map, noise_sampler, q2, max(lod - 2.0, 0.0)).rg;

    // The dial is the average; the region pushes it up or down. The push
    // fades out at both ends so 0 is still clear and 1 still overcast.
    let c = dials.coverage;
    let here = clamp(c + (region.r - 0.5) * REGION_AMOUNT * 4.0 * c * (1.0 - c), 0.0, 1.0);

    // The map is equalised, so `local.r > 1 - here` is true over exactly the
    // fraction `here` of the sky.
    let cover = saturate((local.r - 1.0 + here * (1.0 + COVER_RAMP)) / COVER_RAMP);
    // Clouds are tallest in their middles and where the region is active.
    let top = mix(TOP_MIN, 1.0, 0.5 * (local.g + region.g)) * mix(0.45, 1.0, cover);
    // The floor is not perfectly level either: a little relief gives the
    // underside of a cloud deck something for the light to catch.
    let base = BASE_VARIATION * local.g;
    return vec3<f32>(cover, max(top, base + 0.05), base);
}

// Mip levels for a march step of `step` metres: x for the shape and weather
// textures, y for the detail texture. A longer step reads a blurrier mip, so
// the texture is never sampled more coarsely than it is filtered — that is
// what keeps distant cloud from shimmering.
fn lods_for(step: f32) -> vec2<f32> {
    let shape_texel = dials.feature_size * SHAPE_TILE / 128.0;
    let detail_texel = dials.feature_size * DETAIL_TILE / 64.0;
    return max(vec2<f32>(log2(step / shape_texel), log2(step / detail_texel)) + LOD_BIAS, vec2<f32>(0.0));
}

// How much cloud is at a point: 0..1.
//
//   xz     — horizontal position, metres
//   height — 0 at the bottom of the layer, 1 at the top
//   w      — weather(xz), passed in so callers can skip empty columns first
//   detail — false in the shadow march, where fine detail averages out anyway
fn cloud_density(xz: vec2<f32>, height: f32, w: vec3<f32>, lod: vec2<f32>, detail: bool) -> f32 {
    let cover = w.x;
    // Height within this particular cloud: 0 at its base, 1 at its own top.
    let lifted = height - w.z;
    let hn = lifted / (w.y - w.z);
    if (cover <= 0.0 || hn >= 1.0 || lifted <= 0.0) {
        return 0.0;
    }

    // The profile turns a 2D "is there cloud here" into a 3D shape. It is
    // the fraction of the noise allowed through at each height: it rises
    // sharply from zero at the base (a flat bottom) and tapers towards the
    // top, so only the noise's highest peaks survive up there — rounded,
    // lumpy summits.
    let profile = smoothstep(0.0, BASE_SOFTNESS, lifted) * pow(1.0 - hn, TOP_ROUNDING);
    let allowed = cover * profile;

    // Position in the noise. Height is in metres within the layer, so the
    // noise follows the curve of the planet. Tops lean downwind.
    let metres = vec3<f32>(xz.x, height * (CLOUD_TOP - CLOUD_BOTTOM), xz.y)
        + (WIND + SHAPE_DRIFT) * u.time
        + normalize(WIND) * height * WIND_SHEAR;
    let shape = textureSampleLevel(
        noise_shape, noise_sampler, metres / (dials.feature_size * SHAPE_TILE), lod.x).r;

    // How far the noise rises above the threshold: positive inside the
    // cloud, negative outside, like a contour map's height above sea level.
    var s = shape - (1.0 - allowed);
    var width = EDGE_WIDTH * SHADOW_SOFTNESS;

    if (detail) {
        if (s <= -DETAIL_STRENGTH) {
            return 0.0;
        }
        width = EDGE_WIDTH;
        // Push the surface in and out with small-scale Worley noise — the
        // cauliflower texture. Near the base the noise is used as-is
        // (ragged wisps); higher up it is inverted (rounded billows).
        let q = (metres + SHAPE_DRIFT * u.time * 2.0) / (dials.feature_size * DETAIL_TILE);
        var n = textureSampleLevel(noise_detail, noise_sampler, q, lod.y).r;
        if (lod.y < FINE_DETAIL_LOD) {
            // Close to the camera, the same texture again at a finer scale:
            // the smallest tufts, which only matter where they can be seen.
            let fine = textureSampleLevel(noise_detail, noise_sampler, q * FINE_DETAIL_SCALE, lod.y + 2.0).r;
            n += (fine - 0.5) * FINE_DETAIL_AMOUNT * saturate(FINE_DETAIL_LOD - lod.y);
        }
        let billow = mix(n, 1.0 - n, saturate(hn * 5.0));
        // Centred on zero, so on average the surface stays where it was and
        // the shadow march (which skips this) still agrees with the picture.
        s += (0.5 - billow) * 2.0 * DETAIL_STRENGTH;
    }

    // The edge is `width` wide in noise units: crisp for the view march,
    // deliberately soft for shadows so they vary smoothly.
    return saturate(s / width) * mix(BASE_DENSITY, 1.0, hn);
}

// ===========================================================================
//  Cloud lighting
// ===========================================================================

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));
}

// Optical depth between a cloud sample and the sun: how much cloud the
// sunlight had to cross. `sun_xz` is the sun's horizontal direction and
// `sun_rise` how fast a step towards it gains height, per metre, in layer
// fractions.
fn light_march(xz: vec2<f32>, height: f32, sun_xz: vec2<f32>, sun_rise: f32, first_step: f32, noise: f32) -> f32 {
    var depth = 0.0;
    var travelled = 0.0;
    var step = first_step;
    for (var i = 0; i < LIGHT_STEPS; i++) {
        // Each sample sits at a random point within its step, different for
        // every pixel, so shadows come out as fine grain, not as terraces.
        let s = travelled + (0.5 + (noise - 0.5) * LIGHT_JITTER) * step;
        let h = height + s * sun_rise;
        // Out of the layer: nothing further along can shadow us.
        if (h >= 1.0 || h <= 0.0) {
            break;
        }
        let p = xz + sun_xz * s;
        // One mip coarser than the step warrants: a shadow is an average
        // along a line, so it does not need the detail.
        let lod = lods_for(step * 2.0);
        depth += cloud_density(p, h, weather(p, lod.x), lod, false) * step;
        travelled += step;
        step *= LIGHT_STEP_GROWTH;
    }
    return depth * EXTINCTION * dials.shadow_strength;
}

struct Clouds {
    light: vec3<f32>,   // radiance the clouds send towards the eye
    through: f32,       // fraction of the background that shows through
    distance: f32,      // metres to where most of that light came from
};

// March the cloud shell along a view ray.
//
// `sun_light` is the sun's colour after the atmosphere, at cloud altitude.
fn march_clouds(dir: vec3<f32>, pixel: vec2<f32>, in: VsOut) -> Clouds {
    var out: Clouds;
    out.light = vec3<f32>(0.0);
    out.through = 1.0;
    out.distance = 0.0;

    // The planet is a sphere of radius R whose centre is R + CAMERA_HEIGHT
    // below the camera. The cloud layer is the shell between two larger
    // spheres; the ray enters through the inner one and leaves by the outer.
    // On a flat world the layer would run to a vanishing point at the
    // horizon. On a round one it curves away and drops behind it.
    let big_r = R_GROUND * 1000.0;
    let cam_r = big_r + CAMERA_HEIGHT;
    let b = cam_r * dir.y;
    let t_in = sphere_exit(b, (CAMERA_HEIGHT - CLOUD_BOTTOM) * (2.0 * big_r + CAMERA_HEIGHT + CLOUD_BOTTOM));
    let t_out = sphere_exit(b, (CAMERA_HEIGHT - CLOUD_TOP) * (2.0 * big_r + CAMERA_HEIGHT + CLOUD_TOP));

    // Sunlight at the point where the ray meets the clouds. The sun's
    // height above the *local* horizon there differs from the camera's:
    // clouds far towards a setting sun are still lit after it sets here.
    let nu = dot(dir, u.sun_dir);
    let cloud_r = R_GROUND + 0.0005 * (CLOUD_BOTTOM + CLOUD_TOP);
    let t_mid = 0.0005 * (t_in + t_out);
    let mu_sun = clamp(((R_GROUND + 0.001 * CAMERA_HEIGHT) * u.sun_dir.y + t_mid * nu) / cloud_r, -1.0, 1.0);
    let sun_light = sun_transmittance(cloud_r, mu_sun);

    let sun_xz_len = length(u.sun_dir.xz);
    let sun_xz = u.sun_dir.xz / max(sun_xz_len, 1.0e-4) * sqrt(max(1.0 - mu_sun * mu_sun, 0.0));
    let sun_rise = mu_sun / (CLOUD_TOP - CLOUD_BOTTOM);

    var noise = pixel_noise(pixel);
    if (JITTER_ANIMATE) {
        noise = fract(noise + u.jitter);
    }

    // Two step lengths. `fine` is short enough that one step through solid
    // cloud does not swallow all the light, so a cloud's surface is resolved
    // rather than guessed. `coarse` crosses clear air quickly. Both grow
    // with distance: a cloud twice as far away is half the size on screen.
    var t = t_in + max(STEP_MIN, t_in * STEP_GROWTH) * STEP_COARSE * noise * JITTER;
    var fine_mode = false;
    var misses = 0;
    var weight_sum = 0.0;
    var distance_sum = 0.0;

    for (var i = 0; i < MAX_STEPS; i++) {
        if (t >= t_out || out.through < MIN_TRANSMITTANCE) {
            break;
        }
        let fine = max(STEP_MIN, t * STEP_GROWTH);
        let coarse = fine * STEP_COARSE;
        let dt = select(coarse, fine, fine_mode);

        // Height above the ground at distance t along the ray, again in a
        // form that never subtracts two huge numbers.
        let x = 2.0 * t * b + t * t;
        let altitude = CAMERA_HEIGHT + x / (sqrt(cam_r * cam_r + x) + cam_r);
        let height = (altitude - CLOUD_BOTTOM) / (CLOUD_TOP - CLOUD_BOTTOM);
        let xz = dir.xz * t;

        let lod = lods_for(dt);
        let w = weather(xz, lod.x);
        if (w.x <= 0.0) {
            // Nothing anywhere in this column. Take a longer step still.
            fine_mode = false;
            t += coarse * EMPTY_STEP_SCALE;
            continue;
        }

        let density = cloud_density(xz, height, w, lod, true);
        if (!fine_mode) {
            if (density > 0.0) {
                // Walked into a cloud with a long stride. Go back to the
                // last point known to be clear and approach in short steps.
                fine_mode = true;
                misses = 0;
                t = max(t - coarse + fine, t_in);
            } else {
                t += coarse;
            }
            continue;
        }

        if (density > 0.0) {
            misses = 0;
            let depth = light_march(xz, height, sun_xz, sun_rise, max(LIGHT_STEP_SIZE, dt * 0.8), fract(noise + 0.754877666 * f32(i)));

            // Two measures of how deep in the cloud this sample is. `depth`
            // is the cloud between here and the sun: exact, and full of the
            // shadows of every bump along the way. `overhead` is simply how
            // much cloud lies above, from the weather map: smooth.
            //
            // Light that has bounced once remembers where the sun is and
            // casts sharp shadows. Light that has bounced many times has
            // forgotten, and only knows how buried it is. So each octave
            // leans further from `depth` towards `overhead` — without that,
            // the cloud's own lumps cast streaks right through it and onto
            // its underside, which no real cloud shows.
            let overhead = max(w.y - height, 0.0) * (CLOUD_TOP - CLOUD_BOTTOM)
                * EXTINCTION * dials.shadow_strength * 0.5 * w.x;

            // Sum the scattering octaves: see MS_OCTAVES.
            var sun_scatter = 0.0;
            var attenuation = 1.0;
            var contribution = 1.0;
            var eccentricity = 1.0;
            for (var o = 0; o < MS_OCTAVES; o++) {
                let phase = mix(
                    henyey_greenstein(nu, PHASE_G_BACKWARD * eccentricity),
                    henyey_greenstein(nu, PHASE_G_FORWARD * eccentricity),
                    PHASE_BLEND);
                let forgotten = f32(o) / f32(max(MS_OCTAVES - 1, 1));
                let d = mix(depth, min(depth, overhead * 2.0), forgotten);
                sun_scatter += contribution * phase * exp(-d * attenuation);
                attenuation *= MS_ATTENUATION;
                contribution *= MS_CONTRIBUTION;
                eccentricity *= MS_ECCENTRICITY;
            }
            // The slow diffuse glow: see DIFFUSE_GAIN.
            sun_scatter += DIFFUSE_GAIN / (4.0 * PI)
                / (1.0 + DIFFUSE_FALLOFF * (overhead + DIFFUSE_DIRECTIONAL * depth));

            // Skylight from above and light bounced off the ground from
            // below. Each fills half the sphere of directions; the top of a
            // cloud sees more sky, the base more ground. Both are dimmer
            // deep inside the cloud than at its surface.
            let hn = saturate((height - w.z) / (w.y - w.z));
            let buried = exp(-density * AMBIENT_OCCLUSION);
            let ambient = 0.5 * (in.ambient_sky * mix(AMBIENT_BASE, 1.0, hn)
                + in.ambient_ground * mix(1.0, 0.4, hn)) * AMBIENT_GAIN * buried;

            let source = sun_light * sun_scatter * SUN_GAIN + ambient;

            // Beer's law: the fraction of light surviving a step falls
            // exponentially with how much cloud is in it.
            let step_through = exp(-density * EXTINCTION * dt);
            let weight = out.through * (1.0 - step_through);
            out.light += source * weight;
            out.through *= step_through;
            weight_sum += weight;
            distance_sum += weight * t;
        } else {
            // Clear again. A few misses in a row and we are out of the
            // cloud: back to long strides.
            misses += 1;
            if (misses >= FINE_PATIENCE) {
                fine_mode = false;
            }
        }
        t += dt;
    }

    // The march stops at MIN_TRANSMITTANCE; stretch what is left so that
    // "nearly opaque" becomes exactly opaque with no seam.
    out.through = saturate((out.through - MIN_TRANSMITTANCE) / (1.0 - MIN_TRANSMITTANCE));
    out.distance = distance_sum / max(weight_sum, 1.0e-6);
    return out;
}

// How much of the sun is visible from the camera through the clouds, 0..1.
fn camera_sun_visibility() -> f32 {
    let rise = max(u.sun_dir.y, 0.02);
    let thickness = CLOUD_TOP - CLOUD_BOTTOM;
    let n = 24.0;
    let step = thickness / n / rise;
    let lod = lods_for(step);
    var depth = 0.0;
    for (var i = 0; i < 24; i++) {
        let height = (f32(i) + 0.5) / n;
        let s = (CLOUD_BOTTOM - CAMERA_HEIGHT + height * thickness) / rise;
        let p = u.sun_dir.xz * s;
        depth += cloud_density(p, height, weather(p, lod.x), lod, false);
    }
    return exp(-depth * step * EXTINCTION);
}

// How much direct sun reaches a point on the ground through the clouds.
fn ground_shadow(xz: vec2<f32>) -> f32 {
    let rise = u.sun_dir.y;
    if (rise < 0.03) {
        return 1.0;
    }
    // Walk the sun ray up through the whole layer. With the sun low that is
    // a long slanted path, so the samples must be close enough together not
    // to step over a cloud — otherwise each sample height casts its own
    // separate copy of the shadow.
    var depth = 0.0;
    let thickness = CLOUD_TOP - CLOUD_BOTTOM;
    let n = f32(GROUND_SHADOW_STEPS);
    let step = thickness / n / rise;
    let lod = lods_for(step);
    for (var i = 0; i < GROUND_SHADOW_STEPS; i++) {
        let height = (f32(i) + 0.5) / n;
        let s = (CLOUD_BOTTOM + height * thickness) / rise;
        let p = xz + u.sun_dir.xz * s;
        depth += cloud_density(p, height, weather(p, lod.x), lod, false);
    }
    depth *= step * EXTINCTION;
    // Never fully black: light scatters down through a cloud as well.
    let shadow = max(exp(-depth), 0.25 * exp(-0.15 * depth));
    // Fade the shadows out as the sun nears the horizon, where they would
    // stretch for ever.
    return mix(1.0, shadow, smoothstep(0.03, 0.10, rise));
}

// ===========================================================================
//  Fragment stage
// ===========================================================================

fn view_ray(ndc: vec2<f32>) -> vec3<f32> {
    // `ndc` is [-1,1] across the screen; scaling x by the aspect ratio keeps
    // pixels square.
    let aspect = u.resolution.x / u.resolution.y;
    return normalize(
        u.cam_forward
        + u.cam_right * ndc.x * aspect * u.fov_tan
        + u.cam_up    * ndc.y * u.fov_tan
    );
}

// Read the accumulated image at a point that may fall between pixels.
//
// The obvious way is bilinear filtering: blend the four nearest pixels. But
// each blend blurs a little, and while the camera turns this happens to the
// same image every frame, over and over, until the whole picture goes soft.
//
// Catmull-Rom filtering blends a 4x4 block with weights that go slightly
// negative at the outside, which sharpens exactly as much as the blend
// blurs. Folding pairs of pixels into single bilinear reads (the GPU does
// that blend for free) gets the sixteen pixels in nine fetches. When the
// point is exactly a pixel centre — a still camera — the weights collapse
// to that one pixel and nothing is filtered at all.
fn sample_history(uv: vec2<f32>) -> vec3<f32> {
    if (!HISTORY_SHARP) {
        return textureSampleLevel(history, table_sampler, uv, 0.0).rgb;
    }
    let size = u.resolution;
    let p = uv * size;
    let centre = floor(p - 0.5) + 0.5;   // the pixel centre just before p
    let f = p - centre;                  // how far past it, 0..1

    // Catmull-Rom weights for the four pixels along one axis.
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    // The middle two as one bilinear read, placed between them.
    let w12 = w1 + w2;
    let t0 = (centre - 1.0) / size;
    let t12 = (centre + w2 / w12) / size;
    let t3 = (centre + 2.0) / size;

    var sum = vec3<f32>(0.0);
    sum += textureSampleLevel(history, table_sampler, vec2<f32>(t0.x, t0.y), 0.0).rgb * (w0.x * w0.y);
    sum += textureSampleLevel(history, table_sampler, vec2<f32>(t12.x, t0.y), 0.0).rgb * (w12.x * w0.y);
    sum += textureSampleLevel(history, table_sampler, vec2<f32>(t3.x, t0.y), 0.0).rgb * (w3.x * w0.y);
    sum += textureSampleLevel(history, table_sampler, vec2<f32>(t0.x, t12.y), 0.0).rgb * (w0.x * w12.y);
    sum += textureSampleLevel(history, table_sampler, vec2<f32>(t12.x, t12.y), 0.0).rgb * (w12.x * w12.y);
    sum += textureSampleLevel(history, table_sampler, vec2<f32>(t3.x, t12.y), 0.0).rgb * (w3.x * w12.y);
    sum += textureSampleLevel(history, table_sampler, vec2<f32>(t0.x, t3.y), 0.0).rgb * (w0.x * w3.y);
    sum += textureSampleLevel(history, table_sampler, vec2<f32>(t12.x, t3.y), 0.0).rgb * (w12.x * w3.y);
    sum += textureSampleLevel(history, table_sampler, vec2<f32>(t3.x, t3.y), 0.0).rgb * (w3.x * w3.y);
    // Negative weights can overshoot at a hard edge. Keep it a valid colour.
    return clamp(sum, vec3<f32>(0.0), vec3<f32>(1.0));
}

// ACES filmic curve (Narkowicz's fit): near-linear in the mid-tones, with a
// soft shoulder so sunlit cloud rolls off to white instead of clipping.
fn tone_map(x: vec3<f32>) -> vec3<f32> {
    return saturate((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let pixel = in.clip_pos.xy;
    // Aim through a slightly different point of the pixel each frame. One
    // pixel is 2/resolution wide in ndc, which spans -1..1.
    let dir = view_ray(in.ndc
        + vec2<f32>(u.subpixel_x, u.subpixel_y) * SUBPIXEL_JITTER * 2.0 / u.resolution);

    // The view ray, in the atmosphere's terms (km).
    let cam_h = CAMERA_HEIGHT * 0.001;
    let r0 = R_GROUND + cam_h;
    let mu = dir.y;
    let nu = dot(dir, u.sun_dir);
    let mu_sun0 = u.sun_dir.y;
    let b = r0 * mu;

    var col: vec3<f32>;

    // Does the ray hit the ground? Same quadratic as `sphere_exit`, but for
    // a sphere we are outside of, and we want the near root.
    let c_ground = cam_h * (2.0 * R_GROUND + cam_h);
    let disc = b * b - c_ground;
    if (mu < 0.0 && disc > 0.0) {
        let t_ground = c_ground / (sqrt(disc) - b);
        let mu_sun = clamp((r0 * mu_sun0 + t_ground * nu) / R_GROUND, -1.0, 1.0);
        let sun = sun_transmittance(R_GROUND + 0.001, mu_sun) * max(mu_sun, 0.0)
            * ground_shadow(dir.xz * t_ground * 1000.0);
        let ground = GROUND_COLOR * (sun / PI + in.ambient_sky);
        let air = atmosphere(r0, mu, nu, mu_sun0, 0.0, t_ground, SKY_STEPS_NEAR);
        col = air.light + air.through * ground;
    } else {
        let t_top = sphere_exit(b, (r0 - R_TOP) * (r0 + R_TOP));
        let clouds = march_clouds(dir, pixel, in);

        // The sun itself: a disc half a degree across, far brighter than
        // anything else, dimmer towards its edge ("limb darkening").
        var behind = vec3<f32>(0.0);
        let sun_angle = acos(clamp(nu, -1.0, 1.0));
        let sun_radius = 0.00465;
        let pixel_angle = 2.0 * u.fov_tan / u.resolution.y;
        let disc_cover = SUN_DISC * saturate((sun_radius - sun_angle) / pixel_angle + 0.5);
        if (disc_cover > 0.0) {
            let edge = saturate(sun_angle / sun_radius);
            let limb = 1.0 - 0.6 * (1.0 - sqrt(1.0 - edge * edge));
            behind = vec3<f32>(disc_cover * limb / (PI * sun_radius * sun_radius));
        }

        if (clouds.through < 0.999) {
            // Air in front of the cloud fades and tints it with distance
            // ("aerial perspective"); air behind it is the sky seen through
            // its gaps. Split the ray at the cloud and march each part.
            let t_cloud = clamp(clouds.distance * 0.001, 0.0, t_top);
            let near = atmosphere(r0, mu, nu, mu_sun0, 0.0, t_cloud, SKY_STEPS_NEAR);
            let far = atmosphere(r0, mu, nu, mu_sun0, t_cloud, t_top, SKY_STEPS_FAR);
            let background = far.light + far.through * behind;
            col = near.light + near.through * (clouds.light + clouds.through * background);
        } else {
            let air = atmosphere(r0, mu, nu, mu_sun0, 0.0, t_top, SKY_STEPS);
            col = air.light + air.through * behind;
        }
    }

    // Exposure scales linear radiance like a camera's shutter; the tone map
    // then compresses the bright end.
    // Glare: light from the sun spreading inside the lens. It lies over the
    // whole picture, clouds included, but only while the sun itself is in
    // clear view of the camera.
    let glare_angle = acos(clamp(nu, -1.0, 1.0)) / 0.00465;
    col += in.sun_glare * pow(1.0 + glare_angle * glare_angle, -1.5);

    col = tone_map(col * in.exposure * dials.exposure);

    // Temporal accumulation: blend with what this spot of sky looked like
    // over the previous frames.
    //
    // "This spot" is the catch. If the camera has turned, last frame's
    // result for this direction is at a different place on screen, so the
    // direction is projected through last frame's camera to find it
    // ("reprojection"). The camera only turns, never travels, so that is
    // exact however far away the thing in view is.
    //
    // Blending happens after tone mapping on purpose. Averaged before it,
    // one frame's sample of something enormously bright (the sun's disc
    // through a gap) would outweigh every other frame.
    let centre = view_ray(in.ndc);
    let depth = dot(centre, u.prev_forward);
    let aspect = u.resolution.x / u.resolution.y;
    let was_at = vec2<f32>(
        dot(centre, u.prev_right) / (depth * aspect * u.fov_tan),
        dot(centre, u.prev_up) / (depth * u.fov_tan));
    var uv = vec2<f32>(0.5 + 0.5 * was_at.x, 0.5 - 0.5 * was_at.y);
    if (!REPROJECT) {
        // Blend with the same pixel regardless: correct only while the
        // camera is still. Turn the camera to see what reprojection is for.
        uv = pixel / u.resolution;
    }
    // Sampled unconditionally, then ignored if it turns out not to be
    // usable: this direction was off screen last frame, or there is no
    // history yet.
    let before = sample_history(uv);
    let usable = u.history_frames > 0.0 && depth > 0.0
        && all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0));
    if (usable) {
        // While the history is young, average evenly — frame n gets weight
        // 1/n — so it converges in a few frames instead of fading in.
        let share = max(TEMPORAL_BLEND, 1.0 / (u.history_frames + 1.0));
        col = mix(before, col, share);
    }

    return vec4<f32>(max(col, vec3<f32>(0.0)), 1.0);
}

// ===========================================================================
//  Present: copy the accumulated image to the screen.
// ===========================================================================

@fragment
fn fs_present(in: VsOut) -> @location(0) vec4<f32> {
    // Exactly this pixel's value, no filtering.
    var col = textureLoad(history, vec2<i32>(in.clip_pos.xy), 0).rgb;

    // A smooth sky gradient in 8 bits shows visible bands. Adding less than
    // one 8-bit step of noise before rounding hides them. The target is
    // sRGB-encoded, which is steeper in the darks, hence the scale. This
    // happens here, not before accumulating, or it would be averaged away.
    let grain = pixel_noise(in.clip_pos.xy + vec2<f32>(37.0, 91.0)) - 0.5;
    col += grain / 255.0 * (0.25 + 2.0 * sqrt(col));

    // The target format is sRGB, so the GPU applies the final encoding.
    return vec4<f32>(max(col, vec3<f32>(0.0)), 1.0);
}

// ===========================================================================
//  Field dump (--dump-field): the clouds seen from directly above, as data.
//
//  R is the density a third of the way up the layer, G the same without
//  detail erosion, B the weather map's cover signal.
// ===========================================================================

const FIELD_TILES: f32 = 8.0;    // shape-texture tiles across the dump
const FIELD_HEIGHT: f32 = 0.3;

@fragment
fn fs_field(in: VsOut) -> @location(0) vec4<f32> {
    let xz = in.ndc * 0.5 * FIELD_TILES * dials.feature_size * SHAPE_TILE;
    let lod = vec2<f32>(0.0);
    let w = weather(xz, 0.0);
    let full = cloud_density(xz, FIELD_HEIGHT, w, lod, true);
    let base = cloud_density(xz, FIELD_HEIGHT, w, lod, false);
    return vec4<f32>(full, base, w.x, 1.0);
}
