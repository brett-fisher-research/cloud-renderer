// ===========================================================================
//  Volumetric clouds — single fragment shader.
//  Step 3: sky + noise-shaped clouds raymarched through a slab.
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
const SUN_ANGULAR_SIZE: f32 = 0.020;  // radians, half-angle of the disc
const SUN_COLOR:        vec3<f32> = vec3<f32>(1.0, 0.95, 0.85);
const SUN_INTENSITY:    f32 = 40.0;
const SUN_GLOW:         f32 = 0.40;

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

// --- Density ---
const DENSITY: f32 = 1.0;           // overall multiplier on cloud thickness
const EXTINCTION: f32 = 0.004;      // per metre: how fast cloud blocks light
const FOG_COLOR: vec3<f32> = vec3<f32>(0.85, 0.88, 0.95);

// --- Noise shaping ---
// Size of the largest noise features, in metres. Smaller = busier sky.
const NOISE_FEATURE_SIZE: f32 = 2600.0;
const FBM_OCTAVES: i32 = 5;         // layers of detail; cost scales with this
const FBM_GAIN: f32 = 0.5;          // how much quieter each octave is
const FBM_LACUNARITY: f32 = 2.02;   // how much finer each octave is

// 0 = empty sky, 1 = solid overcast. Around 0.45 gives puffy clouds with gaps.
const COVERAGE: f32 = 0.45;
// Where in the slab the cloud sits: fractions of the layer height.
const CLOUD_BASE_SOFTNESS: f32 = 0.22;  // fade-in above the bottom
const CLOUD_TOP_SOFTNESS:  f32 = 0.55;  // fade-out below the top

// --- Wind ---
const WIND: vec3<f32> = vec3<f32>(-9.0, 0.0, -3.0);  // metres per second

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
    // Angle between this ray and the sun. Working in angle space (rather than
    // in the raw dot product) keeps the sun disc the size we asked for — near
    // zero, a small change in angle is a tiny change in cosine, so
    // thresholding the cosine directly gives a huge soft blob.
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
    let glow = exp(-sun_angle * 9.0) * 0.8 + exp(-sun_angle * 1.6) * 0.2;
    col += SUN_COLOR * SUN_GLOW * glow * (0.5 + 1.2 * dusk);

    // The sun disc itself, with an edge about one pixel wide.
    let edge = 0.0015;
    let disc = 1.0 - smoothstep(SUN_ANGULAR_SIZE - edge, SUN_ANGULAR_SIZE + edge, sun_angle);
    col += SUN_COLOR * SUN_INTENSITY * disc * step(0.0, u.sun_dir.y);

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
    let q = (p + drift) / NOISE_FEATURE_SIZE;

    let n = fbm(q);

    // Coverage: everything below the threshold becomes clear sky, and what
    // remains is rescaled back to [0,1] so the clouds keep their full density
    // instead of thinning out as coverage drops.
    let threshold = 1.0 - COVERAGE;
    let shaped = clamp((n - threshold) / max(COVERAGE, 0.001), 0.0, 1.0);

    return shaped * profile * DENSITY;
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
fn march_clouds(origin: vec3<f32>, dir: vec3<f32>) -> CloudResult {
    var result: CloudResult;
    result.transmittance = 1.0;
    result.scattered = vec3<f32>(0.0);

    let hit = slab_intersect(origin, dir);
    if (hit.y <= hit.x) {
        return result;
    }

    let dt = (hit.y - hit.x) / f32(PRIMARY_STEPS);

    for (var i = 0; i < PRIMARY_STEPS; i = i + 1) {
        // Sample at the middle of each step, which integrates more accurately
        // than sampling at the near edge.
        let t = hit.x + (f32(i) + 0.5) * dt;
        let p = origin + dir * t;

        let density = cloud_density(p);
        if (density <= 0.0) {
            continue;
        }

        let sigma = density * EXTINCTION;
        let step_transmittance = exp(-sigma * dt);

        // Flat ambient colour for now; real sun lighting arrives in step 4.
        let luminance = FOG_COLOR;

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
    let clouds = march_clouds(origin, dir);

    // Composite: whatever of the sky survives through the cloud, plus what the
    // cloud itself scattered towards us.
    var col = sky * clouds.transmittance + clouds.scattered;

    // Tone mapping: the sun disc is far brighter than 1.0, so compress the
    // high end instead of letting it clip to flat white.
    col = col / (1.0 + col);
    // The target format is sRGB, so the GPU applies the final encoding for us.
    return vec4<f32>(col, 1.0);
}
