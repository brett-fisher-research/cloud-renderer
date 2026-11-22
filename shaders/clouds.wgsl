// ===========================================================================
//  Volumetric clouds — single fragment shader.
//  Step 2: sky + a raymarched slab of uniform fog.
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

// --- Density / lighting (placeholder values for step 2) ---
const DENSITY: f32 = 1.0;           // uniform for now; noise comes in step 3
const EXTINCTION: f32 = 0.0004;     // per metre: how fast cloud blocks light
const FOG_COLOR: vec3<f32> = vec3<f32>(0.85, 0.88, 0.95);

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

// Density of the cloud medium at a world-space point.
// Step 2: uniform inside the slab. Step 3 replaces this with fractal noise.
fn cloud_density(p: vec3<f32>) -> f32 {
    return DENSITY;
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
