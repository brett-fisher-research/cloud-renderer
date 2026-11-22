// ===========================================================================
//  Volumetric clouds — single fragment shader.
//  Step 1: sky gradient + sun disc.
// ===========================================================================

// ---------------------------------------------------------------------------
//  TUNABLES — change these and save; the renderer hot-reloads the file.
// ---------------------------------------------------------------------------

// Sky colours. These are *linear* radiance values, not sRGB pixel values:
// everything is tone mapped and then sRGB-encoded at the very end, so a value
// here can exceed 1.0 and the numbers look darker than the final image.
const SKY_ZENITH:  vec3<f32> = vec3<f32>(0.05, 0.17, 0.62); // straight up
const SKY_HORIZON: vec3<f32> = vec3<f32>(0.70, 0.90, 1.25); // at eye level
const GROUND:      vec3<f32> = vec3<f32>(0.055, 0.052, 0.042); // below the horizon

// How tightly the horizon haze hugs the horizon. Bigger = thinner band.
const HORIZON_FALLOFF: f32 = 2.2;

// Sun appearance. The real sun is ~0.53 degrees across; a little bigger reads
// better at this resolution.
const SUN_ANGULAR_SIZE: f32 = 0.020;  // radians, half-angle of the disc
const SUN_COLOR:        vec3<f32> = vec3<f32>(1.0, 0.95, 0.85);
const SUN_INTENSITY:    f32 = 40.0;   // brightness of the disc itself
const SUN_GLOW:         f32 = 0.40;   // broad halo strength around the disc

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
//  Vertex stage: a full-screen triangle.
//
//  Rather than upload geometry, we generate three clip-space points that form
//  a triangle twice the size of the screen. Everything outside the screen is
//  clipped away, so the fragment shader runs once per pixel with no vertex
//  buffer, no index buffer and no CPU-side geometry at all.
// ---------------------------------------------------------------------------

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    // Pixel coordinate in [-1,1], handed to the fragment stage.
    @location(0) ndc: vec2<f32>,
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

// Colour of the empty sky along a view ray.
fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    // Angle between this ray and the sun, in radians. Working in angle space
    // (rather than in the raw dot product) keeps the sun disc the size we
    // asked for — near zero, a small change in angle is a *tiny* change in
    // cosine, so thresholding the cosine directly gives a huge soft blob.
    let cos_sun = clamp(dot(dir, u.sun_dir), -1.0, 1.0);
    let sun_angle = acos(cos_sun);

    // How much of the sun is above the horizon, for warm low-sun tinting.
    let dusk = 1.0 - clamp(u.sun_dir.y * 3.0, 0.0, 1.0);

    var col: vec3<f32>;
    if (dir.y < 0.0) {
        // Below the horizon: flat ground, fading into haze as it recedes.
        let haze = exp(dir.y * 6.0);       // 1 at the horizon -> 0 underfoot
        col = mix(GROUND, SKY_HORIZON * 0.8, haze * 0.65);
    } else {
        // Blend zenith -> horizon. pow() concentrates the pale haze near y = 0.
        let t = pow(1.0 - dir.y, HORIZON_FALLOFF);
        col = mix(SKY_ZENITH, SKY_HORIZON, t);
        // Warm the horizon band when the sun is low.
        col = mix(col, col * vec3<f32>(1.35, 1.05, 0.80), t * dusk);
    }

    // Broad halo around the sun (forward-scattered light in the atmosphere).
    // exp() of the angle gives a smooth falloff that does not depend on the
    // dot-product precision problems above.
    let glow = exp(-sun_angle * 9.0) * 0.8 + exp(-sun_angle * 1.6) * 0.2;
    col += SUN_COLOR * SUN_GLOW * glow * (0.5 + 1.2 * dusk);

    // The sun disc itself, with an edge about one pixel wide.
    let edge = 0.0015;
    let disc = 1.0 - smoothstep(SUN_ANGULAR_SIZE - edge, SUN_ANGULAR_SIZE + edge, sun_angle);
    // Hide it when it has set.
    col += SUN_COLOR * SUN_INTENSITY * disc * step(0.0, u.sun_dir.y);

    return col;
}

// ---------------------------------------------------------------------------
//  Fragment stage
// ---------------------------------------------------------------------------

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Build this pixel's view ray. `ndc` is [-1,1] across the screen; scaling x
    // by the aspect ratio keeps pixels square.
    let aspect = u.resolution.x / u.resolution.y;
    let dir = normalize(
        u.cam_forward
        + u.cam_right * in.ndc.x * aspect * u.fov_tan
        + u.cam_up    * in.ndc.y * u.fov_tan
    );

    var col = sky_color(dir);

    // Tone mapping: the sun disc is far brighter than 1.0, so compress the
    // high end into displayable range instead of letting it clip to white.
    col = col / (1.0 + col);
    // The swapchain is sRGB-encoded, and wgpu converts for us when the format
    // ends in `Srgb`. For non-sRGB targets (our screenshot path) we would need
    // to encode manually — handled in Rust by using an Srgb offscreen format.
    return vec4<f32>(col, 1.0);
}
