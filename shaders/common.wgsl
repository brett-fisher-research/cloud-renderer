// Shared by every shader file: the camera, the per-frame inputs, and a few
// helpers with no better home.
//
// The shader is several files, one per concern, joined into one module before
// compiling (see src/shader.rs). WGSL lets a function or constant be used
// before it is declared, so the files can refer to each other freely:
//
//   common.wgsl   this file
//   sky.wgsl      the atmosphere: what colour the air is along a ray
//   clouds.wgsl   the cloud layer: density, lighting, the march, its shadow
//   terrain.wgsl  the ground: a heightfield marched to a surface
//   frame.wgsl    the frame itself: asks the others what a ray sees, lights
//                 and composites it, tone maps, accumulates over time
//
// A feature (clouds, terrain) answers questions about the world and knows
// nothing about the screen. Only frame.wgsl knows there is a picture.
//
// Units: metres for clouds and terrain, KILOMETRES for the atmosphere (so
// that the planet's radius fits comfortably in a 32-bit float). Colours are
// linear radiance and go far above 1.0; tone mapping happens at the very end.
//
// Every `const` in these files can be overridden headlessly with
// `--def NAME=VALUE`, and the files hot-reload on save.

// Eye height above the ground, in metres.
const CAMERA_HEIGHT: f32 = 150.0;

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

// The five live dials. Must match `struct Params` in src/params.rs exactly.
struct Params {
    coverage: f32,         // 0 = clear sky, 1 = solid overcast
    feature_size: f32,     // metres: width of a typical cloud
    shadow_strength: f32,  // scales self-shadowing
    wispiness: f32,        // 0 = smooth, 1 = ragged: scales the detail noise
    exposure: f32,         // overall brightness, on top of auto exposure
    _pad0: f32,            // unused; these three fill the second 16-byte row
    _pad1: f32,
    _pad2: f32,
};
@group(0) @binding(1) var<uniform> dials: Params;

@group(0) @binding(7) var noise_sampler: sampler;   // repeats
@group(0) @binding(8) var table_sampler: sampler;   // clamps

// ===========================================================================
//  Helpers
// ===========================================================================

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

// Interleaved gradient noise: a cheap per-pixel pseudo-random number whose
// errors are spread so evenly over neighbouring pixels that it reads as fine
// grain, not as blotches.
fn pixel_noise(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))));
}
