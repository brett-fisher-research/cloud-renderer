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
// Samples along the view ray. 64 for a long time, and that was too many:
// moving sixteen of them into the light march below raised accuracy
// against a 256x24-step reference from 0.9912 to 0.9985 SSIM *and* cost
// a millisecond less. The view march was oversampled and the shadow
// march starved.
const PRIMARY_STEPS: i32 = 48;
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
// The shape of the field itself — how many octaves, how fast they fall off —
// is baked into a texture before the first frame, so those constants live in
// `src/noise.rs` rather than here, and changing them needs a rebuild rather
// than a save. The two below control how the baked field is *used*, and still
// hot-reload like everything else.

// How wide the base texture's tile is, as a multiple of the feature size.
// The tile has to be several times the clouds or its repeat is obvious; the
// bake compensates by making the feature-size octave the loudest, so the
// clouds stay compact. Four is the smallest that hides the repeat here.
const TILE_SCALE: f32 = 4.0;

// Detail. A second, finer read that erodes the base's edges. Scaled by a
// prime: at 2, 4 or 8 the two lattices would line up and the erosion would
// reinforce the base's own grid rather than break it up.
const DETAIL_SCALE: f32 = 11.0;
const DETAIL_STRENGTH: f32 = 0.45;

// --- Breaking the texture's repeat ---
//
// The base texture tiles every TILE_SCALE feature sizes: 10.4 km at the
// default dial, against a MAX_DISTANCE of 50 km. So a single view can hold
// five copies of the same sky in each direction, and in perspective a
// repeating lattice on a horizontal plane projects to lines converging on a
// vanishing point — the radial moire at the horizon.
//
// The repeat does not have to be removed, only pushed past the view. Two
// levers, both off at 0.0 so they can be measured one at a time:
//
// WARP is an analytic displacement of the lookup point, costing a few sin
// and cos and no bandwidth at all. Its period is an irrational multiple of
// the tile, so the warped field has no exact period: a shift that lines the
// tiles up leaves the warp out of phase, and the other way round.
//
// WEATHER is a second, much coarser read of the same texture that pushes the
// coverage threshold up and down across the sky. It costs one extra fetch,
// and it is what large-scale structure in a real sky looks like: regions of
// more and less cloud, rather than an even wash of identical puffs.

// The warp's strength, given as a guaranteed bound on its gradient. 0 is off.
//
// This is the parameter that matters, so it is the one exposed. A warp whose
// Jacobian reaches 1.0 somewhere stops being injective: the domain folds back
// through itself and the field creases along the fold, which shows up as a
// hard-edged tear across an otherwise soft cloud. Below 1.0 that cannot
// happen. The displacement amplitude is derived from this bound rather than
// set directly, because amplitude alone says nothing about folding — the same
// displacement is violent at a short wavelength and gentle at a long one.
const WARP_GRADIENT: f32 = 0.9;

// Warp wavelength, in tiles, for each of two octaves.
//
// Long is better than short, which is not obvious. The thing that breaks the
// repeat is how much the displacement *differs* between two points one tile
// apart, and for a sinusoid of wavelength L that difference is proportional
// to L*sin(pi*T/L): it rises with L and saturates at pi*T. At L = 1.618 tiles
// only 48% of the gradient budget turns into useful decorrelation; by 5 tiles
// it is 93%.
//
// Both are irrational multiples of the tile, and of each other, so no small
// whole number of warp periods ever lines up with a whole number of tiles.
const WARP_PERIOD: f32 = 9.8696044;
const WARP_PERIOD_2: f32 = 1.6180339;
// Share of the gradient budget on the long octave. 1.0 spends it all there
// and skips the second octave's six transcendentals entirely.
const WARP_SPLIT: f32 = 1.0;

// Reuse the view sample's warp for every step of the shadow march.
//
// The shadow march walks about a kilometre and the warp varies over tens of
// them, so the displacement it would compute at each step is close to the one
// already computed at the sample it started from. Sharing turns seven warp
// evaluations per sample into one. The shadow is a six-step approximation of
// an integral through cloud; this is far from the coarsest thing about it.
const WARP_SHARED_SHADOW: bool = true;

// --- A second, incommensurately scaled read ---
//
// The warp moves the field about; this changes what is in it. The difference
// matters. A warp is a smooth map, and over any local region a smooth map
// acts like a constant shift — so the field's periodicity is not destroyed,
// only relabelled to a different lag. Measured: the warp alone stalls at an
// in-view autocorrelation of 0.65 against a floor of 0.26, with its residual
// peak sitting at almost exactly six tiles.
//
// Reading the same texture a second time at a scale that is an irrational
// multiple of the first gives a combined field whose period is the two
// periods' common multiple, which does not exist. It costs one fetch.
const MIX_AMOUNT: f32 = 0.5;
// Irrational, and close to 1 so the second read's features are about the same
// size as the first's and the clouds keep their character.
const MIX_SCALE: f32 = 0.6180339;
// An arbitrary shift, so the two reads do not start from the same corner of
// the texture and agree near the origin.
const MIX_OFFSET: vec3<f32> = vec3<f32>(0.37, 0.11, 0.73);
// Whether the shadow march pays for the second read too. Off keeps the
// shadows consistent with a cheaper field, as the detail texture already does.
const MIX_IN_SHADOW: bool = true;
// The same question for the `billow` mode's Worley read. Off, because a
// shadow is an integral of density along a line and the bulges average out of
// it: measured at 2560x1440 on the worst scene, on against off is 16.0 ms
// against 11.9 for no visible difference in the picture.
const BILLOW_IN_SHADOW: bool = false;

// How far the coarse read moves the coverage threshold. 0 disables it.
//
// Kept because it is the only lever here that adds *large-scale structure* —
// regions of more and less cloud — rather than only breaking a repeat. On
// repetition alone it loses to the warp outright; see the commit message.
const WEATHER_AMOUNT: f32 = 0.0;
// Its scale relative to the base read. Below 1 means a coarser field: at
// 0.382 the weather pattern is 2.6 tiles across, so it varies over 27 km
// while the clouds themselves vary over 2.6 km.
const WEATHER_SCALE: f32 = 0.3819660;

// One octave of displacement. Each axis is driven by the *other* axes: a warp
// of x by x alone would only stretch and squeeze the field along x and would
// leave its lattice intact.
//
// The row sums of this function's Jacobian are k*(1 + 0.7*1.7) = 2.19k,
// k*0.6*1.3 = 0.78k and k*(1 + 0.7*1.1) = 1.77k, so scaling the amplitude by
// gradient/(2.19*k) holds the whole Jacobian under `gradient`.
fn warp_octave(p: vec3<f32>, k: f32, gradient: f32) -> vec3<f32> {
    let amp = gradient / (2.19 * k);
    return amp * vec3<f32>(
        sin(p.z * k) + 0.7 * cos(p.y * k * 1.7),
        0.6 * sin(p.x * k * 1.3),
        cos(p.x * k) + 0.7 * sin(p.y * k * 1.1),
    );
}

/// The displacement to add to a lookup point, in metres.
fn warp_at(p: vec3<f32>, tile: f32) -> vec3<f32> {
    if (WARP_GRADIENT <= 0.0) {
        return vec3<f32>(0.0);
    }
    let k1 = 6.2831853 / (tile * WARP_PERIOD);
    var d = warp_octave(p, k1, WARP_GRADIENT * WARP_SPLIT);
    if (WARP_SPLIT < 1.0) {
        let k2 = 6.2831853 / (tile * WARP_PERIOD_2);
        d += warp_octave(p, k2, WARP_GRADIENT * (1.0 - WARP_SPLIT));
    }
    return d;
}

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
//
// Ten rather than six, and nearly free: each step is LIGHT_STEP_GROWTH
// times the last, so the extra ones are long, leave the slab almost at
// once and return zero. Measured, 8 to 12 steps spans 8.80 to 8.95 ms.
// Six was visibly too few for a backlit sky, where the whole image is
// transmittance through cloud: that scene alone went from 0.9656 to
// 0.9920 SSIM on this change. Past ten there is no further gain.
const LIGHT_STEPS: i32 = 10;
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
    algorithm: u32,        // which density field to build: see ALGO_* below
    _pad0: f32,            // a uniform block comes in rows of sixteen bytes;
    _pad1: f32,            // these are the rest of the row `algorithm` opened,
    _pad2: f32,            // and the next three dials to want one are free.
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
//  Noise
//
//  Clouds are shaped by *fractal noise*: a smooth random field summed at
//  several scales. One big smooth layer gives the overall blobs; each finer
//  layer adds wispier detail at half the amplitude. The sum is called fBm
//  (fractional Brownian motion).
//
//  That sum used to be computed here, at every one of the hundreds of samples
//  a pixel takes, and it was 94% of the frame. It is now baked once into two
//  tiling 3D textures by `src/noise.rs`, so a sample is a single filtered
//  fetch. The GPU's texture units do the three-way blend in fixed-function
//  hardware, in parallel with the shader's arithmetic, which is most of why
//  this is now nearly free.
//
//  A 3D texture is a 2D one with a third coordinate: a cube of voxels, read
//  with trilinear filtering. Both of ours *tile*, so one cube covers the whole
//  sky — the sampler's Repeat address mode wraps any coordinate outside 0..1,
//  and the bake hashes each octave's lattice modulo that octave's frequency so
//  the far face of the cube lines up with the near one.
// ---------------------------------------------------------------------------

@group(0) @binding(2) var noise_base: texture_3d<f32>;
@group(0) @binding(3) var noise_detail: texture_3d<f32>;
@group(0) @binding(4) var noise_sampler: sampler;
// Cellular (Worley) noise, used by the `billow` algorithm. Baked at startup
// like the other two; see `bake_worley` in src/noise.rs.
@group(0) @binding(5) var noise_worley: texture_3d<f32>;

// "Interleaved gradient noise": a cheap screen-space dither that looks evenly
// scattered rather than streaky. It depends only on the pixel coordinate, so
// it is stable from frame to frame — a time-varying jitter would make the
// image fizz instead of banding, which is worse without temporal filtering.
fn dither(pixel: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))));
}

// Sample the baked noise field. `q` is in tile units: 1.0 is one full wrap of
// the base texture.
//
// `detail` buys crisper, wispier edges for the price of a second fetch. The
// light march leaves it off: a shadow is an integral of density along a line,
// and detail that fine averages out of it long before it reaches the eye.
// `lod` is a mip level for each texture: x for the base, y for the detail.
// See `mip_for_step`.
fn cloud_noise(q: vec3<f32>, lod: vec2<f32>, detail: bool) -> f32 {
    // textureSampleLevel rather than textureSample: the raymarch loop breaks
    // out early, so it is not uniform control flow, and a sample that works
    // out its own mip level from screen-space derivatives is illegal there.
    // Which suits us, because the right level here has nothing to do with
    // screen space and everything to do with how long a march step is.
    var n = textureSampleLevel(noise_base, noise_sampler, q, lod.x).r;

    if (MIX_AMOUNT > 0.0 && (detail || MIX_IN_SHADOW)) {
        // The second read traverses the texture more slowly, so its features
        // are larger in world space and it can afford a finer mip.
        let mix_lod = max(0.0, lod.x + log2(MIX_SCALE));
        let m = textureSampleLevel(
            noise_base, noise_sampler, q * MIX_SCALE + MIX_OFFSET, mix_lod
        ).r;
        // Blending two fields that do not agree shrinks the spread of the
        // result: two independent values averaged have half the variance.
        // Rescaling about the mean puts it back, so the flattened histogram
        // the bake worked for survives and the coverage dial keeps meaning
        // what it says.
        let w = MIX_AMOUNT;
        let k = 1.0 / sqrt((1.0 - w) * (1.0 - w) + w * w);
        n = 0.5 + (mix(n, m, w) - 0.5) * k;
    }

    if (detail) {
        let d = textureSampleLevel(noise_detail, noise_sampler, q * DETAIL_SCALE, lod.y).r;
        // Erode inwards from the edges: push hardest where the base is
        // already thin and not at all where it is solid, which tears a smooth
        // blob into something with wisps on it.
        //
        // `d - 0.5`, not `d`: the detail field averages 0.5, so subtracting it
        // outright would drag the whole field down and quietly shift what the
        // coverage dial means. Centring it on zero adds detail without moving
        // the average, so coverage keeps the meaning the bake gave it.
        n -= (1.0 - n) * (d - 0.5) * DETAIL_STRENGTH;
    }
    return n;
}

// Which mip level of each noise texture a march of `step` metres should read.
//
// A step of `step` metres covers `step / voxel_size` voxels, and each mip
// level up doubles the size of a voxel, so the level that makes one step
// about one voxel wide is the base-2 log of that ratio. Below level 0 there
// is nothing more to show, hence the max.
fn mip_for_step(step: f32) -> vec2<f32> {
    let tile = dials.feature_size * TILE_SCALE;
    let base_voxel = tile / f32(textureDimensions(noise_base, 0).x);
    let detail_voxel = tile / (DETAIL_SCALE * f32(textureDimensions(noise_detail, 0).x));
    return vec2<f32>(
        max(0.0, log2(step / base_voxel)),
        max(0.0, log2(step / detail_voxel)),
    );
}

// ---------------------------------------------------------------------------
//  THE FIVE ALGORITHMS
//
//  `dials.algorithm` chooses which density field the march samples. Everything
//  else — the slab, the raymarch, the light march, the phase function, the
//  aerial perspective, the tone map — is identical between them. Only the
//  function answering "how much cloud is at this point" changes, which is
//  where the look of a cloud renderer actually lives.
//
//  0 fbm      One connected field over the whole sky, thresholded by
//             coverage. Cheap and dense; at a near-horizon pitch its sheets
//             smear into streaks converging on the vanishing point, because
//             a shallow ray crosses tens of kilometres of the same layer.
//  1 islands  The same field, multiplied by a gate from a much coarser read.
//             Cloud gathers into separate masses with real sky between them,
//             which is what breaks the streaking: a shallow ray now spends
//             most of its length in clear air.
//  2 puffs    No field at all. A horizontal grid of cells, one hash-placed
//             blob in each, eroded by a noise read. Discrete clouds with
//             gaps, and nothing periodic anywhere in it.
//  3 billow   fBm with a Worley field remapped into it, the standard
//             "Perlin-Worley" of production cloud renderers. Worley's level
//             sets bulge outwards where value noise's only undulate, which
//             is what makes a cloud look like cauliflower.
//  4 cumulus  Billow and islands at once: the bulged field inside the gate.
//             The most cloud-like of the field modes, and the most expensive
//             of the five.
// ---------------------------------------------------------------------------

const ALGO_FBM: u32 = 0u;
const ALGO_ISLANDS: u32 = 1u;
const ALGO_PUFFS: u32 = 2u;
const ALGO_BILLOW: u32 = 3u;
const ALGO_CUMULUS: u32 = 4u;

// Two of the modes are really one lever each, and they compose: `cumulus` is
// `billow`'s field shape inside `islands`' gate. Asking the question this way
// rather than with two more enum values keeps the dispatch in one place.
fn uses_islands(algorithm: u32) -> bool {
    return algorithm == ALGO_ISLANDS || algorithm == ALGO_CUMULUS;
}

fn uses_billow(algorithm: u32) -> bool {
    return algorithm == ALGO_BILLOW || algorithm == ALGO_CUMULUS;
}

// --- islands ---
// The gate is a read of the base texture at a fraction of its usual rate, so
// it varies over tens of kilometres rather than over one cloud. At 0.14 the
// gate's own tile is seven times the cloud tile — 73 km at the default dial,
// past MAX_DISTANCE, so the view never holds two copies of it.
const ISLAND_SCALE: f32 = 0.14;
// Read from a coarse mip: the gate wants the shape of the field, not its
// detail, and asking for mip 3 costs the same as asking for mip 0.
const ISLAND_LOD: f32 = 3.0;
// Width of the gate's soft edge. Too sharp and the masses have cut edges;
// too soft and the clear sky never fully clears.
const ISLAND_SOFT: f32 = 0.16;
// A shift, so the gate is not reading the same corner of the texture as the
// clouds and quietly correlating "where cloud is" with "where cloud is".
const ISLAND_OFFSET: vec3<f32> = vec3<f32>(0.61, 0.29, 0.13);
// Two tests in series each get `coverage` to this power, so their product is
// the dial again. The arithmetic says 0.5; the gate's soft edge lets through
// more than a hard test would, and the measured value that puts this mode on
// the same sky cover as mode 0 is higher. See the calibration note in
// CLAUDE.md.
const ISLAND_GAMMA: f32 = 0.72;

// --- puffs ---
// Cell size, in feature sizes. One cloud per cell, so this is how far apart
// the clouds are; the radius below is how big they are.
const PUFF_SPACING: f32 = 1.55;
// Blob radius, as a fraction of a cell. Above ~0.5 neighbouring clouds start
// to merge into a mass again.
const PUFF_RADIUS: f32 = 0.52;
// How far a centre wanders from its cell's middle, in cells. Without this the
// clouds sit on a visible grid.
//
// It is also half of what lets `puff_field` look at four cells instead of
// nine, which is the single biggest thing in this mode's cost. The argument,
// per axis: a centre moves at most PUFF_JITTER/2 = 0.25 cells off its own
// centre, and any cell outside the nearest four is at least 1.0 cells away
// along some axis, leaving 0.75 — while a blob's radius is at most
// PUFF_RADIUS * (1 + PUFF_SIZE_VARY/2) = 0.66. 0.66 < 0.75, so the five cells
// that are not searched provably cannot reach the sample. Raise either
// constant past that and clouds start vanishing as you walk into them.
const PUFF_JITTER: f32 = 0.50;
// How much flatter than wide a cloud is. Cumulus are wider than they are
// tall, and a sphere reads as a balloon.
const PUFF_SQUASH: f32 = 1.9;
// Where the blob's soft shell starts, as a fraction of its radius. The core
// inside this is solid; from here out it falls to nothing at the surface.
const PUFF_EDGE: f32 = 0.25;
// Spread of sizes and heights, so a sky is not one cloud stamped out N times.
const PUFF_SIZE_VARY: f32 = 0.55;
const PUFF_HEIGHT_VARY: f32 = 0.35;
// How hard the erosion noise bites into the shell. This is what turns a
// smooth ellipsoid into something cloud-shaped; 0 leaves bare balloons.
const PUFF_EROSION: f32 = 0.90;
// The erosion read's tile, in feature sizes. Deliberately *not* a whole
// multiple of PUFF_SPACING: if it were, every cloud would be eroded by the
// same part of the texture and the sky would be one shape repeated.
const PUFF_NOISE_SCALE: f32 = 0.6;
// How fast a puff fades at the floor and ceiling of the slab. A puff carries
// its own vertical shape, so this only has to stop it being cut off square.
const PUFF_SLAB_SOFTNESS: f32 = 0.08;
// An occupied cell is not a full cell of cloud: a blob of radius 0.52 cells,
// eroded, covers about two thirds of it. Without this the dial would read a
// third low against every other mode. It clamps at 1, which is the honest
// answer — a sky of separate clouds cannot become overcast, and past about
// 0.65 the dial stops doing anything.
const PUFF_COVER_GAIN: f32 = 1.55;
// The direction the cell grid runs in, as a unit vector. A square lattice
// aligned to the world axes puts its energy on them — measured as an
// axis_bias of 2.53, against 0.99 for the fBm field — which is the same
// defect the warp was built to fix for the value noise. cos/sin of 31.7
// degrees, written out rather than computed, because this is a per-sample
// rotation and two transcendentals per sample is not free.
const PUFF_GRID_DIR: vec2<f32> = vec2<f32>(0.85112, 0.52497);

// --- billow ---
// Worley tile relative to the cloud tile. 1.0 puts a cellular bump on about
// every cloud; higher is bumpier.
const BILLOW_SCALE: f32 = 1.4;
// How far towards the remapped field to go. 0 is plain fBm, 1 all billow.
const BILLOW_AMOUNT: f32 = 1.0;
// The remap both lifts the field and squashes its spread — measured, mean
// 0.675 and standard deviation 0.221 against mode 0's 0.529 and 0.292. Left
// alone that makes the same dial a cloudier sky with softer edges, so the
// mode could not be compared against the others at all. These three put the
// field back on mode 0's mean and spread, which is what the histogram
// flattening in the bake bought and what keeps the coverage dial honest.
const BILLOW_MEAN: f32 = 0.675;
const BILLOW_GAIN: f32 = 1.322;
const BILLOW_TARGET: f32 = 0.529;

// Four-and-four random numbers from a grid cell. One 32-bit avalanche gives
// four usable bytes; a second round of the same mixer gives four more, which
// is cheaper than hashing the cell twice from scratch.
struct CellHash {
    a: vec4<f32>,   // x,z jitter · size · exists
    b: vec4<f32>,   // height · spare
};

fn hash_cell(c: vec2<i32>) -> CellHash {
    var h = u32(c.x) * 0x8da6b343u + u32(c.y) * 0xd8163841u + 0x9e3779b9u;
    // lowbias32: two multiply-and-xorshift rounds with good avalanche, so two
    // cells one apart give completely unrelated bytes.
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    var g = h;
    g ^= g >> 16u;
    g *= 0x7feb352du;
    g ^= g >> 15u;

    var out: CellHash;
    out.a = vec4<f32>(vec4<u32>(h, h >> 8u, h >> 16u, h >> 24u) & vec4<u32>(255u)) * (1.0 / 255.0);
    out.b = vec4<f32>(vec4<u32>(g, g >> 8u, g >> 16u, g >> 24u) & vec4<u32>(255u)) * (1.0 / 255.0);
    return out;
}

// The blob field: 1 deep inside a cloud, 0 in clear sky, before erosion.
//
// Every point is inside at most a few cells' reach, so instead of asking
// "which clouds exist" we ask "which cells near me could hold one" — four of
// them, because the constants are chosen so a fifth provably cannot reach.
// The cost is four hashes and four distances, against the forty hashes a
// five-octave fBm used to cost, and unlike a texture it has no period at all:
// cell (10^6, 3) hashes to something unrelated to cell (0, 3).
fn puff_field(dp: vec3<f32>) -> f32 {
    let cell = dials.feature_size * PUFF_SPACING;
    // Into the grid's own frame, which is rotated against the world's.
    let g = vec2<f32>(
        dot(dp.xz, PUFF_GRID_DIR),
        dot(dp.xz, vec2<f32>(-PUFF_GRID_DIR.y, PUFF_GRID_DIR.x)),
    );
    // The four cells whose *centres* are nearest, not the four the point sits
    // between: centres are at half-integers, so shifting by half a cell before
    // the floor picks the right quad. See PUFF_JITTER for why four is enough.
    let base = vec2<i32>(floor(g / cell - 0.5));
    let thickness = CLOUD_TOP - CLOUD_BOTTOM;
    let mid = (CLOUD_BOTTOM + CLOUD_TOP) * 0.5;

    var d = 0.0;
    for (var dz = 0; dz <= 1; dz = dz + 1) {
        for (var dx = 0; dx <= 1; dx = dx + 1) {
            let c = base + vec2<i32>(dx, dz);
            let h = hash_cell(c);

            // Whether this cell has a cloud in it at all — the dial, scaled
            // by PUFF_COVER_GAIN so that the sky it ends up covering is the
            // dial rather than the cell count being it.
            if (h.a.w >= dials.coverage * PUFF_COVER_GAIN) {
                continue;
            }

            // Kept in grid space, and the sample point with it, so the
            // rotation costs one dot product per sample rather than one
            // rotation back per cell.
            let centre_xz = (vec2<f32>(c) + 0.5 + (h.a.xy - 0.5) * PUFF_JITTER) * cell;
            let r = cell * PUFF_RADIUS * (1.0 + PUFF_SIZE_VARY * (h.a.z - 0.5));
            // Clamp the vertical radius to the slab: at a 10 km feature size a
            // blob would otherwise be taller than the layer is deep and come
            // out as a flat-topped cylinder.
            let ry = min(r / PUFF_SQUASH, thickness * 0.45);
            let cy = mid + (h.b.x - 0.5) * thickness * PUFF_HEIGHT_VARY;

            let v = vec3<f32>(
                (g.x - centre_xz.x) / r,
                (dp.y - cy) / ry,
                (g.y - centre_xz.y) / r,
            );
            // `max`, not a sum: overlapping blobs should merge into one cloud
            // rather than stack into a denser one.
            d = max(d, 1.0 - smoothstep(PUFF_EDGE, 1.0, length(v)));
        }
    }
    return d;
}

// A puff, eroded. The erosion is the whole trick: a smooth ellipsoid reads as
// a balloon, and a noise read that eats into its shell while leaving the core
// alone turns it into a cloud.
fn puff_density(dp: vec3<f32>, lod: vec2<f32>, detail: bool) -> f32 {
    let d = puff_field(dp);
    // Nothing here: skip the fetch. Most samples in a puffy sky take this
    // branch, which is why the mode is affordable at all.
    if (d <= 0.0 || !detail) {
        return d;
    }

    let tile = dials.feature_size * PUFF_NOISE_SCALE;
    // `lod` was worked out for a tile of TILE_SCALE feature sizes. This read
    // crosses its texture faster by that ratio, so it needs a coarser mip by
    // the log of it, or the erosion fizzes in the distance.
    let level = max(0.0, lod.x + log2(TILE_SCALE / PUFF_NOISE_SCALE));
    let n = textureSampleLevel(noise_base, noise_sampler, dp / tile, level).r;

    // Bite into the shell: rescaling by what is left keeps a solid core solid
    // instead of thinning the whole cloud towards transparent.
    let bite = (1.0 - n) * PUFF_EROSION;
    return clamp((d - bite) / max(1.0 - bite, 0.001), 0.0, 1.0);
}

// Perlin-Worley. `n` is the value-noise field; this remaps it so that where a
// Worley feature point sits (w near 0) the floor is lifted and the field
// bulges outwards, and far from one (w near 1) it is left alone.
//
// remap(n, w - 1, 1, 0, 1) is the form it is usually written in: rescale n
// from the range [w-1, 1] onto [0, 1].
fn billow_value(q: vec3<f32>, n: f32, lod: f32, detail: bool) -> f32 {
    if (!detail && !BILLOW_IN_SHADOW) {
        return n;
    }
    // The Worley texture is half the base's resolution, so its voxels are
    // twice as big and it wants one mip level coarser for the same step.
    let level = max(0.0, lod - 1.0 + log2(BILLOW_SCALE));
    let w = textureSampleLevel(noise_worley, noise_sampler, q * BILLOW_SCALE, level).r;
    let pw = (n - (w - 1.0)) / (2.0 - w);
    let normalised = BILLOW_TARGET + (pw - BILLOW_MEAN) * BILLOW_GAIN;
    return clamp(mix(n, normalised, BILLOW_AMOUNT), 0.0, 1.0);
}

// How much cloud the field itself is asked for, once the gate has taken its
// share. Both tests get the dial to the same power, so their product is the
// dial again.
fn island_cover() -> f32 {
    return pow(dials.coverage, ISLAND_GAMMA);
}

// The gate for one view sample, worked out once and handed to every step of
// the shadow march that sample spawns. The same bargain as WARP_SHARED_SHADOW:
// the gate varies over tens of kilometres and the shadow march walks one, so
// the value it would compute for itself is the one it was given. Off the view
// ray it is 1.0, which is what the other three algorithms want anyway.
fn shared_gate(drifted: vec3<f32>, warp: vec3<f32>, tile: f32) -> f32 {
    if (!uses_islands(dials.algorithm)) {
        return 1.0;
    }
    return island_gate((drifted + warp) / tile, island_cover());
}

// The gate that turns an even wash of cloud into separate masses.
//
// Returned in 0..1, averaging `cover`: the base texture's histogram was
// flattened by the bake, so the fraction of it above `1 - cover` is `cover`
// itself, and a smoothstep centred there keeps that average while softening
// the edge.
fn island_gate(q: vec3<f32>, cover: f32) -> f32 {
    let m = textureSampleLevel(
        noise_base, noise_sampler, q * ISLAND_SCALE + ISLAND_OFFSET, ISLAND_LOD
    ).r;
    let t = 1.0 - cover;
    return smoothstep(t - ISLAND_SOFT, t + ISLAND_SOFT, m);
}

// Density of the cloud medium at a world-space point.
//
// The four algorithms differ only here. Each returns a density in 0..1 before
// the overall DENSITY multiplier: what the march integrates is the same shape
// of number whichever one is selected.
fn cloud_density_at(
    p: vec3<f32>,
    lod: vec2<f32>,
    detail: bool,
    warp: vec3<f32>,
    gate: f32,     // the islands gate from `shared_gate`; 1.0 for every other mode
) -> f32 {
    // Height through the layer, 0 at the bottom and 1 at the top.
    let h = (p.y - CLOUD_BOTTOM) / (CLOUD_TOP - CLOUD_BOTTOM);
    if (h < 0.0 || h > 1.0) {
        return 0.0;
    }

    // Scroll the noise field to make the clouds drift. Moving the *sample
    // point* against the wind is the same as moving the clouds with it.
    let tile = dials.feature_size * TILE_SCALE;
    let drifted = p + WIND * u.time;

    if (dials.algorithm == ALGO_PUFFS) {
        // A puff already has a top and a bottom of its own, so the slab only
        // has to stop one hanging out of the layer. The flat-based profile
        // the field modes use would slice every blob off at the knees.
        let slab = smoothstep(0.0, PUFF_SLAB_SOFTNESS, h)
                 * (1.0 - smoothstep(1.0 - PUFF_SLAB_SOFTNESS, 1.0, h));
        return puff_density(drifted, lod, detail) * slab * DENSITY;
    }

    // Vertical profile: clouds have a flattish base and a soft, rounded top,
    // so fade density in quickly above the base and out gradually below the top.
    let profile = smoothstep(0.0, CLOUD_BASE_SOFTNESS, h)
                * (1.0 - smoothstep(CLOUD_TOP_SOFTNESS, 1.0, h));

    let q = (drifted + warp) / tile;
    var n = cloud_noise(q, lod, detail);

    // Coverage: everything below the threshold becomes clear sky, and what
    // remains is rescaled back to [0,1] so the clouds keep their full density
    // instead of thinning out as coverage drops.
    //
    // The bake flattened the field's histogram, so this dial is close to
    // literal: 0.45 means about 45% of the layer is above the threshold.
    // Two independent tests have to pass for a point to be cloud in `islands`,
    // so the sky it covers is the product of the two. Giving each a root of
    // the dial makes that product the dial again — the mode changes where the
    // cloud is, not how much of it there is.
    var cover = dials.coverage;

    if (uses_islands(dials.algorithm)) {
        cover = island_cover();
    }
    if (uses_billow(dials.algorithm)) {
        n = billow_value(q, n, lod.x, detail);
    }

    var threshold = 1.0 - cover;

    // Weather: a coarse read of the same texture nudges that threshold up and
    // down across the sky. Centred on zero, so the dial keeps the meaning the
    // histogram flattening gave it rather than quietly gaining cloud.
    if (WEATHER_AMOUNT > 0.0) {
        let w = cloud_noise(drifted * (WEATHER_SCALE / tile), lod, false);
        threshold += (w - 0.5) * WEATHER_AMOUNT;
    }

    let shaped = clamp((n - threshold) / max(cover, 0.001), 0.0, 1.0);

    return shaped * gate * profile * DENSITY;
}

// The view ray wants every bit of detail the step length can carry.
fn cloud_density(p: vec3<f32>, lod: vec2<f32>, warp: vec3<f32>, gate: f32) -> f32 {
    return cloud_density_at(p, lod, true, warp, gate);
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
// These samples skip the detail texture: see `cloud_noise`.
fn light_march(p: vec3<f32>, lod: vec2<f32>, warp: vec3<f32>, tile: f32, gate: f32) -> f32 {
    var optical_depth = 0.0;
    var t = 0.0;
    var step = LIGHT_STEP_SIZE;

    for (var i = 0; i < LIGHT_STEPS; i = i + 1) {
        // Sample at the middle of the step.
        let sp = p + u.sun_dir * (t + step * 0.5);
        // See WARP_SHARED_SHADOW: reusing the view sample's displacement here
        // turns seven warp evaluations per sample into one.
        var w = warp;
        if (!WARP_SHARED_SHADOW) {
            w = warp_at(sp + WIND * u.time, tile);
        }
        optical_depth += cloud_density_at(sp, lod, false, w, gate) * step;
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
    let tile = dials.feature_size * TILE_SCALE;

    // Both of these are the same for every sample along this ray, so they are
    // worked out once per pixel rather than once per step: the mip level
    // follows from the step length, and the phase function from the angle
    // between the view ray and the sun.
    let lod = mip_for_step(dt);
    // The shadow march's own steps start shorter than the view ray's, so it
    // can afford a sharper level, but never a sharper one than the view ray
    // is using — detail the eye cannot see has no business casting a shadow.
    let light_lod = min(lod, mip_for_step(LIGHT_STEP_SIZE));
    let phase = cloud_phase(dot(dir, u.sun_dir));

    // Where in the first step this pixel starts sampling.
    let offset = mix(0.5, dither(pixel), JITTER);

    for (var i = 0; i < PRIMARY_STEPS; i = i + 1) {
        // Sample partway into each step: 0.5 would be the midpoint, which
        // integrates more accurately than the near edge, and the dither
        // spreads that offset out across neighbouring pixels.
        let t = hit.x + (f32(i) + offset) * dt;
        let p = origin + dir * t;

        // The puff grid has no period to break, so it has no use for the
        // warp's transcendentals either.
        let drifted = p + WIND * u.time;
        var warp = vec3<f32>(0.0);
        if (dials.algorithm != ALGO_PUFFS) {
            warp = warp_at(drifted, tile);
        }
        let gate = shared_gate(drifted, warp, tile);
        let density = cloud_density(p, lod, warp, gate);
        if (density <= 0.0) {
            continue;
        }

        let sigma = density * EXTINCTION;
        let step_transmittance = exp(-sigma * dt);

        // How much sun reaches this point through the cloud above it.
        let sun_visibility = light_march(p, light_lod, warp, tile, gate);
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
//  Field dump — analysis only, never drawn in the window.
//
//  Renders a top-down orthographic slice of the density field straight out of
//  `cloud_density_at`, so a metric computed on it measures the real field
//  rather than a reimplementation of it. `--dump-field` writes it to a PNG in
//  a *linear* format, not sRGB, because a spectrum of gamma-encoded values is
//  a spectrum of the wrong signal.
//
//  The patch is sized in tile units rather than metres, so dumps taken at
//  different feature sizes line up: if the field repeats with the texture,
//  the repeat is always FIELD_TILES cycles across the image, whatever the
//  dial says.
// ---------------------------------------------------------------------------

const FIELD_TILES: f32 = 8.0;    // base-texture tiles across the dump
const FIELD_HEIGHT: f32 = 0.5;   // where in the slab to slice: 0 = bottom

@fragment
fn fs_field(in: VsOut) -> @location(0) vec4<f32> {
    let tile = dials.feature_size * TILE_SCALE;
    let xz = in.ndc * 0.5 * FIELD_TILES * tile;
    let y = CLOUD_BOTTOM + (CLOUD_TOP - CLOUD_BOTTOM) * FIELD_HEIGHT;
    let p = vec3<f32>(xz.x, y, xz.y);

    // Mip 0 throughout: this asks what the field *is*, not what some
    // particular march step length would make of it.
    let lod = vec2<f32>(0.0, 0.0);

    var warp = vec3<f32>(0.0);
    if (dials.algorithm != ALGO_PUFFS) {
        warp = warp_at(p + WIND * u.time, tile);
    }
    let gate = shared_gate(p + WIND * u.time, warp, tile);
    let full = cloud_density_at(p, lod, true, warp, gate);
    let base = cloud_density_at(p, lod, false, warp, gate);

    // The raw field, before the coverage threshold. That threshold is a hard
    // nonlinearity and manufactures harmonics of its own, so the honest input
    // to a spectrum is the field itself. Which field that is depends on the
    // algorithm: for puffs it is the blob placement before erosion, which is
    // the thing whose repetition the metric is asking about.
    let drifted = p + WIND * u.time;
    var raw = 0.0;
    if (dials.algorithm == ALGO_PUFFS) {
        raw = puff_field(drifted);
    } else {
        let q = (drifted + warp) / tile;
        raw = cloud_noise(q, lod, true);
        if (uses_billow(dials.algorithm)) {
            raw = billow_value(q, raw, lod.x, true);
        }
    }

    return vec4<f32>(full, base, raw, 1.0);
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
