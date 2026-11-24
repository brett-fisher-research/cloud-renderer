// The frame: turns what the features report into a picture.
//
// Per pixel it builds a view ray, asks the terrain for the surface the ray
// ends on, asks the clouds what lies in front of that, marches the air on
// either side of the cloud, composites, exposes, tone maps, and blends with
// the frames before. A second fragment entry presents the result.

// ===========================================================================
//  Tunables
// ===========================================================================

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

// --- Sky and camera ---------------------------------------------------------
const SKY_STEPS: i32 = 14;           // atmosphere samples on a clear ray
const SKY_STEPS_NEAR: i32 = 6;       // ...in front of a cloud
const SKY_STEPS_FAR: i32 = 8;        // ...behind it
const SUN_DISC: f32 = 1.0;           // 0 hides the sun's disc
// Lens glare around the sun, as a fraction of the disc's own brightness.
const SUN_GLARE: f32 = 0.004;
// Auto exposure. The sky at dusk is thousands of times darker than at noon;
// a camera compensates. 1.0 would compensate fully and make dusk look like
// day; 0 not at all.
const EXPOSURE_BASE: f32 = 5.5;
const EXPOSURE_ADAPTATION: f32 = 0.65;

// The accumulated image: last frame's in `fs_main`, this frame's in
// `fs_present`. Tone-mapped linear colour.
@group(0) @binding(9) var history: texture_2d<f32>;

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
    let r = R_GROUND + 0.001 * (dials.cloud_base + 0.5 * CLOUD_THICKNESS);
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
    let sun_on_cloud = sun_transmittance(R_GROUND + 0.001 * (dials.cloud_base + 0.5 * CLOUD_THICKNESS), u.sun_dir.y);

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

// Radiance leaving a surface towards the eye: direct sun, dimmed by the air
// and by any cloud above, plus skylight. The surface is matte, so the eye's
// direction does not matter: light of irradiance E gives albedo * E / pi.
fn light_surface(surface: Surface, dir: vec3<f32>, in: VsOut) -> vec3<f32> {
    let r0 = R_GROUND + CAMERA_HEIGHT * 0.001;
    let r = R_GROUND + surface.altitude * 0.001;
    // The sun's height above the horizon *there*, which on a round planet
    // differs from its height here.
    let mu_sun = clamp(
        (r0 * u.sun_dir.y + surface.distance * 0.001 * dot(dir, u.sun_dir)) / r, -1.0, 1.0);
    // The normal is in the camera's frame; correct its up for that too.
    let facing = max(dot(surface.normal, u.sun_dir) + mu_sun - u.sun_dir.y, 0.0);
    let sun = sun_transmittance(r + 0.001, mu_sun) * facing
        * cloud_shadow(dir.xz * surface.distance, surface.altitude);
    // A slope sees less sky than level ground does.
    let sky = in.ambient_sky * (0.5 + 0.5 * surface.normal.y);
    return surface.albedo * (sun / PI + sky);
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

    // Where the ray ends: on the ground, or at the top of the atmosphere.
    let surface = terrain_surface(dir, fract(pixel_noise(pixel) + u.jitter));
    var t_end = sphere_exit(b, (r0 - R_TOP) * (r0 + R_TOP));
    var steps = SKY_STEPS;
    // What is at the end of it: the lit ground, or the sun's disc on black.
    var behind = vec3<f32>(0.0);
    if (surface.hit) {
        t_end = surface.distance * 0.001;
        steps = SKY_STEPS_NEAR;
        behind = light_surface(surface, dir, in);
    } else {
        // The sun itself: a disc half a degree across, far brighter than
        // anything else, dimmer towards its edge ("limb darkening").
        let sun_angle = acos(clamp(nu, -1.0, 1.0));
        let sun_radius = 0.00465;
        let pixel_angle = 2.0 * u.fov_tan / u.resolution.y;
        let disc_cover = SUN_DISC * saturate((sun_radius - sun_angle) / pixel_angle + 0.5);
        if (disc_cover > 0.0) {
            let edge = saturate(sun_angle / sun_radius);
            let limb = 1.0 - 0.6 * (1.0 - sqrt(1.0 - edge * edge));
            behind = vec3<f32>(disc_cover * limb / (PI * sun_radius * sun_radius));
        }
    }

    // Clouds in front of that, and nothing behind it: the march stops at the
    // surface, so cloud hidden by a mountain is never paid for.
    let clouds = march_clouds(dir, pixel, in, t_end * 1000.0);

    var col: vec3<f32>;
    if (clouds.through < 0.999) {
        // Air in front of the cloud fades and tints it with distance
        // ("aerial perspective"); air behind it is the sky, or the hillside,
        // seen through its gaps. Split the ray at the cloud and march each
        // part.
        let t_cloud = clamp(clouds.distance * 0.001, 0.0, t_end);
        let near = atmosphere(r0, mu, nu, mu_sun0, 0.0, t_cloud, SKY_STEPS_NEAR);
        let far = atmosphere(r0, mu, nu, mu_sun0, t_cloud, t_end, SKY_STEPS_FAR);
        let background = far.light + far.through * behind;
        col = near.light + near.through * (clouds.light + clouds.through * background);
    } else {
        let air = atmosphere(r0, mu, nu, mu_sun0, 0.0, t_end, steps);
        col = air.light + air.through * behind;
    }

    // Glare: light from the sun spreading inside the lens. It lies over the
    // whole picture, clouds included, but only while the sun itself is in
    // clear view of the camera.
    let glare_angle = acos(clamp(nu, -1.0, 1.0)) / 0.00465;
    col += in.sun_glare * pow(1.0 + glare_angle * glare_angle, -1.5);

    // Exposure scales linear radiance like a camera's shutter; the tone map
    // then compresses the bright end.
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
