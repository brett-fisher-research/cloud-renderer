// The cloud layer: one model in the style of Guerrilla's Nubis.
//
// Answers three questions: how much cloud is at a point (`cloud_density`),
// what a view ray sees of it up to a given distance (`march_clouds`), and how
// much sun gets through it to a point below (`cloud_shadow`).

// ===========================================================================
//  Tunables
// ===========================================================================

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
// How strongly the detail texture eats into the cloud's edges when the
// wispiness dial is at 1. The dial's default, 0.5, gives 0.32.
const DETAIL_STRENGTH_MAX: f32 = 0.64;
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

const GROUND_SHADOW_STEPS: i32 = 12; // cloud samples for shadows on the ground

@group(0) @binding(2) var noise_shape: texture_3d<f32>;
@group(0) @binding(3) var noise_detail: texture_3d<f32>;
@group(0) @binding(4) var weather_map: texture_2d<f32>;

// ===========================================================================
//  Cloud density
// ===========================================================================

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
        let detail_strength = dials.wispiness * DETAIL_STRENGTH_MAX;
        if (s <= -detail_strength) {
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
        s += (0.5 - billow) * 2.0 * detail_strength;
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

// March the cloud shell along a view ray, as far as `t_max` metres: whatever
// the ray ends on. Cloud beyond that is hidden and is not marched.
fn march_clouds(dir: vec3<f32>, pixel: vec2<f32>, in: VsOut, t_max: f32) -> Clouds {
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
    if (t_in >= t_max) {
        // The ray ends before it reaches the layer.
        return out;
    }
    let t_shell = sphere_exit(b, (CAMERA_HEIGHT - CLOUD_TOP) * (2.0 * big_r + CAMERA_HEIGHT + CLOUD_TOP));
    let t_out = min(t_shell, t_max);

    // Sunlight at the point where the ray meets the clouds. The sun's
    // height above the *local* horizon there differs from the camera's:
    // clouds far towards a setting sun are still lit after it sets here.
    let nu = dot(dir, u.sun_dir);
    let cloud_r = R_GROUND + 0.0005 * (CLOUD_BOTTOM + CLOUD_TOP);
    let t_mid = 0.0005 * (t_in + t_shell);
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

// How much direct sun reaches a point through the clouds above it. `xz` is
// the point's map position and `altitude` its height above the sphere, both
// in metres.
fn cloud_shadow(xz: vec2<f32>, altitude: f32) -> f32 {
    let rise = u.sun_dir.y;
    // Only the part of the layer above the point can shade it.
    let base = max(CLOUD_BOTTOM, altitude);
    let thickness = CLOUD_TOP - base;
    if (rise < 0.03 || thickness <= 0.0) {
        return 1.0;
    }
    // Walk the sun ray up through the layer. With the sun low that is a long
    // slanted path, so the samples must be close enough together not to step
    // over a cloud — otherwise each sample height casts its own separate
    // copy of the shadow.
    var depth = 0.0;
    let n = f32(GROUND_SHADOW_STEPS);
    let step = thickness / n / rise;
    let lod = lods_for(step);
    for (var i = 0; i < GROUND_SHADOW_STEPS; i++) {
        let up = (f32(i) + 0.5) / n * thickness;
        let height = (base - CLOUD_BOTTOM + up) / (CLOUD_TOP - CLOUD_BOTTOM);
        let s = (base - altitude + up) / rise;
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
