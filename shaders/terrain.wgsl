// The ground: a heightfield over the round planet.
//
// Answers one question: where does a view ray end, and what is the surface
// like there (`terrain_surface`)? It does no lighting; frame.wgsl lights
// whatever surface it is handed.
//
// Graphics note: there is no mesh. The heightmap (src/terrain.rs) says how
// high the ground is at each map position, and a ray is walked forwards until
// it is below that height. Two things make the walk affordable:
//
//   - Over ground it is close to, a ray takes steps proportional to its
//     height above the ground: it cannot hit something it is far above.
//   - Across open air it consults a coarse "nothing here is taller than X"
//     copy of the map and strides a whole coarse texel at a time. A ray
//     crossing the plain reaches the mountains in a handful of steps.

// ===========================================================================
//  Tunables
// ===========================================================================

// Off, the ground is the bare sphere: the picture from before terrain
// existed, which is the control for measuring what terrain costs.
const TERRAIN_ENABLED: bool = true;
// Metres across the heightmap. Must match EXTENT in src/terrain.rs.
const TERRAIN_EXTENT: f32 = 96000.0;

// --- The march ---------------------------------------------------------------
// A step is this fraction of the ray's height above the ground. Lower is
// safer on steep slopes (a ray can step through a thin ridge) and slower.
const TERRAIN_STEP_SLOPE: f32 = 0.4;
// ...but never shorter than this fraction of the distance travelled: far
// away, a pixel covers more ground than a short step would resolve.
const TERRAIN_STEP_MIN: f32 = 0.002;
const TERRAIN_MAX_STEPS: i32 = 400;
// Bisections to pin the surface down once a step lands below it.
const TERRAIN_REFINE_STEPS: i32 = 6;
// Vary each step's length per pixel and per frame, so a missed ridge becomes
// grain for accumulation to average, not a hole. 0 = off.
const TERRAIN_JITTER: f32 = 0.3;
// The coarsest and finest mip of the bound map used for striding. Each level
// doubles the stride; the finest sets where striding hands over to stepping.
const TERRAIN_SKIP_COARSEST: f32 = 7.0;
const TERRAIN_SKIP_FINEST: f32 = 2.0;
// Heightmap mip relative to the pixel's footprint on the ground.
const TERRAIN_LOD_BIAS: f32 = 0.0;

// --- Materials ---------------------------------------------------------------
const GROUND_COLOR: vec3<f32> = vec3<f32>(0.16, 0.17, 0.11);
const ROCK_COLOR: vec3<f32> = vec3<f32>(0.19, 0.17, 0.15);
// Slopes (0 flat, 1 vertical) between which grass gives way to bare rock.
const ROCK_SLOPE_MIN: f32 = 0.10;
const ROCK_SLOPE_MAX: f32 = 0.35;

// R + G: height in metres. B: an upper bound on the height of everything
// within one texel (of the mip sampled) of the sample point.
@group(0) @binding(10) var terrain_map: texture_2d<f32>;

// ===========================================================================
//  The surface a ray ends on
// ===========================================================================

struct Surface {
    hit: bool,
    distance: f32,        // metres along the ray
    altitude: f32,        // metres above the sphere
    normal: vec3<f32>,    // unit, pointing out of the ground
    albedo: vec3<f32>,    // fraction of light reflected, per colour
};

fn terrain_height(xz: vec2<f32>, lod: f32) -> f32 {
    let s = textureSampleLevel(terrain_map, table_sampler, xz / TERRAIN_EXTENT + 0.5, lod);
    return s.r + s.g;
}

// Height of a view ray above the sphere at distance `t`, in a form that never
// subtracts two planet-sized numbers. `b` is the camera's radius times the
// ray's cosine to straight-up.
fn ray_altitude(b: f32, cam_r: f32, t: f32) -> f32 {
    let x = 2.0 * t * b + t * t;
    return CAMERA_HEIGHT + x / (sqrt(cam_r * cam_r + x) + cam_r);
}

fn terrain_surface(dir: vec3<f32>, noise: f32) -> Surface {
    var out: Surface;
    out.hit = false;
    out.distance = 0.0;
    out.altitude = 0.0;
    out.normal = vec3<f32>(0.0, 1.0, 0.0);
    out.albedo = GROUND_COLOR;

    // The bare sphere first: the same quadratic as `sphere_exit`, but for a
    // sphere we are outside of, and we want the near root. Worked in
    // kilometres like the atmosphere. Terrain is never below the sphere, so
    // this is also the furthest any ray needs marching.
    let cam_km = CAMERA_HEIGHT * 0.001;
    let b_km = (R_GROUND + cam_km) * dir.y;
    let c_km = cam_km * (2.0 * R_GROUND + cam_km);
    let disc = b_km * b_km - c_km;
    let meets_sphere = dir.y < 0.0 && disc > 0.0;
    var t_sphere = 0.0;
    if (meets_sphere) {
        t_sphere = c_km / (sqrt(disc) - b_km) * 1000.0;
        out.hit = true;
        out.distance = t_sphere;
    }
    if (!TERRAIN_ENABLED) {
        return out;
    }

    let big_r = R_GROUND * 1000.0;
    let cam_r = big_r + CAMERA_HEIGHT;
    let b = cam_r * dir.y;

    // The top mip's bound is the height of the tallest summit. A ray is done
    // once it climbs above that, or runs off the map.
    let summit = textureLoad(terrain_map, vec2<i32>(0), i32(textureNumLevels(terrain_map)) - 1).b;
    var t_end = 0.5 * TERRAIN_EXTENT / max(length(dir.xz), 1.0e-4);
    if (meets_sphere) {
        t_end = min(t_end, t_sphere);
    } else if (summit > CAMERA_HEIGHT) {
        t_end = min(t_end, sphere_exit(b, (CAMERA_HEIGHT - summit) * (2.0 * big_r + CAMERA_HEIGHT + summit)));
    } else {
        return out;
    }

    let texel = TERRAIN_EXTENT / f32(textureDimensions(terrain_map).x);
    let pixel_angle = 2.0 * u.fov_tan / u.resolution.y;
    // The ray's lowest point is where it stops descending, at t = -b.
    let t_lowest = max(-b, 0.0);

    var t = 0.0;
    var t_above = 0.0;   // the last distance known to be above the ground
    var level = TERRAIN_SKIP_COARSEST;
    var below = false;
    // Out of steps counts as a hit. The rays that run out are the ones
    // skimming a long valley floor a few metres up; calling that the floor is
    // a far smaller error than calling it a hole with sky behind it.
    var spent = true;
    for (var i = 0; i < TERRAIN_MAX_STEPS; i++) {
        if (t >= t_end) {
            spent = false;
            break;
        }
        let p = dir.xz * t;

        // Stride: if the ray stays above everything near here for a whole
        // texel of this level, cross it untested and try a coarser level
        // next. Otherwise drop a level and look again.
        let stride = texel * exp2(level);
        let lowest = ray_altitude(b, cam_r, clamp(t_lowest, t, t + stride));
        let bound = textureSampleLevel(terrain_map, table_sampler, p / TERRAIN_EXTENT + 0.5, level).b;
        if (lowest > bound) {
            t += stride;
            t_above = t;
            level = min(level + 1.0, TERRAIN_SKIP_COARSEST);
            continue;
        }
        if (level > TERRAIN_SKIP_FINEST) {
            level -= 1.0;
            continue;
        }

        // Step: close to the ground. Compare against the real height.
        let lod = max(log2(t * pixel_angle / texel) + TERRAIN_LOD_BIAS, 0.0);
        let clearance = ray_altitude(b, cam_r, t) - terrain_height(p, lod);
        if (clearance <= 0.0) {
            below = true;
            spent = false;
            break;
        }
        t_above = t;
        let wobble = 1.0 - TERRAIN_JITTER * fract(noise + 0.618034 * f32(i));
        t += max(clearance * TERRAIN_STEP_SLOPE, t * TERRAIN_STEP_MIN) * wobble;
    }
    if (!below && !spent) {
        // Never went under: the bare sphere if the ray reaches it, else sky.
        return out;
    }

    // The surface is between the last point above and this one below. Halve
    // the gap a few times.
    var t_below = t;
    let lod = max(log2(t * pixel_angle / texel) + TERRAIN_LOD_BIAS, 0.0);
    for (var i = 0; i < TERRAIN_REFINE_STEPS; i++) {
        let mid = 0.5 * (t_above + t_below);
        if (ray_altitude(b, cam_r, mid) > terrain_height(dir.xz * mid, lod)) {
            t_above = mid;
        } else {
            t_below = mid;
        }
    }
    let t_hit = 0.5 * (t_above + t_below);
    let p = dir.xz * t_hit;

    // The normal from the slope: how much the height changes one texel to
    // either side, each way.
    let e = texel * exp2(lod);
    let dx = terrain_height(p + vec2<f32>(e, 0.0), lod) - terrain_height(p - vec2<f32>(e, 0.0), lod);
    let dz = terrain_height(p + vec2<f32>(0.0, e), lod) - terrain_height(p - vec2<f32>(0.0, e), lod);
    let normal = normalize(vec3<f32>(-dx, 2.0 * e, -dz));

    out.hit = true;
    out.distance = t_hit;
    out.altitude = terrain_height(p, lod);
    out.normal = normal;
    out.albedo = mix(GROUND_COLOR, ROCK_COLOR, smoothstep(ROCK_SLOPE_MIN, ROCK_SLOPE_MAX, 1.0 - normal.y));
    return out;
}
