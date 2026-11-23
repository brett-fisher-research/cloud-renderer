# cloud-renderer

A small real-time volumetric cloud renderer in Rust, built as a graphics-programming learning
project. A window shows a physically based sky over a flat plain, with a layer of sunlit cumulus
drifting across it, casting shadows on the ground.

Brett is an experienced web developer and new to graphics programming. Introduce new graphics
concepts briefly when they first appear, in replies and in code comments.

## Architecture

The split is deliberate: the CPU side stays small and boring so that all the interesting work
lives in one shader you can edit while the program runs.

### CPU side (Rust)

Opens a window, bakes five textures once at startup, and each frame writes two small uniform
buffers and draws one full-screen triangle twice: once for the clouds, once to present them. It
does no geometry, no scene graph and no per-pixel work.

- `src/main.rs` — argument parsing, picks windowed or headless mode.
- `src/gpu.rs` — wgpu instance, adapter, device and queue. Vulkan-only; logs every adapter it sees.
- `src/noise.rs` — bakes the three tiling noise textures: cloud shape, edge detail, weather map.
- `src/atmosphere.rs` — bakes the two sky tables: transmittance and multiple scattering.
- `src/camera.rs` — look-around camera and the `Uniforms` struct sent to the GPU.
- `src/params.rs` — the four live dials and their slider ranges.
- `src/ui.rs` — the egui panel the dials are drawn on.
- `src/renderer.rs` — pipelines, bindings, the history textures and per-frame temporal state,
  and shader compilation with hot-reload.
- `src/app.rs` — winit event loop, mouse input, FPS counter.
- `src/screenshot.rs`, `src/bench.rs` — headless still and headless GPU-timed benchmark.
- `src/defines.rs`, `src/watch.rs` — `--def` constant overrides and shader file watching.

### Shader side (WGSL)

`shaders/clouds.wgsl` holds everything visual. Every tunable is a named `const` in one block at
the top and changes on save. Four values are live dials instead, in `struct Params`: coverage,
feature size, shadow strength and exposure.

Per pixel, the fragment shader:

1. Builds a view ray from the camera basis.
2. If the ray hits the ground: shades it with sun, skylight and the shadow of the clouds above.
3. Otherwise marches the cloud layer, a spherical shell between `CLOUD_BOTTOM` and `CLOUD_TOP`
   around a round planet. The shell curves down behind the horizon, so there is no vanishing
   point for the layer to pile into.
4. Marches the atmosphere twice, in front of the cloud and behind it. The near part fades distant
   cloud into haze (aerial perspective); the far part is the sky seen through the gaps.
5. Composites, adds lens glare around the sun, exposes and tone maps (ACES).
6. Blends the result with the accumulated previous frames. See Temporal accumulation below.

A second fragment entry, `fs_present`, copies the accumulated image to the screen and dithers it.

The vertex shader runs three times a frame, not four million, so it computes what is the same for
every pixel: mean sky colour (ambient light), the ground's bounce light, auto exposure, and how
much of the sun the camera can see. These reach the fragment shader as `flat` varyings.

### The cloud model

One model, in the style of Guerrilla's Nubis (Horizon Zero Dawn). `cloud_density` answers "how
much cloud is at this point" from three textures:

- Weather map (2D, read top-down, twice). The first read says where clouds are and how tall. The
  second, zoomed out and rotated, raises or lowers coverage by region, so clouds gather in groups
  and no two copies of the tile look alike.
- Shape (3D Perlin-Worley). Thresholded against a height profile that rises sharply at the base
  and tapers towards the top: flat bottoms, rounded lumpy summits.
- Detail (3D Worley). Pushes the surface in and out for the cauliflower texture. Read a second
  time at a finer scale near the camera only.

The march takes long strides through clear air and backs up to short steps on entering cloud.
Both lengths grow with distance, and the texture mip follows the step length, which is what keeps
distant cloud from shimmering.

Lighting at each cloud sample:

- A light march of `LIGHT_STEPS` samples towards the sun gives the optical depth. It sees a
  softer cloud edge than the view march (`SHADOW_SOFTNESS`) so shadows vary smoothly.
- Multiple scattering is approximated by `MS_OCTAVES` octaves, each dimmer, deeper-reaching and
  less directional, plus a slow diffuse glow (`DIFFUSE_GAIN`).
- Later octaves and the glow use the cloud depth overhead (from the weather map) instead of the
  exact path to the sun. With the sun path alone, every bump cast a streak through the cloud.
- Ambient light is the mean sky from above and ground bounce from below.

### The sky

Hillaire's 2020 atmosphere model. `src/atmosphere.rs` bakes a transmittance table (how much
sunlight reaches a point at a given altitude and sun angle) and a multiple-scattering table. The
shader marches Rayleigh, Mie and ozone along the view ray with one lookup into each per step.
Sunset reddening, twilight, and the planet's shadow all fall out of it. The sun's colour on the
clouds comes from the same transmittance table.

### Temporal accumulation

Every ray starts its march at a random offset, which hides banding but leaves grain. Instead of
more samples per frame, each frame is blended with the frames before it, so a pixel shows the
average of roughly the last twenty.

- Two history textures (half-float) swap roles each frame. The cloud pass reads one and writes
  the other; the present pass copies the one just written to the screen.
- `TEMPORAL_BLEND` (0.1) is the new frame's share. While the history is young the shader averages
  evenly instead (frame n gets 1/n), so a resize or shader reload converges in a few frames.
- Reprojection: when the camera turns, last frame's value for a direction is elsewhere on screen,
  so the direction is projected through last frame's camera basis. The camera never travels, so
  this is exact at any distance and there is no history rejection to tune.
- The history is read with a 9-tap Catmull-Rom filter (`HISTORY_SHARP`). Bilinear blurs a little
  on every read, and during a pan that compounds until the whole picture goes soft.
- Sub-pixel jitter (`SUBPIXEL_JITTER`) aims each frame's rays through a different point of the
  pixel, so the same average anti-aliases cloud edges and the horizon.
- Blending happens after tone mapping, so one frame's glimpse of the sun's disc cannot outweigh
  every other frame.
- `Renderer::update` owns the temporal state: previous camera basis, frame counter, the jitter
  sequences (computed in f64 on the CPU) and the history size. Callers pass a plain camera.
- Drifting cloud and dial changes are not reprojected; they simply lag by the averaging window,
  about a fifth of a second. No smear is visible on the nearest clouds at the default wind.

### Conventions worth knowing

- Units are metres and seconds for clouds, kilometres for the atmosphere. The camera sits at
  `CAMERA_HEIGHT` (150 m, a hilltop) so the ground reaches a horizon 44 km away and fades into haze.
- Colours are linear radiance with the sun's illuminance as 1.0. Tone mapping and sRGB encoding
  happen at the very end.
- `Uniforms` in `src/camera.rs` and in the shader must match exactly. Each `vec3` takes a 16-byte
  row and uses 12, so one float follows it; four of those slots carry temporal values (`jitter`,
  `history_frames`, `subpixel_x`, `subpixel_y`). A mismatch gives silent garbage. `Params` is four
  plain floats, exactly one row; a fifth dial needs three padding floats beside it on both sides.
- The atmosphere constants are duplicated in `src/atmosphere.rs` and at the top of the shader and
  must match, because the tables are baked from one and read by the other.
- Bindings in `@group(0)`: uniforms 0 and 1, shape 2, detail 3, weather 4, transmittance 5,
  multiple scattering 6, repeating sampler 7, clamping sampler 8, history 9.
- The weather map is half-float. As 8-bit, its thresholded edges followed quantisation steps.
- Sphere intersections and altitudes use rearranged formulas (`sphere_exit`) that never subtract
  two planet-sized numbers; the textbook forms lose all precision in 32-bit floats.
- Noise shape lives in `src/noise.rs` and needs a rebuild to change. How it is used (`*_TILE`,
  `DETAIL_STRENGTH`, `EDGE_WIDTH`) is shader constants and hot-reloads.

## Measuring

Everything runs headless, so a change can be judged without anyone at the keyboard.

- `tools/shots <name> [flags]` renders twelve standard views and tiles them into
  `out/<name>.png`. Look at this after any change to the image.
- `tools/perf [flags]` prints GPU milliseconds for fourteen scenes at 2560x1440.
- `tools/temporal [view flags]` scores temporal accumulation three ways, each with a control:
  error against a converged reference, flicker between consecutive frames, and error at the end
  of a camera pan. Run it after touching anything temporal.
- `tools/lab run <name> [--def K=V]...` runs the scene matrix, scores it and appends a row per
  scene to `docs/experiments.jsonl`. `tools/lab report` prints the frontier so far.
- `--def NAME=VALUE` overrides any shader `const` for one run. A name matching nothing is a hard
  error. The tools pass it through.
- `--screenshot` accumulates 32 frames by default, with the clock advancing as at 60 fps, so a
  still shows what the window shows. `--accumulate 1` gives one raw frame. `--pan YAW,PITCH`
  turns the camera at that rate (radians per second) over the lead-up frames, ending on the
  requested view; that is how reprojection is tested headlessly.
- `--bench` times the whole frame, cloud pass plus present pass, with a full history.
- `--dump-field out.png` renders the density top-down to a linear target: R is density a third of
  the way up the layer, G the same without detail, B the weather map's cover signal.

`tools/fieldstat` scores a field dump for repetition (`tile_lock`, `acorr_peak`, `axis_bias`) and
`tools/imgstat` scores a render for contrast and detail. Current field scores: `acorr_peak` 0.40
against a floor of 0.26, `tile_lock` 0.15, `axis_bias` 1.41. The `detail` score fell by about a
fifth when accumulation went in, because it had been counting grain as detail.

Current temporal scores, RMS in 8-bit levels at 1920x1080, whole image and edge pixels:

| | all | edges |
| --- | --- | --- |
| error, one raw frame | 1.28 | 6.03 |
| error, accumulated | 0.41 | 1.27 |
| error, control (a second reference) | 0.16 | 0.74 |
| flicker, no accumulation | 1.91 | 9.12 |
| flicker, accumulated | 0.22 | 0.84 |
| pan at 0.5 rad/s | 0.72 | 2.90 |
| pan, bilinear history | 1.11 | 4.71 |
| pan, control (no reprojection) | 12.22 | 16.13 |

Three cautions, all learned the hard way:

- SSIM is only meaningful when the two images are meant to depict the same thing, such as more
  march steps against fewer.
- Every metric needs a control with a known answer before it is believed.
- A control must be able to fail independently. The first temporal reference was the renderer
  averaging 512 frames itself; it stalls in half-float precision and comes out biased at edges.
  Its control was a second copy made the same way, which shared the bias and agreed perfectly.
  The reference is now averaged on the CPU from independent raw frames.

## Running it

```sh
cargo run --release                        # open a window at 1920x1080
cargo run --release -- --size 2560x1440    # any size, in real pixels
cargo run --release -- --fullscreen        # borderless, at the monitor's resolution
cargo run --release -- --no-vsync          # uncapped, for benchmarking
cargo run --release -- --bench             # headless GPU-timed benchmark, no window
cargo run --release -- --def LIGHT_STEPS=12 --bench    # override any shader const
cargo run --release -- --screenshot out/shot.png --time 120 --sun 0.9 --yaw 0.6 --pitch 0.3
```

Controls: drag to look around, shift-drag to swing the sun around the compass, scroll to raise or
lower the sun (it can set), Esc to quit. The title bar shows frame rate, resolution and sun
elevation.

The panel in the top-left has the four dials:

- coverage — how much of the sky is cloud, 0 clear to 1 overcast.
- feature size — how wide a typical cloud is, 500 m to 6 km.
- shadow — self-shadowing strength; low glows, high gives heavy dark undersides.
- exposure — brightness, on top of the automatic exposure that follows the sun down.

Each dial has a flag (`--coverage`, `--feature-size`, `--shadow`, `--exposure`), and `--time`,
`--sun`, `--yaw`, `--pitch` set the scene, so any look can be reproduced headlessly. `--sun`,
`--yaw` and `--pitch` are radians. `out/` is gitignored.

Edit `shaders/clouds.wgsl` while it runs and the change appears on save. If the shader fails to
compile, the error prints with the offending line and the last working version keeps rendering.

After building anything that changes the image, run `tools/shots`, look at the sheet, and fix
what looks wrong before reporting the work as done.

## Performance

The target is 60 fps at 2560x1440, so the frame is budgeted at 13 ms. Measured on an AMD
RX 7800 XT (RADV, Mesa 24.2.8) with `--bench`, which times the frame with GPU timestamp queries.
An effect is only real above 0.016 ms.

| scene | ms | | scene | ms |
| --- | --- | --- | --- | --- |
| overcast, 500 m features | 10.5 | | default | 5.6 |
| overcast, looking up | 9.1 | | horizon | 4.8 |
| into the sun | 6.5 | | sparse (coverage 0.15) | 2.8 |
| looking straight up | 6.9 | | ground only | 1.4 |

Worst case found is 10.5 ms, 2.5 ms inside the budget. Cost follows how much of the screen is
cloud surface: each ray that enters cloud takes a handful of short steps, each with a light march.

Where the time goes, and the levers:

- `LIGHT_STEPS` scales the light march, which is two thirds of the frame: the overcast scene is
  9.1 ms with eight light steps, 5.7 with four and 3.1 with none.
- `STEP_MIN` and `STEP_GROWTH` set the in-cloud step. They were halved, for about 2 ms, to remove
  a hatch pattern before accumulation existed. With accumulation the coarser pair (`STEP_MIN=28`,
  `STEP_GROWTH=0.004`) flickers barely more (0.30 against 0.25) and takes the overcast scene from
  9.1 to 7.0 ms, but the converged picture differs: 1.7% of pixels by more than three levels,
  mostly thin edge features the longer strides miss. Kept fine; the 2 ms is there if needed.
- Temporal accumulation costs 0.15 to 0.25 ms: the history reads and the present pass.
- A completely clear sky costs 2.1 ms for the whole screen.

## Roadmap

In rough order:

1. Light the air under the clouds correctly. The atmosphere march assumes sunlit air everywhere,
   which gives overcast skies and shadowed ground a blue-violet cast.
2. Spend what accumulation bought. Jittering the light march (`LIGHT_JITTER`) was turned off
   because it showed as stipple; averaged over frames it should instead remove the faint shadow
   terraces. Rendering the clouds at half resolution is the larger version of the same idea.
3. More cloud types: a second, higher layer of thin cirrus, and anvil tops for the tallest
   cumulus. Both are a profile and a weather-map channel, not new machinery.
4. Bake the textures with compute shaders. Worth half a second of startup, and it would put the
   noise back where it can hot-reload.
5. A ground worth looking at: terrain, or at least texture.
6. Replace the noise field with a real fluid simulation.

## Worktrees

Every Claude session that changes files works in its own git worktree, so parallel sessions never
share a checkout.

- At the start of a session, before the first edit, enter a worktree with the `EnterWorktree`
  tool. A session already launched with `claude -w` is in one.
- Worktrees live in `.claude/worktrees/<name>`, which is gitignored. Create them there and nowhere
  else.
- Read-only sessions, such as answering a question about the code, can stay in the main checkout.

## Working agreements

- Build in small steps and get each one running before starting the next.
- Commit after each working step with a short descriptive message.
- Keep tunable values as named constants at the top of the shader rather than inline numbers.
- Be generous with logs, so a problem seen while running can be pasted back as text.
- Keep this file current as the project changes.
- Record fixed bugs in the commit message: the symptom, the root cause, the fix, and anything
  tried that did not work.
- Anything that changes the image or the frame time goes through `tools/lab run`, on the whole
  scene matrix, before and after. A win in one view that regresses another is not a win.
- State the hypothesis and the prediction before the run, not after. Several predictions here
  turned out wrong — skipping the second read in shadows was expected to be a bad trade and
  measured as a good one; the step split was assumed sound and was the largest single error in
  the renderer. Writing the prediction down first is what made those visible.
- Never trust a new metric until it has scored something whose answer is already known.
