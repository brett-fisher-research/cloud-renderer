# cloud-renderer

A small real-time renderer in Rust that is growing into a custom game engine. A window shows a
physically based sky over a plain ringed by mountains, with a layer of sunlit cumulus drifting
across it, casting shadows on the ground. The same program runs in a native window and, compiled
to WebAssembly, in a browser tab.

Brett is an experienced web developer and new to graphics programming. Introduce new graphics
concepts briefly when they first appear, in replies and in code comments.

## Engine direction

Brett intends to build games on this. Clouds and terrain are features he wants to add to a game
like plugins, so every change is held to engine standards:

- Separation of concerns. A feature is one Rust module that bakes its data and one shader file
  that answers questions about the world. It knows nothing about the screen, the frame, or
  another feature's internals. Only `shaders/frame.wgsl` knows there is a picture.
- Abstractions over concretions. The renderer receives a list of `Baked` textures and never
  learns what they hold. The frame receives a `Surface` or a `Clouds` struct and never learns
  how it was found.
- Pay only for what is seen. A feature's march takes the distance at which the ray ends, so
  cloud behind a mountain is never marched.

What is not yet a plugin, so nobody mistakes it for one: features are named in two lists
(`Renderer::with_entry` and `FILES` in `src/shader.rs`), they share one bind group, the dials in
`Params` are all cloud dials, and `src/noise.rs` and `src/atmosphere.rs` are the cloud and sky
features under their old names. Move towards a feature being one registration; do it when a
third feature arrives, with the real shape of three in hand.

## Architecture

The split is deliberate: the CPU side stays small and boring so that all the interesting work
lives in shaders you can edit while the program runs.

### CPU side (Rust)

Opens a window, has each feature bake its textures once at startup, and each frame writes two
small uniform buffers and draws one full-screen triangle twice: once for the scene, once to
present it. It does no geometry, no scene graph and no per-pixel work.

- `src/main.rs` — argument parsing, picks windowed or headless mode; the web entry point.
- `src/gpu.rs` — wgpu instance, adapter, device and queue. Uses the platform's native backend
  (Vulkan on Linux, Metal on macOS, WebGPU in a browser); logs every adapter it sees.
- `src/baked.rs` — `Baked`: a texture a feature baked and the binding its shader reads it on.
- `src/noise.rs` — clouds: bakes the three tiling noise textures, shape, edge detail, weather map.
- `src/atmosphere.rs` — sky: bakes the two tables, transmittance and multiple scattering.
- `src/terrain.rs` — terrain: bakes the heightmap.
- `src/shader.rs` — joins the shader files into one module and maps compile errors back to the
  file and line they came from.
- `src/camera.rs` — look-around camera and the `Uniforms` struct sent to the GPU.
- `src/params.rs` — the five live dials and their slider ranges.
- `src/ui.rs` — the egui panel the dials and the two sun sliders are drawn on.
- `src/renderer.rs` — pipelines, the bind group built from the features' `Baked` lists, the
  history textures and per-frame temporal state, and hot-reload.
- `src/app.rs` — winit event loop, mouse input, FPS counter, start-up on both platforms.
- `index.html` — the web build's page: a full-tab canvas and a loading line.
- `src/screenshot.rs`, `src/bench.rs` — headless still and headless GPU-timed benchmark.
- `src/defines.rs`, `src/watch.rs` — `--def` constant overrides and shader file watching.

### Shader side (WGSL)

WGSL has no include, so the files in `shaders/` are concatenated into one module. Declaration
order does not matter in WGSL, so they refer to each other freely.

- `common.wgsl` — the camera height, the two uniform blocks, the samplers, small helpers.
- `sky.wgsl` — the atmosphere: `atmosphere()` and `sun_transmittance()`.
- `clouds.wgsl` — the cloud layer: `march_clouds()` and `cloud_shadow()`.
- `terrain.wgsl` — the ground: `terrain_surface()`.
- `frame.wgsl` — the vertex stage, lighting of surfaces, compositing, tone mapping, temporal
  accumulation and the present pass.

Every tunable is a named `const` at the top of its file and changes on save. Five values are
live dials instead, in `struct Params`: coverage, feature size, shadow strength, wispiness and
exposure.

Per pixel, `fs_main` in `frame.wgsl`:

1. Builds a view ray from the camera basis.
2. Asks the terrain where the ray ends. If it ends on a surface, lights it with sun, skylight
   and the shadow of the clouds above.
3. Marches the cloud layer as far as that end and no further. The layer is a spherical shell
   between `CLOUD_BOTTOM` and `CLOUD_TOP` around a round planet; it curves down behind the
   horizon, so there is no vanishing point for it to pile into.
4. Marches the atmosphere in front of the cloud and behind it. The near part fades distant cloud
   into haze (aerial perspective); the far part is the sky, or the hillside, seen through gaps.
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
- Detail (3D Worley). Pushes the surface in and out for the cauliflower texture, by an amount the
  wispiness dial scales. Read a second time at a finer scale near the camera only.

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

### The terrain

A heightfield: one height per map position, so a ray is tested against the ground with a single
texture fetch and there is no mesh. `src/terrain.rs` bakes a 2048x2048 map of a 96 km square
centred on the camera: a flat plain out to about 5 km, then a ring of mountains up to 2800 m,
which puts the higher summits inside the cloud layer.

- Shape: ridged multifractal noise. Folding each noise layer at zero turns its zero crossings
  into creases that read as ridgelines. There is no erosion yet, so no gullies or drainage.
- Height is stored across two half-float channels, a coarse part and a remainder. One half
  float gives steps of over a metre at this relief, which lighting shows as terraces.
- A third channel holds, at every mip, an upper bound on the height of everything near the
  sample. A ray higher than that strides a whole texel of that mip untested, so crossing the
  plain to the mountains takes a handful of steps.
- Close to the ground the ray steps by a fraction of its clearance (`TERRAIN_STEP_SLOPE`), then
  bisects to pin the surface. A ray that runs out of steps counts as a hit, not as sky.
- The bare sphere is terrain of height zero: `terrain_surface` returns it where there is no
  relief, and `--def TERRAIN_ENABLED=false` gives the picture from before terrain existed.
- Shading is flat: one matte colour that goes from grass to rock with slope, lit by sun and sky.
  Mountains do not yet shadow themselves or each other.

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
- `Uniforms` in `src/camera.rs` and in `shaders/common.wgsl` must match exactly. Each `vec3` takes a 16-byte
  row and uses 12, so one float follows it; four of those slots carry temporal values (`jitter`,
  `history_frames`, `subpixel_x`, `subpixel_y`). A mismatch gives silent garbage. `Params` is five
  plain floats and three of padding, two rows; the next three dials take the padding's places.
- The atmosphere constants are duplicated in `src/atmosphere.rs` and in `shaders/sky.wgsl` and
  must match, because the tables are baked from one and read by the other.
- Bindings in `@group(0)`: uniforms 0 and 1, shape 2, detail 3, weather 4, transmittance 5,
  multiple scattering 6, repeating sampler 7, clamping sampler 8, history 9, terrain 10. A
  feature's texture bindings are declared in its own shader file and its own Rust module.
- `TERRAIN_EXTENT` in `shaders/terrain.wgsl` and `EXTENT` in `src/terrain.rs` must match.
- The weather map is half-float. As 8-bit, its thresholded edges followed quantisation steps.
- Sphere intersections and altitudes use rearranged formulas (`sphere_exit`) that never subtract
  two planet-sized numbers; the textbook forms lose all precision in 32-bit floats.
- Terrain shape lives in `src/terrain.rs` and needs a rebuild to change; how it is marched and
  coloured is shader constants.
- Noise shape lives in `src/noise.rs` and needs a rebuild to change. How it is used (`*_TILE`,
  `DETAIL_STRENGTH_MAX`, `EDGE_WIDTH`) is shader constants and hot-reloads.

## Measuring

Everything runs headless, so a change can be judged without anyone at the keyboard.

- `tools/shots <name> [flags]` renders fifteen standard views and tiles them into
  `out/<name>.png`. Look at this after any change to the image.
- `tools/perf [flags]` prints GPU milliseconds for seventeen scenes at 2560x1440.
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
lower the sun (it can set), Esc to quit. The panel's two sun sliders set the same two angles. The title bar shows frame rate, resolution and sun
elevation.

The panel in the top-left has the five dials:

- coverage — how much of the sky is cloud, 0 clear to 1 overcast.
- feature size — how wide a typical cloud is, 500 m to 6 km.
- shadow — self-shadowing strength; low glows, high gives heavy dark undersides.
- wispiness — how ragged the surface is, 0 smooth blobs to 1 torn with loose puffs. The default,
  0.5, is the old fixed look. At 1 the overcast scene costs about 5% more (9.0 to 9.4 ms).
- exposure — brightness, on top of the automatic exposure that follows the sun down.

Each dial has a flag (`--coverage`, `--feature-size`, `--shadow`, `--wispiness`,
`--exposure`), and `--time`,
`--sun`, `--yaw`, `--pitch` set the scene, so any look can be reproduced headlessly. `--sun`,
`--yaw` and `--pitch` are radians. `out/` is gitignored.

Edit any file in `shaders/` while it runs and the change appears on save. If the shader fails to
compile, the error prints with the offending file and line and the last working version keeps
rendering.

After building anything that changes the image, run `tools/shots`, look at the sheet, and fix
what looks wrong before reporting the work as done.

## Web build

The web build is the same binary compiled to WebAssembly (WASM: a compact bytecode browsers run
at near-native speed) and drawing through WebGPU, the browser's version of the API wgpu wraps.
The shader is identical; WGSL is WebGPU's own shading language.

```sh
brew install trunk && rustup target add wasm32-unknown-unknown    # once
trunk serve --release      # build, serve on http://localhost:8080, rebuild on save
trunk build --release      # static files in dist/, hostable anywhere
```

Live at https://clouds.brettfisher.dev, a Cloudflare Workers static-assets project (`wrangler.jsonc`).
Deploy from this machine with `trunk build --release && npx wrangler deploy`. That is the only
way it deploys: the repo is deliberately not connected to Cloudflare's git builds, so a push
does not redeploy. On Linux, install trunk with `cargo install trunk --locked`; wrangler needs
`npx wrangler login` once per machine.

It has the window, the camera and the dials. Everything that needs an OS is compiled out with
`cfg(target_arch = "wasm32")`: the command line, `--bench`, `--screenshot`, `--def`, and shader
hot-reload (the shader is built into the binary, so `trunk serve` reloads the page instead).

What differs from native, and why:

- Start-up is async. A browser hands out the GPU through promises and never lets the main thread
  block, so `Gpu::new` and `Renderer::new` are `async`. Native callers wrap them in
  `pollster::block_on`; on web `App::resumed` spawns the set-up and `about_to_wait` collects it.
- The noise bake is single-threaded, since a page cannot spawn threads without special server
  headers. It takes about 2.3 s on an M5 Pro, during which the tab is frozen; `index.html` shows a
  loading line that Rust removes on the first frame, or replaces with the error (usually a
  browser without WebGPU). Baking with compute shaders (roadmap) would remove the wait.
- Resolution is capped at 1920x1080's worth of pixels (`WEB_MAX_PIXELS` in `src/app.rs`). A tab
  on a high-DPI screen is easily five million pixels. Above the cap the surface is smaller than
  the canvas and the browser stretches it. The egui panel is told the surface's pixel density
  (`render_scale` in `src/ui.rs`) so it stays the right size and the pointer still lines up.
- A canvas has no sRGB surface format, so the passes draw into an sRGB *view* of it
  (`target_format` in `src/app.rs`). Native surfaces are already sRGB and take the same path.
- The terrain heightmap is 1024x1024 on the web, a quarter of the native texels, because the
  bake is single-threaded there. Its bake time and look in a browser are not yet measured.
- It needs WebGPU: Chrome, Edge, Brave and recent Safari. There is no WebGL fallback.

Measured in headless Brave on an M5 Pro: 3.4 s from page load to first frame, then a steady
60 fps at 1860x1115 (a 1728x1036 tab at 2x, capped) in the default scene. The `.wasm` is 6.9 MB.

`tools/` only drives the native binary. A change to the Rust side should be built for both
targets (`cargo build --release --target wasm32-unknown-unknown` is the quick check) before it
is called done.

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

Terrain is free at this stage. Measured before and after on the ten lab scenes, no scene got
slower: the three mountain scenes moved by less than 0.25 ms (4.34 to 4.35, 5.98 to 5.73, 4.49
to 4.38) and overcast looking up fell from 9.10 to 8.60 ms, because mountains now hide cloud
near the horizon and hidden cloud is not marched. The heightmap bake adds 0.28 s to start-up.
The table above predates terrain; the lab notebook has the current rows (`terrain-flat-shaded`).

Worst case found is 10.5 ms, 2.5 ms inside the budget. Cost follows how much of the screen is
cloud surface: each ray that enters cloud takes a handful of short steps, each with a light march.

On a Mac it runs on Metal, about 2.5 times slower: an Apple M5 Pro measures 14.1 ms for the
default scene, 22.3 for overcast looking up, 7.5 sparse and 2.9 ground only, at 2560x1440. That
is over the budget at 1440p; at 1920x1080 the default scene is about 8.5 ms. Apple GPUs do not run
passes in recorded order, so their timestamp queries do not bracket a frame (they read 30 ms with
a minimum of 0.00 for an 8.5 ms frame). On Metal `--bench` times each batch of eight frames on
the wall clock instead, so its min and p95 are batch means, and the numbers are not comparable to
the timestamp ones above to better than a few percent.

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

Terrain, in order:

1. Erosion. Bake hydraulic erosion (simulated water carving gullies and drainage) into the
   heightmap. It is what makes mountains read as real; ridged noise alone reads as soft.
2. Shadows and materials. A short march towards the sun so mountains shadow themselves, then
   rock, grass and snow by slope and altitude. Cloud shadows on steep slopes show as stripes,
   one per shadow sample, and want fixing here.
3. Detail finer than the 47 m texel, added in the shader near the camera.

Everything else, in rough order:

1. Light the air under the clouds correctly. The atmosphere march assumes sunlit air everywhere,
   which gives overcast skies and shadowed ground a blue-violet cast.
2. Spend what accumulation bought. Jittering the light march (`LIGHT_JITTER`) was turned off
   because it showed as stipple; averaged over frames it should instead remove the faint shadow
   terraces. Rendering the clouds at half resolution is the larger version of the same idea.
3. More cloud types: a second, higher layer of thin cirrus, and anvil tops for the tallest
   cumulus. Both are a profile and a weather-map channel, not new machinery.
4. Bake the textures with compute shaders. Worth half a second of startup natively and over two
   seconds of frozen tab on the web, and it would put the noise back where it can hot-reload.
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
- Keep tunable values as named constants at the top of the feature's shader file rather than inline numbers.
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
