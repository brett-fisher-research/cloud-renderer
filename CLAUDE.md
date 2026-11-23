# cloud-renderer

A small real-time volumetric cloud renderer in Rust, built as a graphics-programming learning
project. A window shows a sky with a layer of puffy, sunlit, slowly drifting clouds you can look
around in.

Brett is an experienced web developer and new to graphics programming. Introduce new graphics
concepts briefly when they first appear, in replies and in code comments.

## Architecture

The split is deliberate: the CPU side stays small and boring so that all the interesting work
lives in one shader you can edit while the program runs.

### CPU side (Rust)

Opens a window, draws one full-screen triangle, and writes a uniform buffer each frame. It does no
geometry, no scene graph and no per-pixel work.

- `src/main.rs` — argument parsing, picks windowed or headless mode.
- `src/gpu.rs` — wgpu instance, adapter, device and queue. Vulkan-only; logs every adapter it sees.
- `src/noise.rs` — bakes the tiling 3D noise textures the shader samples, once at startup:
  two value-noise fields and one Worley field.
- `src/camera.rs` — look-around camera and the `Uniforms` struct sent to the GPU.
- `src/params.rs` — the four live-tunable cloud dials, their slider ranges, and the
  `Algorithm` enum naming the five density fields.
- `src/ui.rs` — the egui panel the dials and the algorithm picker are drawn on.
- `src/renderer.rs` — pipeline, uniform buffer, and shader compilation with hot-reload.
- `src/app.rs` — winit event loop, mouse input, FPS counter.
- `src/screenshot.rs` — headless render of one frame to a PNG.
- `src/watch.rs` — filesystem watching for the shader.

### Shader side (WGSL)

`shaders/clouds.wgsl` holds everything visual. Its tunable constants sit in one block at the top.
Four of them — coverage, noise feature size, shadow strength and exposure — are not constants but
live dials in a second uniform buffer (`struct Params`, `@group(0) @binding(1)`), driven by the
on-screen panel, along with `algorithm`, which picks one of five density fields. Everything else
is still a `const` and changes on save via hot-reload. The fragment shader, per pixel:

1. Builds a view ray from the camera basis in the uniforms.
2. Colours the sky: a zenith-to-horizon gradient, a warm low-sun tint, and a halo around the sun.
   There is deliberately no sun disc; a hard-edged circle read as pasted on over the soft clouds.
3. Intersects the ray with a horizontal slab between `CLOUD_BOTTOM` and `CLOUD_TOP`.
4. Raymarches that slab in `PRIMARY_STEPS` steps, asking the selected algorithm for a density at
   each sample — see *The five algorithms* below — scrolled over time by `WIND`. The mip level
   comes from the step length, not from screen space, which is what keeps distant cloud from
   fizzing. For the field-based algorithms the lookup point is warped and the base texture is
   read twice at incommensurate scales, so the texture's repeat falls outside the view — see
   *Breaking the repeat* below.
5. At each sample, marches `LIGHT_STEPS` towards the sun and applies Beer's law to find how much
   sunlight arrives, then weights it by a two-lobe Henyey-Greenstein phase function and adds
   ambient skylight.
6. Fades each sample towards the sky colour for that ray with distance (`AERIAL_DISTANCE`),
   which is what makes distant clouds recede instead of piling into a stripe at the horizon.
7. Composites the cloud over the sky using accumulated transmittance, then tone maps.

### Conventions worth knowing

- Units are metres and seconds. The camera sits at `CAMERA_HEIGHT` above a flat ground plane.
- Colours in the shader are linear radiance and may exceed 1.0. Tone mapping and sRGB encoding
  happen at the very end, so constants look darker than the image they produce.
- `Uniforms` in `src/camera.rs` and `struct Uniforms` in the shader must match exactly, including
  the padding floats after each `vec3`. A mismatch gives silent garbage, not an error. The same
  goes for `Params` in `src/params.rs`. It holds plain 4-byte scalars so there is no alignment
  to get wrong, only a size that has to stay a multiple of 16: four dials filled the first row,
  `algorithm` opened a second, and `_pad[3]` is the rest of it. Three more dials are free; the
  fourth costs another row on both sides.
- The params uniform is named `dials` in the shader, not `p`, because `cloud_density` already uses
  `p` for the sample point.
- `@group(0)` carries the two uniform buffers at bindings 0 and 1 and the noise at 2, 3, 4 and 5:
  base texture, detail texture, shared sampler, Worley texture. The sampler's address mode is
  `Repeat`, which is what lets one cube of noise cover the sky.
- The *shape* of the noise field — octave count and the amplitude of each — lives in
  `src/noise.rs`, because it is baked before the first frame. Changing it needs a rebuild, not a
  save. How the field is *used* (`TILE_SCALE`, `DETAIL_SCALE`, `DETAIL_STRENGTH`) is still shader
  constants and still hot-reloads.

### The five algorithms

The complaint that started this: at a near-horizon pitch the sky was long streaks converging on a
vanishing point, and cloud covered all of it. Two separate causes, and the fix for each is a
different density field rather than a different dial, so the field became a live choice:
`dials.algorithm`, a dropdown at the top of the panel and `--algorithm NAME` headlessly.

Every mode marches the same slab with the same lighting, aerial perspective and tone mapping.
Only the function answering *how much cloud is at this point* changes.

| mode | field | worst ms | acorr | axis |
| --- | --- | --- | --- | --- |
| `fbm` | thresholded fBm, one connected field everywhere | 11.34 | 0.534 | 0.99 |
| `islands` | fBm times a coarse gate: masses with sky between | 12.02 | 0.386 | 1.19 |
| `puffs` | hash-placed blobs on a grid, eroded by noise | **10.09** | **0.374** | 0.88 |
| `billow` | Perlin-Worley: a cellular field remapped into the fBm | 11.95 | 0.534 | 0.99 |
| `cumulus` | billow inside the islands gate | 12.62 | 0.386 | 1.19 |

Worst scene is `overcast-up` for all five, at 2560x1440 against the 13 ms budget. All fit.

**What each one answers.** The streaking is not a noise defect: a ray a few degrees above the
horizon crosses tens of kilometres of a 1500 m slab, so it integrates one long tube of the same
layer and neighbouring pixels integrate almost the same tube. Any field that fills the sky
smears that way. `islands` and `cumulus` attack it by making most of the sky empty, so a shallow
ray spends most of its length in clear air. `puffs` removes it outright, because there is no
field to smear — a ray either hits a blob or does not.

**`puffs` is the answer to "simple clouds for a game".** Discrete cumulus with blue sky between
them, the cheapest worst case of the five, and a `tile_lock` of exactly 0.000: the shape has no
period anywhere in it, only the erosion read does. Its coverage dial is the most literal yet —
the value is the fraction of grid cells holding a cloud — and it saturates near 0.65, which is
honest, because a sky of separate clouds cannot become overcast.

**Keeping the dials comparable cost more thought than the fields did.** Flipping modes is only
useful if `coverage 0.5` means the same sky in each, and by default it did not: measured as the
fraction of a field dump above zero, `fbm` gave 0.53, `billow` 0.78 and `puffs` 0.32. Each mode
now carries a measured calibration — `BILLOW_MEAN`/`BILLOW_GAIN` put the remapped field back on
mode 0's mean and spread, `ISLAND_GAMMA` splits the dial between two tests in series, and
`PUFF_COVER_GAIN` accounts for a round blob not filling its square cell. All five now land
within 0.5 ± 0.05.

**Two things the harness caught that the eye did not.**

- `puffs` scored `axis_bias` 2.53 against the fBm field's 0.99: its square cell grid was leaking
  into the field exactly as the value-noise lattice once did. Rotating the grid 31.7 degrees
  against the world axes took it to 0.88 and left `acorr_peak` at 0.374 — which is the honest
  reading of what the rotation does. It does not destroy the lattice, it stops the lattice being
  aligned with the wind and the cardinal directions a player looks along. It also measured 0.85 ms
  *faster* on the worst scene and slower on two mid scenes; the cause is not established.
- The notebook's `base` channel means something different for `billow` and `cumulus`. That
  channel is the density with `detail` off, and `BILLOW_IN_SHADOW` is false, so the Worley read
  is off there too: those rows score the plain fBm field. The `raw` channel is the honest one
  for those modes.

**Costs, and what paid them down.** The first working versions were 21.8, 16.0 and 14.6 ms —
`puffs`, `billow` and `islands` all over budget. Three changes, each the same idea as something
already in the renderer:

- `puffs` searches four cells, not nine. The constants are chosen so a blob reaches at most 0.91
  cells from its own centre while the fifth-nearest cell centre is at least 1.0 away, so the five
  cells it skips provably cannot contribute. 21.8 ms to 11.1.
- `billow` skips its Worley read in the shadow march (`BILLOW_IN_SHADOW`), the same bargain the
  detail texture already strikes: a shadow is an integral along a line and the bulges average out
  of it. 16.0 ms to 11.9, no visible difference.
- `islands` computes its gate once per view sample and hands it to the shadow march, the same
  bargain as `WARP_SHARED_SHADOW`: the gate varies over tens of kilometres and the march walks
  one. 14.6 ms to 12.0.

**What is still wrong with them.** `islands` and `cumulus` reduce the horizon streaking but do
not remove it — where the gate is open the field is still a sheet, and a near-horizon view is
mostly that. `puffs` clouds are uniform in size and read as pebbly near the horizon, and they
have no anvil or flat base of their own beyond what the slab imposes.

### Breaking the repeat

The base texture tiles every `TILE_SCALE` feature sizes — 10.4 km at the default dial, against a
50 km `MAX_DISTANCE`. Left alone, a view holds five copies of the same sky in each direction, and
in perspective a repeating lattice on a horizontal plane projects to lines converging on a
vanishing point: a radial moiré at the horizon. Two levers push the repeat past the view.

- **`WARP_GRADIENT`** displaces the lookup point analytically, for a few transcendentals and no
  bandwidth. Its parameter is a bound on the warp's *Jacobian*, not an amplitude, because
  amplitude alone says nothing about folding — above 1.0 the domain folds back through itself and
  tears a hard crease across a cloud. A long wavelength beats a short one: decorrelation goes as
  `L*sin(pi*T/L)`, so 1.618 tiles spends 48% of the gradient budget usefully where 9.87 spends 93%.
- **`MIX_AMOUNT`** blends a second read of the same texture at an irrational relative scale, for
  one extra fetch. The warp alone is not enough and cannot be: a smooth map acts locally like a
  constant shift, so it relabels periodicity to another lag rather than destroying it.

The warp also fixed the cubic value-noise lattice leaking its axes into the field, which was a
second defect nobody had noticed: `axis_bias` 3.21 to 1.01.

## Measuring

Three instruments, all headless, so a change can be judged without anyone at the keyboard.

- `--def NAME=VALUE` rewrites any `const` in `clouds.wgsl` between reading it and compiling it.
  The file on disk is untouched, the override survives hot-reload, and the bench line prints it.
  A define matching no declaration is a hard error: silently measuring the default would be a
  worse failure than a crash.
- `--dump-field out.png` renders the density field top-down through the same bindings as the real
  renderer, to a *linear* target. R is full density, G base only, B the raw field before the
  coverage threshold. Sized in tile units, so dumps at different feature sizes compare directly.
- `tools/lab run <name> [--algo NAME] [--def K=V]...` runs the whole scene matrix, scores it and
  appends a row per scene to `docs/experiments.jsonl`, which is only ever appended to.
  `tools/lab noise-floor` says how much the benchmark wobbles; `tools/lab report` prints the
  frontier so far.

`tools/fieldstat` scores a field dump three ways. `tile_lock` is the share of spectral energy on
the repeat sublattice — a field repeating every `image/T` pixels has a transform that is exactly
zero off it, so this needs no threshold. `acorr_peak` is the largest normalised autocorrelation
past the field's own feature size, which catches a repeat at any period; `acorr_view` restricts it
to lags where both copies could appear in one frame. `axis_bias` is cardinal-axis spectral energy
over the all-direction average, which is where a cubic lattice shows up.

`tools/imgstat` guards the picture: a repetition fix can always cheat by blurring. `contrast`,
`detail` (RMS of the image minus a 2 px blur of itself) and `slope`, plus SSIM against `--ref`.

Two cautions, both learned the hard way:

- **SSIM is only meaningful when the two images are meant to depict the same thing**, such as more
  march steps against fewer. Scoring a new noise field against a render of the old one measures
  how much the field changed, which is the thing being asked for.
- **Every metric needs a control with a known answer.** `fieldstat` was validated against white
  noise, a tiled block and axis-aligned stripes before being believed. The floor for
  `acorr_peak` is 0.26, measured on a synthetic field with the same radial spectrum and random
  phases — not 0, because a field with large-scale structure correlates with itself.

## Running it

```sh
cargo run --release                        # open a window at 1920x1080
cargo run --release -- --size 2560x1440    # any size, in real pixels
cargo run --release -- --fullscreen        # borderless, at the monitor's resolution
cargo run --release -- --no-vsync          # uncapped, for benchmarking
cargo run --release -- --bench             # headless GPU-timed benchmark, no window
cargo run --release -- --def LIGHT_STEPS=16 --bench    # override any shader const
cargo run --release -- --algorithm puffs               # pick a density algorithm
cargo run --release -- --dump-field out/field.png      # density field, for analysis
```

The window starts at 1920x1080 and can be dragged to any size; the surface is reconfigured on
resize, so the shader always fills whatever the window currently is.

Controls: drag the mouse to look around, shift-drag to swing the sun around the compass, scroll to
raise or lower the sun, Esc to quit. The title bar shows frame rate, resolution and sun elevation;
frame rate is also logged every two seconds.

A panel in the top-left picks the **algorithm** — `fbm`, `islands`, `puffs`, `billow` or
`cumulus`, described in *The five algorithms* above — and has sliders for the four dials that
change the look the most: **coverage** (how much of the sky is cloud, now close to literal, and
calibrated so it means the same in every algorithm — 0.45 is about 45% of the layer), **feature
size** (how big the clouds are, 300 m to 10 km), **shadow** (how strongly
they shadow themselves — low fakes the multiple scattering that keeps real clouds bright) and
**exposure** (overall brightness). The panel swallows mouse input while the pointer is over it, so
dragging a slider does not also spin the camera.

Edit `shaders/clouds.wgsl` while it runs and the change appears on save. If the shader fails to
compile, the error prints with the offending line and the last working version keeps rendering.

### Screenshots

Headless mode renders one frame at 1920x1080 and exits, with no window. `--size` applies here
too, so `--size 3840x2160` gives a 4K still of a scene that would not run at that size. Use it to
check visual changes without needing someone at the keyboard.

```sh
cargo run --release -- --screenshot out/shot.png
cargo run --release -- --screenshot out/shot.png --time 120 --sun 0.9 --yaw 0.6 --pitch 0.3
```

`--time` is animation seconds, and `--sun`, `--yaw` and `--pitch` are radians. The dials have
flags too — `--algorithm`, `--coverage`, `--feature-size`, `--shadow`, `--exposure` — so a look
found by dragging sliders can be reproduced headlessly. `out/*.png` is gitignored.

After building anything that changes the image, render a screenshot, look at it, and fix what
looks wrong before reporting the work as done. When something looks off, measure it before calling
it a bug: a disc that looks elliptical off-axis, for example, is correct rectilinear projection.

## Performance

`--bench` is the measuring instrument: a headless mode that renders the cloud pass over and
over into an offscreen texture and times it with GPU timestamp queries, so the number is the
GPU's own clock around that one pass, with no window, vsync, compositor or egui in it.

```sh
cargo run --release -- --bench                              # 48 frames at 1920x1080
cargo run --release -- --bench --size 2560x1440 --coverage 0.85 --pitch 0.55 --label worst
```

It takes the same scene flags as `--screenshot`, so a scenario can be pinned down exactly and
re-measured after a change. It reports median, min, p95 and mean; the median is the one to
quote. Its 1280x720 reading agrees with the hand-timed `--no-vsync` numbers below to within a
rounding error, which is what first validated it.

Measured on an AMD RX 7800 XT (RADV, Mesa 24.2.8). The target is **60 fps at 2560x1440**, so the
cloud pass is budgeted at 13 ms, leaving the rest of the 16.7 ms for vsync, egui and drift.

The benchmark is far tighter than it needs to be: ten identical runs give a standard deviation of
0.008 ms, 0.12%. **An effect is only real above 0.016 ms, or 0.25%.** Anything smaller is noise.

Current, at 2560x1440, worst scene first:

| scene | ms | | scene | ms |
| --- | --- | --- | --- | --- |
| overcast-up (coverage 0.85, pitch 0.55) | 11.4 | | small-features (300 m) | 8.8 |
| backlit (sun 0.08) | 9.9 | | default | 5.2 |
| reported (coverage 0.77) | 8.9 | | clear (coverage 0.15) | 3.0 |

Worst case 11.4 ms, 1.6 ms inside the budget. Coverage is no longer the dial that decides whether
the frame fits; the sun's position is, because a backlit sky is all transmittance and every
sample pays the full light march.

That table is the `fbm` algorithm. The other four cost between 10.1 and 12.6 ms on the same worst
scene; the per-mode numbers are in *The five algorithms*.

### How it got here

It was ten times slower. The frame cost 16.4 ms at 1920x1080 in the default view — against a
16.7 ms vsync budget, so the GPU never idled — and 33.7 ms looking up at coverage 0.85.

`--bench` found where it went, by deleting one piece of work at a time at coverage 0.70:

| Frame as built | 24.1 ms |
| --- | --- |
| ...with the sun light march removed | 4.3 ms |
| ...with the noise replaced by a constant | 1.5 ms |

**94% of the frame was value noise, and 82% of it was noise inside the light march.** Everything
else together — sky, ray setup, the compositing loop, tone mapping — was 1.5 ms. The fix had to
be the noise or it was not worth doing.

Baking it into 3D textures did it: worst case 33.7 ms to 4.0 ms. A sample is now one filtered
fetch, done by the texture units in fixed-function hardware alongside the shader's arithmetic,
instead of five octaves of hash-and-interpolate. Two further things fell out of it that were not
available before: mip levels chosen from the step length, which prefilter distant cloud instead
of letting it alias, and a flattened histogram, which makes the coverage dial close to literal.

Things measured and rejected along the way, all recorded so they are not retried:

- **Adaptive empty-space skipping** (23.2 ms) and **a cheap probe before the full fBm** (23.9 ms),
  both worth about 4%. Empty space was never where the time went; the light march was, and it
  only runs where there is already cloud.
- **Cheaper noise in the shadow march alone**, 5 octaves down to 2: 33.7 ms to 17.9 ms, and
  visually almost free. A good change, made redundant by the textures.
- **A hybrid** keeping the coarse octaves procedural and baking only the fine ones. It solves
  tiling but only reaches 11.2 ms, because the coarse procedural octaves were most of the cost.
- **A weather map** — a coarse read of the noise modulating the coverage threshold. One extra
  fetch, and it loses to the warp on every repetition metric: its autocorrelation barely moves
  from 1.000, because it shifts how much cloud there is while leaving the shapes repeating
  exactly. Kept behind `WEATHER_AMOUNT = 0` only because it is the one lever that adds
  large-scale structure rather than merely breaking a repeat.
- **A second warp octave** (`WARP_SPLIT` below 1.0). In-view autocorrelation 0.467 against 0.473
  for a single octave, and `axis_bias` worse at 1.38 against 1.01. Six more transcendentals for
  nothing.
- **Skipping the second texture read in the shadow march.** Cheaper, at 8.3 ms against 11.4, but
  spending the same time on light steps instead is more accurate. Dominated on the frontier.

Cost now scales with `PRIMARY_STEPS` times `LIGHT_STEPS`, and barely with the noise at all.

### The step budget was split wrong

That product is not the whole story, and getting the split wrong cost both quality and time.
The march ran 64 view steps to 6 light steps for a long time. Moving sixteen samples from the
first to the second — **48 view steps to 10 light steps** — raised accuracy against a 256x24-step
reference from 0.9912 to 0.9985 SSIM *and* cost a millisecond less. The view march was
oversampled and the shadow march starved.

Extra light steps are nearly free: each is `LIGHT_STEP_GROWTH` times the last, so the later ones
are long, leave the slab at once and return zero. Eight to twelve light steps spans 8.80 to
8.95 ms. Past ten there is no further gain, and `PRIMARY_STEPS` above 48 buys nothing measurable
for up to 1.2 ms.

The scene that cared most was `backlit`, where the whole image is transmittance through cloud:
0.9656 to 0.9920 SSIM on this change alone. A matrix of one scene would have missed it.

## Roadmap

In rough order:

1. A powder effect darkening the parts of a cloud facing the sun, for the way light scatters out
   of thin edges. The step counts half of this item is done and measured; see *The step budget
   was split wrong*.
2. Close the rest of the repetition gap for the field-based algorithms. `fbm` sits at an
   autocorrelation of 0.53 against a floor of 0.26; `islands` and `puffs` reach 0.39 and 0.37,
   so the non-tiling content a warp cannot provide is what moves it. For `puffs` the remaining
   self-similarity is the cell grid itself, which rotation moves but does not remove — a jittered
   count of blobs per cell, or two interleaved grids at different spacings, would.
3. Make the `puffs` clouds less uniform: they are one shape at one size, and real cumulus come in
   clusters with flat bases. Two or three blobs per cell, and the per-cloud hash already spare in
   `CellHash` used to offset the erosion read, are the cheap versions.
4. Bake the noise with a compute shader instead of on the CPU. Worth 500 ms of startup across the
   three textures, and it would put the fields back in the shader where they can hot-reload.
5. Improve the sky and atmosphere: real Rayleigh and Mie scattering instead of a gradient. Aerial
   perspective on distant clouds is done; a spherical cloud shell instead of a flat slab would let
   the layer drop below the horizon naturally rather than stopping at `MAX_DISTANCE`.
6. Replace the noise field with a real fluid simulation.

Dropped: rendering at lower resolution with temporal reprojection. It existed to buy frame budget,
and the textures bought ten times more without the reprojection artifacts or a second render
target. `JITTER` stays for what it already does.

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
