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
- `src/noise.rs` — bakes the tiling 3D noise textures the shader samples, once at startup.
- `src/camera.rs` — look-around camera and the `Uniforms` struct sent to the GPU.
- `src/params.rs` — the four live-tunable cloud dials and their slider ranges.
- `src/ui.rs` — the egui panel those dials are drawn on.
- `src/renderer.rs` — pipeline, uniform buffer, and shader compilation with hot-reload.
- `src/app.rs` — winit event loop, mouse input, FPS counter.
- `src/screenshot.rs` — headless render of one frame to a PNG.
- `src/watch.rs` — filesystem watching for the shader.

### Shader side (WGSL)

`shaders/clouds.wgsl` holds everything visual. Its tunable constants sit in one block at the top.
Four of them — coverage, noise feature size, shadow strength and exposure — are not constants but
live dials in a second uniform buffer (`struct Params`, `@group(0) @binding(1)`), driven by the
on-screen panel. Everything else is still a `const` and changes on save via hot-reload.
The fragment shader, per pixel:

1. Builds a view ray from the camera basis in the uniforms.
2. Colours the sky: a zenith-to-horizon gradient, a warm low-sun tint, and a halo around the sun.
   There is deliberately no sun disc; a hard-edged circle read as pasted on over the soft clouds.
3. Intersects the ray with a horizontal slab between `CLOUD_BOTTOM` and `CLOUD_TOP`.
4. Raymarches that slab in `PRIMARY_STEPS` steps, sampling density from two tiling 3D noise
   textures shaped by a coverage threshold and a vertical profile, scrolled over time by `WIND`.
   The mip level comes from the step length, not from screen space, which is what keeps distant
   cloud from fizzing.
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
  goes for `Params` in `src/params.rs`; it is kept to plain `f32`s so there is no padding to get
  wrong, which is why a fifth dial means adding three more to keep the size a multiple of 16.
- The params uniform is named `dials` in the shader, not `p`, because `cloud_density` already uses
  `p` for the sample point.
- `@group(0)` carries the two uniform buffers at bindings 0 and 1 and the noise at 2, 3 and 4:
  base texture, detail texture, shared sampler. The sampler's address mode is `Repeat`, which is
  what lets one cube of noise cover the sky.
- The *shape* of the noise field — octave count and the amplitude of each — lives in
  `src/noise.rs`, because it is baked before the first frame. Changing it needs a rebuild, not a
  save. How the field is *used* (`TILE_SCALE`, `DETAIL_SCALE`, `DETAIL_STRENGTH`) is still shader
  constants and still hot-reloads.

## Running it

```sh
cargo run --release                        # open a window at 1920x1080
cargo run --release -- --size 2560x1440    # any size, in real pixels
cargo run --release -- --fullscreen        # borderless, at the monitor's resolution
cargo run --release -- --no-vsync          # uncapped, for benchmarking
cargo run --release -- --bench             # headless GPU-timed benchmark, no window
```

The window starts at 1920x1080 and can be dragged to any size; the surface is reconfigured on
resize, so the shader always fills whatever the window currently is.

Controls: drag the mouse to look around, shift-drag to swing the sun around the compass, scroll to
raise or lower the sun, Esc to quit. The title bar shows frame rate, resolution and sun elevation;
frame rate is also logged every two seconds.

A panel in the top-left has sliders for the four dials that change the look the most: **coverage**
(how much of the sky is cloud, now close to literal — 0.45 means about 45% of the layer is above
the cloud threshold), **feature size** (how big the clouds are; the floor is 1500 m because of the
noise texture's repeat, see the roadmap), **shadow** (how strongly
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

`--time` is animation seconds, and `--sun`, `--yaw` and `--pitch` are radians. The four dials have
flags too — `--coverage`, `--feature-size`, `--shadow`, `--exposure` — so a look found by dragging
sliders can be reproduced headlessly. `out/*.png` is gitignored.

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

Measured on an AMD RX 7800 XT (RADV, Mesa 24.2.8). The expensive view is not the default one:
looking up at high coverage, long slanted rays cross the whole slab and nothing terminates early.

| | 1920x1080 | 2560x1440 | 3840x2160 |
| --- | --- | --- | --- |
| Default view (coverage 0.45) | 2.1 ms | 3.6 ms | 7.7 ms |
| Worst case (coverage 0.85, pitch 0.55) | 4.0 ms (248 fps) | 6.8 ms (147 fps) | 14.5 ms (69 fps) |

Coverage barely matters any more: 2.1 ms for clear sky against 3.3 ms for solid overcast at
1920x1080. It used to be the dial that decided whether the frame fit in its budget.

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

Cost now scales with `PRIMARY_STEPS` times `LIGHT_STEPS`, and barely with the noise at all. There
is roughly four times the frame budget spare at 1920x1080, which is where the quality in the
roadmap below can be spent.

## Roadmap

In rough order:

1. Spend the frame budget the textures freed on quality: more `PRIMARY_STEPS` and `LIGHT_STEPS`,
   and a powder effect darkening the parts of a cloud facing the sun, for the way light scatters
   out of thin edges.
2. Break up the texture's repeat, so the feature-size dial can go back below 1500 m. The field
   tiles every four feature sizes, and at small sizes that repeat reads as a radial moiré at the
   horizon. A per-cell offset on the lookup, or a cheap procedural warp of the sample point, is
   the usual answer — not a bigger texture.
3. Bake the noise with a compute shader instead of on the CPU. Worth 350 ms of startup, and it
   would put the noise field back in the shader where it can hot-reload. The field itself could
   become Worley or Perlin-Worley at the same time, which is what makes clouds look billowy
   rather than soft; it costs nothing at runtime now that it is baked.
4. Improve the sky and atmosphere: real Rayleigh and Mie scattering instead of a gradient. Aerial
   perspective on distant clouds is done; a spherical cloud shell instead of a flat slab would let
   the layer drop below the horizon naturally rather than stopping at `MAX_DISTANCE`.
5. Replace the noise field with a real fluid simulation.

Dropped: rendering at lower resolution with temporal reprojection. It existed to buy frame budget,
and the textures bought ten times more without the reprojection artifacts or a second render
target. `JITTER` stays for what it already does.

## Working agreements

- Build in small steps and get each one running before starting the next.
- Commit after each working step with a short descriptive message.
- Keep tunable values as named constants at the top of the shader rather than inline numbers.
- Be generous with logs, so a problem seen while running can be pasted back as text.
- Keep this file current as the project changes.
- Record fixed bugs in the commit message: the symptom, the root cause, the fix, and anything
  tried that did not work.
