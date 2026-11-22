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
- `src/camera.rs` — look-around camera and the `Uniforms` struct sent to the GPU.
- `src/renderer.rs` — pipeline, uniform buffer, and shader compilation with hot-reload.
- `src/app.rs` — winit event loop, mouse input, FPS counter.
- `src/screenshot.rs` — headless render of one frame to a PNG.
- `src/watch.rs` — filesystem watching for the shader.

### Shader side (WGSL)

`shaders/clouds.wgsl` holds everything visual. Its tunable constants sit in one block at the top.
The fragment shader, per pixel:

1. Builds a view ray from the camera basis in the uniforms.
2. Colours the sky: a zenith-to-horizon gradient, a warm low-sun tint, and a halo around the sun.
   There is deliberately no sun disc; a hard-edged circle read as pasted on over the soft clouds.
3. Intersects the ray with a horizontal slab between `CLOUD_BOTTOM` and `CLOUD_TOP`.
4. Raymarches that slab in `PRIMARY_STEPS` steps, sampling density from fBm value noise shaped by
   a coverage threshold and a vertical profile, scrolled over time by `WIND`.
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
  the padding floats after each `vec3`. A mismatch gives silent garbage, not an error.

## Running it

```sh
cargo run --release                   # open a window
cargo run --release -- --no-vsync     # uncapped, for benchmarking
```

Controls: drag the mouse to look around, scroll to raise or lower the sun, Esc to quit. The title
bar shows frame rate, resolution and sun elevation; frame rate is also logged every two seconds.

Edit `shaders/clouds.wgsl` while it runs and the change appears on save. If the shader fails to
compile, the error prints with the offending line and the last working version keeps rendering.

### Screenshots

Headless mode renders one frame at 1280x720 and exits, with no window. Use it to check visual
changes without needing someone at the keyboard.

```sh
cargo run --release -- --screenshot out/shot.png
cargo run --release -- --screenshot out/shot.png --time 120 --sun 0.9 --yaw 0.6 --pitch 0.3
```

`--time` is animation seconds, and `--sun`, `--yaw` and `--pitch` are radians. `out/*.png` is
gitignored.

After building anything that changes the image, render a screenshot, look at it, and fix what
looks wrong before reporting the work as done. When something looks off, measure it before calling
it a bug: a disc that looks elliptical off-axis, for example, is correct rectilinear projection.

## Performance

Measured at 1280x720 on an AMD RX 7800 XT (RADV, Mesa 24.2.8), via `--no-vsync`:

| Stage | Frame time |
| --- | --- |
| Sky only | ~0.1 ms |
| Noise clouds, no lighting | 2.3 ms |
| With sun light march | 7.8 ms |
| With aerial perspective | 8.0 ms |

Cost scales with `PRIMARY_STEPS` times `LIGHT_STEPS` times `FBM_OCTAVES`, since density is
recomputed procedurally at every sample. That product is the first thing to cut when adding
expensive effects.

## Roadmap

In rough order:

1. Bake Worley and Perlin-Worley noise into 3D textures with a compute shader. This replaces the
   per-sample fBm, which is both the main cost and the reason the clouds look soft rather than
   billowy.
2. Add a powder effect, darkening the parts of a cloud facing the sun to capture the way light
   scatters out of thin edges.
3. Render at lower resolution with temporal reprojection, reusing the previous frame to afford
   more steps per pixel. `JITTER` in the shader is already the groundwork for this.
4. Improve the sky and atmosphere: real Rayleigh and Mie scattering instead of a gradient. Aerial
   perspective on distant clouds is done; a spherical cloud shell instead of a flat slab would let
   the layer drop below the horizon naturally rather than stopping at `MAX_DISTANCE`.
5. Replace the noise field with a real fluid simulation.

## Working agreements

- Build in small steps and get each one running before starting the next.
- Commit after each working step with a short descriptive message.
- Keep tunable values as named constants at the top of the shader rather than inline numbers.
- Be generous with logs, so a problem seen while running can be pasted back as text.
- Keep this file current as the project changes.
- Record fixed bugs in the commit message: the symptom, the root cause, the fix, and anything
  tried that did not work.
