//! A small real-time volumetric cloud renderer.
//!
//! The CPU side is deliberately thin: open a window, draw one full-screen
//! triangle, and feed a uniform buffer to the fragment shader each frame.
//! All the actual cloud rendering lives in `shaders/clouds.wgsl`.
//!
//! Usage:
//!   cargo run --release
//!   cargo run --release -- --screenshot out.png [--time 10.0] [--sun 0.4]
//!   trunk serve --release        (the same program in a browser tab)
//!
//! The web build is this same binary compiled to WebAssembly. It has the
//! window and the dials; the command line, the headless tools and shader
//! hot-reload need a real OS and are compiled out (`cfg(target_arch)`).

// The web build leaves out everything that only the native tools call, so
// some fields and helpers go unused there. That is expected, not a smell.
#![cfg_attr(target_arch = "wasm32", allow(dead_code, unused_imports))]

mod app;
mod atmosphere;
mod baked;
#[cfg(not(target_arch = "wasm32"))]
mod bench;
mod camera;
mod defines;
mod noise;
mod gpu;
mod params;
mod renderer;
mod shader;
mod terrain;
#[cfg(not(target_arch = "wasm32"))]
mod screenshot;
mod ui;
#[cfg(not(target_arch = "wasm32"))]
mod watch;

#[cfg(not(target_arch = "wasm32"))]
use anyhow::{Context, Result};

/// Render size when nothing else is asked for: both the window's starting
/// size and the headless screenshot. 1920x1080 on a 2560x1440 desktop leaves
/// room for the title bar; `--size` or `--fullscreen` go bigger.
#[cfg(not(target_arch = "wasm32"))]
const DEFAULT_SIZE: (u32, u32) = (1920, 1080);

#[cfg(not(target_arch = "wasm32"))]
struct Args {
    screenshot: Option<String>,
    /// Dump the density field rather than the rendered sky.
    field: bool,
    /// Headless benchmark instead of a window or a screenshot.
    bench: bool,
    bench_frames: u32,
    /// Frames a screenshot accumulates before the one it saves.
    accumulate: u32,
    /// Camera turn rate during a screenshot's lead-up frames (yaw, pitch).
    pan: (f32, f32),
    /// Free-text tag printed with the benchmark line.
    label: String,
    /// Render size in real pixels: window size, or screenshot size.
    size: (u32, u32),
    /// Open the window borderless-fullscreen at the monitor's own resolution.
    fullscreen: bool,
    time: f32,
    sun: Option<f32>,
    yaw: Option<f32>,
    pitch: Option<f32>,
    vsync: bool,
    /// `--def NAME=VALUE` overrides applied to the shader before it compiles.
    defines: defines::Defines,
    /// Starting values for the five live dials. In windowed mode these are just
    /// where the sliders begin; headlessly they are the whole story.
    params: params::Params,
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_args() -> Result<Args> {
    let mut args = Args {
        screenshot: None,
        field: false,
        bench: false,
        bench_frames: 48,
        accumulate: 32,
        pan: (0.0, 0.0),
        label: String::from("bench"),
        size: DEFAULT_SIZE,
        fullscreen: false,
        time: 0.0,
        sun: None,
        yaw: None,
        pitch: None,
        vsync: true,
        defines: defines::Defines::default(),
        params: params::Params::default(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--screenshot" => {
                args.screenshot = Some(it.next().context("--screenshot needs a file path")?);
            }
            "--dump-field" => {
                args.screenshot = Some(it.next().context("--dump-field needs a file path")?);
                args.field = true;
                // Square by default: the analysis compares the two axes of
                // the field against each other, so a 16:9 patch would make
                // one of them shorter for no reason.
                if args.size == DEFAULT_SIZE {
                    args.size = (1024, 1024);
                }
            }
            "--size" => {
                let v = it.next().context("--size needs WxH, e.g. 1920x1080")?;
                args.size = parse_size(&v).with_context(|| format!("bad --size: {v}"))?;
            }
            "--fullscreen" => args.fullscreen = true,
            "--bench" => args.bench = true,
            "--bench-frames" => {
                let v = it.next().context("--bench-frames needs a count")?;
                args.bench_frames = v.parse().with_context(|| format!("bad --bench-frames: {v}"))?;
            }
            "--accumulate" => {
                let v = it.next().context("--accumulate needs a frame count")?;
                args.accumulate = v.parse().with_context(|| format!("bad --accumulate: {v}"))?;
            }
            "--pan" => {
                let v = it.next().context("--pan needs YAW,PITCH in radians per second")?;
                let (y, p) = v.split_once(',').context("--pan needs YAW,PITCH")?;
                args.pan = (y.trim().parse()?, p.trim().parse()?);
            }
            "--label" => {
                args.label = it.next().context("--label needs a string")?;
            }
            "--time" => {
                let v = it.next().context("--time needs a number (seconds)")?;
                args.time = v.parse().with_context(|| format!("bad --time: {v}"))?;
            }
            "--sun" => {
                let v = it.next().context("--sun needs a number (radians)")?;
                args.sun = Some(v.parse().with_context(|| format!("bad --sun: {v}"))?);
            }
            "--yaw" => {
                let v = it.next().context("--yaw needs a number (radians)")?;
                args.yaw = Some(v.parse().with_context(|| format!("bad --yaw: {v}"))?);
            }
            "--pitch" => {
                let v = it.next().context("--pitch needs a number (radians)")?;
                args.pitch = Some(v.parse().with_context(|| format!("bad --pitch: {v}"))?);
            }
            "--coverage" => {
                let v = it.next().context("--coverage needs a number (0..1)")?;
                args.params.coverage = v.parse().with_context(|| format!("bad --coverage: {v}"))?;
            }
            "--feature-size" => {
                let v = it.next().context("--feature-size needs a number (metres)")?;
                args.params.feature_size = v
                    .parse()
                    .with_context(|| format!("bad --feature-size: {v}"))?;
            }
            "--shadow" => {
                let v = it.next().context("--shadow needs a number (0..2)")?;
                args.params.shadow_strength =
                    v.parse().with_context(|| format!("bad --shadow: {v}"))?;
            }
            "--cloud-base" => {
                let v = it.next().context("--cloud-base needs a number (metres)")?;
                args.params.cloud_base = v
                    .parse::<f32>()
                    .with_context(|| format!("bad --cloud-base: {v}"))?
                    .clamp(*params::Params::CLOUD_BASE_RANGE.start(), *params::Params::CLOUD_BASE_RANGE.end());
            }
            "--wispiness" => {
                let v = it.next().context("--wispiness needs a number (0..1)")?;
                args.params.wispiness = v.parse().with_context(|| format!("bad --wispiness: {v}"))?;
            }
            "--exposure" => {
                let v = it.next().context("--exposure needs a number (0.1..4)")?;
                args.params.exposure = v.parse().with_context(|| format!("bad --exposure: {v}"))?;
            }
            "--def" => {
                let v = it.next().context("--def needs NAME=VALUE")?;
                args.defines.push_arg(&v)?;
            }
            "--no-vsync" => args.vsync = false,
            "--help" | "-h" => {
                println!(
                    "cloud-renderer\n\n\
                     (no args)                       open a window\n\
                     --size <WxH>                    render size in pixels (default 1920x1080)\n\
                     --fullscreen                    open borderless at the monitor's resolution\n\
                     --screenshot <out.png>          render one frame headlessly to a PNG\n\
                     --dump-field <out.png>          top-down density field, linear, for analysis\n\
                                                     (R full, G no detail, B weather cover; 1024^2)\n\
                     --time <seconds>                animation time for the screenshot\n\
                     --accumulate <n>                frames a screenshot blends, as the window\n\
                                                     does over time (default 32; 1 = one raw frame)\n\
                     --pan <yaw,pitch>               turn the camera at this rate (rad/s) over those\n\
                                                     frames, ending on the requested view\n\
                     --sun <radians>                 sun elevation (0 = horizon, 1.57 = zenith)\n\
                     --yaw, --pitch <radians>        camera direction for the screenshot\n\
                     --no-vsync                      uncap the frame rate (benchmarking)\n\
                     --bench                         headless GPU-timed benchmark, no window\n\
                     --bench-frames <n>              frames to time (default 48)\n\
                     --label <text>                  tag printed with the benchmark line\n\
                     --def NAME=VALUE                override a `const` in the shader\n\
                                                     (repeatable; the file on disk is untouched)\n\
                     \n\
                     The six live dials. In a window these set where the sliders start:\n\
                     --coverage <0..1>               how much of the sky is cloud\n\
                     --feature-size <metres>         how big the clouds are\n\
                     --shadow <0..2>                 self-shadowing strength\n\
                     --cloud-base <metres>           altitude of the clouds' undersides\n\
                     --wispiness <0..1>              how ragged the cloud surface is\n\
                     --exposure <0.1..4>             overall brightness\n"
                );
                std::process::exit(0);
            }
            other => anyhow::bail!("unknown argument: {other} (try --help)"),
        }
    }
    Ok(args)
}

/// "1920x1080" -> (1920, 1080). Accepts a capital X too, since shells make it
/// easy to type either.
#[cfg(not(target_arch = "wasm32"))]
fn parse_size(v: &str) -> Result<(u32, u32)> {
    let (w, h) = v
        .to_lowercase()
        .split_once('x')
        .map(|(w, h)| (w.trim().to_string(), h.trim().to_string()))
        .context("expected WxH, e.g. 1920x1080")?;
    let (w, h): (u32, u32) = (w.parse()?, h.parse()?);
    if w == 0 || h == 0 {
        anyhow::bail!("both dimensions must be non-zero");
    }
    Ok((w, h))
}

/// Web entry point. There is no command line, so everything starts at its
/// default; the canvas sets the size. Logs go to the browser console, and a
/// panic prints there with a stack trace instead of vanishing.
#[cfg(target_arch = "wasm32")]
fn main() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).expect("logger already set");
    app::App::run(
        true,
        params::Params::default(),
        (0, 0),
        false,
        defines::Defines::default(),
    )
    .expect("failed to start the event loop");
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> Result<()> {
    // Default to info-level logs; override with RUST_LOG=debug.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = parse_args()?;

    if args.bench {
        return bench::run(&bench::BenchOptions {
            width: args.size.0,
            height: args.size.1,
            frames: args.bench_frames,
            time: args.time,
            sun_elevation: args.sun,
            yaw: args.yaw,
            pitch: args.pitch,
            params: args.params,
            label: args.label,
            defines: args.defines,
        });
    }

    match args.screenshot {
        Some(path) => screenshot::render_to_png(
            &path,
            &screenshot::ShotOptions {
                width: args.size.0,
                height: args.size.1,
                time: args.time,
                sun_elevation: args.sun,
                yaw: args.yaw,
                pitch: args.pitch,
                params: args.params,
                defines: args.defines,
                field: args.field,
                frames: args.accumulate,
                pan: args.pan,
            },
        ),
        None => app::App::run(
            args.vsync,
            args.params,
            args.size,
            args.fullscreen,
            args.defines,
        ),
    }
}
