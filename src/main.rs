//! A small real-time volumetric cloud renderer.
//!
//! The CPU side is deliberately thin: open a window, draw one full-screen
//! triangle, and feed a uniform buffer to the fragment shader each frame.
//! All the actual cloud rendering lives in `shaders/clouds.wgsl`.
//!
//! Usage:
//!   cargo run --release
//!   cargo run --release -- --screenshot out.png [--time 10.0] [--sun 0.4]

mod app;
mod bench;
mod camera;
mod noise;
mod gpu;
mod params;
mod renderer;
mod screenshot;
mod ui;
mod watch;

use anyhow::{Context, Result};

/// Render size when nothing else is asked for: both the window's starting
/// size and the headless screenshot. 1920x1080 on a 2560x1440 desktop leaves
/// room for the title bar; `--size` or `--fullscreen` go bigger.
const DEFAULT_SIZE: (u32, u32) = (1920, 1080);

struct Args {
    screenshot: Option<String>,
    /// Headless benchmark instead of a window or a screenshot.
    bench: bool,
    bench_frames: u32,
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
    /// Starting values for the four live dials. In windowed mode these are just
    /// where the sliders begin; headlessly they are the whole story.
    params: params::Params,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        screenshot: None,
        bench: false,
        bench_frames: 48,
        label: String::from("bench"),
        size: DEFAULT_SIZE,
        fullscreen: false,
        time: 0.0,
        sun: None,
        yaw: None,
        pitch: None,
        vsync: true,
        params: params::Params::default(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--screenshot" => {
                args.screenshot = Some(it.next().context("--screenshot needs a file path")?);
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
            "--exposure" => {
                let v = it.next().context("--exposure needs a number (0.1..4)")?;
                args.params.exposure = v.parse().with_context(|| format!("bad --exposure: {v}"))?;
            }
            "--no-vsync" => args.vsync = false,
            "--help" | "-h" => {
                println!(
                    "cloud-renderer\n\n\
                     (no args)                       open a window\n\
                     --size <WxH>                    render size in pixels (default 1920x1080)\n\
                     --fullscreen                    open borderless at the monitor's resolution\n\
                     --screenshot <out.png>          render one frame headlessly to a PNG\n\
                     --time <seconds>                animation time for the screenshot\n\
                     --sun <radians>                 sun elevation (0 = horizon, 1.57 = zenith)\n\
                     --yaw, --pitch <radians>        camera direction for the screenshot\n\
                     --no-vsync                      uncap the frame rate (benchmarking)\n\
                     --bench                         headless GPU-timed benchmark, no window\n\
                     --bench-frames <n>              frames to time (default 48)\n\
                     --label <text>                  tag printed with the benchmark line\n\
                     \n\
                     The four live dials. In a window these set where the sliders start:\n\
                     --coverage <0..1>               how much of the sky is cloud\n\
                     --feature-size <metres>         how big the clouds are\n\
                     --shadow <0..2>                 self-shadowing strength\n\
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
            },
        ),
        None => app::App::run(args.vsync, args.params, args.size, args.fullscreen),
    }
}
