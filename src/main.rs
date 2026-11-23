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
mod camera;
mod gpu;
mod params;
mod renderer;
mod screenshot;
mod ui;
mod watch;

use anyhow::{Context, Result};

struct Args {
    screenshot: Option<String>,
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
                     --screenshot <out.png>          render one 1280x720 frame headlessly\n\
                     --time <seconds>                animation time for the screenshot\n\
                     --sun <radians>                 sun elevation (0 = horizon, 1.57 = zenith)\n\
                     --yaw, --pitch <radians>        camera direction for the screenshot\n\
                     --no-vsync                      uncap the frame rate (benchmarking)\n\
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

fn main() -> Result<()> {
    // Default to info-level logs; override with RUST_LOG=debug.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let args = parse_args()?;

    match args.screenshot {
        Some(path) => screenshot::render_to_png(
            &path,
            &screenshot::ShotOptions {
                time: args.time,
                sun_elevation: args.sun,
                yaw: args.yaw,
                pitch: args.pitch,
                params: args.params,
            },
        ),
        None => app::App::run(args.vsync, args.params),
    }
}
