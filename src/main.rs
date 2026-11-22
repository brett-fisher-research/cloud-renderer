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
mod renderer;
mod screenshot;
mod watch;

use anyhow::{Context, Result};

struct Args {
    screenshot: Option<String>,
    time: f32,
    sun: Option<f32>,
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        screenshot: None,
        time: 0.0,
        sun: None,
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
            "--help" | "-h" => {
                println!(
                    "cloud-renderer\n\n\
                     (no args)                       open a window\n\
                     --screenshot <out.png>          render one 1280x720 frame headlessly\n\
                     --time <seconds>                animation time for the screenshot\n\
                     --sun <radians>                 sun elevation (0 = horizon, 1.57 = zenith)\n"
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
        Some(path) => screenshot::render_to_png(&path, args.time, args.sun),
        None => app::App::run(),
    }
}
