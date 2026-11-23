//! Windowed mode: winit event loop, mouse input, FPS counter, shader watching.

use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Instant;

use anyhow::{Context, Result};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::ModifiersState;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::camera::Camera;
use crate::gpu::{Gpu, create_instance};
use crate::params::Params;
use crate::renderer::Renderer;
use crate::ui::Ui;
use crate::watch;

/// Everything that only exists once a window has been created.
struct Active {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: Gpu,
    renderer: Renderer,
    /// The dial panel drawn over the clouds.
    ui: Ui,
    /// Shader-file-changed notifications from the watcher thread.
    shader_changes: Receiver<()>,
}

pub struct App {
    active: Option<Active>,
    /// false = present as fast as the GPU can, for benchmarking.
    vsync: bool,
    camera: Camera,
    /// The four live cloud dials, driven by the on-screen panel.
    params: Params,
    start: Instant,

    // Input state.
    dragging: bool,
    last_cursor: Option<(f64, f64)>,
    /// Live modifier keys; shift turns a drag into a sun-compass drag.
    modifiers: ModifiersState,

    // FPS counter state.
    frames: u32,
    last_fps_update: Instant,
    last_fps_log: Instant,
    fps: f32,
}

impl App {
    pub fn new() -> Self {
        Self {
            active: None,
            vsync: true,
            camera: Camera::default(),
            params: Params::default(),
            start: Instant::now(),
            dragging: false,
            last_cursor: None,
            modifiers: ModifiersState::empty(),
            frames: 0,
            last_fps_update: Instant::now(),
            last_fps_log: Instant::now(),
            fps: 0.0,
        }
    }

    pub fn run(vsync: bool, params: Params) -> Result<()> {
        let event_loop = EventLoop::new().context("failed to create event loop")?;
        // Poll (rather than Wait) so we render continuously — the clouds drift
        // even when there is no input.
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut app = Self::new();
        app.vsync = vsync;
        app.params = params;
        event_loop.run_app(&mut app).context("event loop failed")?;
        Ok(())
    }

    fn render(&mut self) {
        let Some(active) = self.active.as_mut() else {
            return;
        };

        // Acquire the next image from the swapchain to draw into. In wgpu 30
        // this returns a status enum rather than a Result.
        let frame = match active.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            // Suboptimal still gives us a usable texture (e.g. mid-resize).
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            // Lost/Outdated happen on resize: reconfigure and try next frame.
            status @ (wgpu::CurrentSurfaceTexture::Lost
            | wgpu::CurrentSurfaceTexture::Outdated) => {
                log::warn!("surface {status:?}, reconfiguring");
                active.surface.configure(&active.gpu.device, &active.config);
                return;
            }
            // Occluded = window hidden; Timeout = compositor busy. Skip a frame.
            status => {
                log::debug!("skipping frame: {status:?}");
                return;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let t = self.start.elapsed().as_secs_f32();
        let uniforms = self
            .camera
            .uniforms(active.config.width, active.config.height, t);
        active.renderer.update(&active.gpu, &uniforms);
        active.renderer.update_params(&active.gpu, &self.params);

        let mut encoder =
            active
                .gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frame encoder"),
                });
        active.renderer.draw(&mut encoder, &view);
        // The panel draws second, over the top, in a pass that loads rather
        // than clears what the clouds just wrote.
        active.ui.draw(
            &active.gpu,
            &active.window,
            &mut encoder,
            &view,
            (active.config.width, active.config.height),
            &mut self.params,
            self.fps,
        );
        active.gpu.queue.submit(Some(encoder.finish()));
        // In wgpu 30 presenting goes through the queue, not the frame itself.
        active.gpu.queue.present(frame);

        self.tick_fps();
    }

    /// Update the window title about once a second.
    fn tick_fps(&mut self) {
        self.frames += 1;
        let elapsed = self.last_fps_update.elapsed();
        if elapsed.as_secs_f32() >= 0.5 {
            self.fps = self.frames as f32 / elapsed.as_secs_f32();
            self.frames = 0;
            self.last_fps_update = Instant::now();

            if let Some(active) = &self.active {
                let sun_deg = self.camera.sun_elevation.to_degrees();
                active.window.set_title(&format!(
                    "cloud-renderer — {:.0} fps — {}x{} — sun {:.0}°",
                    self.fps, active.config.width, active.config.height, sun_deg
                ));
            }

            // Also log it, so a slow frame rate is visible without the window.
            if self.last_fps_log.elapsed().as_secs_f32() >= 2.0 {
                self.last_fps_log = Instant::now();
                log::info!("{:.1} fps ({:.2} ms/frame)", self.fps, 1000.0 / self.fps.max(0.001));
            }
        }
    }

    /// If the shader file changed, try to rebuild. Failures are logged and the
    /// previous pipeline keeps rendering.
    fn poll_shader_changes(&mut self) {
        let Some(active) = self.active.as_mut() else {
            return;
        };
        // Drain the channel: an editor save often emits several events.
        let mut changed = false;
        while active.shader_changes.try_recv().is_ok() {
            changed = true;
        }
        if !changed {
            return;
        }

        log::info!("shader changed on disk, rebuilding...");
        match active.renderer.reload_shader(&active.gpu) {
            Ok(()) => log::info!("hot-reload succeeded"),
            Err(e) => log::error!("hot-reload FAILED, keeping previous shader:\n{e}"),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.active.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("cloud-renderer — starting...")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );

        let instance = create_instance();
        // The surface borrows the window; cloning the Arc gives it 'static.
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");
        let gpu = Gpu::new(instance, Some(&surface)).expect("failed to initialise GPU");

        let size = window.inner_size();
        let mut config = surface
            .get_default_config(&gpu.adapter, size.width.max(1), size.height.max(1))
            .expect("surface is not supported by this adapter");
        // Fifo = vsync, which is what we want for a smooth look-around.
        // Immediate uncaps the frame rate so we can see real GPU cost.
        config.present_mode = if self.vsync {
            wgpu::PresentMode::Fifo
        } else {
            wgpu::PresentMode::Immediate
        };
        surface.configure(&gpu.device, &config);
        log::info!(
            "surface configured: {}x{} format {:?} present {:?}",
            config.width,
            config.height,
            config.format,
            config.present_mode
        );

        let renderer = Renderer::new(&gpu, config.format).expect("failed to build renderer");
        let ui = Ui::new(&gpu, &window, config.format);
        let shader_changes = watch::watch_file(renderer.shader_path());

        log::info!(
            "controls: drag = look around, shift-drag = sun compass, \
             scroll = sun elevation, Esc = quit"
        );

        self.active = Some(Active {
            window,
            surface,
            config,
            gpu,
            renderer,
            ui,
            shader_changes,
        });
        self.start = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        // egui gets first refusal on every event. `consumed` is true when the
        // pointer is over the panel or a slider has grabbed the drag, and the
        // camera must then stay still — otherwise nudging a slider would also
        // spin the sky.
        let consumed = match self.active.as_mut() {
            Some(active) => {
                let window = active.window.clone();
                active.ui.on_event(&window, &event)
            }
            None => false,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::ModifiersChanged(mods) => self.modifiers = mods.state(),

            WindowEvent::KeyboardInput { event, .. } => {
                // Esc quits even while typing in the panel; there is nothing
                // in there to type into.
                if event.state == ElementState::Pressed
                    && event.logical_key == Key::Named(NamedKey::Escape)
                {
                    event_loop.exit();
                }
            }

            WindowEvent::Resized(size) => {
                if let Some(active) = self.active.as_mut() {
                    active.config.width = size.width.max(1);
                    active.config.height = size.height.max(1);
                    active.surface.configure(&active.gpu.device, &active.config);
                    log::info!("resized to {}x{}", active.config.width, active.config.height);
                }
            }

            WindowEvent::MouseInput { state, button, .. } => {
                if button == MouseButton::Left {
                    match state {
                        // Only *start* a drag if egui did not want the click,
                        // so pressing on a slider does not also grab the camera.
                        ElementState::Pressed => self.dragging = !consumed,
                        // Always *end* one, consumed or not. Releasing over the
                        // panel after a drag that began on the sky is consumed
                        // by egui, and ignoring it would leave the camera stuck
                        // spinning with every later mouse move.
                        ElementState::Released => {
                            self.dragging = false;
                            self.last_cursor = None;
                        }
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                let now = (position.x, position.y);
                if self.dragging {
                    if let Some(prev) = self.last_cursor {
                        let (dx, dy) = ((now.0 - prev.0) as f32, (now.1 - prev.1) as f32);
                        if self.modifiers.shift_key() {
                            // Shift-drag walks the sun around the compass. It
                            // changes the image more than any slider: it moves
                            // the silver lining from one side of a cloud to the
                            // other, and swings every shadow with it.
                            self.camera.drag_sun_azimuth(dx);
                        } else {
                            self.camera.look(dx, dy);
                        }
                    }
                }
                self.last_cursor = Some(now);
            }

            WindowEvent::CursorLeft { .. } => self.last_cursor = None,

            WindowEvent::MouseWheel { delta, .. } if !consumed => {
                let notches = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 60.0,
                };
                self.camera.scroll_sun(notches);
                log::debug!(
                    "sun elevation now {:.1}°",
                    self.camera.sun_elevation.to_degrees()
                );
            }

            WindowEvent::RedrawRequested => self.render(),

            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        self.poll_shader_changes();
        // Drive the next frame; ControlFlow::Poll means this runs continuously.
        if let Some(active) = &self.active {
            active.window.request_redraw();
        }
    }
}
