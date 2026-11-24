//! Windowed mode: winit event loop, mouse input, FPS counter, shader watching.
//!
//! The same code drives a desktop window and a browser canvas. Two things
//! differ on the web, both marked with `cfg(target_arch = "wasm32")`:
//!
//! - Start-up is asynchronous. A browser hands out the GPU through promises
//!   and never lets the main thread block, so the set-up runs as a spawned
//!   future and the event loop picks the result up when it is ready.
//! - The picture is drawn at a capped resolution and stretched to the canvas
//!   by the browser. See [`WEB_MAX_PIXELS`].

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::mpsc::Receiver;

use web_time::Instant;

use anyhow::{Context, Result};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::ModifiersState;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::camera::Camera;
use crate::gpu::{Gpu, create_instance};
use crate::defines::Defines;
use crate::params::Params;
use crate::renderer::Renderer;
use crate::ui::Ui;
#[cfg(not(target_arch = "wasm32"))]
use crate::watch;

/// The most pixels the web build draws per frame: 1920x1080's worth.
///
/// A browser tab on a high-DPI laptop screen is easily five million real
/// pixels, well past what the frame budget was measured for. Above this the
/// surface is made smaller than the canvas, keeping its shape, and the
/// browser scales it up. Native windows are sized by `--size` and are not
/// capped.
const WEB_MAX_PIXELS: f32 = 1920.0 * 1080.0;

/// The size to draw at for this window: its real pixel size, capped on web.
fn surface_size(window: &Window) -> (u32, u32) {
    let size = window.inner_size();
    let (w, h) = (size.width.max(1) as f32, size.height.max(1) as f32);
    let scale = if cfg!(target_arch = "wasm32") {
        (WEB_MAX_PIXELS / (w * h)).sqrt().min(1.0)
    } else {
        1.0
    };
    (
        ((w * scale).round() as u32).max(1),
        ((h * scale).round() as u32).max(1),
    )
}

/// Everything that only exists once a window has been created.
struct Active {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    /// The format the passes draw into: the surface's, sRGB-encoded.
    target_format: wgpu::TextureFormat,
    gpu: Gpu,
    renderer: Renderer,
    /// The dial panel drawn over the clouds.
    ui: Ui,
    /// Shader-file-changed notifications from the watcher thread.
    #[cfg(not(target_arch = "wasm32"))]
    shader_changes: Receiver<()>,
}

impl Active {
    /// Open the GPU on `window` and build the renderer: the slow part of
    /// start-up, mostly baking noise.
    async fn new(window: Arc<Window>, vsync: bool, defines: Defines) -> Result<Self> {
        let instance = create_instance();
        // The surface borrows the window; cloning the Arc gives it 'static.
        let surface = instance
            .create_surface(window.clone())
            .context("failed to create surface")?;
        let gpu = Gpu::new(instance, Some(&surface)).await?;

        let (width, height) = surface_size(&window);
        let mut config = surface
            .get_default_config(&gpu.adapter, width, height)
            .context("surface is not supported by this adapter")?;
        // Fifo = vsync, which is what we want for a smooth look-around.
        // Immediate uncaps the frame rate so we can see real GPU cost. A
        // browser only ever presents in step with the display.
        config.present_mode = if vsync || cfg!(target_arch = "wasm32") {
            wgpu::PresentMode::Fifo
        } else {
            wgpu::PresentMode::Immediate
        };
        // The shader writes linear light and relies on an sRGB target to
        // encode it. Desktop surfaces offer an sRGB format directly. A
        // browser canvas does not, but lets us *view* its plain format as
        // sRGB, which does the same encoding on write.
        let target_format = config.format.add_srgb_suffix();
        if target_format != config.format {
            config.view_formats.push(target_format);
        }
        surface.configure(&gpu.device, &config);
        log::info!(
            "surface configured: {}x{} format {:?} (drawn as {:?}) present {:?}",
            config.width,
            config.height,
            config.format,
            target_format,
            config.present_mode
        );

        let renderer = Renderer::new(&gpu, target_format, &defines).await?;
        let ui = Ui::new(&gpu, &window, target_format);
        #[cfg(not(target_arch = "wasm32"))]
        let shader_changes = watch::watch_shaders(renderer.shader_dir());

        Ok(Self {
            window,
            surface,
            config,
            target_format,
            gpu,
            renderer,
            ui,
            #[cfg(not(target_arch = "wasm32"))]
            shader_changes,
        })
    }

    /// Match the surface to the window after a resize.
    fn resize(&mut self) {
        let (width, height) = surface_size(&self.window);
        if (width, height) == (self.config.width, self.config.height) {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.gpu.device, &self.config);
        log::info!("resized to {width}x{height}");
    }
}

/// Replace the page's "loading" line with `message`, or remove it.
#[cfg(target_arch = "wasm32")]
fn set_loading(message: Option<&str>) {
    let element = web_sys::window()
        .and_then(|w| w.document())
        .and_then(|d| d.get_element_by_id("loading"));
    if let Some(element) = element {
        match message {
            Some(text) => element.set_text_content(Some(text)),
            None => element.remove(),
        }
    }
}

pub struct App {
    active: Option<Active>,
    /// True once start-up has begun, so a second `resumed` does not repeat it.
    starting: bool,
    /// Where the web build's asynchronous start-up leaves its result.
    #[cfg(target_arch = "wasm32")]
    pending: std::rc::Rc<std::cell::RefCell<Option<Result<Active>>>>,
    /// false = present as fast as the GPU can, for benchmarking.
    vsync: bool,
    camera: Camera,
    /// The five live cloud dials, driven by the on-screen panel.
    params: Params,
    /// Starting window size in real pixels. Resizing the window afterwards
    /// just reconfigures the surface, so this is only where it begins.
    /// Unused on web, where the page's CSS sizes the canvas.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    size: (u32, u32),
    /// Skip the window size and fill the monitor instead.
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    fullscreen: bool,
    /// `--def` shader constant overrides for this run.
    defines: Defines,
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
            starting: false,
            #[cfg(target_arch = "wasm32")]
            pending: Default::default(),
            vsync: true,
            camera: Camera::default(),
            params: Params::default(),
            size: (1280, 720),
            fullscreen: false,
            defines: Defines::default(),
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

    pub fn run(
        vsync: bool,
        params: Params,
        size: (u32, u32),
        fullscreen: bool,
        defines: Defines,
    ) -> Result<()> {
        let event_loop = EventLoop::new().context("failed to create event loop")?;
        // Poll (rather than Wait) so we render continuously — the clouds drift
        // even when there is no input.
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut app = Self::new();
        app.vsync = vsync;
        app.params = params;
        app.size = size;
        app.fullscreen = fullscreen;
        app.defines = defines;
        #[cfg(not(target_arch = "wasm32"))]
        event_loop.run_app(&mut app).context("event loop failed")?;
        // On web the loop cannot own the thread. `spawn_app` hooks it into
        // the browser's own event loop and returns at once.
        #[cfg(target_arch = "wasm32")]
        {
            use winit::platform::web::EventLoopExtWebSys;
            event_loop.spawn_app(app);
        }
        Ok(())
    }

    /// Take the finished start-up and begin rendering, or report why not.
    fn install(&mut self, result: Result<Active>) {
        match result {
            Ok(mut active) => {
                // On web the canvas can change size while start-up is still
                // running, and those resize events had nobody to go to.
                active.resize();
                log::info!(
                    "controls: drag = look around, shift-drag = sun compass, \
                     scroll = sun elevation, Esc = quit"
                );
                #[cfg(target_arch = "wasm32")]
                set_loading(None);
                self.active = Some(active);
                self.start = Instant::now();
            }
            Err(e) => {
                log::error!("start-up failed: {e:#}");
                // A page should say what is wrong rather than stay blank.
                // The usual cause is a browser without WebGPU.
                #[cfg(target_arch = "wasm32")]
                set_loading(Some(&format!(
                    "Could not start: {e:#}. This needs a browser with WebGPU \
                     (recent Chrome, Edge or Safari)."
                )));
                #[cfg(not(target_arch = "wasm32"))]
                panic!("start-up failed: {e:#}");
            }
        }
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

        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor {
            format: Some(active.target_format),
            ..Default::default()
        });

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
            active.config.width as f32 / active.window.inner_size().width.max(1) as f32,
            &mut self.params,
            &mut self.camera,
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
    #[cfg(not(target_arch = "wasm32"))]
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
        if self.starting {
            return;
        }
        self.starting = true;

        #[cfg(not(target_arch = "wasm32"))]
        let attrs = {
            // PhysicalSize, not LogicalSize: --size should mean the pixel count
            // the fragment shader actually has to fill, not a DPI-scaled number.
            let attrs = Window::default_attributes()
                .with_title("cloud-renderer — starting...")
                .with_inner_size(winit::dpi::PhysicalSize::new(self.size.0, self.size.1));
            if self.fullscreen {
                // Borderless(None) = the monitor the window lands on, at its
                // current mode, so there is no mode switch to wait for.
                attrs.with_fullscreen(Some(winit::window::Fullscreen::Borderless(None)))
            } else {
                attrs
            }
        };
        // On web the "window" is the <canvas id="canvas"> already on the
        // page, and the page's CSS decides how big it is.
        #[cfg(target_arch = "wasm32")]
        let attrs = {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;
            let canvas = web_sys::window()
                .and_then(|w| w.document())
                .and_then(|d| d.get_element_by_id("canvas"))
                .and_then(|e| e.dyn_into::<web_sys::HtmlCanvasElement>().ok());
            if canvas.is_none() {
                log::error!("no <canvas id=\"canvas\"> on the page");
            }
            Window::default_attributes()
                .with_title("cloud-renderer")
                .with_canvas(canvas)
        };
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );

        let init = Active::new(window, self.vsync, self.defines.clone());
        // Natively the GPU answers at once, so wait here. In a browser, run
        // it as a task; `about_to_wait` collects the result.
        #[cfg(not(target_arch = "wasm32"))]
        self.install(pollster::block_on(init));
        #[cfg(target_arch = "wasm32")]
        {
            let pending = self.pending.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let result = init.await;
                *pending.borrow_mut() = Some(result);
            });
        }
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
                // (Not on web: a page has nothing to quit to.)
                if event.state == ElementState::Pressed
                    && event.logical_key == Key::Named(NamedKey::Escape)
                    && cfg!(not(target_arch = "wasm32"))
                {
                    event_loop.exit();
                }
            }

            WindowEvent::Resized(_) => {
                if let Some(active) = self.active.as_mut() {
                    active.resize();
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
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_shader_changes();
        #[cfg(target_arch = "wasm32")]
        {
            let finished = self.pending.borrow_mut().take();
            if let Some(result) = finished {
                self.install(result);
            }
        }
        // Drive the next frame; ControlFlow::Poll means this runs continuously.
        if let Some(active) = &self.active {
            active.window.request_redraw();
        }
    }
}
