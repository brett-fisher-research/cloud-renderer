//! The on-screen dial panel, drawn with egui on top of the clouds.
//!
//! Graphics note: egui is an *immediate mode* GUI. There is no widget tree to
//! keep in sync — every frame you call `ui.add(Slider::new(&mut value, ...))`,
//! it reads the current value, draws a slider, and writes the new value back
//! if you dragged it. The whole panel is rebuilt from scratch each frame, which
//! is exactly the model that suits a renderer that already redraws constantly.
//!
//! Drawing it costs one extra render pass over the same swapchain texture,
//! loading rather than clearing what the cloud pass already wrote.

use std::sync::Arc;

use egui_wgpu::ScreenDescriptor;
use winit::window::Window;

use crate::gpu::Gpu;
use crate::params::{Algorithm, Params};

pub struct Ui {
    ctx: egui::Context,
    state: egui_winit::State,
    renderer: egui_wgpu::Renderer,
    /// Collapsed panels are easy to lose; keep it open to start with.
    open: bool,
}

impl Ui {
    pub fn new(gpu: &Gpu, window: &Arc<Window>, target_format: wgpu::TextureFormat) -> Self {
        let ctx = egui::Context::default();
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            None,
            // Our textures are small; the adapter limit is a safe ceiling.
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );

        let renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            target_format,
            egui_wgpu::RendererOptions {
                // No MSAA and no depth buffer: the cloud pass has neither.
                msaa_samples: 1,
                depth_stencil_format: None,
                ..Default::default()
            },
        );

        Self {
            ctx,
            state,
            renderer,
            open: true,
        }
    }

    /// Feed a window event to egui.
    ///
    /// Returns true if egui wants the event for itself — if the pointer is over
    /// the panel, dragging a slider must not also spin the camera.
    pub fn on_event(&mut self, window: &Window, event: &winit::event::WindowEvent) -> bool {
        self.state.on_window_event(window, event).consumed
    }

    /// Build the panel and draw it over `view`.
    ///
    /// `params` is edited in place: egui writes a new value straight into the
    /// struct the moment a slider moves, and the next frame uploads it.
    pub fn draw(
        &mut self,
        gpu: &Gpu,
        window: &Arc<Window>,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        size: (u32, u32),
        params: &mut Params,
        fps: f32,
    ) {
        let raw_input = self.state.take_egui_input(window);

        let mut reset = false;
        let full_output = self.ctx.run_pass(raw_input, |ctx| {
            egui::Window::new("cloud dials")
                .default_open(self.open)
                .default_pos([16.0, 16.0])
                .resizable(false)
                .show(ctx, |ui| {
                    ui.label(format!("{fps:.0} fps"));
                    ui.separator();

                    // The algorithm picker. This is a different kind of knob
                    // from the sliders under it: those move a number the
                    // shader reads, this changes which density field it
                    // builds at all. Both arrive the same way, as a value in
                    // the params uniform, so switching costs one 32-byte
                    // upload and no recompile.
                    let mut algorithm = params.algorithm();
                    egui::ComboBox::from_label("algorithm")
                        .selected_text(algorithm.name())
                        .show_ui(ui, |ui| {
                            for option in Algorithm::ALL {
                                ui.selectable_value(&mut algorithm, option, option.name())
                                    .on_hover_text(option.help());
                            }
                        })
                        .response
                        .on_hover_text(algorithm.help());
                    if algorithm != params.algorithm() {
                        params.algorithm = algorithm as u32;
                        log::info!("algorithm: {}", algorithm.name());
                    }
                    ui.label(
                        egui::RichText::new(algorithm.help())
                            .small()
                            .weak(),
                    );
                    ui.separator();

                    ui.add(
                        egui::Slider::new(&mut params.coverage, Params::COVERAGE_RANGE)
                            .text("coverage")
                            .fixed_decimals(2),
                    )
                    .on_hover_text("How much of the sky is cloud.\n0 = clear, 1 = solid overcast.");

                    ui.add(
                        egui::Slider::new(&mut params.feature_size, Params::FEATURE_SIZE_RANGE)
                            .text("feature size")
                            .suffix(" m")
                            .logarithmic(true)
                            .fixed_decimals(0),
                    )
                    .on_hover_text(
                        "Size of the largest noise features: how big the clouds are.\n\
                         Small = busy popcorn, large = continental masses.",
                    );

                    ui.add(
                        egui::Slider::new(
                            &mut params.shadow_strength,
                            Params::SHADOW_STRENGTH_RANGE,
                        )
                        .text("shadow")
                        .fixed_decimals(2),
                    )
                    .on_hover_text(
                        "Self-shadowing strength.\n\
                         Low fakes the light that bounces around inside a real cloud, which \
                         single scattering misses; high leaves a flat grey mass.",
                    );

                    ui.add(
                        egui::Slider::new(&mut params.exposure, Params::EXPOSURE_RANGE)
                            .text("exposure")
                            .logarithmic(true)
                            .fixed_decimals(2),
                    )
                    .on_hover_text("Overall brightness, applied just before tone mapping.");

                    ui.separator();
                    reset = ui.button("reset").clicked();
                    ui.label(
                        egui::RichText::new(
                            "drag: look · scroll: sun height\nshift-drag: sun compass",
                        )
                        .small()
                        .weak(),
                    );
                });
        });

        if reset {
            *params = Params::default();
            log::info!("dials reset to defaults");
        }

        self.state
            .handle_platform_output(window, full_output.platform_output);

        // Turn egui's shapes into triangles, then hand them to the GPU. Fonts
        // and icons arrive as texture deltas that have to be applied first.
        let pixels_per_point = self.ctx.pixels_per_point();
        let primitives = self.ctx.tessellate(full_output.shapes, pixels_per_point);
        for (id, deltas) in &full_output.textures_delta.set {
            // One texture can arrive as several partial updates in a frame.
            for delta in deltas {
                self.renderer
                    .update_texture(&gpu.device, &gpu.queue, *id, delta);
            }
        }

        let screen = ScreenDescriptor {
            size_in_pixels: [size.0, size.1],
            pixels_per_point,
        };
        self.renderer
            .update_buffers(&gpu.device, &gpu.queue, encoder, &primitives, &screen);

        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Load, not Clear: the clouds are already in there.
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // egui's renderer wants a 'static pass; the encoder outlives it.
            self.renderer
                .render(&mut pass.forget_lifetime(), &primitives, &screen);
        }

        for id in &full_output.textures_delta.free {
            self.renderer.free_texture(id);
        }
    }
}
