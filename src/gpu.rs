//! wgpu setup shared by both the windowed and headless (screenshot) modes.

use anyhow::{Context, Result};

/// The long-lived wgpu objects every mode needs.
///
/// Graphics 101:
/// - `Instance` is the library entry point; it talks to the OS graphics APIs.
/// - `Adapter` is a handle to a *physical* GPU (we ask for the discrete AMD one).
/// - `Device` is our logical connection to it, used to create resources
///   (buffers, textures, pipelines).
/// - `Queue` is where we submit recorded work for the GPU to actually execute.
pub struct Gpu {
    /// Kept alive for the lifetime of the renderer: surfaces are created from
    /// it, so it must outlive them.
    #[allow(dead_code)]
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// Create an instance restricted to Vulkan, which is what Mesa/RADV exposes on
/// this machine. Restricting the backend keeps the adapter list predictable.
pub fn create_instance() -> wgpu::Instance {
    log::info!("creating wgpu instance (Vulkan backend)");
    wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        // The display handle is only needed by the GL/WebGPU backends; Vulkan
        // picks up the X11/Wayland connection from the surface itself.
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    })
}

impl Gpu {
    /// Pick an adapter and open a device on it.
    ///
    /// `compatible_surface` is `Some` in windowed mode so wgpu only offers us
    /// adapters that can actually present to that window, and `None` for
    /// headless screenshots where there is no surface at all.
    pub fn new(
        instance: wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self> {
        // Log everything we can see; this doubles as our "is the Vulkan driver
        // working?" check, standing in for `vulkaninfo`.
        for a in pollster::block_on(instance.enumerate_adapters(wgpu::Backends::VULKAN)) {
            let info = a.get_info();
            log::info!(
                "available adapter: {:?} {} ({:?}, driver: {} {})",
                info.device_type,
                info.name,
                info.backend,
                info.driver,
                info.driver_info
            );
        }

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            // Prefer the discrete GPU over the Raphael iGPU.
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface,
            ..Default::default()
        }))
        .context("no suitable Vulkan adapter found (is mesa-vulkan-drivers installed?)")?;

        let info = adapter.get_info();
        log::info!(
            "using adapter: {} ({:?}, {:?}) driver: {} {}",
            info.name,
            info.device_type,
            info.backend,
            info.driver,
            info.driver_info
        );

        // Timestamp queries let the GPU time its own work (see `--bench`).
        // Optional: ask for it only if this adapter has it, so a driver
        // without it still runs everything else.
        let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        if features.is_empty() {
            log::warn!("adapter has no TIMESTAMP_QUERY; --bench will not work");
        }

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("cloud-renderer device"),
            required_features: features,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
            ..Default::default()
        }))
        .context("failed to open a wgpu device")?;

        // Without this, a validation error that escapes an error scope panics
        // the process. We want to log it and keep running (see shader reload).
        device.on_uncaptured_error(std::sync::Arc::new(|err| {
            log::error!("wgpu uncaptured error: {err}");
        }));

        Ok(Self {
            instance,
            adapter,
            device,
            queue,
        })
    }
}
