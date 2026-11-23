//! A ground-level look-around camera, plus the uniform block we hand the shader.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

/// Vertical field of view in degrees.
const FOV_Y_DEG: f32 = 65.0;

/// Clamp pitch just short of straight up/down so the basis never degenerates.
const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - 0.02;

/// The data block the fragment shader reads every frame.
///
/// Graphics 101: a *uniform buffer* is a small blob of read-only data that is
/// the same ("uniform") for every pixel in a draw call. WGSL requires `vec3`
/// to be 16-byte aligned, so each one gets an explicit padding float — if the
/// Rust layout and the WGSL layout disagree you get silent garbage, not an error.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Uniforms {
    /// Render target size in pixels.
    pub resolution: [f32; 2],
    /// Seconds since start, used to scroll the noise (wind).
    pub time: f32,
    /// tan(fov_y / 2), so the shader can build ray directions with one multiply.
    pub fov_tan: f32,

    /// Camera basis: where we are looking...
    pub cam_forward: [f32; 3],
    pub _pad0: f32,
    /// ...and the screen-right and screen-up axes that go with it.
    pub cam_right: [f32; 3],
    pub _pad1: f32,
    pub cam_up: [f32; 3],
    pub _pad2: f32,

    /// Unit vector pointing *towards* the sun.
    pub sun_dir: [f32; 3],
    pub _pad3: f32,
}

/// Mouse-driven look-around camera. It never moves; only the orientation and
/// the sun angle change, which is all the MVP needs.
pub struct Camera {
    /// Horizontal angle in radians.
    pub yaw: f32,
    /// Vertical angle in radians; positive looks up.
    pub pitch: f32,
    /// Sun elevation in radians: 0 = on the horizon, pi/2 = directly overhead.
    pub sun_elevation: f32,
    /// Sun compass angle in radians.
    pub sun_azimuth: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            // Start tilted up a bit so the cloud layer is in frame.
            pitch: 0.25,
            // A low-ish sun gives the nicest silver lining to start with.
            sun_elevation: 0.30,
            sun_azimuth: 0.6,
        }
    }
}

impl Camera {
    /// Apply a mouse drag, in pixels.
    pub fn look(&mut self, dx: f32, dy: f32) {
        const SENSITIVITY: f32 = 0.0035;
        self.yaw -= dx * SENSITIVITY;
        self.pitch = (self.pitch - dy * SENSITIVITY).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Apply a horizontal mouse drag to the sun's compass bearing.
    ///
    /// Elevation is "time of day"; azimuth is which way you are facing when
    /// you see it. Moving it swings every cloud shadow and slides the silver
    /// lining around the cloud edges.
    pub fn drag_sun_azimuth(&mut self, dx: f32) {
        const SUN_DRAG_SENSITIVITY: f32 = 0.004;
        self.sun_azimuth += dx * SUN_DRAG_SENSITIVITY;
    }

    /// Apply a scroll notch, moving the sun up or down ("time of day").
    pub fn scroll_sun(&mut self, notches: f32) {
        const SUN_SPEED: f32 = 0.04;
        // Allow the sun to dip below the horizon for dusk.
        self.sun_elevation = (self.sun_elevation + notches * SUN_SPEED).clamp(-0.25, 1.55);
    }

    /// Forward direction of the camera in world space (+Y is up).
    pub fn forward(&self) -> Vec3 {
        let (sp, cp) = self.pitch.sin_cos();
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(cp * sy, sp, -cp * cy).normalize()
    }

    pub fn sun_dir(&self) -> Vec3 {
        let (se, ce) = self.sun_elevation.sin_cos();
        let (sa, ca) = self.sun_azimuth.sin_cos();
        Vec3::new(ce * sa, se, -ce * ca).normalize()
    }

    /// Pack the current state into the GPU uniform block.
    pub fn uniforms(&self, width: u32, height: u32, time: f32) -> Uniforms {
        let forward = self.forward();
        // Build an orthonormal screen basis. Using world-up as the reference is
        // fine because pitch is clamped away from vertical.
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward).normalize();

        Uniforms {
            resolution: [width as f32, height as f32],
            time,
            fov_tan: (FOV_Y_DEG.to_radians() * 0.5).tan(),
            cam_forward: forward.into(),
            _pad0: 0.0,
            cam_right: right.into(),
            _pad1: 0.0,
            cam_up: up.into(),
            _pad2: 0.0,
            sun_dir: self.sun_dir().into(),
            _pad3: 0.0,
        }
    }
}
