//! The live-tunable cloud dials.
//!
//! Graphics note: this is a *second* uniform buffer rather than extra fields on
//! [`crate::camera::Uniforms`], for two reasons. The camera block is rewritten
//! every frame from the mouse; these change only when you move a slider. And
//! keeping them apart means one struct of plain `f32`s with no `vec3`s in it,
//! so there is no 16-byte alignment padding to keep in sync — the trap that
//! `Uniforms` has to work around.

use bytemuck::{Pod, Zeroable};

/// Matches `struct Params` in `shaders/clouds.wgsl`.
///
/// A uniform block is a multiple of 16 bytes, so dials come in fours. There
/// are five, so three `_pad` floats fill the second row; the next three dials
/// take their places.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Params {
    /// How much of the sky is cloud. 0 = clear, 1 = solid overcast.
    pub coverage: f32,
    /// How wide a typical cloud is, in metres. Small = popcorn, large =
    /// towering masses that fill the layer.
    pub feature_size: f32,
    /// Scales self-shadowing. Low values let light through the clouds so
    /// they glow; high values give dark, heavy undersides.
    pub shadow_strength: f32,
    /// How far the fine noise pushes the cloud surface in and out. 0 = smooth
    /// blobs, 1 = ragged and torn.
    pub wispiness: f32,
    /// Overall brightness multiplier, applied just before tone mapping, on
    /// top of the automatic exposure.
    pub exposure: f32,
    /// Unused; fills the second 16-byte row.
    pub _pad: [f32; 3],
}

impl Default for Params {
    fn default() -> Self {
        Self {
            coverage: 0.45,
            feature_size: 1500.0,
            shadow_strength: 1.0,
            wispiness: 0.5,
            exposure: 0.5,
            _pad: [0.0; 3],
        }
    }
}

/// Slider bounds, kept next to the values they constrain so the UI and any
/// command-line clamping agree.
impl Params {
    pub const COVERAGE_RANGE: std::ops::RangeInclusive<f32> = 0.0..=1.0;
    pub const FEATURE_SIZE_RANGE: std::ops::RangeInclusive<f32> = 500.0..=6000.0;
    pub const SHADOW_STRENGTH_RANGE: std::ops::RangeInclusive<f32> = 0.0..=2.0;
    pub const WISPINESS_RANGE: std::ops::RangeInclusive<f32> = 0.0..=1.0;
    pub const EXPOSURE_RANGE: std::ops::RangeInclusive<f32> = 0.1..=4.0;
}
