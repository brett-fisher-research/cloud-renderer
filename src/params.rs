//! The live-tunable cloud dials.
//!
//! These four values used to be `const`s at the top of `clouds.wgsl`. They now
//! live in their own uniform buffer so they can change while the program runs,
//! without recompiling the shader.
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
/// Four `f32`s is exactly 16 bytes, which is also the minimum size and the
/// alignment WGSL wants for a uniform block, so no padding is needed. Adding a
/// fifth field means adding three more to keep that true.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Params {
    /// How much of the sky is cloud. 0 = clear, 1 = solid overcast.
    pub coverage: f32,
    /// Size of the largest noise features in metres: how *big* the clouds are.
    /// Small = busy popcorn, large = continental masses. The bottom of the
    /// range is limited by the baked noise texture; see `FEATURE_SIZE_RANGE`.
    pub feature_size: f32,
    /// Scales self-shadowing. Below 1.0 fakes the extra brightness real clouds
    /// get from light bouncing around inside them, which single scattering does
    /// not capture; high values leave the clouds a flat grey mass.
    pub shadow_strength: f32,
    /// Overall brightness multiplier, applied just before tone mapping.
    pub exposure: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            coverage: 0.45,
            feature_size: 2600.0,
            shadow_strength: 0.60,
            exposure: 1.0,
        }
    }
}

/// Slider bounds, kept next to the values they constrain so the UI and any
/// command-line clamping agree.
impl Params {
    pub const COVERAGE_RANGE: std::ops::RangeInclusive<f32> = 0.0..=1.0;
    /// Back to 300 m. The floor was 1500 m for as long as the noise was a
    /// plain tiling texture: the tile is four times the feature size, so at
    /// 300 m the field wrapped every 1.2 km, and across a 50 km view that is
    /// forty repeats, which read as a radial moiré converging at the horizon.
    ///
    /// The shader now warps the lookup point and blends a second read at an
    /// incommensurate scale, which pushes the repeat past the view. Measured
    /// at 300 m: tile_lock 0.051 and in-view autocorrelation 0.485, against
    /// 1.000 and 1.000 before, and the render at a near-horizon pitch shows
    /// no moiré at all. See the repetition section of CLAUDE.md.
    pub const FEATURE_SIZE_RANGE: std::ops::RangeInclusive<f32> = 300.0..=10000.0;
    pub const SHADOW_STRENGTH_RANGE: std::ops::RangeInclusive<f32> = 0.0..=2.0;
    pub const EXPOSURE_RANGE: std::ops::RangeInclusive<f32> = 0.1..=4.0;
}
