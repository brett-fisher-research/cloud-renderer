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
/// A uniform block is a multiple of 16 bytes, so these come in fours. The
/// first four `f32`s filled the first sixteen exactly; `algorithm` opened a
/// second row, and the three `_pad` slots are the rest of it. They are spare
/// dials waiting to happen — a fifth real one costs nothing until there are
/// four of them.
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
    /// Which density field to build the clouds from: an [`Algorithm`] as a
    /// plain `u32`, because that is what crosses into the shader.
    pub algorithm: u32,
    /// Unused; see the note on the struct.
    pub _pad: [f32; 3],
}

impl Default for Params {
    fn default() -> Self {
        Self {
            coverage: 0.45,
            feature_size: 2600.0,
            shadow_strength: 0.60,
            exposure: 1.0,
            algorithm: Algorithm::Fbm as u32,
            _pad: [0.0; 3],
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

    pub fn algorithm(&self) -> Algorithm {
        Algorithm::from(self.algorithm)
    }
}

/// Which density field the clouds are built from.
///
/// Every mode marches the same slab with the same lighting, aerial perspective
/// and tone mapping. The only thing that changes is the function answering
/// "how much cloud is at this point", which is where the *look* of a cloud
/// renderer actually lives — so this is the one knob worth being able to flip
/// between while the program runs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Algorithm {
    /// Thresholded fBm over the whole slab: one connected field everywhere.
    #[default]
    Fbm = 0,
    /// The same field, gated by a much coarser one so cloud gathers into
    /// separate masses with real clear sky between them.
    Islands = 1,
    /// No field at all: one blob per cell of a horizontal grid, placed and
    /// sized by a hash, then eroded by noise. Discrete clouds, no repeat.
    Puffs = 2,
    /// Perlin-Worley: the fBm with a cellular field remapped into it, which
    /// is what gives clouds cauliflower bulges rather than smooth lumps.
    Billow = 3,
    /// Billow inside the islands gate: cauliflower masses with sky between.
    Cumulus = 4,
}

impl Algorithm {
    pub const ALL: [Algorithm; 5] = [
        Algorithm::Fbm,
        Algorithm::Islands,
        Algorithm::Puffs,
        Algorithm::Billow,
        Algorithm::Cumulus,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Algorithm::Fbm => "fbm",
            Algorithm::Islands => "islands",
            Algorithm::Puffs => "puffs",
            Algorithm::Billow => "billow",
            Algorithm::Cumulus => "cumulus",
        }
    }

    /// One line for the panel's tooltip and for `--help`.
    pub fn help(self) -> &'static str {
        match self {
            Algorithm::Fbm => "Thresholded fBm: one connected field over the whole sky.",
            Algorithm::Islands => {
                "fBm gated by a coarse mask: cloud gathers into separate masses \
                 with clear sky between."
            }
            Algorithm::Puffs => {
                "Hash-placed blobs on a grid, eroded by noise: discrete clouds, \
                 no field and no repeat."
            }
            Algorithm::Billow => {
                "Perlin-Worley: a cellular field remapped into the fBm, for \
                 cauliflower bulges."
            }
            Algorithm::Cumulus => {
                "Billow inside the islands gate: cauliflower masses with real \
                 sky between them."
            }
        }
    }

    pub fn from_name(s: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|a| a.name() == s)
    }

    /// Every name, for error messages and `--help`.
    pub fn names() -> String {
        Self::ALL
            .iter()
            .map(|a| a.name())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl From<u32> for Algorithm {
    fn from(v: u32) -> Self {
        Self::ALL
            .into_iter()
            .find(|a| *a as u32 == v)
            .unwrap_or(Algorithm::Fbm)
    }
}

