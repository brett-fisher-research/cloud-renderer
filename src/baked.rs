//! What a feature hands the renderer: a texture it baked, and where the
//! shader expects to find it.
//!
//! The renderer builds its bind group from a list of these and knows nothing
//! about what any of them holds. Adding a feature's texture is one more entry
//! in the list and one `@binding` line in that feature's shader file.

/// A texture baked once at startup and sampled by the shader.
pub struct Baked {
    /// The `@binding` number in `@group(0)`.
    pub binding: u32,
    pub dimension: wgpu::TextureViewDimension,
    /// A view keeps its texture alive, so the view is all that is held.
    pub view: wgpu::TextureView,
}

impl Baked {
    pub fn new(binding: u32, dimension: wgpu::TextureViewDimension, texture: &wgpu::Texture) -> Self {
        Self {
            binding,
            dimension,
            view: texture.create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }
}
