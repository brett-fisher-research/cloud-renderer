//! The shader source: several WGSL files joined into one module.
//!
//! WGSL has no `#include`. Each concern lives in its own file under
//! `shaders/` (see the top of `common.wgsl` for which does what) and the
//! files are concatenated before compiling. Declaration order does not matter
//! in WGSL, so neither does the order here.
//!
//! Natively the files are read from disk so they can be edited while the
//! program runs. A web page has no disk, so there they are compiled into the
//! binary.

use std::path::{Path, PathBuf};

use anyhow::Result;

/// Where the shader files live, relative to the crate root.
pub const SHADER_DIR: &str = "shaders";

/// The label the joined module is compiled under. Compile errors name it,
/// which is how [`Source::locate_errors`] finds the line numbers to rewrite.
pub const MODULE_LABEL: &str = "scene.wgsl";

/// Every file in the module. Adding a feature's shader is one line here.
macro_rules! shader_files {
    ($($name:literal),* $(,)?) => {
        pub const FILES: &[&str] = &[$($name),*];
        #[cfg(target_arch = "wasm32")]
        const BUILT_IN: &[&str] = &[$(include_str!(concat!("../shaders/", $name))),*];
    };
}
shader_files!("common.wgsl", "sky.wgsl", "clouds.wgsl", "terrain.wgsl", "frame.wgsl");

/// The joined source, remembering where each file landed in it.
pub struct Source {
    pub text: String,
    /// Each file's name and the 1-based line of the joined text it starts on.
    starts: Vec<(&'static str, usize)>,
}

impl Source {
    fn join(parts: impl Iterator<Item = (&'static str, String)>) -> Self {
        let mut text = String::new();
        let mut starts = Vec::new();
        let mut line = 1;
        for (name, mut part) in parts {
            if !part.ends_with('\n') {
                part.push('\n');
            }
            starts.push((name, line));
            line += part.matches('\n').count();
            text.push_str(&part);
        }
        Self { text, starts }
    }

    /// Which file a line of the joined text came from, and its line there.
    fn locate(&self, line: usize) -> (&'static str, usize) {
        let (name, start) = self
            .starts
            .iter()
            .rev()
            .find(|(_, start)| *start <= line)
            .copied()
            .unwrap_or(("?", 1));
        (name, line - start + 1)
    }

    /// Rewrite `scene.wgsl:LINE` in a compiler diagnostic to the file and
    /// line the code actually lives at, so an error can be clicked through.
    pub fn locate_errors(&self, message: &str) -> String {
        let marker = format!("{MODULE_LABEL}:");
        let mut out = String::with_capacity(message.len());
        let mut rest = message;
        while let Some(at) = rest.find(&marker) {
            out.push_str(&rest[..at]);
            let after = &rest[at + marker.len()..];
            let digits = after.bytes().take_while(u8::is_ascii_digit).count();
            match after[..digits].parse::<usize>() {
                Ok(line) => {
                    let (name, local) = self.locate(line);
                    out.push_str(&format!("{SHADER_DIR}/{name}:{local}"));
                }
                Err(_) => out.push_str(&marker),
            }
            rest = &after[digits..];
        }
        out.push_str(rest);
        out
    }
}

/// Read and join every shader file.
pub fn load(dir: &Path) -> Result<Source> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        use anyhow::Context;
        log::info!("loading {} shader files from {}", FILES.len(), dir.display());
        let mut parts = Vec::new();
        for name in FILES {
            let path = dir.join(name);
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("could not read {}", path.display()))?;
            parts.push((*name, text));
        }
        Ok(Source::join(parts.into_iter()))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = dir;
        log::info!("using the shader built into the binary");
        Ok(Source::join(FILES.iter().copied().zip(BUILT_IN.iter().map(|s| s.to_string()))))
    }
}

/// Find `shaders/` whether we were launched from the crate root or from
/// somewhere else (e.g. a different cwd under `cargo run`).
pub fn resolve_dir() -> PathBuf {
    let direct = PathBuf::from(SHADER_DIR);
    if direct.is_dir() {
        return direct;
    }
    let from_manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join(SHADER_DIR);
    if from_manifest.is_dir() {
        return from_manifest;
    }
    direct
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errors_are_rewritten_to_the_file_they_came_from() {
        let source = Source::join(
            [("a.wgsl", "one\ntwo\n".to_string()), ("b.wgsl", "three\nfour".to_string())]
                .into_iter(),
        );
        assert_eq!(source.locate(2), ("a.wgsl", 2));
        assert_eq!(source.locate(3), ("b.wgsl", 1));
        assert_eq!(
            source.locate_errors("  ┌─ scene.wgsl:4:7\n"),
            "  ┌─ shaders/b.wgsl:2:7\n"
        );
    }
}
