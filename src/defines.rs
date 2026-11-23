//! Overriding shader constants from the command line.
//!
//! The shader keeps its tunables as `const`s at the top of the file, which is
//! right for hot-reloading by hand but wrong for a sweep: measuring eight step
//! counts should not mean eight edits and eight rebuilds.
//!
//! `--def NAME=VALUE` rewrites the matching `const` line in the WGSL source
//! after it is read from disk and before it is compiled. The file on disk is
//! never touched, so a sweep cannot leave the working tree dirty, and the
//! substitution is logged so a run's output says what it actually ran.
//!
//! A define that matches nothing is an error rather than a warning. In a sweep
//! the whole point is that the variable changed; silently measuring the
//! default eight times and calling it a flat response would be worse than
//! crashing.

use anyhow::{bail, Result};

/// A list of `NAME = VALUE` overrides to apply to the shader source.
#[derive(Clone, Debug, Default)]
pub struct Defines(Vec<(String, String)>);

impl Defines {
    /// Parse one `NAME=VALUE` argument and add it.
    pub fn push_arg(&mut self, arg: &str) -> Result<()> {
        let Some((name, value)) = arg.split_once('=') else {
            bail!("--def needs NAME=VALUE, got {arg:?}");
        };
        let (name, value) = (name.trim(), value.trim());
        if name.is_empty() || value.is_empty() {
            bail!("--def needs a non-empty name and value, got {arg:?}");
        }
        if !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            bail!("--def name must be a WGSL identifier, got {name:?}");
        }
        self.0.push((name.to_string(), value.to_string()));
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// A stable one-line description, for benchmark labels and notebook rows.
    pub fn summary(&self) -> String {
        self.0
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Apply every override to `source`, returning the rewritten WGSL.
    pub fn apply(&self, source: &str) -> Result<String> {
        let mut out = source.to_string();
        for (name, value) in &self.0 {
            out = substitute(&out, name, value)?;
        }
        Ok(out)
    }
}

/// Replace the initialiser of `const NAME: T = ...;` with `value`.
///
/// Matches on the declaration rather than on the name alone, so a constant
/// that is also *used* further down the file is only rewritten where it is
/// declared. The type annotation is kept exactly as written: a value of the
/// wrong type then fails WGSL compilation with a real diagnostic, which is a
/// better error than anything this function could invent.
fn substitute(source: &str, name: &str, value: &str) -> Result<String> {
    let mut out = String::with_capacity(source.len() + 32);
    let mut hits = 0usize;

    for line in source.split_inclusive('\n') {
        let body = line.trim_start();
        let matched = body
            .strip_prefix("const ")
            .and_then(|rest| rest.strip_prefix(name))
            // The next character after the name must end the identifier, or
            // `PRIMARY_STEPS` would also match `PRIMARY_STEPS_FINE`.
            .filter(|rest| rest.starts_with(':') || rest.starts_with(' ') || rest.starts_with('='))
            .is_some();

        if !matched {
            out.push_str(line);
            continue;
        }

        let Some(eq) = line.find('=') else {
            bail!("const {name} has no '=' to substitute into: {line:?}");
        };
        let Some(semi) = line[eq..].find(';') else {
            bail!("const {name} declaration does not end with ';' on one line: {line:?}");
        };
        let tail = &line[eq + semi..]; // from ';' to end of line, comments kept

        out.push_str(&line[..eq]);
        out.push_str("= ");
        out.push_str(value);
        out.push_str(tail);
        hits += 1;
    }

    match hits {
        0 => bail!(
            "--def {name}={value} matched no `const {name}` declaration in the shader. \
             Check the spelling; a define that changes nothing would silently \
             invalidate the measurement."
        ),
        1 => Ok(out),
        n => bail!("--def {name} matched {n} declarations; expected exactly one"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defs(pairs: &[&str]) -> Defines {
        let mut d = Defines::default();
        for p in pairs {
            d.push_arg(p).unwrap();
        }
        d
    }

    #[test]
    fn replaces_a_scalar_and_keeps_the_comment() {
        let src = "const PRIMARY_STEPS: i32 = 64;      // samples along the view ray\n";
        let out = defs(&["PRIMARY_STEPS=128"]).apply(src).unwrap();
        assert_eq!(
            out,
            "const PRIMARY_STEPS: i32 = 128;      // samples along the view ray\n"
        );
    }

    #[test]
    fn replaces_a_vector() {
        let src = "const WIND: vec3<f32> = vec3<f32>(-9.0, 0.0, -3.0);\n";
        let out = defs(&["WIND=vec3<f32>(0.0, 0.0, 0.0)"]).apply(src).unwrap();
        assert!(out.contains("= vec3<f32>(0.0, 0.0, 0.0);"));
    }

    #[test]
    fn does_not_match_a_longer_name() {
        let src = "const TILE_SCALE_FINE: f32 = 4.0;\nconst TILE_SCALE: f32 = 4.0;\n";
        let out = defs(&["TILE_SCALE=6.0"]).apply(src).unwrap();
        assert!(out.contains("const TILE_SCALE_FINE: f32 = 4.0;"));
        assert!(out.contains("const TILE_SCALE: f32 = 6.0;"));
    }

    #[test]
    fn does_not_match_a_use_site() {
        let src = "const TILE_SCALE: f32 = 4.0;\nlet t = TILE_SCALE = 9;\n";
        let out = defs(&["TILE_SCALE=6.0"]).apply(src).unwrap();
        assert!(out.contains("let t = TILE_SCALE = 9;"));
    }

    #[test]
    fn an_unmatched_define_is_an_error() {
        let src = "const PRIMARY_STEPS: i32 = 64;\n";
        assert!(defs(&["NO_SUCH_CONST=1"]).apply(src).is_err());
    }

    #[test]
    fn rejects_a_malformed_argument() {
        let mut d = Defines::default();
        assert!(d.push_arg("NOEQUALS").is_err());
        assert!(d.push_arg("=4").is_err());
        assert!(d.push_arg("NAME=").is_err());
        assert!(d.push_arg("bad name=4").is_err());
    }

    #[test]
    fn applies_several_in_order() {
        let src = "const A: f32 = 1.0;\nconst B: i32 = 2;\n";
        let out = defs(&["A=9.0", "B=8"]).apply(src).unwrap();
        assert!(out.contains("const A: f32 = 9.0;"));
        assert!(out.contains("const B: i32 = 8;"));
    }
}
