//! Filesystem watching for shader hot-reload.

use std::path::Path;
use std::sync::mpsc::{Receiver, channel};

use notify::{EventKind, RecursiveMode, Watcher};

/// Watch the shader directory and send `()` whenever a `.wgsl` file in it is
/// written to.
///
/// The watcher is deliberately leaked: it must outlive this function for
/// events to keep arriving, and it lives for the whole program anyway.
pub fn watch_shaders(dir: &Path) -> Receiver<()> {
    let (tx, rx) = channel();
    // Watching the directory rather than each file also survives editors that
    // save by writing a temp file and renaming it over the original, which
    // destroys the inode a file-level watch is attached to.
    let dir = dir.to_path_buf();

    let result = notify::recommended_watcher(move |res: notify::Result<notify::Event>| match res {
        Ok(event) => {
            let relevant = matches!(
                event.kind,
                EventKind::Create(_) | EventKind::Modify(_) | EventKind::Any
            );
            let touches_shader = event
                .paths
                .iter()
                .any(|p| p.extension().is_some_and(|ext| ext == "wgsl"));
            if relevant && touches_shader {
                let _ = tx.send(());
            }
        }
        Err(e) => log::warn!("shader watcher error: {e}"),
    });

    match result {
        Ok(mut watcher) => match watcher.watch(&dir, RecursiveMode::NonRecursive) {
            Ok(()) => {
                log::info!("watching {} for shader changes", dir.display());
                // Keep the watcher alive for the lifetime of the process.
                std::mem::forget(watcher);
            }
            Err(e) => log::warn!("could not watch {}: {e} (hot-reload disabled)", dir.display()),
        },
        Err(e) => log::warn!("could not create file watcher: {e} (hot-reload disabled)"),
    }

    rx
}
