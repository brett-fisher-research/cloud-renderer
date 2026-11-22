//! Filesystem watching for shader hot-reload.

use std::path::Path;
use std::sync::mpsc::{Receiver, channel};

use notify::{EventKind, RecursiveMode, Watcher};

/// Watch `path` and send `()` whenever it is written to.
///
/// The watcher is deliberately leaked: it must outlive this function for
/// events to keep arriving, and it lives for the whole program anyway.
pub fn watch_file(path: &Path) -> Receiver<()> {
    let (tx, rx) = channel();
    let path = path.to_path_buf();

    // Watch the *directory*, not the file: many editors save by writing a temp
    // file and renaming it over the original, which destroys the inode a
    // file-level watch is attached to.
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let watched_name = path.file_name().map(|n| n.to_os_string());

    let result = notify::recommended_watcher(move |res: notify::Result<notify::Event>| match res {
        Ok(event) => {
            let relevant = matches!(
                event.kind,
                EventKind::Create(_) | EventKind::Modify(_) | EventKind::Any
            );
            let touches_shader = event.paths.iter().any(|p| {
                watched_name
                    .as_ref()
                    .is_some_and(|name| p.file_name() == Some(name.as_os_str()))
            });
            if relevant && touches_shader {
                let _ = tx.send(());
            }
        }
        Err(e) => log::warn!("shader watcher error: {e}"),
    });

    match result {
        Ok(mut watcher) => match watcher.watch(&dir, RecursiveMode::NonRecursive) {
            Ok(()) => {
                log::info!("watching {} for changes", path.display());
                // Keep the watcher alive for the lifetime of the process.
                std::mem::forget(watcher);
            }
            Err(e) => log::warn!("could not watch {}: {e} (hot-reload disabled)", dir.display()),
        },
        Err(e) => log::warn!("could not create file watcher: {e} (hot-reload disabled)"),
    }

    rx
}
