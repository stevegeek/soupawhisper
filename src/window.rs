//! Which app has focus, so terminals can get their own paste shortcut. On KDE Plasma a
//! small KWin script reports every window activation to us over D-Bus; elsewhere the
//! focused app is unknown and the default shortcut is used.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use log::{debug, info};
use zbus::blocking::{Connection, Proxy, connection};

const SERVICE: &str = "io.github.stevegeek.SoupaWhisper";
const SCRIPT_NAME: &str = "soupawhisper-active-window";

const SCRIPT: &str = r#"function report(window) {
    if (window) {
        callDBus("io.github.stevegeek.SoupaWhisper", "/", "io.github.stevegeek.SoupaWhisper",
                 "WindowActivated", String(window.resourceClass));
    }
}
workspace.windowActivated.connect(report);
report(workspace.activeWindow);
"#;

struct Receiver {
    class: Arc<Mutex<Option<String>>>,
}

#[zbus::interface(name = "io.github.stevegeek.SoupaWhisper")]
impl Receiver {
    fn window_activated(&self, class: String) {
        debug!("Active window: {class}");
        *self.class.lock().unwrap() = Some(class);
    }
}

pub struct ActiveWindow {
    class: Arc<Mutex<Option<String>>>,
    /// Kept open: it serves the Receiver.
    _conn: Connection,
}

impl ActiveWindow {
    /// The resource class of the focused window, if KWin has reported one.
    pub fn class(&self) -> Option<String> {
        self.class.lock().unwrap().clone()
    }
}

/// Start following the focused window through KWin, writing the script to `dir`.
pub fn watch(dir: &Path) -> Result<ActiveWindow> {
    let class: Arc<Mutex<Option<String>>> = Arc::default();
    let conn = connection::Builder::session()?
        .name(SERVICE)?
        .serve_at("/", Receiver { class: Arc::clone(&class) })?
        .build()
        .context("connecting to the session bus")?;

    std::fs::create_dir_all(dir)?;
    let path = dir.join("kwin-active-window.js");
    std::fs::write(&path, SCRIPT)?;

    let scripting = Proxy::new(&conn, "org.kde.KWin", "/Scripting", "org.kde.kwin.Scripting")?;
    // A previous run's script would report to a stale connection; replace it
    let _: bool = scripting.call("unloadScript", &(SCRIPT_NAME,)).context("talking to KWin")?;
    let id: i32 = scripting.call("loadScript", &(path.to_string_lossy().as_ref(), SCRIPT_NAME))?;
    let _: () = scripting.call("start", &())?;
    info!("Following the focused window through KWin (script {id})");
    Ok(ActiveWindow { class, _conn: conn })
}

/// Whether a window class belongs to one of the configured terminals.
pub fn is_terminal(class: &str, terminals: &[String]) -> bool {
    let class = class.to_lowercase();
    terminals.iter().any(|t| *t == class)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_match_ignores_case() {
        let terminals = vec!["org.kde.konsole".to_string(), "kitty".to_string()];
        assert!(is_terminal("org.kde.konsole", &terminals));
        assert!(is_terminal("Kitty", &terminals));
        assert!(!is_terminal("firefox", &terminals));
    }
}
