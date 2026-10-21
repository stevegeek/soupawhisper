//! Getting text out: clipboard, auto-paste, desktop notifications.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Duration;

use anyhow::{Context, Result};
use log::{debug, warn};

pub fn is_wayland() -> bool {
    std::env::var("XDG_SESSION_TYPE").is_ok_and(|t| t == "wayland") || std::env::var_os("WAYLAND_DISPLAY").is_some()
}

pub fn copy_to_clipboard(text: &str) -> Result<()> {
    let mut cmd = if is_wayland() {
        Command::new("wl-copy")
    } else {
        let mut c = Command::new("xclip");
        c.args(["-selection", "clipboard"]);
        c
    };
    let mut child = cmd.stdin(Stdio::piped()).spawn().context("starting clipboard tool")?;
    child.stdin.take().unwrap().write_all(text.as_bytes())?;
    child.wait()?;
    Ok(())
}

/// Paste into the focused window: the configured shortcut via ydotool on Wayland,
/// or type the text with xdotool on X11.
pub fn auto_type(text: &str, paste_keys: &[String]) {
    std::thread::sleep(Duration::from_millis(150)); // let the clipboard settle
    let result = if is_wayland() {
        Command::new("ydotool").arg("key").args(paste_keys).status()
    } else {
        Command::new("xdotool").args(["type", "--clearmodifiers", text]).status()
    };
    if let Err(e) = result {
        warn!("Auto-type failed: {e}");
    }
}

pub struct Notifier {
    enabled: bool,
    last_id: u32,
}

impl Notifier {
    pub fn new(enabled: bool) -> Notifier {
        Notifier { enabled, last_id: 0 }
    }

    /// A new notification each time: KDE doesn't re-show replaced ones after dismissal.
    pub fn notify(&mut self, title: &str, body: &str, icon: &str, timeout_ms: i32) {
        if !self.enabled {
            return;
        }
        debug!("Notification: {title} - {body}");
        let out = Command::new("gdbus")
            .args([
                "call", "--session",
                "--dest", "org.freedesktop.Notifications",
                "--object-path", "/org/freedesktop/Notifications",
                "--method", "org.freedesktop.Notifications.Notify",
                "SoupaWhisper", "0", icon, title, body, "[]", "{}",
                &timeout_ms.to_string(),
            ])
            .stderr(Stdio::null())
            .output();
        // Output looks like "(uint32 123,)"
        if let Ok(out) = out {
            let s = String::from_utf8_lossy(&out.stdout);
            if let Some(id) = s.trim().strip_prefix("(uint32 ").and_then(|r| r.strip_suffix(",)")) {
                self.last_id = id.parse().unwrap_or(0);
            }
        }
    }

    /// Close the most recent notification, if any.
    pub fn close_last(&mut self) {
        if !self.enabled || self.last_id == 0 {
            return;
        }
        let _ = Command::new("gdbus")
            .args([
                "call", "--session",
                "--dest", "org.freedesktop.Notifications",
                "--object-path", "/org/freedesktop/Notifications",
                "--method", "org.freedesktop.Notifications.CloseNotification",
                &self.last_id.to_string(),
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
}
