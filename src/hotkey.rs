//! Watches every keyboard for the hotkey. Virtual keyboards (RustDesk, Toshy/xwaykeyz,
//! ydotoold) get destroyed and re-created at runtime, so new devices are picked up and
//! vanished ones dropped.

use std::collections::{HashMap, HashSet};
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use evdev::{Device, EventSummary, KeyCode};
use log::{debug, info, warn};

use crate::app::Event;

fn is_keyboard(device: &Device) -> bool {
    device
        .supported_keys()
        .is_some_and(|keys| keys.contains(KeyCode::KEY_F1) || keys.contains(KeyCode::KEY_F10))
}

/// Every /dev/input/event* node with its inode, which changes when a device number is
/// reused by a new device.
fn device_nodes() -> HashMap<PathBuf, u64> {
    std::fs::read_dir("/dev/input")
        .map(|dir| {
            dir.filter_map(|e| e.ok())
                .filter(|e| e.file_name().to_string_lossy().starts_with("event"))
                .filter_map(|e| Some((e.path(), e.metadata().ok()?.ino())))
                .collect()
        })
        .unwrap_or_default()
}

/// Devices already looked at: keyboards being read, and other devices (by node identity)
/// not worth reopening every second.
#[derive(Default)]
struct Seen {
    watched: HashSet<PathBuf>,
    skipped: HashMap<PathBuf, u64>,
    warned: HashSet<(PathBuf, u64)>,
}

impl Seen {
    /// Nodes worth opening: not read already, and not the same node skipped before.
    /// Forgets skipped nodes that have gone, so a reused device number is checked again.
    fn to_check(&mut self, nodes: &HashMap<PathBuf, u64>) -> Vec<(PathBuf, u64)> {
        self.skipped.retain(|path, ino| nodes.get(path) == Some(ino));
        self.warned.retain(|(path, ino)| nodes.get(path) == Some(ino));
        let mut out: Vec<_> = nodes
            .iter()
            .filter(|(path, ino)| !self.watched.contains(*path) && self.skipped.get(*path) != Some(ino))
            .map(|(path, ino)| (path.clone(), *ino))
            .collect();
        out.sort();
        out
    }
}

/// Spawn the watcher. Returns how many keyboards were found at start.
pub fn spawn(hotkey: KeyCode, tx: Sender<Event>) -> usize {
    let seen: Arc<Mutex<Seen>> = Arc::default();
    let initial = scan(hotkey, &tx, &seen);
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(1));
            scan(hotkey, &tx, &seen);
        }
    });
    initial
}

/// Start a reader thread for every keyboard not already watched.
fn scan(hotkey: KeyCode, tx: &Sender<Event>, seen: &Arc<Mutex<Seen>>) -> usize {
    let mut started = 0;
    let candidates = seen.lock().unwrap().to_check(&device_nodes());
    for (path, ino) in candidates {
        let device = match Device::open(&path) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                // udev may not have granted access yet; retry, but warn once per device
                if seen.lock().unwrap().warned.insert((path.clone(), ino)) {
                    warn!("Permission denied: {}", path.display());
                }
                continue;
            }
            Err(_) => continue,
        };
        if !is_keyboard(&device) {
            seen.lock().unwrap().skipped.insert(path, ino);
            continue;
        }
        let name = device.name().unwrap_or("?").to_string();
        info!("Monitoring keyboard device: {name} ({})", path.display());
        seen.lock().unwrap().watched.insert(path.clone());
        started += 1;

        let tx = tx.clone();
        let seen = Arc::clone(seen);
        std::thread::spawn(move || {
            read_device(device, &path, hotkey, &tx);
            warn!("Keyboard device gone, no longer monitoring: {name} ({})", path.display());
            // Forget the path so a re-created device reusing it gets picked up
            seen.lock().unwrap().watched.remove(&path);
            let _ = tx.send(Event::DeviceGone(path));
        });
    }
    started
}

fn read_device(mut device: Device, path: &PathBuf, hotkey: KeyCode, tx: &Sender<Event>) {
    loop {
        let events = match device.fetch_events() {
            Ok(events) => events,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(_) => return,
        };
        for event in events {
            if let EventSummary::Key(_, code, value) = event.destructure() {
                if code != hotkey {
                    continue;
                }
                let msg = match value {
                    1 => Event::KeyDown(path.clone()),
                    0 => Event::KeyUp,
                    _ => continue, // autorepeat
                };
                debug!("Hotkey {}", if value == 1 { "pressed" } else { "released" });
                if tx.send(msg).is_err() {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nodes(list: &[(&str, u64)]) -> HashMap<PathBuf, u64> {
        list.iter().map(|(p, i)| (PathBuf::from(p), *i)).collect()
    }

    #[test]
    fn skipped_device_number_reused_by_a_new_device_is_checked_again() {
        let mut seen = Seen::default();
        let first = nodes(&[("/dev/input/event9", 100)]);
        assert_eq!(seen.to_check(&first).len(), 1);
        seen.skipped.insert("/dev/input/event9".into(), 100); // a mouse, say
        assert!(seen.to_check(&first).is_empty(), "the same node is not reopened");

        // Toshy's virtual keyboard comes back on the same number between two scans
        let reused = nodes(&[("/dev/input/event9", 200)]);
        assert_eq!(seen.to_check(&reused), [("/dev/input/event9".into(), 200)]);
    }

    #[test]
    fn vanished_devices_are_forgotten() {
        let mut seen = Seen::default();
        seen.skipped.insert("/dev/input/event3".into(), 1);
        seen.warned.insert(("/dev/input/event4".into(), 2));
        seen.to_check(&nodes(&[]));
        assert!(seen.skipped.is_empty() && seen.warned.is_empty());
        // Back with the same inode (unlikely, but harmless): checked again
        assert_eq!(seen.to_check(&nodes(&[("/dev/input/event3", 1)])).len(), 1);
    }

    #[test]
    fn watched_keyboards_are_not_reopened() {
        let mut seen = Seen::default();
        seen.watched.insert("/dev/input/event5".into());
        assert!(seen.to_check(&nodes(&[("/dev/input/event5", 7)])).is_empty());
    }
}
