//! Watches every keyboard for the hotkey. Virtual keyboards (RustDesk, Toshy/xwaykeyz,
//! ydotoold) get destroyed and re-created at runtime, so new devices are picked up and
//! vanished ones dropped.

use std::collections::HashSet;
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

fn device_paths() -> HashSet<PathBuf> {
    std::fs::read_dir("/dev/input")
        .map(|dir| {
            dir.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.file_name().is_some_and(|n| n.to_string_lossy().starts_with("event")))
                .collect()
        })
        .unwrap_or_default()
}

/// Spawn the watcher. Returns how many keyboards were found at start.
pub fn spawn(hotkey: KeyCode, tx: Sender<Event>) -> usize {
    let watched: Arc<Mutex<HashSet<PathBuf>>> = Arc::default();
    let initial = scan(hotkey, &tx, &watched);
    std::thread::spawn(move || {
        loop {
            std::thread::sleep(Duration::from_secs(1));
            scan(hotkey, &tx, &watched);
        }
    });
    initial
}

/// Start a reader thread for every keyboard not already watched.
fn scan(hotkey: KeyCode, tx: &Sender<Event>, watched: &Arc<Mutex<HashSet<PathBuf>>>) -> usize {
    let mut started = 0;
    for path in device_paths() {
        if watched.lock().unwrap().contains(&path) {
            continue;
        }
        let device = match Device::open(&path) {
            Ok(d) => d,
            Err(e) if e.kind() == std::io::ErrorKind::PermissionDenied => {
                // Remember it so the warning isn't repeated every second
                warn!("Permission denied: {}", path.display());
                watched.lock().unwrap().insert(path);
                continue;
            }
            Err(_) => continue,
        };
        if !is_keyboard(&device) {
            watched.lock().unwrap().insert(path);
            continue;
        }
        let name = device.name().unwrap_or("?").to_string();
        info!("Monitoring keyboard device: {name} ({})", path.display());
        watched.lock().unwrap().insert(path.clone());
        started += 1;

        let tx = tx.clone();
        let watched = Arc::clone(watched);
        std::thread::spawn(move || {
            read_device(device, &path, hotkey, &tx);
            warn!("Keyboard device gone, no longer monitoring: {name} ({})", path.display());
            // Forget the path so a re-created device reusing it gets picked up
            watched.lock().unwrap().remove(&path);
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
