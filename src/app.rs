//! The dictation loop: hotkey and tray events in, recordings out as text.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender};
use std::time::{Duration, Instant};

use log::{debug, error, info, warn};

use crate::config::{Choice, Config, Device, EngineKind, RecordMode};
use crate::engine::Engine;
use crate::output::{self, Notifier};
use crate::recorder::Recording;
use crate::tray::{self, Status, TrayHandle};
use crate::window::{self, ActiveWindow};

pub enum Event {
    KeyDown(PathBuf),
    KeyUp,
    DeviceGone(PathBuf),
    ToggleRecording,
    SetChoice(Choice),
    Loaded(Choice, Result<Engine, String>),
}

/// What the hotkey asks for; the caller does the recording.
#[derive(Debug, PartialEq, Eq)]
pub enum Action {
    None,
    Start,
    Stop,
    /// A short tap in auto mode: keep recording until the next tap.
    BecameToggle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyState {
    Idle,
    Holding,
    Toggled,
    Stopping,
}

/// Hold vs. toggle logic for the hotkey, kept free of side effects so it can be tested.
pub struct Keys {
    mode: RecordMode,
    long_press: Duration,
    state: KeyState,
    down_on: Option<PathBuf>,
    pressed_at: Instant,
}

impl Keys {
    pub fn new(mode: RecordMode, long_press_s: f64) -> Keys {
        Keys {
            mode,
            long_press: Duration::from_secs_f64(long_press_s.max(0.0)),
            state: KeyState::Idle,
            down_on: None,
            pressed_at: Instant::now(),
        }
    }

    pub fn press(&mut self, device: PathBuf, now: Instant) -> Action {
        if self.down_on.is_some() {
            return Action::None; // same press seen on another device
        }
        self.down_on = Some(device);
        self.pressed_at = now;
        match self.state {
            KeyState::Idle => {
                self.state = KeyState::Holding;
                Action::Start
            }
            // Holding here means the release was lost with its device
            KeyState::Toggled | KeyState::Holding => {
                self.state = KeyState::Stopping; // stop on release
                Action::None
            }
            KeyState::Stopping => Action::None,
        }
    }

    pub fn release(&mut self, now: Instant) -> Action {
        if self.down_on.take().is_none() {
            return Action::None;
        }
        let held = now.duration_since(self.pressed_at);
        match self.state {
            KeyState::Holding => match self.mode {
                RecordMode::Toggle => {
                    self.state = KeyState::Toggled;
                    Action::None
                }
                RecordMode::Auto if held < self.long_press => {
                    self.state = KeyState::Toggled;
                    Action::BecameToggle
                }
                _ => {
                    self.state = KeyState::Idle;
                    Action::Stop
                }
            },
            KeyState::Stopping => {
                self.state = KeyState::Idle;
                Action::Stop
            }
            _ => Action::None,
        }
    }

    /// The device the key is held on vanished: its release will never arrive.
    pub fn device_gone(&mut self, device: &PathBuf) {
        if self.down_on.as_ref() == Some(device) {
            self.down_on = None;
        }
    }

    /// Recording could not start.
    pub fn reset(&mut self) {
        self.state = KeyState::Idle;
    }

    /// The tray's start/stop button behaves like the toggle-mode hotkey.
    pub fn tray_toggle(&mut self, recording: bool) -> Action {
        if recording {
            self.state = KeyState::Idle;
            Action::Stop
        } else {
            self.state = KeyState::Toggled; // the hotkey can stop it too
            Action::Start
        }
    }
}

pub struct App {
    cfg: Config,
    tx: Sender<Event>,
    keys: Keys,
    notifier: Notifier,
    tray: Option<TrayHandle>,
    /// The focused app, when the desktop can tell us (KDE Plasma).
    active_window: Option<ActiveWindow>,
    /// The engine wanted; the loaded one may differ until a switch completes.
    choice: Choice,
    engine: Option<Engine>,
    loading: bool,
    failed: bool,
    recording: Option<(Recording, Instant)>,
    /// Audio waiting for the model to finish loading.
    pending: Option<Vec<f32>>,
    last_used: Instant,
}

impl App {
    pub fn new(cfg: Config, tx: Sender<Event>) -> App {
        let tray = tray::spawn(&cfg, tx.clone());
        let active_window = match &cfg.terminal_paste_keys {
            Some(_) if cfg.auto_type && output::is_wayland() => {
                let dir = crate::engine::models_dir().parent().unwrap().to_path_buf();
                window::watch(&dir)
                    .inspect_err(|e| info!("Focused window unknown ({e:#}); pasting with paste_keys everywhere"))
                    .ok()
            }
            _ => None,
        };
        App {
            active_window,
            keys: Keys::new(cfg.record_mode, cfg.long_press),
            notifier: Notifier::new(cfg.notifications),
            choice: cfg.choice,
            cfg,
            tx,
            tray,
            engine: None,
            loading: false,
            failed: false,
            recording: None,
            pending: None,
            last_used: Instant::now(),
        }
    }

    fn set_tray(&self, status: Status) {
        if let Some(tray) = &self.tray {
            let choice = self.engine.as_ref().map(|e| e.choice).unwrap_or(self.choice);
            let since = self.recording.as_ref().map(|(_, t)| *t);
            tray.update(move |t| {
                t.status = status;
                t.choice = choice;
                t.recording_since = since;
            });
        }
    }

    fn idle_status(&self) -> Status {
        if self.failed {
            Status::Error
        } else if self.engine.is_some() {
            Status::Ready
        } else if self.loading {
            Status::Loading
        } else {
            Status::Unloaded
        }
    }

    fn load(&mut self, choice: Choice) {
        if self.loading {
            return;
        }
        self.loading = true;
        self.failed = false;
        self.choice = choice;
        if self.recording.is_none() {
            self.set_tray(Status::Loading);
        }
        let cfg = self.cfg.clone();
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let result = Engine::load(&cfg, choice).map_err(|e| format!("{e:#}"));
            let _ = tx.send(Event::Loaded(choice, result));
        });
    }

    fn on_loaded(&mut self, choice: Choice, result: Result<Engine, String>) {
        self.loading = false;
        match result {
            Ok(engine) => {
                info!("Ready for dictation ({})", choice.label());
                self.engine = Some(engine);
                self.last_used = Instant::now();
            }
            Err(e) => {
                error!("Failed to load {}: {e}", choice.label());
                let fallback = Choice { engine: EngineKind::Whisper, device: Device::Cpu };
                if choice != fallback {
                    // e.g. the GPU is full: keep dictation working on Whisper · CPU
                    let short: String = e.chars().take(80).collect();
                    self.notifier.notify(
                        &format!("Couldn't load {}", choice.label()),
                        &format!("{short}\nFalling back to {}", fallback.label()),
                        "dialog-warning",
                        6000,
                    );
                    self.load(fallback);
                    return;
                }
                self.failed = true;
            }
        }
        if let Some(audio) = self.pending.take() {
            self.transcribe(audio);
        } else if self.recording.is_none() {
            self.set_tray(self.idle_status());
        }
    }

    fn start_recording(&mut self) -> bool {
        if self.recording.is_some() {
            return true;
        }
        if self.failed {
            warn!("Not recording: the model failed to load");
            self.notifier.notify("Error", "Model failed to load", "dialog-error", 3000);
            return false;
        }
        if self.pending.is_some() {
            debug!("Still waiting to transcribe the previous recording");
            return false;
        }
        let recording = match Recording::start(&self.cfg.audio_device) {
            Ok(r) => r,
            Err(e) => {
                error!("Could not start recording: {e:#}");
                self.notifier.notify("Error", "Could not start recording", "dialog-error", 3000);
                return false;
            }
        };
        self.recording = Some((recording, Instant::now()));
        // Load an unloaded model while the user speaks
        if self.engine.is_none() {
            self.load(self.choice);
        }
        self.set_tray(Status::Recording);
        let hint = match self.cfg.record_mode {
            RecordMode::Toggle => format!("Tap {} again to stop", self.cfg.hotkey_name),
            _ => format!("Release {} when done", self.cfg.hotkey_name),
        };
        self.notifier.notify("Recording...", &hint, "audio-input-microphone", 30000);
        true
    }

    fn stop_recording(&mut self) {
        let Some((recording, _)) = self.recording.take() else { return };
        let audio = recording.stop();
        info!("Recording stopped ({:.1}s), transcribing...", audio.len() as f32 / 16000.0);
        if self.engine.is_some() {
            self.transcribe(audio);
        } else if self.loading {
            debug!("Waiting for the model to finish loading...");
            self.pending = Some(audio);
            self.set_tray(Status::Processing);
        } else {
            self.notifier.notify("Error", "Model failed to load", "dialog-error", 3000);
            self.set_tray(self.idle_status());
        }
    }

    fn transcribe(&mut self, audio: Vec<f32>) {
        if self.engine.is_none() {
            return;
        }
        self.set_tray(Status::Processing);
        let engine = self.engine.as_mut().unwrap();
        let start = Instant::now();
        let result = engine.transcribe(&audio);
        info!("Transcription completed in {:.2}s ({})", start.elapsed().as_secs_f32(), engine.choice.label());
        self.last_used = Instant::now();
        match result {
            Ok(text) if !text.is_empty() => {
                let preview: String = text.chars().take(80).collect();
                info!("Transcribed ({} chars): {preview}{}", text.len(), if text.len() > 80 { "..." } else { "" });
                if let Err(e) = output::copy_to_clipboard(&text) {
                    error!("Clipboard failed: {e:#}");
                }
                if self.cfg.auto_type {
                    output::auto_type(&text, self.paste_keys());
                }
                let body: String = text.chars().take(100).collect();
                let more = if text.chars().count() > 100 { "..." } else { "" };
                self.notifier.notify("Copied!", &format!("{body}{more}"), "emblem-ok-symbolic", 3000);
            }
            Ok(_) => {
                warn!("No speech detected in recording");
                self.notifier.notify("No speech detected", "Try speaking louder", "dialog-warning", 2000);
            }
            Err(e) => {
                error!("Transcription error: {e:#}");
                let short: String = format!("{e}").chars().take(50).collect();
                self.notifier.notify("Error", &short, "dialog-error", 3000);
            }
        }
        self.set_tray(self.idle_status());
    }

    /// The paste shortcut for the focused app: terminals have their own.
    fn paste_keys(&self) -> &[String] {
        let class = self.active_window.as_ref().and_then(ActiveWindow::class);
        match (&self.cfg.terminal_paste_keys, class) {
            (Some(keys), Some(class)) if window::is_terminal(&class, &self.cfg.terminal_apps) => {
                debug!("Pasting into terminal {class}");
                keys
            }
            _ => &self.cfg.paste_keys,
        }
    }

    fn act(&mut self, action: Action) {
        match action {
            Action::Start => {
                if !self.start_recording() {
                    self.keys.reset();
                }
            }
            Action::Stop => self.stop_recording(),
            Action::BecameToggle => {
                info!("Toggle mode: recording until {} is tapped again", self.cfg.hotkey_name);
                // Swap the "Release ... when done" notification shown at start
                self.notifier.close_last();
                let hint = format!("Tap {} again to stop", self.cfg.hotkey_name);
                self.notifier.notify("Recording...", &hint, "audio-input-microphone", 30000);
            }
            Action::None => {}
        }
    }

    fn switch(&mut self, choice: Choice) {
        if self.engine.as_ref().is_some_and(|e| e.choice == choice) && !self.loading {
            return;
        }
        info!("Switching engine to {}", choice.label());
        if let Err(e) = self.cfg.save_choice(choice) {
            warn!("Could not save engine choice: {e}");
        }
        if self.loading {
            // Picked up once the current load finishes
            self.choice = choice;
            return;
        }
        self.engine = None; // free RAM/VRAM before loading the next one
        self.load(choice);
    }

    fn tick(&mut self) {
        if let Some((_, since)) = &self.recording {
            let since = *since;
            if let Some(tray) = &self.tray {
                tray.update(move |t| t.recording_since = Some(since)); // refresh the timer
            }
        }
        // A switch requested during a load
        if !self.loading && self.recording.is_none() && self.engine.as_ref().is_some_and(|e| e.choice != self.choice) {
            let choice = self.choice;
            self.engine = None;
            self.load(choice);
        }
        let idle_limit = Duration::from_secs(self.cfg.unload_after_min * 60);
        if !idle_limit.is_zero()
            && self.engine.is_some()
            && self.recording.is_none()
            && self.last_used.elapsed() >= idle_limit
        {
            info!("Unloading model after {} idle minute(s)", self.cfg.unload_after_min);
            self.engine = None;
            self.set_tray(Status::Unloaded);
        }
    }

    pub fn run(mut self, rx: Receiver<Event>) {
        self.load(self.choice);
        loop {
            let event = match rx.recv_timeout(Duration::from_secs(1)) {
                Ok(event) => event,
                Err(RecvTimeoutError::Timeout) => {
                    self.tick();
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => return,
            };
            match event {
                Event::KeyDown(device) => {
                    let action = self.keys.press(device, Instant::now());
                    self.act(action);
                }
                Event::KeyUp => {
                    let action = self.keys.release(Instant::now());
                    self.act(action);
                }
                Event::DeviceGone(device) => self.keys.device_gone(&device),
                Event::ToggleRecording => {
                    let action = self.keys.tray_toggle(self.recording.is_some());
                    self.act(action);
                }
                Event::SetChoice(choice) => {
                    if self.recording.is_some() || self.pending.is_some() {
                        // Never swap the model under a recording; tick() picks it up
                        if let Err(e) = self.cfg.save_choice(choice) {
                            warn!("Could not save engine choice: {e}");
                        }
                        self.choice = choice;
                    } else {
                        self.switch(choice);
                    }
                }
                Event::Loaded(choice, result) => self.on_loaded(choice, result),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(n: u8) -> PathBuf {
        PathBuf::from(format!("/dev/input/event{n}"))
    }

    fn at(base: Instant, ms: u64) -> Instant {
        base + Duration::from_millis(ms)
    }

    #[test]
    fn hold_mode_records_while_held() {
        let t = Instant::now();
        let mut k = Keys::new(RecordMode::Hold, 0.4);
        assert_eq!(k.press(dev(1), t), Action::Start);
        assert_eq!(k.release(at(t, 100)), Action::Stop);
    }

    #[test]
    fn toggle_mode_taps_start_and_stop() {
        let t = Instant::now();
        let mut k = Keys::new(RecordMode::Toggle, 0.4);
        assert_eq!(k.press(dev(1), t), Action::Start);
        assert_eq!(k.release(at(t, 2000)), Action::None);
        assert_eq!(k.press(dev(1), at(t, 5000)), Action::None);
        assert_eq!(k.release(at(t, 5100)), Action::Stop);
    }

    #[test]
    fn auto_mode_tap_toggles_and_long_press_holds() {
        let t = Instant::now();
        let mut k = Keys::new(RecordMode::Auto, 0.4);
        assert_eq!(k.press(dev(1), t), Action::Start);
        assert_eq!(k.release(at(t, 150)), Action::BecameToggle);
        assert_eq!(k.press(dev(1), at(t, 3000)), Action::None);
        assert_eq!(k.release(at(t, 3100)), Action::Stop);

        assert_eq!(k.press(dev(1), at(t, 9000)), Action::Start);
        assert_eq!(k.release(at(t, 9800)), Action::Stop);
    }

    #[test]
    fn same_press_on_a_second_device_is_ignored() {
        let t = Instant::now();
        let mut k = Keys::new(RecordMode::Hold, 0.4);
        assert_eq!(k.press(dev(1), t), Action::Start);
        assert_eq!(k.press(dev(2), t), Action::None);
        assert_eq!(k.release(at(t, 500)), Action::Stop);
        assert_eq!(k.release(at(t, 501)), Action::None);
    }

    #[test]
    fn vanished_device_does_not_swallow_the_next_press() {
        let t = Instant::now();
        let mut k = Keys::new(RecordMode::Hold, 0.4);
        assert_eq!(k.press(dev(1), t), Action::Start);
        k.device_gone(&dev(1));
        // The lost release left us holding; the next press stops on release
        assert_eq!(k.press(dev(2), at(t, 1000)), Action::None);
        assert_eq!(k.release(at(t, 1100)), Action::Stop);
    }

    #[test]
    fn failed_start_resets() {
        let t = Instant::now();
        let mut k = Keys::new(RecordMode::Toggle, 0.4);
        assert_eq!(k.press(dev(1), t), Action::Start);
        k.reset();
        assert_eq!(k.release(at(t, 100)), Action::None);
        assert_eq!(k.press(dev(1), at(t, 200)), Action::Start);
    }

    #[test]
    fn tray_button_toggles_and_hotkey_can_stop() {
        let t = Instant::now();
        let mut k = Keys::new(RecordMode::Hold, 0.4);
        assert_eq!(k.tray_toggle(false), Action::Start);
        assert_eq!(k.press(dev(1), t), Action::None);
        assert_eq!(k.release(at(t, 100)), Action::Stop);
        assert_eq!(k.tray_toggle(false), Action::Start);
        assert_eq!(k.tray_toggle(true), Action::Stop);
    }
}
