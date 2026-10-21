//! Status icon and menu (StatusNotifierItem, as KDE shows it).

use std::sync::mpsc::Sender;
use std::time::Instant;

use ksni::blocking::TrayMethods;
use ksni::menu::{RadioGroup, RadioItem, StandardItem};
use ksni::{MenuItem, Tray};
use log::warn;

use crate::app::Event;
use crate::config::{Choice, Config, Device, EngineKind};

const CHOICES: [Choice; 4] = [
    Choice { engine: EngineKind::Whisper, device: Device::Cpu },
    Choice { engine: EngineKind::Whisper, device: Device::Gpu },
    Choice { engine: EngineKind::Granite, device: Device::Cpu },
    Choice { engine: EngineKind::Granite, device: Device::Gpu },
];

pub type TrayHandle = ksni::blocking::Handle<DictationTray>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Loading,
    Ready,
    /// Idle with the model freed; it reloads on the next recording.
    Unloaded,
    Recording,
    Processing,
    Error,
}

pub struct DictationTray {
    pub status: Status,
    pub choice: Choice,
    pub recording_since: Option<Instant>,
    hotkey: String,
    config_path: String,
    tx: Sender<Event>,
}

impl DictationTray {
    fn engine(&self) -> String {
        self.choice.label()
    }

    fn send(&self, event: Event) {
        let _ = self.tx.send(event);
    }

    fn status_line(&self) -> String {
        let engine = self.engine();
        match self.status {
            Status::Loading => format!("Loading {engine}..."),
            Status::Ready => format!("Ready · {engine}"),
            Status::Unloaded => format!("Ready · {engine} (model unloaded)"),
            Status::Recording => format!("Recording · {engine}"),
            Status::Processing => format!("Transcribing with {engine}..."),
            Status::Error => "Model failed to load".into(),
        }
    }
}

impl Tray for DictationTray {
    fn id(&self) -> String {
        "soupawhisper".into()
    }

    fn category(&self) -> ksni::Category {
        ksni::Category::ApplicationStatus
    }

    fn icon_name(&self) -> String {
        match self.status {
            Status::Loading => "content-loading-symbolic",
            Status::Ready | Status::Unloaded => "audio-input-microphone",
            Status::Recording => "media-record",
            Status::Processing => "system-run",
            Status::Error => "dialog-error",
        }
        .into()
    }

    fn attention_icon_name(&self) -> String {
        "media-record".into()
    }

    fn status(&self) -> ksni::Status {
        // NeedsAttention: KDE shows the icon even if tray icons are collapsed
        if self.status == Status::Recording { ksni::Status::NeedsAttention } else { ksni::Status::Active }
    }

    fn title(&self) -> String {
        format!("SoupaWhisper - {}", self.status_line())
    }

    fn tool_tip(&self) -> ksni::ToolTip {
        ksni::ToolTip { title: self.title(), ..Default::default() }
    }

    /// Middle click: same as the start/stop item.
    fn secondary_activate(&mut self, _x: i32, _y: i32) {
        if matches!(self.status, Status::Ready | Status::Unloaded | Status::Recording) {
            self.send(Event::ToggleRecording);
        }
    }

    fn menu(&self) -> Vec<MenuItem<Self>> {
        let record_label = match self.recording_since {
            Some(since) => {
                let s = since.elapsed().as_secs();
                format!("■ Stop recording ({}:{:02})", s / 60, s % 60)
            }
            None => "● Start recording".into(),
        };
        vec![
            StandardItem {
                label: record_label,
                enabled: matches!(self.status, Status::Ready | Status::Unloaded | Status::Recording),
                activate: Box::new(|t: &mut Self| t.send(Event::ToggleRecording)),
                ..Default::default()
            }
            .into(),
            StandardItem { label: self.status_line(), enabled: false, ..Default::default() }.into(),
            MenuItem::Separator,
            RadioGroup {
                selected: CHOICES.iter().position(|c| *c == self.choice).unwrap_or(0),
                select: Box::new(|t: &mut Self, i| {
                    t.choice = CHOICES[i];
                    t.send(Event::SetChoice(CHOICES[i]));
                }),
                options: CHOICES
                    .iter()
                    .map(|c| {
                        let slow = *c == Choice { engine: EngineKind::Granite, device: Device::Cpu };
                        RadioItem { label: format!("{}{}", c.label(), if slow { " (slow)" } else { "" }), ..Default::default() }
                    })
                    .collect(),
            }
            .into(),
            MenuItem::Separator,
            StandardItem {
                label: format!("SoupaWhisper v{} · Hotkey: {}", env!("CARGO_PKG_VERSION"), self.hotkey),
                enabled: false,
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: "Open Config".into(),
                activate: Box::new(|t: &mut Self| {
                    let _ = std::process::Command::new("xdg-open").arg(&t.config_path).spawn();
                }),
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: "Restart".into(),
                activate: Box::new(|_| crate::restart()),
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            StandardItem {
                label: "Quit".into(),
                icon_name: "application-exit".into(),
                activate: Box::new(|_| crate::quit()),
                ..Default::default()
            }
            .into(),
        ]
    }
}

pub fn spawn(cfg: &Config, tx: Sender<Event>) -> Option<TrayHandle> {
    let tray = DictationTray {
        status: Status::Loading,
        choice: cfg.choice,
        recording_since: None,
        hotkey: cfg.hotkey_name.clone(),
        config_path: cfg.path.to_string_lossy().into_owned(),
        tx,
    };
    match tray.spawn() {
        Ok(handle) => Some(handle),
        Err(e) => {
            warn!("No system tray: {e}");
            None
        }
    }
}
