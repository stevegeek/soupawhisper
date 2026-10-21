//! `~/.config/soupawhisper/config.ini`, plus `state.ini` where the engine picked from the
//! tray is remembered.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use evdev::KeyCode;
use log::warn;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordMode {
    /// Record while the key is held.
    Hold,
    /// Tap to start, tap again to stop.
    Toggle,
    /// A tap shorter than `long_press` toggles, a longer hold records while held.
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineKind {
    Whisper,
    /// IBM Granite Speech: English-focused, with keyword biasing.
    Granite,
}

impl EngineKind {
    pub fn label(self) -> &'static str {
        match self {
            EngineKind::Whisper => "Whisper",
            EngineKind::Granite => "Granite",
        }
    }

    fn parse(s: &str) -> Option<EngineKind> {
        match s.trim().to_lowercase().as_str() {
            "whisper" => Some(EngineKind::Whisper),
            "granite" => Some(EngineKind::Granite),
            _ => None,
        }
    }

    fn as_ini(self) -> &'static str {
        match self {
            EngineKind::Whisper => "whisper",
            EngineKind::Granite => "granite",
        }
    }
}

/// An engine on a device, as picked from the tray.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Choice {
    pub engine: EngineKind,
    pub device: Device,
}

impl Choice {
    pub fn label(self) -> String {
        format!("{} · {}", self.engine.label(), self.device.label())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Gpu,
}

impl Device {
    pub fn label(self) -> &'static str {
        match self {
            Device::Cpu => "CPU",
            Device::Gpu => "GPU",
        }
    }

    /// The config and state spell the GPU as "cuda".
    fn parse(s: &str) -> Option<Device> {
        match s.trim().to_lowercase().as_str() {
            "cpu" => Some(Device::Cpu),
            "cuda" | "gpu" | "vulkan" => Some(Device::Gpu),
            _ => None,
        }
    }

    fn as_ini(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Gpu => "cuda",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Language {
    /// Full auto-detect.
    Auto,
    /// A single forced language.
    Forced(String),
    /// Auto-detect, but only among these.
    OneOf(Vec<String>),
}

impl Language {
    pub fn parse(s: &str) -> Language {
        let s = s.trim().to_lowercase();
        if s == "auto" || s.is_empty() {
            return Language::Auto;
        }
        let parts: Vec<String> = s.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect();
        match parts.len() {
            0 => Language::Auto,
            1 => Language::Forced(parts.into_iter().next().unwrap()),
            _ => Language::OneOf(parts),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub path: PathBuf,
    /// Whisper model: a name like "large-v3-turbo", or a path to a GGUF file.
    pub model: String,
    /// Granite model: a Hugging Face id like "ibm-granite/granite-speech-4.1-2b", or a GGUF path.
    pub granite_model: String,
    /// GGUF quantization downloaded for model names (Q8_0, Q5_K_M, F16, ...).
    pub quant: String,
    pub language: Language,
    /// Names and jargon to favour: Granite's keyword list, Whisper's initial prompt.
    pub keywords: Vec<String>,
    pub threads: usize,
    /// Free the model after this many idle minutes (0 = keep it loaded).
    pub unload_after_min: u64,
    pub hotkey: KeyCode,
    pub hotkey_name: String,
    pub auto_type: bool,
    pub notifications: bool,
    pub audio_device: String,
    pub paste_keys: Vec<String>,
    pub record_mode: RecordMode,
    pub long_press: f64,
    pub vad_threshold: f32,
    pub vad_min_speech_ms: i32,
    pub choice: Choice,
}

/// A minimal reader for configparser-style INI files.
pub fn parse_ini(text: &str) -> HashMap<(String, String), String> {
    let mut out = HashMap::new();
    let mut section = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with(';') {
            continue;
        }
        if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
            section = name.trim().to_string();
        } else if let Some((key, value)) = line.split_once(['=', ':']) {
            out.insert((section.clone(), key.trim().to_lowercase()), value.trim().to_string());
        }
    }
    out
}

struct Ini(HashMap<(String, String), String>);

impl Ini {
    fn load(path: &Path) -> Ini {
        Ini(fs::read_to_string(path).map(|t| parse_ini(&t)).unwrap_or_default())
    }

    fn get(&self, section: &str, key: &str) -> Option<&str> {
        self.0.get(&(section.to_string(), key.to_string())).map(String::as_str)
    }

    fn str(&self, section: &str, key: &str, default: &str) -> String {
        self.get(section, key).unwrap_or(default).to_string()
    }

    fn bool(&self, section: &str, key: &str, default: bool) -> bool {
        match self.get(section, key).map(|v| v.to_lowercase()) {
            Some(v) if ["1", "yes", "true", "on"].contains(&v.as_str()) => true,
            Some(v) if ["0", "no", "false", "off"].contains(&v.as_str()) => false,
            _ => default,
        }
    }

    fn num<T: std::str::FromStr>(&self, section: &str, key: &str, default: T) -> T {
        self.get(section, key).and_then(|v| v.parse().ok()).unwrap_or(default)
    }
}

pub fn config_dir() -> PathBuf {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".config"));
    base.join("soupawhisper")
}

pub fn home() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("/"))
}

fn state_path(config_path: &Path) -> PathBuf {
    config_path.with_file_name("state.ini")
}

impl Config {
    pub fn load(path: &Path) -> Config {
        let ini = Ini::load(path);
        let state = Ini::load(&state_path(path));

        let key_name = ini.str("hotkey", "key", "f12").to_lowercase();
        let hotkey = key_code(&key_name).unwrap_or_else(|| {
            warn!("Unknown key: {key_name}, defaulting to F12");
            KeyCode::KEY_F12
        });

        let record_mode = match ini.str("behavior", "record_mode", "auto").trim().to_lowercase().as_str() {
            "hold" => RecordMode::Hold,
            "toggle" => RecordMode::Toggle,
            "auto" => RecordMode::Auto,
            other => {
                warn!("Unknown record_mode: {other}, expected one of auto, hold, toggle; using auto");
                RecordMode::Auto
            }
        };

        // The tray choice in state.ini wins over config.ini
        let configured = ini.str("whisper", "device", "cpu");
        let engine_str = state
            .get("engine", "engine")
            .map(str::to_string)
            .unwrap_or_else(|| ini.str("engine", "engine", "whisper"));
        let device_str = state
            .get("engine", "device")
            .map(str::to_string)
            .unwrap_or_else(|| ini.str("engine", "device", &configured));
        let choice = match (EngineKind::parse(&engine_str), Device::parse(&device_str)) {
            (Some(engine), Some(device)) => Choice { engine, device },
            _ => {
                warn!("Unknown engine/device: {engine_str}/{device_str}; using whisper/cpu");
                Choice { engine: EngineKind::Whisper, device: Device::Cpu }
            }
        };

        // [granite] keywords is where they lived when only Granite used them
        let keywords = ini.get("engine", "keywords").or(ini.get("granite", "keywords")).unwrap_or("").to_string();
        let keywords = keywords.split(',').map(str::trim).filter(|k| !k.is_empty()).map(str::to_string).collect();

        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);

        Config {
            path: path.to_path_buf(),
            model: ini.str("whisper", "model", "base.en"),
            granite_model: ini.str("granite", "model", "ibm-granite/granite-speech-4.1-2b"),
            quant: ini.str("engine", "quant", "Q8_0"),
            language: Language::parse(&ini.str("whisper", "language", "auto")),
            keywords,
            threads: ini.num("whisper", "threads", (cores / 2).clamp(1, 8)),
            unload_after_min: ini.num("whisper", "unload_after", 0),
            hotkey,
            hotkey_name: key_name.to_uppercase(),
            auto_type: ini.bool("behavior", "auto_type", true),
            notifications: ini.bool("behavior", "notifications", true),
            audio_device: ini.str("audio", "device", "default"),
            paste_keys: parse_paste_keys(&ini.str("behavior", "paste_keys", "ctrl+v")),
            record_mode,
            long_press: ini.num("behavior", "long_press", 0.4),
            vad_threshold: ini.num("vad", "threshold", 0.5),
            vad_min_speech_ms: ini.num("vad", "min_speech_ms", 0),
            choice,
        }
    }

    /// Remember the engine picked from the tray across restarts.
    pub fn save_choice(&self, choice: Choice) -> std::io::Result<()> {
        let path = state_path(&self.path);
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let (engine, device) = (choice.engine.as_ini(), choice.device.as_ini());
        fs::write(path, format!("[engine]\nengine = {engine}\ndevice = {device}\n\n"))
    }
}

pub fn key_code(name: &str) -> Option<KeyCode> {
    let name = name.to_lowercase();
    if let Some(n) = name.strip_prefix('f').and_then(|n| n.parse::<u16>().ok()) {
        // KEY_F1..F10 = 59..68, F11/F12 = 87/88, F13..F24 = 183..194
        return match n {
            1..=10 => Some(KeyCode::new(58 + n)),
            11 | 12 => Some(KeyCode::new(76 + n)),
            13..=24 => Some(KeyCode::new(170 + n)),
            _ => None,
        };
    }
    Some(match name.as_str() {
        "scroll_lock" => KeyCode::KEY_SCROLLLOCK,
        "pause" => KeyCode::KEY_PAUSE,
        "insert" => KeyCode::KEY_INSERT,
        "home" => KeyCode::KEY_HOME,
        "end" => KeyCode::KEY_END,
        "pageup" => KeyCode::KEY_PAGEUP,
        "pagedown" => KeyCode::KEY_PAGEDOWN,
        _ => return None,
    })
}

fn paste_keycode(name: &str) -> Option<u16> {
    Some(match name {
        "ctrl" | "control" => 29,
        "alt" => 56,
        "shift" => 42,
        "super" | "meta" | "cmd" | "command" => 125,
        _ => {
            let c = name.chars().next().filter(|c| name.len() == 1 && c.is_ascii_lowercase())?;
            // Linux keycodes for a..z
            const LETTERS: [u16; 26] = [
                30, 48, 46, 32, 18, 33, 34, 35, 23, 36, 37, 38, 50, 49, 24, 25, 16, 19, 31, 20, 22, 47, 17, 45, 21, 44,
            ];
            LETTERS[(c as u8 - b'a') as usize]
        }
    })
}

/// "super+v" → ydotool args: press all keys, then release them in reverse.
pub fn parse_paste_keys(spec: &str) -> Vec<String> {
    let mut codes: Vec<u16> = Vec::new();
    for part in spec.split('+').map(|p| p.trim().to_lowercase()) {
        match paste_keycode(&part) {
            Some(code) => codes.push(code),
            None => warn!("Unknown key in paste_keys: {part}"),
        }
    }
    if codes.is_empty() {
        warn!("No valid paste keys, defaulting to Ctrl+V");
        codes = vec![29, 47];
    }
    let presses = codes.iter().map(|c| format!("{c}:1"));
    let releases = codes.iter().rev().map(|c| format!("{c}:0"));
    presses.chain(releases).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_forms() {
        assert_eq!(Language::parse("auto"), Language::Auto);
        assert_eq!(Language::parse(" EN "), Language::Forced("en".into()));
        assert_eq!(Language::parse("en, it,el"), Language::OneOf(vec!["en".into(), "it".into(), "el".into()]));
    }

    #[test]
    fn paste_keys_press_then_release_in_reverse() {
        assert_eq!(parse_paste_keys("cmd+v"), ["125:1", "47:1", "47:0", "125:0"]);
        assert_eq!(parse_paste_keys("ctrl+shift+v"), ["29:1", "42:1", "47:1", "47:0", "42:0", "29:0"]);
        assert_eq!(parse_paste_keys("ctrl+b"), ["29:1", "48:1", "48:0", "29:0"]);
        assert_eq!(parse_paste_keys("bogus"), ["29:1", "47:1", "47:0", "29:0"]);
    }

    #[test]
    fn key_names_map_to_evdev_codes() {
        assert_eq!(key_code("f1"), Some(KeyCode::KEY_F1));
        assert_eq!(key_code("F10"), Some(KeyCode::KEY_F10));
        assert_eq!(key_code("f12"), Some(KeyCode::KEY_F12));
        assert_eq!(key_code("f13"), Some(KeyCode::KEY_F13));
        assert_eq!(key_code("f19"), Some(KeyCode::KEY_F19));
        assert_eq!(key_code("f24"), Some(KeyCode::KEY_F24));
        assert_eq!(key_code("scroll_lock"), Some(KeyCode::KEY_SCROLLLOCK));
        assert_eq!(key_code("f25"), None);
        assert_eq!(key_code("banana"), None);
    }

    #[test]
    fn engine_keywords_win_over_granite_keywords() {
        let dir = std::env::temp_dir().join(format!("soupawhisper-kw-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.ini");
        fs::write(&path, "[engine]\nkeywords = Ruby, Hotwire\n[granite]\nkeywords = Rails\n").unwrap();
        assert_eq!(Config::load(&path).keywords, ["Ruby", "Hotwire"]);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn reads_config_and_state() {
        let dir = std::env::temp_dir().join(format!("soupawhisper-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("config.ini");
        fs::write(
            &path,
            "[whisper]\n# comment\nmodel = large-v3-turbo\ndevice = cpu\nlanguage = en,it,el\n\n\
             [hotkey]\nkey = f19\n[behavior]\nauto_type = false\npaste_keys = cmd+v\nrecord_mode = toggle\n\
             [granite]\nkeywords = Rails, Sidekiq ,\n",
        )
        .unwrap();
        fs::write(dir.join("state.ini"), "[engine]\nengine = granite\ndevice = cuda\n").unwrap();

        let cfg = Config::load(&path);
        assert_eq!(cfg.model, "large-v3-turbo");
        assert_eq!(cfg.hotkey, KeyCode::KEY_F19);
        assert_eq!(cfg.hotkey_name, "F19");
        assert!(!cfg.auto_type);
        assert!(cfg.notifications);
        assert_eq!(cfg.record_mode, RecordMode::Toggle);
        assert_eq!(
            cfg.choice,
            Choice { engine: EngineKind::Granite, device: Device::Gpu },
            "state.ini wins over config.ini"
        );
        assert_eq!(cfg.keywords, ["Rails", "Sidekiq"]);
        assert_eq!(cfg.granite_model, "ibm-granite/granite-speech-4.1-2b");
        assert_eq!(cfg.quant, "Q8_0");
        assert_eq!(cfg.language, Language::OneOf(vec!["en".into(), "it".into(), "el".into()]));

        let whisper_cpu = Choice { engine: EngineKind::Whisper, device: Device::Cpu };
        cfg.save_choice(whisper_cpu).unwrap();
        assert_eq!(Config::load(&path).choice, whisper_cpu);
        fs::remove_dir_all(dir).unwrap();
    }
}
