//! SoupaWhisper: push-to-talk voice dictation on whisper.cpp.

mod app;
mod config;
mod engine;
mod hotkey;
mod output;
mod recorder;
mod tray;
mod window;

use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::{Command, ExitCode};
use std::time::Instant;

use anyhow::{Result, bail};
use log::{error, info};

use crate::config::{Config, Device, EngineKind};

const USAGE: &str = "\
Usage: soupawhisper [--config PATH]
       soupawhisper --transcribe FILE.wav [--engine whisper|granite] [--device cpu|gpu]
                    [--model NAME] [--config PATH]

Push-to-talk voice dictation. With --transcribe, print the text of a 16 kHz mono WAV and exit.";

pub fn restart() {
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("soupawhisper"));
    info!("Restart requested from tray");
    let err = Command::new(exe).args(std::env::args_os().skip(1)).exec();
    error!("Restart failed: {err}");
}

pub fn quit() {
    info!("Quit requested from tray");
    std::process::exit(0);
}

fn has(cmd: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|paths| std::env::split_paths(&paths).any(|dir| dir.join(cmd).is_file()))
}

fn check_dependencies(cfg: &Config) -> Result<()> {
    let wayland = output::is_wayland();
    let mut needed = vec![("arecord", "alsa-utils")];
    needed.push(if wayland { ("wl-copy", "wl-clipboard") } else { ("xclip", "xclip") });
    if cfg.auto_type {
        needed.push(if wayland { ("ydotool", "ydotool") } else { ("xdotool", "xdotool") });
    }
    let missing: Vec<_> = needed.into_iter().filter(|(cmd, _)| !has(cmd)).collect();
    if !missing.is_empty() {
        for (cmd, pkg) in &missing {
            error!("Missing: {cmd} - install: sudo pacman -S {pkg}");
        }
        bail!("missing dependencies");
    }
    Ok(())
}

fn transcribe_file(cfg: &Config, wav: &PathBuf) -> Result<()> {
    let audio = engine::read_wav(wav)?;
    let mut engine = engine::Engine::load(cfg, cfg.choice)?;
    let start = Instant::now();
    let text = engine.transcribe(&audio)?;
    info!(
        "Transcribed {:.1}s of audio in {:.2}s ({})",
        audio.len() as f32 / engine::SAMPLE_RATE as f32,
        start.elapsed().as_secs_f32(),
        cfg.choice.label()
    );
    println!("{text}");
    Ok(())
}

fn run() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut config_path = config::config_dir().join("config.ini");
    let mut wav = None;
    let mut device = None;
    let mut engine = None;
    let mut model = None;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-v" | "--version" => {
                println!("SoupaWhisper {} (transcribe.cpp {})", env!("CARGO_PKG_VERSION"), transcribe_cpp::version());
                return Ok(());
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            "--config" => config_path = args.next().map(PathBuf::from).unwrap_or(config_path),
            "--transcribe" => wav = args.next().map(PathBuf::from),
            "--device" => device = args.next(),
            "--engine" => engine = args.next(),
            "--model" => model = args.next(),
            other => bail!("unknown argument: {other}\n\n{USAGE}"),
        }
    }

    let mut cfg = Config::load(&config_path);
    match device.as_deref() {
        Some("gpu" | "cuda") => cfg.choice.device = Device::Gpu,
        Some("cpu") => cfg.choice.device = Device::Cpu,
        Some(other) => bail!("unknown device: {other}"),
        None => {}
    }
    match engine.as_deref() {
        Some("whisper") => cfg.choice.engine = EngineKind::Whisper,
        Some("granite") => cfg.choice.engine = EngineKind::Granite,
        Some(other) => bail!("unknown engine: {other}"),
        None => {}
    }
    if let Some(m) = model {
        match cfg.choice.engine {
            EngineKind::Whisper => cfg.model = m,
            EngineKind::Granite => cfg.granite_model = m,
        }
    }
    if let Some(wav) = wav {
        return transcribe_file(&cfg, &wav);
    }

    info!("SoupaWhisper v{} starting", env!("CARGO_PKG_VERSION"));
    info!("Session type: {}", if output::is_wayland() { "Wayland" } else { "X11" });
    info!("Config file: {}", cfg.path.display());
    info!("Hotkey: {} · {}", cfg.hotkey_name, cfg.choice.label());
    check_dependencies(&cfg)?;

    let (tx, rx) = std::sync::mpsc::channel();
    if hotkey::spawn(cfg.hotkey, tx.clone()) == 0 {
        error!("No keyboard devices found");
        error!("Make sure you have permission to read /dev/input/event* devices:");
        error!("  sudo usermod -aG input $USER   (then log out and back in)");
        bail!("no keyboards");
    }
    app::App::new(cfg, tx).run(rx);
    Ok(())
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("soupawhisper=debug,transcribe_cpp=warn"))
        .format_target(false)
        .init();
    transcribe_cpp::init_logging();
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            error!("{e:#}");
            ExitCode::FAILURE
        }
    }
}
