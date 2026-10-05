#!/usr/bin/env python3
"""
SoupaWhisper - Voice dictation tool using faster-whisper.
Hold the hotkey to record, release to transcribe and copy to clipboard.
Supports both X11 and Wayland.
"""

import argparse
import configparser
import logging
import subprocess
import tempfile
import threading
import time
import signal
import sys
import os
import queue
import select
from enum import Enum
from pathlib import Path

# System tray support using GTK AppIndicator (like Toshy)
HAS_TRAY = False
try:
    import gi
    gi.require_version('Gtk', '3.0')
    try:
        gi.require_version('AyatanaAppIndicator3', '0.1')
        from gi.repository import AyatanaAppIndicator3 as AppIndicator3
    except (ValueError, ImportError):
        gi.require_version('AppIndicator3', '0.1')
        from gi.repository import AppIndicator3
    from gi.repository import Gtk, GLib
    HAS_TRAY = True
except (ImportError, ValueError) as e:
    pass

# Configure logging for systemd journal (stdout)
# Use unbuffered output and include timestamp for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
# Force unbuffered stdout for systemd
sys.stdout.reconfigure(line_buffering=True)
log = logging.getLogger("soupawhisper")

# Preload CUDA libraries from pip packages before importing faster_whisper
def preload_cuda_libs():
    """Preload pip-installed CUDA libraries using ctypes."""
    import ctypes
    import glob
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib

        # Find and load cublas
        cublas_path = nvidia.cublas.lib.__path__[0]
        log.debug(f"Loading CUDA cublas from: {cublas_path}")
        for lib in glob.glob(f"{cublas_path}/libcublas.so*"):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                log.debug(f"Loaded: {lib}")
            except OSError as e:
                log.debug(f"Failed to load {lib}: {e}")

        # Find and load cudnn
        cudnn_path = nvidia.cudnn.lib.__path__[0]
        log.debug(f"Loading CUDA cudnn from: {cudnn_path}")
        for lib in glob.glob(f"{cudnn_path}/libcudnn*.so*"):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                log.debug(f"Loaded: {lib}")
            except OSError as e:
                log.debug(f"Failed to load {lib}: {e}")
    except ImportError:
        log.debug("CUDA libs not installed via pip, using system libs")

preload_cuda_libs()

import evdev
import numpy as np
from evdev import ecodes
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps

__version__ = "0.3.0"

# System tray icon names (using system theme icons)
TRAY_ICONS = {
    "ready": "audio-input-microphone",
    "recording": "media-record",
    "processing": "system-run",
    "loading": "content-loading-symbolic",
    "error": "dialog-error",
}

# Load configuration
CONFIG_PATH = Path.home() / ".config" / "soupawhisper" / "config.ini"

# Detect Wayland
IS_WAYLAND = os.environ.get("XDG_SESSION_TYPE") == "wayland" or "WAYLAND_DISPLAY" in os.environ

# Map key names to evdev keycodes
KEY_MAP = {
    "f1": ecodes.KEY_F1, "f2": ecodes.KEY_F2, "f3": ecodes.KEY_F3, "f4": ecodes.KEY_F4,
    "f5": ecodes.KEY_F5, "f6": ecodes.KEY_F6, "f7": ecodes.KEY_F7, "f8": ecodes.KEY_F8,
    "f9": ecodes.KEY_F9, "f10": ecodes.KEY_F10, "f11": ecodes.KEY_F11, "f12": ecodes.KEY_F12,
    "f13": ecodes.KEY_F13, "f14": ecodes.KEY_F14, "f15": ecodes.KEY_F15, "f16": ecodes.KEY_F16,
    "f17": ecodes.KEY_F17, "f18": ecodes.KEY_F18, "f19": ecodes.KEY_F19, "f20": ecodes.KEY_F20,
    "scroll_lock": ecodes.KEY_SCROLLLOCK, "pause": ecodes.KEY_PAUSE,
    "insert": ecodes.KEY_INSERT, "home": ecodes.KEY_HOME, "end": ecodes.KEY_END,
    "pageup": ecodes.KEY_PAGEUP, "pagedown": ecodes.KEY_PAGEDOWN,
}


def load_config():
    config = configparser.ConfigParser()

    # Defaults
    defaults = {
        "model": "base.en",
        "device": "cpu",
        "compute_type": "int8",
        "key": "f12",
        "auto_type": "true",
        "notifications": "true",
        "audio_device": "default",
        "paste_keys": "ctrl+v",
        "language": "auto",
        "long_press": 0.4,
        "record_mode": "auto",
        # faster-whisper's own VAD defaults
        "vad_threshold": 0.5,
        "vad_min_speech_ms": 0,
        "engine": "whisper",
        "granite_model": "ibm-granite/granite-speech-4.1-2b",
        "granite_keywords": "",
    }

    if CONFIG_PATH.exists():
        log.debug(f"Loading config from: {CONFIG_PATH}")
        config.read(CONFIG_PATH)
    else:
        log.debug(f"Config file not found, using defaults: {CONFIG_PATH}")

    whisper_device = config.get("whisper", "device", fallback=defaults["device"])
    # The engine picked from the tray menu is saved here and wins over config.ini
    state = configparser.ConfigParser()
    state.read(STATE_PATH)

    cfg = {
        "model": config.get("whisper", "model", fallback=defaults["model"]),
        "device": whisper_device,
        "compute_type": config.get("whisper", "compute_type", fallback=defaults["compute_type"]),
        "key": config.get("hotkey", "key", fallback=defaults["key"]),
        "auto_type": config.getboolean("behavior", "auto_type", fallback=True),
        "notifications": config.getboolean("behavior", "notifications", fallback=True),
        "audio_device": config.get("audio", "device", fallback=defaults["audio_device"]),
        "paste_keys": config.get("behavior", "paste_keys", fallback=defaults["paste_keys"]),
        "language": config.get("whisper", "language", fallback=defaults["language"]),
        "long_press": config.getfloat("behavior", "long_press", fallback=defaults["long_press"]),
        "record_mode": config.get("behavior", "record_mode", fallback=defaults["record_mode"]).strip().lower(),
        "vad_threshold": config.getfloat("vad", "threshold", fallback=defaults["vad_threshold"]),
        "vad_min_speech_ms": config.getint("vad", "min_speech_ms", fallback=defaults["vad_min_speech_ms"]),
        "engine": state.get("engine", "engine", fallback=config.get("engine", "engine", fallback=defaults["engine"])).strip().lower(),
        "engine_device": state.get("engine", "device", fallback=config.get("engine", "device", fallback=whisper_device)).strip().lower(),
        "granite_model": config.get("granite", "model", fallback=defaults["granite_model"]),
        "granite_keywords": [k.strip() for k in config.get("granite", "keywords", fallback=defaults["granite_keywords"]).split(",") if k.strip()],
    }
    if cfg["record_mode"] not in RECORD_MODES:
        log.warning(f"Unknown record_mode: {cfg['record_mode']}, expected one of {', '.join(RECORD_MODES)}; using auto")
        cfg["record_mode"] = "auto"
    if cfg["engine"] not in ENGINE_NAMES or cfg["engine_device"] not in DEVICE_NAMES:
        log.warning(f"Unknown engine/device: {cfg['engine']}/{cfg['engine_device']}; using whisper/cpu")
        cfg["engine"], cfg["engine_device"] = "whisper", "cpu"
    log.debug(f"Config loaded: model={cfg['model']}, device={cfg['device']}, compute_type={cfg['compute_type']}, language={cfg['language']}, key={cfg['key']}, auto_type={cfg['auto_type']}, notifications={cfg['notifications']}, audio_device={cfg['audio_device']}, paste_keys={cfg['paste_keys']}, record_mode={cfg['record_mode']}, long_press={cfg['long_press']}, vad_threshold={cfg['vad_threshold']}, vad_min_speech_ms={cfg['vad_min_speech_ms']}, engine={cfg['engine']}, engine_device={cfg['engine_device']}, granite_model={cfg['granite_model']}, granite_keywords={len(cfg['granite_keywords'])}")
    return cfg


def save_engine_choice(engine, device):
    """Remember the engine picked from the tray across restarts."""
    state = configparser.ConfigParser()
    state["engine"] = {"engine": engine, "device": device}
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        state.write(f)


# hold: record while held; toggle: tap to start, tap to stop;
# auto: a tap shorter than long_press toggles, a longer hold records while held
RECORD_MODES = ("auto", "hold", "toggle")
ENGINE_NAMES = {"whisper": "Whisper", "granite": "Granite"}
DEVICE_NAMES = {"cpu": "CPU", "cuda": "GPU"}
STATE_PATH = CONFIG_PATH.parent / "state.ini"

CONFIG = load_config()


def get_hotkey_code(key_name):
    """Map key name to evdev keycode."""
    key_name = key_name.lower()
    if key_name in KEY_MAP:
        log.debug(f"Mapped hotkey '{key_name}' to keycode {KEY_MAP[key_name]}")
        return KEY_MAP[key_name]
    else:
        log.warning(f"Unknown key: {key_name}, defaulting to F12")
        return ecodes.KEY_F12


HOTKEY_CODE = get_hotkey_code(CONFIG["key"])
HOTKEY_NAME = CONFIG["key"].upper()
MODEL_SIZE = CONFIG["model"]
DEVICE = CONFIG["device"]
COMPUTE_TYPE = CONFIG["compute_type"]
AUTO_TYPE = CONFIG["auto_type"]
NOTIFICATIONS = CONFIG["notifications"]
AUDIO_DEVICE = CONFIG["audio_device"]
LONG_PRESS = CONFIG["long_press"]
RECORD_MODE = CONFIG["record_mode"]
VAD_PARAMETERS = {
    "threshold": CONFIG["vad_threshold"],
    "min_speech_duration_ms": CONFIG["vad_min_speech_ms"],
}
GRANITE_MODEL = CONFIG["granite_model"]
GRANITE_KEYWORDS = CONFIG["granite_keywords"]


class State(Enum):
    IDLE = "idle"
    HOLDING = "holding"
    TOGGLED = "toggled"
    STOPPING = "stopping"

# Parse language config: "auto", "en", or "en,it,el" (comma-separated allowed languages)
def parse_language_config(lang_str):
    """Parse language config into (language, allowed_languages) tuple."""
    lang_str = lang_str.strip().lower()
    if lang_str == "auto":
        return None, None  # Full auto-detect
    parts = [p.strip() for p in lang_str.split(",")]
    if len(parts) == 1:
        return parts[0], None  # Single forced language
    return None, parts  # Auto-detect with allowed list

LANGUAGE, ALLOWED_LANGUAGES = parse_language_config(CONFIG["language"])
if LANGUAGE:
    log.debug(f"Language forced to: {LANGUAGE}")
elif ALLOWED_LANGUAGES:
    log.debug(f"Language auto-detect limited to: {ALLOWED_LANGUAGES}")
else:
    log.debug("Language: full auto-detect")

# Keycode map for ydotool paste shortcut
PASTE_KEYCODE_MAP = {
    "ctrl": 29, "control": 29,
    "alt": 56,
    "shift": 42,
    "super": 125, "meta": 125, "cmd": 125, "command": 125,
    "a": 30, "b": 31, "c": 46, "d": 32, "e": 18, "f": 33, "g": 34, "h": 35,
    "i": 23, "j": 36, "k": 37, "l": 38, "m": 50, "n": 49, "o": 24, "p": 25,
    "q": 16, "r": 19, "s": 31, "t": 20, "u": 22, "v": 47, "w": 17, "x": 45,
    "y": 21, "z": 44,
}

def parse_paste_keys(paste_keys_str):
    """Parse paste keys config (e.g. 'super+v') into ydotool args."""
    parts = [p.strip().lower() for p in paste_keys_str.split("+")]
    keycodes = []
    for part in parts:
        if part in PASTE_KEYCODE_MAP:
            keycodes.append(PASTE_KEYCODE_MAP[part])
        else:
            log.warning(f"Unknown key in paste_keys: {part}")
    if not keycodes:
        log.warning("No valid paste keys, defaulting to Ctrl+V")
        keycodes = [29, 47]  # Ctrl+V
    # Build ydotool sequence: press all, release in reverse
    args = []
    for kc in keycodes:
        args.append(f"{kc}:1")  # press
    for kc in reversed(keycodes):
        args.append(f"{kc}:0")  # release
    return args

PASTE_YDOTOOL_ARGS = parse_paste_keys(CONFIG["paste_keys"])
PASTE_TERMINAL_ARGS = parse_paste_keys("ctrl+shift+v")
log.debug(f"Paste keys '{CONFIG['paste_keys']}' -> ydotool args: {PASTE_YDOTOOL_ARGS}")

# Terminal app classes that use Ctrl+Shift+V for paste
TERMINAL_APPS = {
    "org.kde.konsole", "konsole",
    "alacritty", "org.alacritty.Alacritty",
    "kitty", "org.kde.yakuake",
    "gnome-terminal", "gnome-terminal-server",
    "xfce4-terminal", "terminator", "tilix",
    "foot", "wezterm",
}

def get_active_window_class():
    """Get the resource class of the active window via KWin D-Bus."""
    try:
        result = subprocess.run(
            ["gdbus", "call", "--session",
             "--dest", "org.kde.KWin",
             "--object-path", "/KWin",
             "--method", "org.kde.KWin.queryWindowInfo"],
            capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0:
            # Parse resourceClass from output
            import re
            match = re.search(r"'resourceClass': <'([^']*)'", result.stdout)
            if match:
                return match.group(1)
    except Exception as e:
        log.debug(f"Failed to get active window: {e}")
    return None

def get_paste_keys_for_window(window_class):
    """Get appropriate paste keys based on window class."""
    if window_class:
        log.debug(f"Checking window class: {window_class}")
        if window_class.lower() in {t.lower() for t in TERMINAL_APPS}:
            log.debug("Using terminal paste keys (Ctrl+Shift+V)")
            return PASTE_TERMINAL_ARGS
    log.debug(f"Using default paste keys (Ctrl+V)")
    return PASTE_YDOTOOL_ARGS


def find_keyboard_devices(paths=None):
    """Find keyboard input devices (among `paths`, or all input devices)."""
    log.debug("Scanning for keyboard input devices...")
    devices = []
    all_devices = evdev.list_devices() if paths is None else list(paths)
    log.debug(f"Found {len(all_devices)} input devices total")
    for path in all_devices:
        try:
            device = evdev.InputDevice(path)
            caps = device.capabilities()
            # Check if device has EV_KEY capability and has F-keys
            if ecodes.EV_KEY in caps:
                keys = caps[ecodes.EV_KEY]
                if ecodes.KEY_F1 in keys or ecodes.KEY_F10 in keys:
                    devices.append(device)
                    log.debug(f"Added keyboard device: {device.name} ({path})")
                else:
                    log.debug(f"Skipped (no F-keys): {device.name} ({path})")
            else:
                log.debug(f"Skipped (no EV_KEY): {path}")
        except PermissionError:
            log.warning(f"Permission denied: {path}")
        except OSError as e:
            log.debug(f"OSError for {path}: {e}")
    log.info(f"Found {len(devices)} keyboard device(s)")
    return devices


class WhisperEngine:
    """faster-whisper; runs well on CPU."""

    def __init__(self, device):
        # compute_type from config applies to the configured device; otherwise pick per device
        compute_type = COMPUTE_TYPE if device == DEVICE else ("float16" if device == "cuda" else "int8")
        log.info(f"Loading Whisper model ({MODEL_SIZE}) on {device} with {compute_type}...")
        self.model = WhisperModel(MODEL_SIZE, device=device, compute_type=compute_type)

    def transcribe(self, path):
        # Determine language for transcription
        use_language = LANGUAGE
        if ALLOWED_LANGUAGES and not LANGUAGE:
            # First pass: detect language
            _, detect_info = self.model.transcribe(path, beam_size=1, vad_filter=True)
            detected = detect_info.language
            log.debug(f"Detected language: {detected}")
            if detected not in ALLOWED_LANGUAGES:
                log.info(f"Detected '{detected}' not in allowed {ALLOWED_LANGUAGES}, using '{ALLOWED_LANGUAGES[0]}'")
                use_language = ALLOWED_LANGUAGES[0]
            else:
                use_language = detected

        segments, info = self.model.transcribe(
            path,
            beam_size=5,
            vad_filter=True,
            vad_parameters=VAD_PARAMETERS,
            language=use_language,
            # Whisper invents filler ("yeah", "bye bye bye") over silence and noise.
            # Don't let one window's hallucination prime the next into a loop, and
            # skip silent gaps once a hallucination is detected around them.
            condition_on_previous_text=False,
            word_timestamps=True,
            hallucination_silence_threshold=2,
        )
        return " ".join(segment.text.strip() for segment in segments)

    def close(self):
        del self.model


class GraniteEngine:
    """IBM Granite Speech (English-focused, keyword biasing); best on GPU."""

    SAMPLE_RATE = 16000
    MAX_CHUNK_S = 30  # the card documents no long-form support, so feed it VAD-split chunks

    def __init__(self, device):
        # Heavy imports only when Granite is actually used
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.torch = torch
        self.device = device
        # bf16 on CPU is ~2.5x faster than fp32 on AVX-512 BF16 CPUs (Zen 4/5)
        dtype = torch.bfloat16
        log.info(f"Loading Granite model ({GRANITE_MODEL}) on {device} with bfloat16...")
        self.processor = AutoProcessor.from_pretrained(GRANITE_MODEL)
        self.tokenizer = self.processor.tokenizer
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(GRANITE_MODEL, dtype=dtype).to(device).eval()

        instruction = "transcribe the speech with proper punctuation and capitalization."
        if GRANITE_KEYWORDS:
            instruction += " Keywords: " + ", ".join(GRANITE_KEYWORDS)
        self.prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": "<|audio|>" + instruction}],
            tokenize=False, add_generation_prompt=True,
        )
        # The first generate() is several times slower (kernel warm-up); pay it now
        self._generate(np.zeros(self.SAMPLE_RATE, dtype=np.float32))

    def _generate(self, audio):
        wav = self.torch.from_numpy(audio).unsqueeze(0)
        inputs = self.processor(self.prompt, wav, device=self.device, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=300, do_sample=False, num_beams=1)
        return self.tokenizer.batch_decode(
            out[:, inputs["input_ids"].shape[-1]:], skip_special_tokens=True
        )[0].strip()

    def _speech_chunks(self, audio):
        """Drop silence and group the remaining speech into chunks of at most MAX_CHUNK_S."""
        options = VadOptions(**VAD_PARAMETERS, max_speech_duration_s=self.MAX_CHUNK_S)
        max_len = self.MAX_CHUNK_S * self.SAMPLE_RATE
        gap = np.zeros(int(0.2 * self.SAMPLE_RATE), dtype=np.float32)  # keep words apart
        chunks, current = [], []
        for ts in get_speech_timestamps(audio, options):
            piece = audio[ts["start"]:ts["end"]]
            if current and sum(len(p) for p in current) + len(gap) + len(piece) > max_len:
                chunks.append(np.concatenate(current))
                current = []
            current += [gap, piece] if current else [piece]
        if current:
            chunks.append(np.concatenate(current))
        return chunks

    def transcribe(self, path):
        audio = decode_audio(path, sampling_rate=self.SAMPLE_RATE)
        chunks = self._speech_chunks(audio)
        log.debug(f"Granite: {len(chunks)} speech chunk(s), {sum(len(c) for c in chunks) / self.SAMPLE_RATE:.1f}s of speech")
        return " ".join(t for t in (self._generate(c) for c in chunks) if t)

    def close(self):
        del self.model
        import gc
        gc.collect()
        if self.device == "cuda":
            self.torch.cuda.empty_cache()  # give the VRAM back (e.g. to LM Studio)


ENGINES = {"whisper": WhisperEngine, "granite": GraniteEngine}


def engine_label(engine, device):
    return f"{ENGINE_NAMES[engine]} · {DEVICE_NAMES[device]}"


class Dictation:
    def __init__(self):
        log.debug("Initializing Dictation instance")
        self.recording = False
        self.record_process = None
        self.temp_file = None
        self.record_started_at = 0.0
        self.engine = None
        self.engine_name = CONFIG["engine"]
        self.engine_device = CONFIG["engine_device"]
        self.pending_engine = None  # (engine, device) to switch to once idle
        self.model_loaded = threading.Event()
        self.model_error = None
        self.running = True
        # Tray actions run on the GTK thread; hand them to the event loop thread
        self.requests = queue.SimpleQueue()
        # Toggle vs. hold-to-record state
        self.state = State.IDLE
        self.key_down = False
        self.key_device_path = None  # Device the hotkey is held on
        self.press_time = 0.0
        self.notification_id = 0  # For replacing notifications
        self.target_window_class = None  # Window to paste into

        # System tray icon
        self.indicator = None
        if HAS_TRAY:
            self._setup_tray()

        # Load model in background
        threading.Thread(target=self._load_engine, args=(self.engine_name, self.engine_device), daemon=True).start()

    def _setup_tray(self):
        """Set up the system tray icon with menu using GTK AppIndicator."""
        log.debug("Setting up system tray icon (AppIndicator)")

        # Create the indicator
        self.indicator = AppIndicator3.Indicator.new(
            "soupawhisper",
            TRAY_ICONS["loading"],
            AppIndicator3.IndicatorCategory.APPLICATION_STATUS
        )
        self.indicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)
        self.indicator.set_title("SoupaWhisper - Loading...")

        # Create menu
        menu = Gtk.Menu()

        # Start/stop button; also what a middle click on the tray icon does
        self.item_record = Gtk.MenuItem(label="● Start recording")
        self.item_record.connect("activate", lambda _: self.requests.put(("toggle_recording",)))
        menu.append(self.item_record)
        self.indicator.set_secondary_activate_target(self.item_record)

        self.item_status = Gtk.MenuItem(label="Loading...")
        self.item_status.set_sensitive(False)
        menu.append(self.item_status)

        menu.append(Gtk.SeparatorMenuItem())

        # Engine · device choices
        self.engine_items = {}
        group = None
        for engine in ENGINE_NAMES:
            for device in DEVICE_NAMES:
                label = engine_label(engine, device)
                if (engine, device) == ("granite", "cpu"):
                    label += " (slow)"
                item = Gtk.RadioMenuItem.new_with_label_from_widget(group, label)
                group = item
                item.set_active((engine, device) == (self.engine_name, self.engine_device))
                item.connect("toggled", self._on_engine_toggled, engine, device)
                menu.append(item)
                self.engine_items[(engine, device)] = item
        self._syncing_engine_items = False

        menu.append(Gtk.SeparatorMenuItem())

        # Header items (non-clickable info)
        item_version = Gtk.MenuItem(label=f"SoupaWhisper v{__version__} · Hotkey: {HOTKEY_NAME}")
        item_version.set_sensitive(False)
        menu.append(item_version)

        # Open Config
        item_config = Gtk.MenuItem(label="Open Config")
        item_config.connect("activate", self._on_open_config)
        menu.append(item_config)

        # Restart
        item_restart = Gtk.MenuItem(label="Restart")
        item_restart.connect("activate", self._on_restart)
        menu.append(item_restart)

        menu.append(Gtk.SeparatorMenuItem())

        # Quit
        item_quit = Gtk.MenuItem(label="Quit")
        item_quit.connect("activate", self._on_quit)
        menu.append(item_quit)

        menu.show_all()
        self.indicator.set_menu(menu)

        # Run GTK main loop in background thread
        self.gtk_thread = threading.Thread(target=self._run_gtk, daemon=True)
        self.gtk_thread.start()
        log.debug("System tray icon started (AppIndicator)")

    def _run_gtk(self):
        """Run GTK main loop in background."""
        Gtk.main()

    def _on_open_config(self, widget):
        log.debug("Opening config file")
        subprocess.Popen(["xdg-open", str(CONFIG_PATH)])

    def _on_restart(self, widget):
        log.info("Restart requested from tray")
        GLib.idle_add(Gtk.main_quit)
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def _on_quit(self, widget):
        log.info("Quit requested from tray")
        GLib.idle_add(Gtk.main_quit)
        self.stop()

    def _on_engine_toggled(self, item, engine, device):
        if item.get_active() and not self._syncing_engine_items:
            self.requests.put(("switch_engine", engine, device))

    def _update_tray(self, state):
        """Update tray icon and menu: 'loading', 'ready', 'recording', 'processing', 'error'."""
        if not getattr(self, "indicator", None):
            return
        GLib.idle_add(self._apply_tray_state, state)

    def _apply_tray_state(self, state):
        """Runs on the GTK thread."""
        engine = engine_label(self.engine_name, self.engine_device)
        titles = {
            "loading": f"SoupaWhisper - Loading {engine}...",
            "ready": f"SoupaWhisper - Ready ({HOTKEY_NAME}) · {engine}",
            "recording": "SoupaWhisper - Recording...",
            "processing": f"SoupaWhisper - Transcribing with {engine}...",
            "error": "SoupaWhisper - Model failed to load",
        }
        statuses = {
            "loading": f"Loading {engine}...",
            "ready": f"Ready · {engine}",
            "recording": f"Recording · {engine}",
            "processing": f"Transcribing with {engine}...",
            "error": "Model failed to load",
        }
        self.indicator.set_icon_full(TRAY_ICONS.get(state, TRAY_ICONS["loading"]), titles.get(state, "SoupaWhisper"))
        self.indicator.set_title(titles.get(state, "SoupaWhisper"))
        self.item_status.set_label(statuses.get(state, ""))

        if state == "recording":
            # NeedsAttention: KDE shows the icon even if tray icons are hidden in the overflow
            self.indicator.set_attention_icon_full(TRAY_ICONS["recording"], "Recording")
            self.indicator.set_status(AppIndicator3.IndicatorStatus.ATTENTION)
            self._tick_record_label()
            GLib.timeout_add_seconds(1, self._tick_record_label)
        else:
            self.indicator.set_status(AppIndicator3.IndicatorStatus.ACTIVE)
            self.item_record.set_label("● Start recording")
        self.item_record.set_sensitive(state in ("ready", "recording"))

        # Reflect the engine actually loaded (e.g. after falling back to Whisper · CPU)
        self._syncing_engine_items = True
        self.engine_items[(self.engine_name, self.engine_device)].set_active(True)
        self._syncing_engine_items = False
        return False  # one-shot idle callback

    def _tick_record_label(self):
        if not self.recording:
            return False  # stops the timer
        elapsed = int(time.time() - self.record_started_at)
        self.item_record.set_label(f"■ Stop recording ({elapsed // 60}:{elapsed % 60:02d})")
        return True

    def _load_engine(self, engine, device):
        log.debug(f"Model loading thread started ({engine_label(engine, device)})")
        self._update_tray("loading")
        try:
            start_time = time.time()
            self.engine = ENGINES[engine](device)
            self.engine_name, self.engine_device = engine, device
            self.model_error = None
            elapsed = time.time() - start_time
            self.model_loaded.set()
            log.info(f"Model loaded successfully in {elapsed:.1f}s - Ready for dictation! ({engine_label(engine, device)})")
            self._update_tray("ready")
        except Exception as e:
            log.error(f"Failed to load {engine_label(engine, device)}: {e}")
            if "cudnn" in str(e).lower() or "cuda" in str(e).lower():
                log.error("Hint: the GPU may be busy (e.g. an LLM is loaded); switch to a CPU engine from the tray.")
            if (engine, device) != ("whisper", "cpu"):
                # e.g. GPU out of memory: keep dictation working on Whisper · CPU
                self.notify(f"Couldn't load {engine_label(engine, device)}",
                            f"{str(e)[:80]}\nFalling back to Whisper · CPU", "dialog-warning", 6000)
                self._load_engine("whisper", "cpu")
                return
            self.model_error = str(e)
            self.model_loaded.set()
            self._update_tray("error")

    def switch_engine(self, engine, device):
        """Swap the transcription engine; call only from the event loop thread while idle."""
        if (engine, device) == (self.engine_name, self.engine_device) and self.engine:
            return
        log.info(f"Switching engine to {engine_label(engine, device)}")
        save_engine_choice(engine, device)
        self.model_loaded.clear()
        if self.engine:
            self.engine.close()  # free RAM/VRAM before loading the next one
            self.engine = None
        threading.Thread(target=self._load_engine, args=(engine, device), daemon=True).start()

    def toggle_recording_from_tray(self):
        """Start/stop button in the tray menu: behaves like the toggle-mode hotkey."""
        if self.recording:
            self.state = State.IDLE
            self.stop_recording()
        else:
            self.start_recording()
            if self.recording:
                self.state = State.TOGGLED  # the hotkey can stop it too

    def handle_requests(self):
        """Run tray actions queued from the GTK thread."""
        while True:
            try:
                action, *args = self.requests.get_nowait()
            except queue.Empty:
                break
            if action == "toggle_recording":
                self.toggle_recording_from_tray()
            elif action == "switch_engine":
                self.pending_engine = tuple(args)
        # Never swap the model under an active recording
        if self.pending_engine and not self.recording:
            engine, device = self.pending_engine
            self.pending_engine = None
            self.switch_engine(engine, device)

    def close_notification(self):
        """Close the most recent notification, if any."""
        if not NOTIFICATIONS or not self.notification_id:
            return
        subprocess.run(
            [
                "gdbus", "call", "--session",
                "--dest", "org.freedesktop.Notifications",
                "--object-path", "/org/freedesktop/Notifications",
                "--method", "org.freedesktop.Notifications.CloseNotification",
                str(self.notification_id),
            ],
            capture_output=True
        )

    def notify(self, title, message, icon="dialog-information", timeout=2000):
        """Send a desktop notification that replaces the previous one."""
        if not NOTIFICATIONS:
            log.debug(f"Notification suppressed (disabled): {title}")
            return
        log.debug(f"Sending notification: {title} - {message}")
        try:
            # Use gdbus to call notification daemon directly - supports replacement on KDE
            result = subprocess.run(
                [
                    "gdbus", "call", "--session",
                    "--dest", "org.freedesktop.Notifications",
                    "--object-path", "/org/freedesktop/Notifications",
                    "--method", "org.freedesktop.Notifications.Notify",
                    "SoupaWhisper",  # app_name
                    "0",  # replaces_id (always new - KDE doesn't show replaced notifications after dismiss)
                    icon,  # app_icon
                    title,  # summary
                    message,  # body
                    "[]",  # actions
                    "{}",  # hints
                    str(timeout),  # timeout
                ],
                capture_output=True,
                text=True
            )
            # Parse the returned notification ID for future replacement
            # Output format: (uint32 123,)
            if result.stdout:
                import re
                match = re.search(r'\(uint32 (\d+),\)', result.stdout)
                if match:
                    self.notification_id = int(match.group(1))
                    log.debug(f"Notification sent with id={self.notification_id}")
        except Exception as e:
            log.debug(f"gdbus notification failed: {e}, falling back to notify-send")
            # Fallback to notify-send
            subprocess.run(
                ["notify-send", "-a", "SoupaWhisper", "-i", icon, "-t", str(timeout), title, message],
                capture_output=True
            )

    def start_recording(self):
        if self.recording:
            log.debug("start_recording called but already recording, ignoring")
            return
        if self.model_error:
            log.warning("start_recording called but model failed to load")
            return

        self.recording = True
        self.record_started_at = time.time()
        self._update_tray("recording")
        self.temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        self.temp_file.close()
        log.debug(f"Created temp file: {self.temp_file.name}")

        # Record using arecord (ALSA)
        log.debug(f"Starting arecord subprocess (device={AUDIO_DEVICE})")
        arecord_cmd = ["arecord"]
        if AUDIO_DEVICE != "default":
            arecord_cmd.extend(["-D", AUDIO_DEVICE])
        arecord_cmd.extend([
            "-f", "S16_LE",  # Format: 16-bit little-endian
            "-r", "16000",   # Sample rate: 16kHz (what Whisper expects)
            "-c", "1",       # Mono
            "-t", "wav",
            self.temp_file.name
        ])
        self.record_process = subprocess.Popen(
            arecord_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        log.info(f"Recording started (pid={self.record_process.pid})")
        if RECORD_MODE == "toggle":
            hint = f"Tap {HOTKEY_NAME} again to stop"
        else:
            hint = f"Release {HOTKEY_NAME} when done"
        self.notify("Recording...", hint, "audio-input-microphone", 30000)

    def stop_recording(self):
        if not self.recording:
            log.debug("stop_recording called but not recording, ignoring")
            return

        self.recording = False
        log.debug("Stopping recording")

        if self.record_process:
            log.debug(f"Terminating arecord (pid={self.record_process.pid})")
            self.record_process.terminate()
            self.record_process.wait()
            self.record_process = None

        log.info("Recording stopped, transcribing...")
        self._update_tray("processing")

        # Wait for model if not loaded yet
        if not self.model_loaded.is_set():
            log.debug("Waiting for model to finish loading...")
        self.model_loaded.wait()

        if self.model_error:
            log.error(f"Cannot transcribe: model failed to load")
            self.notify("Error", "Model failed to load", "dialog-error", 3000)
            if self.temp_file and os.path.exists(self.temp_file.name):
                os.unlink(self.temp_file.name)
            self._update_tray("error")
            return

        # Transcribe
        try:
            start_time = time.time()
            log.debug(f"Starting transcription of {self.temp_file.name} with {engine_label(self.engine_name, self.engine_device)}")

            # Check file size
            file_size = os.path.getsize(self.temp_file.name)
            log.debug(f"Audio file size: {file_size} bytes")

            text = self.engine.transcribe(self.temp_file.name)
            elapsed = time.time() - start_time
            log.info(f"Transcription completed in {elapsed:.2f}s ({engine_label(self.engine_name, self.engine_device)})")

            if text:
                log.info(f"Transcribed ({len(text)} chars): {text[:80]}{'...' if len(text) > 80 else ''}")

                # Copy to clipboard
                log.debug(f"Copying to clipboard ({'Wayland' if IS_WAYLAND else 'X11'})")
                if IS_WAYLAND:
                    process = subprocess.Popen(["wl-copy"], stdin=subprocess.PIPE)
                else:
                    process = subprocess.Popen(
                        ["xclip", "-selection", "clipboard"],
                        stdin=subprocess.PIPE
                    )
                process.communicate(input=text.encode())

                # Type it into the active input field
                if AUTO_TYPE:
                    log.debug(f"Auto-typing with paste keys: {PASTE_YDOTOOL_ARGS}")
                    time.sleep(0.15)  # Small delay to ensure clipboard is ready
                    if IS_WAYLAND:
                        subprocess.run(["ydotool", "key"] + PASTE_YDOTOOL_ARGS)
                    else:
                        subprocess.run(["xdotool", "type", "--clearmodifiers", text])

                self.notify("Copied!", text[:100] + ("..." if len(text) > 100 else ""), "emblem-ok-symbolic", 3000)
            else:
                log.warning("No speech detected in recording")
                self.notify("No speech detected", "Try speaking louder", "dialog-warning", 2000)

        except Exception as e:
            log.error(f"Transcription error: {e}", exc_info=True)
            self.notify("Error", str(e)[:50], "dialog-error", 3000)
        finally:
            # Cleanup temp file
            if self.temp_file and os.path.exists(self.temp_file.name):
                log.debug(f"Cleaning up temp file: {self.temp_file.name}")
                os.unlink(self.temp_file.name)
            self._update_tray("ready")

    def on_hotkey_press(self, device_path):
        if self.key_down:  # same press seen on another device
            return
        self.key_down = True
        self.key_device_path = device_path
        self.press_time = time.monotonic()
        if self.state is State.IDLE:
            self.start_recording()  # always start recording
            if self.recording:
                self.state = State.HOLDING
        elif self.state in (State.TOGGLED, State.HOLDING):
            # HOLDING here means the release was lost with its device
            self.state = State.STOPPING  # will stop on release

    def on_hotkey_release(self):
        if not self.key_down:
            return
        self.key_down = False
        self.key_device_path = None
        duration = time.monotonic() - self.press_time
        if self.state is State.HOLDING:
            if RECORD_MODE == "toggle":
                self.state = State.TOGGLED  # first press → keep recording
            elif RECORD_MODE == "auto" and duration < LONG_PRESS:
                self.state = State.TOGGLED  # tap → stay recording
                log.info(f"Toggle mode: recording until {HOTKEY_NAME} is tapped again")
                # Swap the "Release ... when done" notification shown at start
                self.close_notification()
                self.notify("Recording...", f"Tap {HOTKEY_NAME} again to stop", "audio-input-microphone", 30000)
            else:
                self.state = State.IDLE
                self.stop_recording()  # hold → stop
        elif self.state is State.STOPPING:
            self.state = State.IDLE
            self.stop_recording()  # second press → stop

    def stop(self):
        log.info("Shutting down...")
        self.running = False
        if HAS_TRAY:
            try:
                GLib.idle_add(Gtk.main_quit)
            except:
                pass
        os._exit(0)

    def run(self):
        """Main event loop using evdev for keyboard input."""
        devices = find_keyboard_devices()
        if not devices:
            log.error("No keyboard devices found")
            log.error("Make sure you have permission to read /dev/input/event* devices")
            log.error("You may need to add your user to the 'input' group:")
            log.error("  sudo usermod -aG input $USER")
            log.error("Then log out and back in.")
            sys.exit(1)

        log.info(f"Monitoring {len(devices)} keyboard device(s):")
        for d in devices:
            log.info(f"  - {d.name}")

        # Create a dict mapping fd to device
        fd_to_device = {dev.fd: dev for dev in devices}
        # Virtual keyboards (RustDesk, Toshy/xwaykeyz, ydotoold) get destroyed and
        # re-created at runtime, so watch for new devices and drop vanished ones.
        known_paths = set(evdev.list_devices())
        last_scan = time.monotonic()
        log.debug(f"Event loop starting, waiting for {HOTKEY_NAME} key events...")

        while self.running:
            self.handle_requests()

            if time.monotonic() - last_scan >= 1.0:
                last_scan = time.monotonic()
                current_paths = set(evdev.list_devices())
                new_paths = current_paths - known_paths
                known_paths = current_paths
                if new_paths:
                    for dev in find_keyboard_devices(new_paths):
                        log.info(f"Now monitoring keyboard device: {dev.name} ({dev.path})")
                        fd_to_device[dev.fd] = dev

            # Use select to wait for events from any device
            r, _, _ = select.select(fd_to_device.keys(), [], [], 0.1)
            for fd in r:
                device = fd_to_device[fd]
                try:
                    for event in device.read():
                        if event.type == ecodes.EV_KEY and event.code == HOTKEY_CODE:
                            if event.value == 1:  # Key pressed
                                log.debug(f"Hotkey {HOTKEY_NAME} pressed")
                                self.on_hotkey_press(device.path)
                            elif event.value == 0:  # Key released
                                log.debug(f"Hotkey {HOTKEY_NAME} released")
                                self.on_hotkey_release()
                except BlockingIOError:
                    pass
                except OSError as e:
                    log.warning(f"Keyboard device gone, no longer monitoring: {device.name} ({device.path}): {e}")
                    del fd_to_device[fd]
                    # Forget the path so a re-created device reusing it gets picked up
                    known_paths.discard(device.path)
                    if self.key_down and self.key_device_path == device.path:
                        # Its release will never arrive; don't ignore the next press
                        self.key_down = False
                        self.key_device_path = None
                    try:
                        device.close()
                    except OSError:
                        pass


def check_dependencies():
    """Check that required system commands are available."""
    log.debug("Checking system dependencies...")
    missing = []

    # Audio recording
    if subprocess.run(["which", "arecord"], capture_output=True).returncode != 0:
        missing.append(("arecord", "alsa-utils"))
    else:
        log.debug("Found: arecord")

    # Clipboard
    if IS_WAYLAND:
        if subprocess.run(["which", "wl-copy"], capture_output=True).returncode != 0:
            missing.append(("wl-copy", "wl-clipboard"))
        else:
            log.debug("Found: wl-copy")
    else:
        if subprocess.run(["which", "xclip"], capture_output=True).returncode != 0:
            missing.append(("xclip", "xclip"))
        else:
            log.debug("Found: xclip")

    # Auto-typing
    if AUTO_TYPE:
        if IS_WAYLAND:
            if subprocess.run(["which", "ydotool"], capture_output=True).returncode != 0:
                missing.append(("ydotool", "ydotool"))
            else:
                log.debug("Found: ydotool")
        else:
            if subprocess.run(["which", "xdotool"], capture_output=True).returncode != 0:
                missing.append(("xdotool", "xdotool"))
            else:
                log.debug("Found: xdotool")

    if missing:
        log.error("Missing dependencies:")
        for cmd, pkg in missing:
            log.error(f"  {cmd} - install: sudo pacman -S {pkg}")
        sys.exit(1)
    log.debug("All dependencies satisfied")


def main():
    parser = argparse.ArgumentParser(
        description="SoupaWhisper - Push-to-talk voice dictation"
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"SoupaWhisper {__version__}"
    )
    parser.parse_args()

    log.info(f"SoupaWhisper v{__version__} starting")
    log.info(f"Session type: {'Wayland' if IS_WAYLAND else 'X11'}")
    log.info(f"Config file: {CONFIG_PATH}")
    log.info(f"Hotkey: {HOTKEY_NAME}")

    check_dependencies()

    dictation = Dictation()

    # Handle Ctrl+C gracefully
    def handle_sigint(sig, frame):
        log.debug("Received SIGINT")
        dictation.stop()

    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)
    log.debug("Signal handlers installed (SIGINT, SIGTERM)")

    dictation.run()


if __name__ == "__main__":
    main()
