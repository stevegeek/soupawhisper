# SoupaWhisper

A simple push-to-talk voice dictation tool for Linux using faster-whisper. Hold a key to record, release to transcribe, and it automatically copies to clipboard and types into the active input.

Supports both **X11** and **Wayland** (including KDE Plasma).

**Features:**
- Push-to-talk recording with configurable hotkey (F1-F24)
- Auto-paste into active window after transcription
- System tray icon with status indicator (gray=loading, green=ready, red=recording)
- Configurable audio device, language detection, and paste keys
- Desktop notifications

## Requirements

- Python 3.10+
- Poetry
- Linux with X11 or Wayland (ALSA audio)

## Supported Distros

- Ubuntu / Pop!_OS / Debian (apt)
- Fedora (dnf)
- Arch Linux (pacman)
- openSUSE (zypper)

## Supported Desktop Environments

- GNOME (X11/Wayland)
- KDE Plasma (X11/Wayland)
- Sway, Hyprland, and other wlroots-based compositors
- Any X11-based desktop

## Installation

```bash
git clone https://github.com/ksred/soupawhisper.git
cd soupawhisper
chmod +x install.sh
./install.sh
```

The installer will:
1. Detect your package manager
2. Install system dependencies
3. Install Python dependencies via Poetry
4. Set up the config file
5. Optionally install as a systemd service

### Manual Installation

#### X11

```bash
# Ubuntu/Debian
sudo apt install alsa-utils xclip xdotool libnotify-bin

# Fedora
sudo dnf install alsa-utils xclip xdotool libnotify

# Arch
sudo pacman -S alsa-utils xclip xdotool libnotify

# Then install Python deps
poetry install
```

#### Wayland (KDE, GNOME, Sway, etc.)

```bash
# Arch
sudo pacman -S alsa-utils wl-clipboard ydotool libnotify

# Enable ydotool daemon (required for auto-typing)
systemctl --user enable --now ydotool

# Add yourself to input group (for keyboard monitoring)
sudo usermod -aG input $USER
# Log out and back in for group changes to take effect

# Then install Python deps
poetry install
```

### GPU Support (Optional)

For NVIDIA GPU acceleration with CUDA 12:

```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install libcudnn9-cuda-12
```

Then edit `~/.config/soupawhisper/config.ini`:
```ini
device = cuda
compute_type = float16 (or float32)
```

#### CUDA 13 Users

If you have CUDA 13 installed, the bundled cuBLAS/cuDNN won't work with faster-whisper (which expects CUDA 12). Install the CUDA 12 libraries via pip:

```bash
poetry run pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

The script will automatically detect and use these pip-installed libraries.

## Usage

```bash
poetry run python dictate.py
```

- Hold **F12** to record
- Release to transcribe → copies to clipboard and types into active input
- Or tap **F12** to start recording and tap again to transcribe (see `record_mode`)
- Press **Ctrl+C** to quit (when running manually)

### Tray menu

- **● Start / ■ Stop recording**: same as tapping the hotkey; shows elapsed time while recording.
  Middle-click the tray icon does the same. While recording the icon is shown as needing
  attention, so it stays visible even if tray icons are collapsed.
- **Engine · device**: switch between Whisper and Granite on CPU or GPU without restarting.
  The previous model is unloaded first (freeing GPU memory for other apps, e.g. a local LLM).
  If a GPU engine can't load (say the GPU is full), SoupaWhisper falls back to Whisper · CPU.
  The choice is remembered in `~/.config/soupawhisper/state.ini`.

### Granite engine (optional)

[IBM Granite Speech 4.1 2B](https://huggingface.co/ibm-granite/granite-speech-4.1-2b) is an
English-focused speech model with punctuation and **keyword biasing**: list the names, acronyms
and jargon you use under `[granite] keywords` and it will favour them. It needs ~5 GB of GPU
memory; on CPU it works but is several times slower than Whisper.

```bash
.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements-granite.txt
```

The model (~5 GB) downloads the first time Granite is selected.

## Run as a systemd Service

Use the service control script to manage SoupaWhisper as a background service:

```bash
./service.sh install   # Install and enable the service
./service.sh start     # Start the service
./service.sh stop      # Stop the service
./service.sh restart   # Restart the service
./service.sh status    # Check service status
./service.sh logs      # Follow service logs
./service.sh uninstall # Remove the service
```

The service will auto-start on login once installed.

Alternatively, the main installer can also set up the service:

```bash
./install.sh  # Select 'y' when prompted for systemd
```

## Configuration

Edit `~/.config/soupawhisper/config.ini`:

```ini
[whisper]
# Model size: tiny.en, base.en, small.en, medium.en, large-v3
model = base.en

# Device: cpu or cuda (cuda requires cuDNN)
device = cpu

# Compute type: int8 for CPU, float16 (or float32) for GPU
compute_type = int8

# Language detection:
#   auto       - Full auto-detect (may detect wrong language)
#   en         - Force English only
#   en,it,el   - Auto-detect but limit to these languages (falls back to first if other detected)
language = en

[hotkey]
# Key to hold for recording: f1-f20, scroll_lock, pause, insert, home, end, pageup, pagedown
# Apple keyboards with extended F-keys can use f13-f20
key = f12

[behavior]
# Type text into active input field
auto_type = true

# Show desktop notification
notifications = true

# Paste shortcut for auto-type (e.g. ctrl+v, super+v, ctrl+shift+v)
# Note: Terminals typically use ctrl+shift+v, other apps use ctrl+v
# The app auto-detects terminals (Konsole, Alacritty, etc.) and uses ctrl+shift+v for them
paste_keys = ctrl+v

# Recording mode:
#   hold   - hold the key to record, release to transcribe
#   toggle - tap the key to start recording, tap again to transcribe
#   auto   - a quick tap toggles, a longer hold records while held
record_mode = auto

# auto mode only: seconds before entering "hold-to-record" mode, otherwise "toggle" mode
long_press = 0.4

[audio]
# Audio device for recording
#   default      - Use system default (may not work with PipeWire)
#   plughw:0,0   - Use specific ALSA device directly (bypasses PipeWire)
# Run 'arecord -l' to list available devices
device = default
```

### Context-Aware Paste

The auto-paste feature automatically detects terminal applications (Konsole, Alacritty, Kitty, etc.) and uses `Ctrl+Shift+V` for them, while using the configured `paste_keys` for other applications.

### Direct Audio Device

If you're using PipeWire and your microphone isn't being detected as an input source, you can bypass PipeWire by specifying the ALSA device directly:

```bash
# List available capture devices
arecord -l

# Use device from card 0, device 0
device = plughw:0,0
```

Create the config directory and file if it doesn't exist:
```bash
mkdir -p ~/.config/soupawhisper
cp /path/to/soupawhisper/config.example.ini ~/.config/soupawhisper/config.ini
```

## Troubleshooting

**No audio recording:**
```bash
# Check your input device
arecord -l

# Test recording
arecord -d 3 test.wav && aplay test.wav
```

**PipeWire not detecting microphone:**

If your microphone shows up in `arecord -l` but not in `pactl list sources`, PipeWire isn't exposing it as an input. Configure the audio device directly in config:
```ini
[audio]
device = plughw:0,0
```

**Auto-paste not working:**

The paste feature uses `ydotool` which injects keyboard events at the evdev level. If you use a key remapper like Toshy/xwaykeyz:
- ydotool bypasses the remapper, so apps see raw keycodes
- Configure `paste_keys` to match what apps expect (usually `ctrl+v`)
- Terminals auto-detect and use `ctrl+shift+v`

**Wrong language detected:**

With multilingual models (large-v3), Whisper may detect the wrong language. Configure allowed languages:
```ini
[whisper]
language = en,it,el  # Only allow English, Italian, Greek
```

**Permission issues with keyboard:**
```bash
sudo usermod -aG input $USER
# Then log out and back in
```

**cuDNN errors with GPU:**
```
Unable to load any of {libcudnn_ops.so.9...}
```
Install cuDNN 9 (see GPU Support section above) or switch to CPU mode.

## Model Sizes

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny.en | ~75MB | Fastest | Basic |
| base.en | ~150MB | Fast | Good |
| small.en | ~500MB | Medium | Better |
| medium.en | ~1.5GB | Slower | Great |
| large-v3 | ~3GB | Slowest | Best |

For dictation, `base.en` or `small.en` is usually the sweet spot.

## Known Limitations

### Auto-paste only works reliably in terminals

The auto-paste feature uses `ydotool` to send keyboard shortcuts. Currently:

- **Terminals** (Konsole, Alacritty, etc.): Auto-paste works with `Ctrl+Shift+V`
- **Other apps** (LibreOffice, browsers, etc.): Auto-paste may not work; use manual `Cmd+V` / `Ctrl+V`

The issue is that different apps expect different paste shortcuts:
- Terminals: `Ctrl+Shift+V`
- Regular apps: `Ctrl+V`
- Apps with key remappers (Toshy): More complex

Attempts to auto-detect the active window via KWin D-Bus (`queryWindowInfo`) cause timeouts and UI freezes.

### Outstanding work

- [ ] Find a reliable, non-blocking way to detect the active window class on KDE Wayland
- [ ] Investigate alternative paste methods (Shift+Insert, menu navigation)
- [ ] Consider app-specific paste key configuration
