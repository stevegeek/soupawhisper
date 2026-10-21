# SoupaWhisper

Push-to-talk voice dictation for Linux. Press a key, speak, and the text is typed into whatever
window has focus.

SoupaWhisper runs speech recognition locally with **Whisper** or **IBM Granite Speech**, on the GPU
(CUDA or Vulkan) or the CPU, through [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp).
It is a single Rust binary: models load in about a second and nothing leaves your machine.

**Features**

- Push-to-talk on a configurable hotkey (F1-F24, Scroll Lock, Pause, ...): hold to record, tap to
  toggle, or `auto` (a tap toggles, a long press holds)
- Types the text into the focused window and copies it to the clipboard
- Two engines, switchable from the tray: Whisper (99 languages) and Granite Speech (English-focused,
  very accurate), each on GPU or CPU, falling back to Whisper · CPU if the GPU can't load
- Keyword biasing for names and jargon (`Sidekiq`, `PostgreSQL`, your product names...)
- Silero voice activity detection, so silence and noise don't turn into invented words
- Language limits: auto-detect only among the languages you speak
- Tray icon with status, a start/stop button with a timer, and the engine menu
- Wayland (KDE Plasma, GNOME, wlroots) and X11; keyboards that come and go (virtual keyboards,
  Bluetooth, RustDesk) are picked up automatically

## Requirements

- Linux with ALSA (`arecord`)
- Wayland: `wl-clipboard` and `ydotool` (with `ydotoold` running); X11: `xclip` and `xdotool`
- Read access to `/dev/input/event*`: add yourself to the `input` group
- To build: Rust (stable), CMake and a C++ compiler; for GPU support the CUDA toolkit or the
  Vulkan SDK

On Arch:

```bash
sudo pacman -S rust cmake alsa-utils wl-clipboard ydotool   # plus cuda, or vulkan-devel shaderc
systemctl --user enable --now ydotool
sudo usermod -aG input $USER    # then log out and back in
```

## Install

```bash
git clone https://github.com/stevegeek/soupawhisper-wayland.git soupawhisper
cd soupawhisper
./install.sh            # CUDA build; or --vulkan, or --cpu
systemctl --user enable --now soupawhisper
```

`install.sh` builds the release binary, installs it to `~/.local/bin/soupawhisper`, writes
`~/.config/systemd/user/soupawhisper.service`, and creates `~/.config/soupawhisper/config.ini` from
[`config.example.ini`](config.example.ini) if you don't have one. `./install.sh --uninstall` removes
the binary and the service and keeps your config and models.

Models are downloaded on first use to `~/.local/share/soupawhisper/models`: GGUF builds from
[handy-computer on Hugging Face](https://huggingface.co/handy-computer) (Whisper large-v3-turbo is
0.9 GB at Q8_0, Granite Speech 4.1 2B 2.6 GB) and the 1 MB Silero VAD model.

## Usage

- Press the hotkey (F12 by default) and speak; release it (or tap it again in toggle mode) to
  transcribe. The text is pasted into the focused window and left on the clipboard.
- **Tray menu:** start/stop recording (middle-click the icon does the same), the current status,
  the engine · device choice, Open Config, Restart and Quit. While recording, the icon asks for
  attention so it stays visible even with tray icons collapsed.
- **Logs:** `journalctl --user -u soupawhisper -f`

Transcribe a file without the daemon (16 kHz mono 16-bit WAV):

```bash
soupawhisper --transcribe clip.wav --engine granite --device gpu
```

## Configuration

`~/.config/soupawhisper/config.ini`; [`config.example.ini`](config.example.ini) documents every
option. The main ones:

| Section · key | Meaning |
|---|---|
| `[engine] engine`, `device` | `whisper` or `granite`, on `cpu` or `cuda`. The tray choice is saved to `state.ini` and wins over these. |
| `[engine] keywords` | Comma-separated names and jargon to favour: Granite's keyword list, Whisper's initial prompt. |
| `[engine] quant` | GGUF quantization to download: `Q8_0` (default), `Q5_K_M`, `F16`, `BF16`. |
| `[whisper] model` | `large-v3-turbo`, `base.en`, ... or a path to a `.gguf`. |
| `[whisper] language` | `auto`, a single code (`en`), or a list (`en,it,el`) to detect among. |
| `[whisper] unload_after` | Minutes of idleness before the model is freed; it reloads while you speak. `0` keeps it loaded. |
| `[granite] model` | `ibm-granite/granite-speech-4.1-2b` or a path to a `.gguf`. Granite covers en, fr, de, es, pt and ja. |
| `[hotkey] key` | `f1`-`f24`, `scroll_lock`, `pause`, `insert`, `home`, `end`, `pageup`, `pagedown`. |
| `[behavior] record_mode` | `hold`, `toggle` or `auto` (with `long_press` seconds). |
| `[behavior] paste_keys` | Paste shortcut sent by ydotool, e.g. `ctrl+v`, `ctrl+shift+v`, `super+v`. |
| `[vad] threshold`, `min_speech_ms` | Raise them if breaths or keyboard noise get transcribed. |

## Development

```bash
cargo test
cargo build --release                       # CUDA (the default feature)
cargo build --release --no-default-features # CPU only
```

`.cargo/config.toml` builds transcribe.cpp for this machine's CPU and GPU only and turns off its
optional system BLAS (Arch's netlib `libblas` lacks the cblas symbols). `build.rs` links the CUDA
runtime and cuBLAS, which transcribe-cpp-sys 0.4.0's link manifest leaves out.

| Module | Does |
|---|---|
| `src/app.rs` | The event loop and the hold/toggle hotkey state machine |
| `src/engine.rs` | Model download and loading, VAD, transcription, language handling |
| `src/config.rs` | `config.ini` / `state.ini` |
| `src/hotkey.rs` | evdev keyboard watching, including hot-plugged devices |
| `src/recorder.rs` | `arecord` capture into memory |
| `src/output.rs` | Clipboard, auto-paste, notifications |
| `src/tray.rs` | The StatusNotifierItem tray icon and menu |

## History

SoupaWhisper was created by [Kyle Redelinghuys](https://github.com/ksred) as a small Python tool
built on [faster-whisper](https://github.com/SYSTRAN/faster-whisper):
[ksred/soupawhisper](https://github.com/ksred/soupawhisper), with contributions from Bhadraksh
Bhargava and Zachariah Carmichael. This fork, by Stephen Ierodiaconou, added Wayland and KDE Plasma
support, toggle and auto record modes, hallucination fixes, a tray with engine switching, and IBM
Granite Speech running on PyTorch.

In October 2026 it was ported to Rust on transcribe.cpp, which runs both Whisper and Granite on
ggml, so neither engine needs Python or PyTorch any more. The last Python version is at commit
[`0cbb8b6`](https://github.com/stevegeek/soupawhisper-wayland/tree/0cbb8b6) if you need it.

Measured on one machine (Ryzen 9 9950X, RTX 5070 Ti) with the same 5-12 s clips and settings,
Python's faster-whisper with int8 and Granite in bf16 against the Rust build's Q8_0 GGUFs:

| | Python | Rust |
|---|---|---|
| Granite · GPU: model load | 18-21 s | ~1 s |
| Granite · GPU: per clip | 0.40-0.95 s | 0.25-0.35 s |
| Granite · GPU: VRAM | 4.9 GB | 3.5 GB |
| Whisper large-v3-turbo · CPU: per clip | 4.3-4.5 s | 2.8-3.0 s |
| Whisper large-v3-turbo · GPU: per clip | not measured | 0.13-0.21 s, 1.3 GB VRAM |
| Process memory (Granite · GPU) | 0.4 GB resident + 1.7 GB swapped, after a week | 0.8 GB resident, mostly the memory-mapped model |
| Installed size | 7.2 GB virtualenv | 51 MB binary (CUDA build) |

Transcripts matched between the two, except that Whisper now also gets the keyword list, so it
writes "Sidekiq" and "SparkLoop" where the Python version wrote "sidekick" and "Spark Loop".

## License

MIT, see [LICENSE](LICENSE). Copyright (c) 2024 Kyle.
