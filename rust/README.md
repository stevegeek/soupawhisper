# SoupaWhisper (Rust)

A port of `dictate.py` to Rust on [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp),
which runs both **Whisper** and **IBM Granite Speech** on ggml, with CUDA or on the CPU. No Python,
PyTorch or CUDA wheels: one binary that loads its model in about a second.

It is a drop-in replacement: it reads the same `~/.config/soupawhisper/config.ini` and `state.ini`, and
behaves the same way (hotkey hold/toggle/auto modes, `arecord`, VAD, `wl-copy` + `ydotool` paste,
notifications, tray with engine · device switching and GPU → CPU fallback).

## Install

```bash
./install.sh            # CUDA build (needs the CUDA toolkit in /opt/cuda)
./install.sh --vulkan   # or Vulkan
./install.sh --cpu      # or CPU only

systemctl --user disable --now soupawhisper && systemctl --user enable --now soupawhisper-rs
```

`soupawhisper-rs.service` conflicts with `soupawhisper.service`, so only one listens to the hotkey.
Switch back with `systemctl --user disable --now soupawhisper-rs && systemctl --user enable --now soupawhisper`.

Models are GGUFs from [handy-computer on Hugging Face](https://huggingface.co/handy-computer), downloaded to
`~/.local/share/soupawhisper/models` on first use (plus the Silero VAD model).

## Config

Everything in the Python version's config is read the same way. Differences:

- `[whisper] model` is a Whisper size (`large-v3-turbo`, `base.en`, ...) or a path to a `.gguf`.
- `[granite] model` is the Hugging Face id (`ibm-granite/granite-speech-4.1-2b`) or a `.gguf` path.
- `[granite] keywords` bias **both** engines: Granite gets IBM's `Keywords:` list, Whisper an initial prompt.
- `compute_type` is replaced by `[engine] quant` (default `Q8_0`; `Q5_K_M` is smaller, `BF16`/`F16` larger).
- `[whisper] threads` sets CPU threads (default half the cores, at most 8).
- `[whisper] unload_after = <minutes>` frees the model when idle; it reloads while you speak the next time.
- `[whisper] language = en,it,el` re-runs in the first listed language when another one is detected.
  Granite transcribes en/fr/de/es/pt/ja only.

## Try a file

```bash
soupawhisper-rs --transcribe clip.wav --engine granite --device gpu   # 16 kHz mono 16-bit WAV
```

## Build notes

`.cargo/config.toml` builds transcribe.cpp for this machine's CPU and GPU only and turns off its
optional system BLAS (Arch's netlib `libblas` lacks the cblas symbols). `build.rs` links the CUDA
runtime and cuBLAS, which transcribe-cpp-sys 0.4.0's link manifest leaves out.
