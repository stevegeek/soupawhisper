//! Transcription with transcribe.cpp: Silero VAD drops the silence, then Whisper or
//! Granite Speech transcribes the rest.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use log::{debug, info, warn};
use transcribe_cpp::{
    Backend, Feature, Model, ModelOptions, RunExtension, RunOptions, Session, SessionOptions, Transcript,
    VadOptions, VadSession, VadSessionOptions, WhisperRunOptions,
};

use crate::config::{Choice, Config, Device, EngineKind, Language};

pub const SAMPLE_RATE: usize = 16_000;
const VAD_MODEL: &str = "ggml-silero-v5.1.2.bin";
const VAD_URL: &str = "https://huggingface.co/ggml-org/whisper-vad/resolve/main";
const MODEL_URL: &str = "https://huggingface.co/handy-computer";
/// Silence put back between speech segments so words don't run together.
const GAP_SECONDS: f32 = 0.2;
/// Granite has no long-form support: past ~30 s per call it starts repeating itself.
const GRANITE_CHUNK_SECONDS: usize = 30;
/// Whisper handles long audio in 30 s windows itself.
const WHISPER_CHUNK_SECONDS: usize = 300;

pub fn models_dir() -> PathBuf {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| crate::config::home().join(".local/share"));
    base.join("soupawhisper/models")
}

/// The GGUF slug for a configured model name: "large-v3-turbo" → "whisper-large-v3-turbo",
/// "ibm-granite/granite-speech-4.1-2b" → "granite-speech-4.1-2b".
pub fn model_slug(engine: EngineKind, name: &str) -> String {
    let name = name.rsplit('/').next().unwrap_or(name);
    match engine {
        EngineKind::Whisper if !name.starts_with("whisper-") => format!("whisper-{name}"),
        _ => name.to_string(),
    }
}

/// The model's GGUF file, downloaded from handy-computer on Hugging Face if missing.
pub fn model_path(cfg: &Config, engine: EngineKind) -> Result<PathBuf> {
    let name = match engine {
        EngineKind::Whisper => &cfg.model,
        EngineKind::Granite => &cfg.granite_model,
    };
    if name.ends_with(".gguf") || name.ends_with(".bin") {
        return Ok(PathBuf::from(name)); // an explicit file
    }
    let slug = model_slug(engine, name);
    let file = format!("{slug}-{}.gguf", cfg.quant);
    ensure_downloaded(&models_dir().join(&file), &format!("{MODEL_URL}/{slug}-gguf/resolve/main/{file}"))
}

fn vad_model_path() -> Result<PathBuf> {
    ensure_downloaded(&models_dir().join(VAD_MODEL), &format!("{VAD_URL}/{VAD_MODEL}"))
}

fn ensure_downloaded(path: &Path, url: &str) -> Result<PathBuf> {
    if path.exists() {
        return Ok(path.to_path_buf());
    }
    std::fs::create_dir_all(path.parent().unwrap())?;
    info!("Downloading {url}");
    let part = path.with_extension("part");
    let status = Command::new("curl")
        .args(["-fL", "--retry", "3", "-o"])
        .arg(&part)
        .arg(url)
        .status()
        .context("running curl")?;
    if !status.success() {
        bail!("download failed: {url}");
    }
    std::fs::rename(&part, path)?;
    Ok(path.to_path_buf())
}

pub struct Engine {
    pub choice: Choice,
    session: Session,
    vad: VadSession,
    /// Kept alive for the sessions.
    _models: (Model, Model),
    language: Language,
    /// Languages this model can transcribe (empty = unknown).
    languages: Vec<String>,
    keywords: Vec<String>,
    use_vocabulary: bool,
    use_initial_prompt: bool,
    vad_threshold: f32,
    vad_min_speech_ms: i32,
}

impl Engine {
    pub fn load(cfg: &Config, choice: Choice) -> Result<Engine> {
        let start = Instant::now();
        let path = model_path(cfg, choice.engine)?;
        info!("Loading {} from {}...", choice.label(), path.display());

        let mut options = ModelOptions::default();
        options.backend = match choice.device {
            Device::Cpu => Backend::Cpu,
            Device::Gpu if transcribe_cpp::backend_available(Backend::Cuda) => Backend::Cuda,
            Device::Gpu if transcribe_cpp::backend_available(Backend::Vulkan) => Backend::Vulkan,
            Device::Gpu => bail!("this build has no GPU backend"),
        };
        let model = Model::load_with(&path, &options).with_context(|| format!("loading {}", path.display()))?;
        let mut session_options = SessionOptions::default();
        session_options.n_threads = cfg.threads as i32;
        let session = model.session_with(&session_options)?;

        // VAD is tiny; keep it on the CPU so it never competes for VRAM
        let mut vad_options = ModelOptions::default();
        vad_options.backend = Backend::Cpu;
        let vad_model = Model::load_with(vad_model_path()?, &vad_options).context("loading VAD model")?;
        let mut vad_session_options = VadSessionOptions::default();
        vad_session_options.n_threads = cfg.threads as i32;
        let vad = vad_model.vad_session_with(&vad_session_options)?;

        let use_vocabulary = !cfg.keywords.is_empty() && model.supports(Feature::Vocabulary);
        let use_initial_prompt = !cfg.keywords.is_empty() && !use_vocabulary && model.supports(Feature::InitialPrompt);
        if !cfg.keywords.is_empty() && !use_vocabulary && !use_initial_prompt {
            warn!("{} can't take keywords; ignoring them", choice.label());
        } else if !cfg.keywords.is_empty() {
            let how = if use_vocabulary { "vocabulary" } else { "initial prompt" };
            debug!("{} keyword(s) passed as {how}", cfg.keywords.len());
        }
        let languages = model.capabilities().map(|c| c.languages).unwrap_or_default();
        let features = [
            (Feature::InitialPrompt, "initial-prompt"),
            (Feature::Pnc, "pnc"),
            (Feature::Vocabulary, "vocabulary"),
            (Feature::ContextPrompt, "context-prompt"),
            (Feature::Instruct, "instruct"),
            (Feature::TranscriptPrefix, "prefix"),
        ];
        let features: Vec<&str> = features.iter().filter(|(f, _)| model.supports(*f)).map(|(_, n)| *n).collect();
        debug!("{} supports: {} · languages: {}", model.arch(), features.join(", "), languages.join(","));

        info!(
            "Model loaded in {:.1}s ({} on {})",
            start.elapsed().as_secs_f32(),
            choice.label(),
            model.backend()
        );
        Ok(Engine {
            choice,
            session,
            vad,
            _models: (model, vad_model),
            language: cfg.language.clone(),
            languages,
            keywords: cfg.keywords.clone(),
            use_vocabulary,
            use_initial_prompt,
            vad_threshold: cfg.vad_threshold,
            vad_min_speech_ms: cfg.vad_min_speech_ms,
        })
    }

    /// Only the speech, split into chunks the model accepts, with short gaps between
    /// segments. Empty if there is none.
    fn speech_chunks(&mut self, audio: &[f32]) -> Result<Vec<Vec<f32>>> {
        let max_seconds = match self.choice.engine {
            EngineKind::Granite => GRANITE_CHUNK_SECONDS,
            EngineKind::Whisper => WHISPER_CHUNK_SECONDS,
        };
        let mut options = VadOptions::default();
        options.threshold = self.vad_threshold as f64;
        options.min_speech_ms = self.vad_min_speech_ms;
        // Split long stretches of speech too, a little under the limit to leave room for padding
        options.max_speech_ms = (max_seconds * 1000 - 1000) as i32;
        let segments = self.vad.run(audio, &options)?.segments;
        let pieces = segments.iter().map(|s| (s.start_sample.max(0) as usize, (s.end_sample.max(0) as usize).min(audio.len())));
        Ok(join_speech(audio, pieces, max_seconds * SAMPLE_RATE))
    }

    /// One model call. When the model repeats itself or runs out of room, keep what it
    /// produced rather than losing the whole chunk.
    fn run_once(&mut self, chunk: &[f32], options: &RunOptions) -> Result<Transcript> {
        match self.session.run(chunk, options) {
            Ok(transcript) => Ok(transcript),
            Err(e) => match e.partial() {
                Some(partial) if !partial.text.trim().is_empty() => {
                    warn!("Keeping a partial transcript: {e}");
                    Ok(partial.clone())
                }
                _ => Err(e.into()),
            },
        }
    }

    fn run_options(&self, language: Option<String>) -> RunOptions {
        let mut options = RunOptions::default();
        options.language = language;
        if self.use_vocabulary {
            options.vocabulary = self.keywords.clone();
        }
        if self.choice.engine == EngineKind::Whisper {
            // Whisper invents filler over silence and noise; don't let one window's
            // hallucination prime the next
            let mut whisper = WhisperRunOptions::default();
            whisper.condition_on_prev_tokens = Some(false);
            if self.use_initial_prompt {
                whisper.initial_prompt = Some(format!("{}.", self.keywords.join(", ")));
            }
            options.family = Some(RunExtension::Whisper(whisper));
        }
        options
    }

    /// The language to ask for up front (None = let the model detect it).
    fn language_hint(&self) -> Option<String> {
        match &self.language {
            Language::Forced(lang) if self.supports_language(lang) => Some(lang.clone()),
            _ => None,
        }
    }

    fn supports_language(&self, lang: &str) -> bool {
        self.languages.is_empty() || self.languages.iter().any(|l| l == lang)
    }

    fn run(&mut self, chunk: &[f32]) -> Result<String> {
        let hint = self.language_hint();
        let transcript = self.run_once(chunk, &self.run_options(hint.clone()))?;
        // Detected a language outside the allowed set: redo it in the first allowed one
        if let (None, Language::OneOf(allowed)) = (&hint, &self.language) {
            if let Some(detected) = &transcript.language {
                debug!("Detected language: {detected}");
                if !allowed.contains(detected) && self.supports_language(&allowed[0]) {
                    info!("Detected '{detected}' not in allowed {allowed:?}, using '{}'", allowed[0]);
                    let retry = self.run_options(Some(allowed[0].clone()));
                    return Ok(self.run_once(chunk, &retry)?.text.trim().to_string());
                }
            }
        }
        Ok(transcript.text.trim().to_string())
    }

    pub fn transcribe(&mut self, audio: &[f32]) -> Result<String> {
        let chunks = self.speech_chunks(audio)?;
        let speech: usize = chunks.iter().map(Vec::len).sum();
        debug!(
            "VAD kept {:.1}s of {:.1}s in {} chunk(s)",
            speech as f32 / SAMPLE_RATE as f32,
            audio.len() as f32 / SAMPLE_RATE as f32,
            chunks.len()
        );
        let mut texts = Vec::new();
        for chunk in chunks {
            let text = self.run(&chunk)?;
            if !text.is_empty() {
                texts.push(text);
            }
        }
        Ok(texts.join(" "))
    }
}

/// Concatenate speech `pieces` (sample ranges) with short gaps, starting a new chunk
/// before one would exceed `max_len` samples. Short results are padded to a second,
/// the shortest input every model family accepts.
pub fn join_speech(audio: &[f32], pieces: impl Iterator<Item = (usize, usize)>, max_len: usize) -> Vec<Vec<f32>> {
    let gap = vec![0.0; (GAP_SECONDS * SAMPLE_RATE as f32) as usize];
    let mut chunks: Vec<Vec<f32>> = Vec::new();
    let mut current: Vec<f32> = Vec::new();
    for (start, end) in pieces {
        if start >= end {
            continue;
        }
        let piece = &audio[start..end];
        if !current.is_empty() && current.len() + gap.len() + piece.len() > max_len {
            chunks.push(std::mem::take(&mut current));
        }
        if !current.is_empty() {
            current.extend_from_slice(&gap);
        }
        current.extend_from_slice(piece);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    for chunk in &mut chunks {
        if chunk.len() < SAMPLE_RATE {
            chunk.resize(SAMPLE_RATE, 0.0);
        }
    }
    chunks
}

/// Write 16 kHz mono 16-bit WAV, the format read_wav and --transcribe take.
pub fn write_wav(path: &Path, samples: &[f32]) -> Result<()> {
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + samples.len() * 2);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&[1, 0, 1, 0]); // PCM, mono
    wav.extend_from_slice(&(SAMPLE_RATE as u32).to_le_bytes());
    wav.extend_from_slice(&(SAMPLE_RATE as u32 * 2).to_le_bytes());
    wav.extend_from_slice(&[2, 0, 16, 0]);
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        wav.extend_from_slice(&((s.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, wav).with_context(|| format!("writing {}", path.display()))
}

/// 16-bit little-endian mono PCM → f32 samples in [-1, 1).
pub fn pcm16_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0).collect()
}

/// Read a 16 kHz mono 16-bit WAV (what arecord writes and the models want).
pub fn read_wav(path: &Path) -> Result<Vec<f32>> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        bail!("{} is not a WAV file", path.display());
    }
    let mut pos = 12;
    let mut format = None;
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let len = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = &data[pos + 8..(pos + 8 + len).min(data.len())];
        match id {
            b"fmt " if body.len() >= 16 => {
                let channels = u16::from_le_bytes([body[2], body[3]]);
                let rate = u32::from_le_bytes(body[4..8].try_into().unwrap());
                let bits = u16::from_le_bytes([body[14], body[15]]);
                format = Some((channels, rate, bits));
            }
            b"data" => {
                match format {
                    Some((1, 16_000, 16)) => {}
                    Some((c, r, b)) => bail!("need 16 kHz mono 16-bit WAV, got {r} Hz, {c} channel(s), {b}-bit"),
                    None => bail!("WAV has no fmt chunk before data"),
                }
                return Ok(pcm16_to_f32(body));
            }
            _ => {}
        }
        pos += 8 + len + (len & 1);
    }
    bail!("WAV has no data chunk")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugs_for_configured_names() {
        assert_eq!(model_slug(EngineKind::Whisper, "large-v3-turbo"), "whisper-large-v3-turbo");
        assert_eq!(model_slug(EngineKind::Whisper, "whisper-tiny.en"), "whisper-tiny.en");
        assert_eq!(model_slug(EngineKind::Granite, "ibm-granite/granite-speech-4.1-2b"), "granite-speech-4.1-2b");
    }

    #[test]
    fn pcm_conversion() {
        let bytes = [0x00, 0x00, 0xff, 0x7f, 0x00, 0x80, 0x00, 0x40];
        assert_eq!(pcm16_to_f32(&bytes), [0.0, 32767.0 / 32768.0, -1.0, 0.5]);
    }

    #[test]
    fn speech_is_joined_with_gaps_and_split_at_the_limit() {
        let audio: Vec<f32> = (0..100_000).map(|i| i as f32).collect();
        let gap = (GAP_SECONDS * SAMPLE_RATE as f32) as usize;

        let one = join_speech(&audio, [(10, 20_010), (30_000, 50_000)].into_iter(), usize::MAX);
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].len(), 20_000 + gap + 20_000);
        assert_eq!(one[0][0], 10.0);
        assert_eq!(one[0][20_000], 0.0, "gap between segments");
        assert_eq!(one[0][20_000 + gap], 30_000.0);

        let split = join_speech(&audio, [(0, 20_000), (30_000, 50_000)].into_iter(), 30_000);
        assert_eq!(split.len(), 2);
        assert_eq!(split[1][0], 30_000.0);

        assert!(join_speech(&audio, std::iter::empty(), 30_000).is_empty());
        let short = join_speech(&audio, [(0, 100)].into_iter(), 30_000);
        assert_eq!(short[0].len(), SAMPLE_RATE, "padded to a second");
    }

    #[test]
    fn written_wav_reads_back() {
        let path = std::env::temp_dir().join(format!("sw-write-{}.wav", std::process::id()));
        write_wav(&path, &[0.0, 0.5, -0.5]).unwrap();
        let back = read_wav(&path).unwrap();
        assert!(back.iter().zip([0.0, 0.5, -0.5]).all(|(a, b)| (a - b).abs() < 1e-3));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn wav_roundtrip() {
        let path = std::env::temp_dir().join(format!("sw-test-{}.wav", std::process::id()));
        let samples: [i16; 3] = [0, 16384, -16384];
        let mut wav = Vec::new();
        let data_len = (samples.len() * 2) as u32;
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36 + data_len).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&[1, 0, 1, 0]); // PCM, mono
        wav.extend_from_slice(&16000u32.to_le_bytes());
        wav.extend_from_slice(&32000u32.to_le_bytes());
        wav.extend_from_slice(&[2, 0, 16, 0]);
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&data_len.to_le_bytes());
        for s in samples {
            wav.extend_from_slice(&s.to_le_bytes());
        }
        std::fs::write(&path, wav).unwrap();
        assert_eq!(read_wav(&path).unwrap(), [0.0, 0.5, -0.5]);
        std::fs::remove_file(path).unwrap();
    }
}
