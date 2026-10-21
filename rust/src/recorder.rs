//! Microphone capture through arecord (ALSA), straight into memory.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::thread::JoinHandle;

use anyhow::{Context, Result};
use log::{debug, info};

use crate::engine::pcm16_to_f32;

pub struct Recording {
    child: Child,
    reader: JoinHandle<Vec<u8>>,
}

impl Recording {
    pub fn start(device: &str) -> Result<Recording> {
        let mut cmd = Command::new("arecord");
        if device != "default" {
            cmd.args(["-D", device]);
        }
        // 16-bit little-endian, 16 kHz mono: what Whisper expects
        cmd.args(["-q", "-f", "S16_LE", "-r", "16000", "-c", "1", "-t", "raw"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = cmd.spawn().context("starting arecord")?;
        let mut stdout = child.stdout.take().unwrap();
        let reader = std::thread::spawn(move || {
            let mut buf = Vec::new();
            let _ = stdout.read_to_end(&mut buf);
            buf
        });
        info!("Recording started (arecord pid={})", child.id());
        Ok(Recording { child, reader })
    }

    /// Stop arecord and return what it captured.
    pub fn stop(mut self) -> Vec<f32> {
        // SIGTERM lets arecord flush what it has
        unsafe { libc::kill(self.child.id() as i32, libc::SIGTERM) };
        let _ = self.child.wait();
        let bytes = self.reader.join().unwrap_or_default();
        debug!("Captured {} bytes of audio", bytes.len());
        pcm16_to_f32(&bytes)
    }
}
