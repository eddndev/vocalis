use hound::{self, WavReader};
use anyhow::{Context, Result};
use log::{info, warn};

use crate::error::VocalisError;

/// Loads a WAV file and returns its raw audio samples as a vector of f32.
/// Ensures the audio is 16kHz and converts to mono if stereo.
///
/// `path`: The path to the WAV file.
///
/// Returns a `Vec<f32>` containing the audio samples.
pub fn load_wav_to_f32(path: &str, target_sample_rate: u32, expected_length: usize) -> Result<Vec<f32>, VocalisError> {
    info!("Loading WAV file: {}", path);
    let mut reader = WavReader::open(path).context(format!("Failed to open WAV file: {}", path))?;
    let spec = reader.spec();

    if spec.sample_rate != target_sample_rate {
        warn!(
            "WAV file sample rate mismatch: Expected {}Hz, got {}Hz. Resampling is not yet implemented.",
            target_sample_rate, spec.sample_rate
        );
        return Err(VocalisError::AudioError(format!(
            "Unsupported sample rate: Expected {}Hz, got {}Hz. Resampling not implemented.",
            target_sample_rate, spec.sample_rate
        )));
    }

    let samples: Vec<f32> = reader
        .samples::<i16>() // Assuming 16-bit audio from dataset preprocessing
        .filter_map(|s| s.ok())
        .map(|s| s as f32 / i16::MAX as f32) // Normalize i16 to f32 in range [-1.0, 1.0]
        .collect();

    // If stereo, convert to mono by averaging channels
    let mono_samples = if spec.channels > 1 {
        warn!("Stereo audio detected. Converting to mono by averaging channels.");
        samples
            .chunks_exact(spec.channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    } else {
        samples
    };

    // Pad or crop to expected_length
    let mut processed_samples = mono_samples;
    if processed_samples.len() < expected_length {
        let padding_needed = expected_length - processed_samples.len();
        processed_samples.extend(vec![0.0; padding_needed]);
        info!("Padded audio to {} samples.", expected_length);
    } else if processed_samples.len() > expected_length {
        // Crop from the center, similar to Python preprocessing
        let start = (processed_samples.len() - expected_length) / 2;
        processed_samples = processed_samples[start..start + expected_length].to_vec();
        info!("Cropped audio to {} samples.", expected_length);
    }

    if processed_samples.len() != expected_length {
        return Err(VocalisError::AudioError(format!(
            "Final audio length mismatch: Expected {} samples, got {}. This should not happen after padding/cropping.",
            expected_length, processed_samples.len()
        )));
    }

    info!("WAV file loaded, processed, and normalized.");
    Ok(processed_samples)
}
