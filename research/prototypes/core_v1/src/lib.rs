//! `vocalis_core` is the core library for the Vocalis project,
//! providing functionality for loading audio, performing inference with
//! the trained ONNX model, and returning vocal and gender predictions.

pub mod error;
pub mod model;
pub mod audio;

/// Initializes the logging system.
pub fn init_logger() {
    // Only initialize logger once
    let _ = env_logger::builder().is_test(true).try_init();
    log::info!("Logger initialized.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::VocalisModel;
    use crate::audio;

    #[test]
    fn test_model_inference_with_sample_audio() -> anyhow::Result<()> {
        // Initialize logger for test, if not already initialized
        let _ = env_logger::builder().is_test(true).try_init();

        // 1. Initialize the VocalisModel
        // The model will be loaded once when `init()` is called for the first time
        let vocalis_model = VocalisModel::init()?;
        
        // 2. Load a sample audio file
        let relative_path = "train_lab/dataset/audio/s001_F_a_0000.wav";
        let possible_paths = [relative_path, "../train_lab/dataset/audio/s001_F_a_0000.wav"];
        let sample_audio_path = possible_paths.iter()
            .find(|p| std::path::Path::new(p).exists())
            .ok_or_else(|| anyhow::anyhow!("Could not find sample audio. Searched: {:?}", possible_paths))?;

        let target_sample_rate = 16000;
        let expected_audio_length = 8000; // 0.5 seconds at 16kHz

        let audio_samples = audio::load_wav_to_f32(
            sample_audio_path,
            target_sample_rate,
            expected_audio_length,
        )?;

        // 3. Perform prediction
        let (vowel, gender) = vocalis_model.predict(&audio_samples)?;

        log::info!("Test Prediction: Vowel = {}, Gender = {}", vowel, gender);

        // Basic assertions (you might want more specific checks)
        assert!(!vowel.is_empty());
        assert!(!gender.is_empty());
        
        // We expect 'a' and 'F' based on the filename "s001_F_a_0000.wav"
        // This assumes the model is good enough to predict correctly on this specific sample.
        assert_eq!(vowel, "a");
        assert_eq!(gender, "F");

        Ok(())
    }
}