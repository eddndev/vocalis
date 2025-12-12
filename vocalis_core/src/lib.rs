mod model;
mod inference;
mod dsp;

use wasm_bindgen::prelude::*;
use model::{VocalisModel, PredictionResult};
use std::sync::OnceLock;
use dsp::DspProcessor;

// Singleton para mantener el modelo en memoria
static MODEL: OnceLock<VocalisModel> = OnceLock::new();

// Incrustamos el JSON en el binario
const MODEL_JSON: &str = include_str!("../../research/dsp_lab/models/vocalis_model.json");

// Macro de logging híbrido (WASM vs Native)
macro_rules! log {
    ($($t:tt)*) => {
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&format!($($t)*).into());
        #[cfg(not(target_arch = "wasm32"))]
        println!($($t)*)
    }
}

// --- LÓGICA INTERNA (Pure Rust) ---
// Funciona en Windows (Tests) y en Web

pub fn init_vocalis_internal() -> Result<(), String> {
    if MODEL.get().is_none() {
        let model: VocalisModel = serde_json::from_str(MODEL_JSON)
            .map_err(|e| format!("Error parsing model JSON: {}", e))?;
            
        MODEL.set(model).map_err(|_| "Model already initialized".to_string())?;
        
        log!("Vocalis Core (Rust) Initialized!");
    }
    Ok(())
}

/// NEW: Predict using unified model (25 classes: vowels + syllables)
/// Uses 39-dimensional feature vector (onset + transition + nucleus MFCCs)
pub fn predict_unified_internal(audio_data: &[f32], sample_rate: f32) -> Result<String, String> {
    let model = MODEL.get().ok_or("Model not initialized. Call init_vocalis() first.")?;
    
    // Check if unified models are available
    let (unified_male, unified_female) = match (&model.unified_male, &model.unified_female) {
        (Some(m), Some(f)) => (m, f),
        _ => return Err("Unified models not available in JSON. Re-export with unified models.".to_string()),
    };
    
    // Initialize DSP Processor with 13 MFCCs (will be used 3x for 39 total)
    let dsp_processor = DspProcessor::new(sample_rate as u32, 13);
    let pre_emphasis_coeff = 0.0;
    
    // Extract 39-dim syllable features (onset + transition + nucleus)
    let syllable_features = dsp_processor.extract_syllable_features(audio_data, pre_emphasis_coeff);
    
    log!("Syllable features (39 dims): [{:.2}, {:.2}, {:.2} ... {:.2}, {:.2}, {:.2}]", 
         syllable_features[0], syllable_features[1], syllable_features[2],
         syllable_features[36], syllable_features[37], syllable_features[38]);
    
    // Detect gender using pitch (same logic as before)
    let f0 = dsp_processor.compute_pitch(audio_data);
    let is_male = f0 <= 178.7;
    
    log!("Detected F0: {:.2} Hz -> {}", f0, if is_male { "Male" } else { "Female" });
    
    // Select appropriate unified model based on gender
    let svm_model = if is_male { unified_male } else { unified_female };
    
    // Run SVM inference
    let probabilities = inference::Predictor::predict_proba(&syllable_features, svm_model);
    
    // Get top prediction
    let label = probabilities.first()
        .map(|(l, _)| l.clone())
        .unwrap_or_else(|| "Unknown".to_string());
    
    let gender_str = if is_male { "Masculino" } else { "Femenino" };
    
    let result = PredictionResult {
        vowel: label,  // Holds vowel OR syllable (e.g., "a" or "pa")
        gender: gender_str.to_string(),
        probabilities: probabilities,
    };
    
    serde_json::to_string(&result)
        .map_err(|e| format!("Error serializing prediction result: {}", e))
}

/// Legacy: Predict vowels only (5 classes, 13 dims)
/// Kept for backwards compatibility
pub fn predict_vowel_internal(audio_data: &[f32], sample_rate: f32) -> Result<String, String> {
    let model = MODEL.get().ok_or("Model not initialized. Call init_vocalis() first.")?;
    
    // Try to use legacy vowel models, fall back to unified if not available
    let (vowel_male, vowel_female) = match (&model.vowel_male, &model.vowel_female) {
        (Some(m), Some(f)) => (m, f),
        _ => {
            // Fall back to unified model if vowel-only not available
            log!("Vowel-only models not found, using unified model");
            return predict_unified_internal(audio_data, sample_rate);
        }
    };
    
    // Initialize DSP Processor
    let dsp_processor = DspProcessor::new(sample_rate as u32, vowel_male.svm.support_vectors[0].len());
    let pre_emphasis_coeff = 0.0;

    let mut all_mfccs: Vec<Vec<f32>> = Vec::new();

    // Frame the audio and extract features
    let num_samples = audio_data.len();
    let mut current_sample = 0;

    while current_sample + dsp_processor.frame_length <= num_samples {
        let frame_end = current_sample + dsp_processor.frame_length;
        let audio_frame = &audio_data[current_sample..frame_end];
        
        let mfccs = dsp_processor.extract_features(audio_frame, pre_emphasis_coeff);
        all_mfccs.push(mfccs);

        current_sample += dsp_processor.hop_length;
    }

    if all_mfccs.is_empty() {
        return Err("No valid audio frames to process.".to_string());
    }

    // Bag-of-Frames: Average MFCCs across all frames
    let mut averaged_mfccs = vec![0.0f32; dsp_processor.n_mfcc];
    for mfcc_vec in &all_mfccs {
        for (i, &mfcc_val) in mfcc_vec.iter().enumerate() {
            averaged_mfccs[i] += mfcc_val;
        }
    }
    for mfcc_val in averaged_mfccs.iter_mut() {
        *mfcc_val /= all_mfccs.len() as f32;
    }
    
    // DSP: Extract Pitch
    let f0 = dsp_processor.compute_pitch(audio_data);
    
    log!("Detected F0: {:.2} Hz", f0);
    log!("Averaged MFCCs (first 5): {:?}", &averaged_mfccs[0..5]);

    let mfccs_for_prediction = averaged_mfccs; 
    
    // Gender classification
    let is_male = f0 <= 178.7;
    
    // Select SVM model
    let svm_model = if is_male { vowel_male } else { vowel_female };
    
    // SVM Inference
    let probabilities = inference::Predictor::predict_proba(&mfccs_for_prediction, svm_model);
    
    let vowel = probabilities.first()
        .map(|(label, _)| label.clone())
        .unwrap_or_else(|| "Unknown".to_string());
    
    let gender_str = if is_male { "Masculino" } else { "Femenino" };

    let result = PredictionResult {
        vowel: vowel,
        gender: gender_str.to_string(),
        probabilities: probabilities,
    };

    serde_json::to_string(&result)
        .map_err(|e| format!("Error serializing prediction result: {}", e))
}


// --- INTERFAZ WASM (Wrappers) ---

#[wasm_bindgen]
pub fn init_vocalis() -> Result<(), JsValue> {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
    
    init_vocalis_internal().map_err(|e| JsValue::from_str(&e))
}

/// Legacy vowel prediction (5 classes)
#[wasm_bindgen]
pub fn predict_vowel(audio_data: &[f32], sample_rate: f32) -> Result<String, JsValue> {
    predict_vowel_internal(audio_data, sample_rate).map_err(|e| JsValue::from_str(&e))
}

/// NEW: Unified prediction (25 classes: vowels + syllables)
#[wasm_bindgen]
pub fn predict_unified(audio_data: &[f32], sample_rate: f32) -> Result<String, JsValue> {
    predict_unified_internal(audio_data, sample_rate).map_err(|e| JsValue::from_str(&e))
}