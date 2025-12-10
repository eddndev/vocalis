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
        println!($($t)*);
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

pub fn predict_vowel_internal(audio_data: &[f32], sample_rate: f32) -> Result<String, String> {
    let model = MODEL.get().ok_or("Model not initialized. Call init_vocalis() first.")?;
    
    // Initialize DSP Processor
    let dsp_processor = DspProcessor::new(sample_rate as u32, model.model_male.svm.support_vectors[0].len());
    let pre_emphasis_coeff = 0.0; // Disabled to match Python librosa features

    // --- DEBUG SILENCIADO PARA RELEASE/TEST ---
    
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

    // Debug: Log Raw MFCCs of first frame (solo si es muy necesario)
    // log!("RAW MFCCs (First Frame): {:?}", &all_mfccs[0]);

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
    
    // 1. DSP: Extraer Features (MFCC + Pitch)
    let f0 = dsp_processor.compute_pitch(audio_data);
    
    log!("Detected F0: {:.2} Hz", f0);
    log!("Averaged MFCCs (first 5): {:?}", &averaged_mfccs[0..5]);

    // Replace the simulated MFCCs with the averaged ones
    let mfccs_for_prediction = averaged_mfccs; 
    
    // 2. Clasificación de Género (Regla DSP simple)
    let is_male = f0 <= 178.7;
    let gender_char = if is_male { "M" } else { "F" };
    
    // 3. Selección de Modelo SVM
    let svm_model = if is_male {
        &model.model_male
    } else {
        &model.model_female
    };
    
    // 4. Inferencia SVM (Probabilística)
    let probabilities = inference::Predictor::predict_proba(&mfccs_for_prediction, svm_model);
    
    // El ganador es el primero (ya está ordenado)
    let vowel = probabilities.first()
        .map(|(label, _)| label.clone())
        .unwrap_or_else(|| "Unknown".to_string());
    
    let result = PredictionResult {
        vowel: vowel,
        gender: format!("{} (DSP)", gender_char),
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

#[wasm_bindgen]
pub fn predict_vowel(audio_data: &[f32], sample_rate: f32) -> Result<String, JsValue> {
    predict_vowel_internal(audio_data, sample_rate).map_err(|e| JsValue::from_str(&e))
}