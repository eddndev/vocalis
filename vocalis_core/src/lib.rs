mod model;
mod inference;
mod dsp;

use wasm_bindgen::prelude::*;
use model::{VocalisModel, PredictionResult};
use std::sync::OnceLock;
use dsp::DspProcessor;

// Singleton para mantener el modelo en memoria
static MODEL: OnceLock<VocalisModel> = OnceLock::new();

// Incrustamos el JSON en el binario (Compilación estática)
// Asegúrate de que la ruta sea correcta relativa al crate root
const MODEL_JSON: &str = include_str!("../../research/dsp_lab/models/vocalis_model.json");

#[wasm_bindgen]
pub fn init_vocalis() -> Result<(), JsValue> {
    // Configurar hook de pánico para debugging en consola
    console_error_panic_hook::set_once();
    
    // Cargar y parsear el modelo una sola vez
    if MODEL.get().is_none() {
        let model: VocalisModel = serde_json::from_str(MODEL_JSON)
            .map_err(|e| JsValue::from_str(&format!("Error parsing model JSON: {}", e)))?;
            
        MODEL.set(model).map_err(|_| JsValue::from_str("Model already initialized"))?;
        
        web_sys::console::log_1(&"Vocalis Core (Rust) Initialized!".into());
    }
    
    Ok(())
}

#[wasm_bindgen]
pub fn predict_vowel(audio_data: &[f32], sample_rate: f32) -> Result<String, JsValue> {
    let model = MODEL.get().ok_or("Model not initialized. Call init_vocalis() first.")?;
    
    // Initialize DSP Processor
    let dsp_processor = DspProcessor::new(sample_rate as u32, model.model_male.svm.support_vectors[0].len());
    let pre_emphasis_coeff = 0.0; // Disabled to match librosa.feature.mfcc defaults

    // --- DEBUG: Synthetic 150Hz Sine Wave (matches Python debug script) ---
    // let mut audio_data_synth = Vec::with_capacity(8000);
    // let duration = 0.5;
    // let samples = (sample_rate * duration) as usize;
    // for i in 0..samples {
    //     let t = i as f32 / sample_rate;
    //     let sample = 0.5 * (2.0 * std::f32::consts::PI * 150.0 * t).sin();
    //     audio_data_synth.push(sample);
    // }
    // let audio_data = &audio_data_synth; // Override input
    // web_sys::console::log_1(&format!("DEBUG: Using Synthetic 150Hz Sine Wave (Max: {:.2})", 0.5).into());
    // ----------------------------------------------------------------------

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
        return Err(JsValue::from_str("No valid audio frames to process."));
    }

    // Debug: Log Raw MFCCs of first frame
    web_sys::console::log_1(&format!("RAW MFCCs (First Frame): {:?}", &all_mfccs[0]).into());

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
    
    // Debug logging
    web_sys::console::log_1(&format!("Detected F0: {:.2} Hz", f0).into());
    web_sys::console::log_1(&format!("Averaged MFCCs (first 5): {:?}", &averaged_mfccs[0..5]).into());

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
    
    // 4. Inferencia SVM
    // Pasamos los MFCCs al predictor
    let vowel = inference::Predictor::predict(&mfccs_for_prediction, svm_model);
    
    let result = PredictionResult {
        vowel: vowel,
        gender: format!("{} (DSP)", gender_char),
    };

    serde_json::to_string(&result)
        .map_err(|e| JsValue::from_str(&format!("Error serializing prediction result: {}", e)))
}