mod model;
mod inference;
// mod dsp; // Lo implementaremos en el siguiente paso

use wasm_bindgen::prelude::*;
use model::VocalisModel;
use std::sync::OnceLock;

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
    
    // 1. DSP: Extraer Features (MFCC + Pitch)
    // TODO: Implementar dsp::extract_features(audio_data, sample_rate)
    // Por ahora simulamos features aleatorios para probar la compilación
    let f0 = 150.0; // Simulado
    let mfccs = vec![0.0; 13]; // Simulado
    
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
    // Convertimos f32 a f64 porque el SVM usa f64
    let features_f64: Vec<f64> = mfccs.iter().map(|&x| x as f64).collect();
    let vowel = inference::Predictor::predict(&features_f64, svm_model);
    
    Ok(format!("Gender: {} (DSP), Vowel: {} (SVM)", gender_char, vowel))
}