use once_cell::sync::OnceCell;
use anyhow::{Context, Result};
use log::info;
use std::sync::Mutex;

// Revert to stable ort 1.x imports
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Value,
};
use ndarray::Array2;

use crate::error::VocalisError;

const ONNX_MODEL_PATH: &str = "models/vocalis_model.onnx";
const VOWEL_LABELS: [&str; 5] = ["a", "e", "i", "o", "u"];
const GENDER_LABELS: [&str; 2] = ["M", "F"];

// Lazy static instance of the VocalisModel
static VOCALIS_MODEL: OnceCell<VocalisModel> = OnceCell::new();

pub struct VocalisModel {
    session: Mutex<Session>,
    _input_name: String,
}

impl VocalisModel {
    /// Initializes the model and stores it in the global OnceCell.
    pub fn init() -> Result<&'static VocalisModel, VocalisError> {
        VOCALIS_MODEL.get_or_try_init(|| {
            info!("Initializing VocalisModel with ONNX Runtime (ort 2.0)...");
            
            // 1. Initialize ORT (Global Environment)
            ort::init()
                .with_name("vocalis_ort_env")
                .commit()
                .map_err(|e| VocalisError::ModelError(format!("Failed to init ORT: {}", e)))?;

            // Resolve model path (check current dir and parent dir)
            let possible_paths = [ONNX_MODEL_PATH, "../models/vocalis_model.onnx"];
            let model_path = possible_paths.iter()
                .find(|p| std::path::Path::new(p).exists())
                .ok_or_else(|| VocalisError::ModelError(format!("Could not find ONNX model. Searched: {:?}", possible_paths)))?;

            info!("Loading model from: {}", model_path);

            // 2. Load session
            let session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .commit_from_file(model_path)
                .context(format!("Failed to load ONNX model from {}", model_path))?;
            
            info!("VocalisModel loaded successfully.");

            // Extract input name (assuming 1 input)
            let input_name = session.inputs[0].name.clone();

            Ok(VocalisModel {
                session: Mutex::new(session),
                _input_name: input_name,
            })
        })
    }

    /// Performs inference on raw audio samples.
    pub fn predict(&self, audio_samples: &[f32]) -> Result<(String, String), VocalisError> {
        info!("Performing inference...");

        // Audio length check (approximate check based on training)
        let expected_samples = 8000;
        if audio_samples.len() != expected_samples {
             // Log warning but maybe proceed or resize? For E2E we stick to strict length for now.
             return Err(VocalisError::InputError(format!(
                "Expected {} samples, got {}", expected_samples, audio_samples.len()
            )));
        }

        // 1. Prepare Input Tensor (ndarray)
        // Shape: [1, samples]
        let input_array = Array2::from_shape_vec((1, expected_samples), audio_samples.to_vec())?;

        let input_value = Value::from_array(input_array).context("Failed to create ORT Value")?;

        // Lock the session
        let mut session = self.session.lock().map_err(|e| VocalisError::Other(format!("Failed to lock session: {}", e)))?;

        // 2. Run Inference
        let outputs = session.run(ort::inputs![input_value])
            .context("Failed to run inference")?;

        // 3. Extract Outputs
        // Output 0: Vowel Logits [1, 5]
        // Output 1: Gender Logits [1, 2]
        
        let (_vowel_shape, vowel_data) = outputs[0].try_extract_tensor::<f32>()?;
        let (_gender_shape, gender_data) = outputs[1].try_extract_tensor::<f32>()?;
        
        info!("Vowel Logits (flat): {:?}", vowel_data);
        info!("Gender Logits (flat): {:?}", gender_data);

        // Find ArgMax
        let vowel_idx = vowel_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0);

        let gender_idx = gender_data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap_or(0);

        Ok((
            VOWEL_LABELS.get(vowel_idx).unwrap_or(&"?").to_string(),
            GENDER_LABELS.get(gender_idx).unwrap_or(&"?").to_string(),
        ))
    }
}
