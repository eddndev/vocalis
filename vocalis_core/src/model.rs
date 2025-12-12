use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct VocalisModel {
    // Existing vowel-only models (5 classes, 13 dims)
    #[serde(alias = "model_male")]
    pub vowel_male: Option<GenderModel>,
    #[serde(alias = "model_female")]
    pub vowel_female: Option<GenderModel>,
    
    // NEW: Unified models for vowels + syllables (25 classes, 39 dims)
    pub unified_male: Option<GenderModel>,
    pub unified_female: Option<GenderModel>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct GenderModel {
    pub scaler: ScalerParams,
    pub svm: SvmParams,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ScalerParams {
    pub mean: Vec<f32>,
    pub scale: Vec<f32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SvmParams {
    pub gamma: f32,
    pub intercept: Vec<f32>,
    pub dual_coef: Vec<Vec<f32>>,
    pub support_vectors: Vec<Vec<f32>>,
    pub n_support: Vec<i32>,
    pub classes: Vec<String>,
    #[serde(default, rename = "probA")]
    pub prob_a: Vec<f32>,
    #[serde(default, rename = "probB")]
    pub prob_b: Vec<f32>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PredictionResult {
    pub vowel: String,       // Keeps name for backwards compatibility, but holds label
    pub gender: String,
    pub probabilities: Vec<(String, f32)>,
}

