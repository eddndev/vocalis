use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct VocalisModel {
    pub model_male: GenderModel,
    pub model_female: GenderModel,
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
}

#[derive(Debug, Deserialize, Serialize)]
pub struct PredictionResult {
    pub vowel: String,
    pub gender: String,
}
