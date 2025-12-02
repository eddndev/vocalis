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
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SvmParams {
    pub gamma: f64,
    pub intercept: Vec<f64>,
    pub dual_coef: Vec<Vec<f64>>,
    pub support_vectors: Vec<Vec<f64>>,
    pub n_support: Vec<i32>,
    pub classes: Vec<String>,
}
