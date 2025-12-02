use crate::model::GenderModel;

pub struct Predictor;

impl Predictor {
    /// Aplica la normalización (StandardScaler) al vector de entrada
    /// x_scaled = (x - mean) / scale
    pub fn normalize(features: &[f64], model: &GenderModel) -> Vec<f64> {
        features.iter()
            .zip(model.scaler.mean.iter())
            .zip(model.scaler.scale.iter())
            .map(|((&x, &m), &s)| (x - m) / s)
            .collect()
    }

    /// Kernel RBF (Radial Basis Function)
    /// K(x, y) = exp(-gamma * ||x - y||^2)
    fn kernel_rbf(x1: &[f64], x2: &[f64], gamma: f64) -> f64 {
        let mut sum_sq_diff = 0.0;
        for i in 0..x1.len() {
            let diff = x1[i] - x2[i];
            sum_sq_diff += diff * diff;
        }
        (-gamma * sum_sq_diff).exp()
    }

    /// Predicción SVM (Simplificada)
    /// Esta es la parte crítica. Sklearn usa One-vs-One para multiclase.
    /// Por ahora, implementaremos la lógica genérica de votación.
    pub fn predict(features: &[f64], model: &GenderModel) -> String {
        let normalized = Self::normalize(features, model);
        
        // TODO: Implementar la lógica completa de votación OvO de LIBSVM.
        // Requiere iterar sobre los coeficientes duales y vectores de soporte
        // de forma correcta según n_support.
        
        // Placeholder temporal: Retorna la primera clase para que compile
        model.svm.classes[0].clone()
    }
}
