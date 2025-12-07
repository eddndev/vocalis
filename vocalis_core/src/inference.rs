use crate::model::GenderModel;

pub struct Predictor;

impl Predictor {
    /// Aplica la normalización (StandardScaler) al vector de entrada
    /// x_scaled = (x - mean) / scale
    pub fn normalize(features: &[f32], model: &GenderModel) -> Vec<f32> {
        // HACK: Neutralizar C0 (Energía/Volumen)
        // El volumen del navegador puede variar mucho respecto al entrenamiento.
        // Al forzar C0 = media, anulamos su efecto en el SVM.
        let mut feats = features.to_vec();
        if !feats.is_empty() && !model.scaler.mean.is_empty() {
            feats[0] = model.scaler.mean[0];
        }

        feats.iter()
            .zip(model.scaler.mean.iter())
            .zip(model.scaler.scale.iter())
            .map(|((&x, &m), &s)| (x - m) / s)
            .collect()
    }

    /// Kernel RBF (Radial Basis Function)
    /// K(x, y) = exp(-gamma * ||x - y||^2)
    fn kernel_rbf(x1: &[f32], x2: &[f32], gamma: f32) -> f32 {
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
    pub fn predict(features: &[f32], model: &GenderModel) -> String {
        let normalized = Self::normalize(features, model);
        let n_classes = model.svm.classes.len();
        let n_sv = model.svm.support_vectors.len();
        
        // 1. Pre-calculate Kernel values for all Support Vectors
        // K(x, SV_i)
        let mut k_values = Vec::with_capacity(n_sv);
        for sv in &model.svm.support_vectors {
            k_values.push(Self::kernel_rbf(&normalized, sv, model.svm.gamma));
        }

        // 2. Calculate start indices (offsets) for each class's SVs
        let mut sv_offsets = vec![0; n_classes + 1];
        for i in 0..n_classes {
            sv_offsets[i + 1] = sv_offsets[i] + model.svm.n_support[i] as usize;
        }

        // 3. One-vs-One Voting
        let mut votes = vec![0; n_classes];
        let mut intercept_idx = 0;

        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                let mut sum = 0.0;

                // Sum over SVs of class i
                // Coefs are in dual_coef[j-1] for class i part
                let start_i = sv_offsets[i];
                let end_i = sv_offsets[i + 1];
                for k in start_i..end_i {
                    sum += model.svm.dual_coef[j - 1][k] * k_values[k];
                }

                // Sum over SVs of class j
                // Coefs are in dual_coef[i] for class j part
                let start_j = sv_offsets[j];
                let end_j = sv_offsets[j + 1];
                for k in start_j..end_j {
                    sum += model.svm.dual_coef[i][k] * k_values[k];
                }

                // Add Intercept
                sum += model.svm.intercept[intercept_idx];
                intercept_idx += 1;

                // Vote
                if sum > 0.0 {
                    votes[i] += 1;
                } else {
                    votes[j] += 1;
                }
            }
        }

        // 4. Find Winner
        let mut max_votes = -1;
        let mut winner_idx = 0;

        for (i, &vote) in votes.iter().enumerate() {
            if vote > max_votes {
                max_votes = vote;
                winner_idx = i;
            }
        }

        model.svm.classes[winner_idx].clone()
    }
}
