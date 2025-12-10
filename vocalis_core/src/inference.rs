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


    /// Sigmoid function
    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Multiclass Probability Coupling (Wu et al. 2004)
    /// Solves for p_i such that r_ij approx p_i / (p_i + p_j)
    fn multiclass_probability(k: usize, r: &Vec<Vec<f32>>) -> Vec<f32> {
        let mut p = vec![1.0 / k as f32; k];
        let mut q = vec![vec![0.0; k]; k];
        let eps = 0.005 / k as f32;
        
        let mut iter = 0;
        let max_iter = 100;
        
        // Reconstruct full r matrix (symmetric with 1-p)
        // r_flat contains k*(k-1)/2 values for (0,1), (0,2)...
        // We need easy access.
        
        loop {
            // p_old = p
            let p_old = p.clone();
            
            let mut q_sum = vec![0.0; k];
            
            for i in 0..k {
                for j in 0..k {
                    if i == j { continue; }
                    
                    // Find r_ij from flattened r
                    // Index mapping: for i < j, index = i * (2k - 1 - i) / 2 + j - i - 1
                    
                    // Formula for standard upper triangular indexing
                    // But our input r is likely sequential from the loop
                    // Let's rely on the caller passing a mapping or structured data.
                    // Actually, let's redefine this function to take a closure or fully coupled matrix.
                    // For Simplicity inside this function, let's assume 'r' is a KxK matrix where r[i][j] = P(i|i,j)
                    
                     q[i][j] = 1.0 / (p[i] + p[j]);
                     q_sum[i] += q[i][j];
                }
            }
            
            let mut max_diff = 0.0;
            
            for i in 0..k {
                let mut sum = 0.0;
                for j in 0..k {
                     if i == j { continue; }
                     // r[i][j] = P(i vs j)
                     // r[j][i] should be P(j vs i) = 1 - r[i][j]
                     let r_ij = r[i][j]; // Assume input r is full KxK
                     
                     sum += q[i][j] * r_ij;
                }
                p[i] = sum / q_sum[i];
                
                let diff = (p[i] - p_old[i]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
            
            if max_diff < eps || iter >= max_iter {
                break;
            }
            iter += 1;
        }
        
        p
    }

    pub fn predict_proba(features: &[f32], model: &GenderModel) -> Vec<(String, f32)> {
        let normalized = Self::normalize(features, model);
        let n_classes = model.svm.classes.len();
        let n_sv = model.svm.support_vectors.len();
        
        // Check if we have probability parameters
        let has_probs = !model.svm.probA.is_empty() && !model.svm.probB.is_empty();

        // 1. Pre-calculate Kernel values
        let mut k_values = Vec::with_capacity(n_sv);
        for sv in &model.svm.support_vectors {
            k_values.push(Self::kernel_rbf(&normalized, sv, model.svm.gamma));
        }

        // 2. Offsets
        let mut sv_offsets = vec![0; n_classes + 1];
        for i in 0..n_classes {
            sv_offsets[i + 1] = sv_offsets[i] + model.svm.n_support[i] as usize;
        }

        // 3. Pairwise Decision & Probabilities
        let mut r_matrix = vec![vec![0.0; n_classes]; n_classes];
        let mut votes = vec![0; n_classes];
        
        let mut pair_idx = 0;
        let mut intercept_idx = 0;

        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                let mut sum = 0.0;

                // SVs of i
                let start_i = sv_offsets[i];
                let end_i = sv_offsets[i + 1];
                for k in start_i..end_i {
                    sum += model.svm.dual_coef[j - 1][k] * k_values[k];
                }

                // SVs of j
                let start_j = sv_offsets[j];
                let end_j = sv_offsets[j + 1];
                for k in start_j..end_j {
                    sum += model.svm.dual_coef[i][k] * k_values[k];
                }

                // Intercept
                sum += model.svm.intercept[intercept_idx];
                intercept_idx += 1;

                // Voting (Fallback)
                if sum > 0.0 {
                    votes[i] += 1;
                } else {
                    votes[j] += 1;
                }

                // Probability (Platt Scaling)
                if has_probs {
                    let a = model.svm.probA[pair_idx];
                    let b = model.svm.probB[pair_idx];
                    
                    // f = decision value. 
                    // Platt: P(y=i | f) = 1 / (1 + exp(A*f + B))
                    let f = sum;
                    let prob_i = Self::sigmoid(a * f + b); 
                    let prob_j = 1.0 - prob_i; // P(y=j | f)
                    
                    r_matrix[i][j] = prob_i;
                    r_matrix[j][i] = prob_j;
                }
                
                pair_idx += 1;
            }
        }

        if has_probs {
            let p = Self::multiclass_probability(n_classes, &r_matrix);
            // Map to (Label, Score)
            let mut results: Vec<(String, f32)> = model.svm.classes.iter()
                .zip(p.iter())
                .map(|(label, &score)| (label.clone(), score))
                .collect();
                
            // Sort Descending
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return results;
        } else {
             // Fallback to Voting if no probs
             // Normalize votes to fake "probabilities" (just ratios)
             let total_votes: i32 = n_classes as i32 * (n_classes as i32 - 1) / 2; // Total pairs
             // Actually, votes sum is total pairs.
             
             let mut results: Vec<(String, f32)> = Vec::new();
             for (i, &vote) in votes.iter().enumerate() {
                 // Simple ratio. Not real probability but gives ranking.
                 // Avoid div by zero (shouldn't happen with >1 classes)
                 let score = if total_votes > 0 { vote as f32 / total_votes as f32 } else { 0.0 };
                 results.push((model.svm.classes[i].clone(), score));
             }
             results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
             return results;
        }
    }

    /// Backwards compatible predict (just takes usage of predict_proba winner)
    pub fn predict(features: &[f32], model: &GenderModel) -> String {
        let results = Self::predict_proba(features, model);
        if let Some((first_label, _)) = results.first() {
            first_label.clone()
        } else {
            "Unknown".to_string()
        }
    }
}
