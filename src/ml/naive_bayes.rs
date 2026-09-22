use crate::domain::traits::{AppResult, ModelPredictor, ModelTrainer};
use serde::{Deserialize, Serialize};

/// Gaussian Naive Bayes: P(C_k | X) \propto P(C_k) \prod \mathcal{N}(x_j; \mu_{kj}, \sigma_{kj}^2)
/// Supuesto de independencia condicional entre variables dadas la clase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianNaiveBayesModel {
    pub prior_pos: f64,
    pub prior_neg: f64,
    pub means_pos: Vec<f64>,
    pub vars_pos: Vec<f64>,
    pub means_neg: Vec<f64>,
    pub vars_neg: Vec<f64>,
}

impl ModelPredictor for GaussianNaiveBayesModel {
    fn name(&self) -> &'static str {
        "Gaussian Naive Bayes"
    }

    fn predict_probability(&self, features: &[f64]) -> f64 {
        let n_feat = features.len().min(self.means_pos.len());

        let mut log_p_pos = self.prior_pos.ln();
        let mut log_p_neg = self.prior_neg.ln();

        let pi_const = (2.0 * std::f64::consts::PI).ln();

        for j in 0..n_feat {
            let x = features[j];

            // Positivos (Default Y = 1)
            let var_pos = self.vars_pos[j].max(1e-9);
            let diff_pos = x - self.means_pos[j];
            let log_likelihood_pos = -0.5 * (pi_const + var_pos.ln() + (diff_pos * diff_pos) / var_pos);
            log_p_pos += log_likelihood_pos;

            // Negativos (Solventes Y = 0)
            let var_neg = self.vars_neg[j].max(1e-9);
            let diff_neg = x - self.means_neg[j];
            let log_likelihood_neg = -0.5 * (pi_const + var_neg.ln() + (diff_neg * diff_neg) / var_neg);
            log_p_neg += log_likelihood_neg;
        }

        // Softmax numéricamente estable para dos clases: P(Y=1) = 1 / (1 + exp(log_neg - log_pos))
        let log_diff = log_p_neg - log_p_pos;
        if log_diff >= 35.0 {
            0.0
        } else if log_diff <= -35.0 {
            1.0
        } else {
            1.0 / (1.0 + log_diff.exp())
        }
    }

    fn feature_importances(&self) -> Vec<f64> {
        // En Naive Bayes, una medida canónica de importancia es la divergencia de Kullback-Leibler o distancia de medias normalizada
        let n = self.means_pos.len();
        let mut scores = Vec::with_capacity(n);
        for j in 0..n {
            let var_pool = (self.vars_pos[j] + self.vars_neg[j]) / 2.0;
            let std_pool = var_pool.sqrt().max(1e-6);
            let d = ((self.means_pos[j] - self.means_neg[j]).abs()) / std_pool;
            scores.push(d);
        }

        let sum: f64 = scores.iter().sum();
        if sum > 0.0 {
            scores.iter().map(|&s| s / sum).collect()
        } else {
            vec![1.0 / n as f64; n]
        }
    }
}

pub struct GaussianNaiveBayesTrainer {
    pub var_smoothing: f64,
}

impl Default for GaussianNaiveBayesTrainer {
    fn default() -> Self {
        Self { var_smoothing: 1e-6 }
    }
}

impl ModelTrainer for GaussianNaiveBayesTrainer {
    type Model = GaussianNaiveBayesModel;

    fn train(&self, x: &[Vec<f64>], y: &[u8]) -> AppResult<Self::Model> {
        let n_samples = x.len();
        if n_samples == 0 {
            return Err("Sin muestras para Naive Bayes".into());
        }
        let n_features = x[0].len();

        let mut count_pos = 0usize;
        let mut count_neg = 0usize;

        let mut means_pos = vec![0.0f64; n_features];
        let mut means_neg = vec![0.0f64; n_features];

        for i in 0..n_samples {
            if y[i] == 1 {
                count_pos += 1;
                for j in 0..n_features {
                    means_pos[j] += x[i][j];
                }
            } else {
                count_neg += 1;
                for j in 0..n_features {
                    means_neg[j] += x[i][j];
                }
            }
        }

        if count_pos == 0 || count_neg == 0 {
            return Err("Se requieren muestras de ambas clases para estimar distribuciones gaussianas".into());
        }

        for j in 0..n_features {
            means_pos[j] /= count_pos as f64;
            means_neg[j] /= count_neg as f64;
        }

        let mut vars_pos = vec![0.0f64; n_features];
        let mut vars_neg = vec![0.0f64; n_features];

        for i in 0..n_samples {
            if y[i] == 1 {
                for j in 0..n_features {
                    let diff = x[i][j] - means_pos[j];
                    vars_pos[j] += diff * diff;
                }
            } else {
                for j in 0..n_features {
                    let diff = x[i][j] - means_neg[j];
                    vars_neg[j] += diff * diff;
                }
            }
        }

        for j in 0..n_features {
            vars_pos[j] = (vars_pos[j] / count_pos as f64) + self.var_smoothing;
            vars_neg[j] = (vars_neg[j] / count_neg as f64) + self.var_smoothing;
        }

        let prior_pos = count_pos as f64 / n_samples as f64;
        let prior_neg = count_neg as f64 / n_samples as f64;

        Ok(GaussianNaiveBayesModel {
            prior_pos,
            prior_neg,
            means_pos,
            vars_pos,
            means_neg,
            vars_neg,
        })
    }
}
