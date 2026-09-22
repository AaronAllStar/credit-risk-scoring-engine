use crate::domain::traits::{AppResult, ModelPredictor, ModelTrainer};
use serde::{Deserialize, Serialize};

/// Regresión Logística regularizada con penalización L2 (Ridge) y pesos de balanceo de clases.
/// Función Sigmoide: \sigma(z) = 1 / (1 + e^{-z})
/// Función Objetivo: J(w) = - \sum [ w_1 y_i \ln(p_i) + w_0 (1 - y_i) \ln(1 - p_i) ] + \frac{\lambda}{2} ||w||_2^2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegressionModel {
    pub weights: Vec<f64>,
    pub bias: f64,
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
}

impl ModelPredictor for LogisticRegressionModel {
    fn name(&self) -> &'static str {
        "Logistic Regression (L2 Regularized)"
    }

    fn predict_probability(&self, features: &[f64]) -> f64 {
        let mut z = self.bias;
        for i in 0..features.len().min(self.weights.len()) {
            let std = if self.stds[i] > 1e-8 { self.stds[i] } else { 1.0 };
            let normalized_x = (features[i] - self.means[i]) / std;
            z += self.weights[i] * normalized_x;
        }

        // Sigmoide numéricamente estable contra overflow
        if z >= 35.0 {
            1.0
        } else if z <= -35.0 {
            0.0
        } else {
            1.0 / (1.0 + (-z).exp())
        }
    }

    fn feature_importances(&self) -> Vec<f64> {
        let abs_sum: f64 = self.weights.iter().map(|w| w.abs()).sum();
        if abs_sum > 0.0 {
            self.weights.iter().map(|w| w.abs() / abs_sum).collect()
        } else {
            vec![1.0 / self.weights.len() as f64; self.weights.len()]
        }
    }
}

pub struct LogisticRegressionTrainer {
    pub learning_rate: f64,
    pub max_epochs: usize,
    pub l2_penalty: f64,
    pub batch_size: usize,
}

impl Default for LogisticRegressionTrainer {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            max_epochs: 150,
            l2_penalty: 0.001,
            batch_size: 256,
        }
    }
}

impl ModelTrainer for LogisticRegressionTrainer {
    type Model = LogisticRegressionModel;

    fn train(&self, x: &[Vec<f64>], y: &[u8]) -> AppResult<Self::Model> {
        let n_samples = x.len();
        if n_samples == 0 {
            return Err("El conjunto de datos está vacío".into());
        }
        let n_features = x[0].len();

        // 1. Estandarización de Variables Z-Score: mu y sigma
        let mut means = vec![0.0f64; n_features];
        let mut stds = vec![0.0f64; n_features];

        for row in x {
            for j in 0..n_features {
                means[j] += row[j];
            }
        }
        for j in 0..n_features {
            means[j] /= n_samples as f64;
        }

        for row in x {
            for j in 0..n_features {
                let diff = row[j] - means[j];
                stds[j] += diff * diff;
            }
        }
        for j in 0..n_features {
            stds[j] = (stds[j] / n_samples as f64).sqrt();
            if stds[j] < 1e-7 {
                stds[j] = 1.0;
            }
        }

        // 2. Cálculo de pesos de clase inversamente proporcionales al desbalance
        let n_pos = y.iter().filter(|&&v| v == 1).count().max(1);
        let n_neg = (n_samples - n_pos).max(1);
        let weight_pos = (n_samples as f64) / (2.0 * n_pos as f64);
        let weight_neg = (n_samples as f64) / (2.0 * n_neg as f64);

        // 3. Matriz X estandarizada
        let x_norm: Vec<Vec<f64>> = x
            .iter()
            .map(|row| {
                (0..n_features)
                    .map(|j| (row[j] - means[j]) / stds[j])
                    .collect()
            })
            .collect();

        let mut weights = vec![0.0f64; n_features];
        let mut bias = ((n_pos as f64) / (n_neg as f64)).ln(); // Prior log-odds initialization

        // 4. Descenso de Gradiente con Regularización L2
        for epoch in 0..self.max_epochs {
            let lr = self.learning_rate / (1.0 + 0.01 * epoch as f64);

            let mut grad_w = vec![0.0f64; n_features];
            let mut grad_b = 0.0f64;

            for i in 0..n_samples {
                let mut z = bias;
                for j in 0..n_features {
                    z += weights[j] * x_norm[i][j];
                }

                let p = if z >= 35.0 {
                    1.0
                } else if z <= -35.0 {
                    0.0
                } else {
                    1.0 / (1.0 + (-z).exp())
                };

                let target = y[i] as f64;
                let sample_weight = if y[i] == 1 { weight_pos } else { weight_neg };
                let error = (p - target) * sample_weight;

                for j in 0..n_features {
                    grad_w[j] += error * x_norm[i][j];
                }
                grad_b += error;
            }

            for j in 0..n_features {
                // Gradiente promedio + término de regularización L2 (\lambda * w_j)
                let total_grad = (grad_w[j] / n_samples as f64) + self.l2_penalty * weights[j];
                weights[j] -= lr * total_grad;
            }
            bias -= lr * (grad_b / n_samples as f64);
        }

        Ok(LogisticRegressionModel {
            weights,
            bias,
            means,
            stds,
        })
    }
}
