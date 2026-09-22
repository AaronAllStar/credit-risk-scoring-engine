use serde::{Deserialize, Serialize};

/// Métricas de clasificación binaria con especial soporte para desbalance severo de clases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub model_name: String,
    pub true_positives: usize,
    pub false_positives: usize,
    pub true_negatives: usize,
    pub false_negatives: usize,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub roc_auc: f64,
    pub pr_auc: f64,
    pub total_samples: usize,
    pub positive_rate: f64,
}

pub struct MetricsCalculator;

impl MetricsCalculator {
    /// Evalúa predicciones continuas de probabilidad contra las etiquetas reales.
    pub fn evaluate(model_name: &str, y_true: &[u8], y_probs: &[f64], threshold: f64) -> ModelMetrics {
        assert_eq!(y_true.len(), y_probs.len(), "Longitud de etiquetas y probabilidades debe coincidir");
        let n = y_true.len();
        if n == 0 {
            return ModelMetrics {
                model_name: model_name.to_string(),
                true_positives: 0,
                false_positives: 0,
                true_negatives: 0,
                false_negatives: 0,
                accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                roc_auc: 0.5,
                pr_auc: 0.0,
                total_samples: 0,
                positive_rate: 0.0,
            };
        }

        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_cnt = 0;

        for (&y, &p) in y_true.iter().zip(y_probs.iter()) {
            let pred = if p >= threshold { 1 } else { 0 };
            match (y, pred) {
                (1, 1) => tp += 1,
                (0, 1) => fp += 1,
                (0, 0) => tn += 1,
                (1, 0) => fn_cnt += 1,
                _ => {}
            }
        }

        let accuracy = (tp + tn) as f64 / n as f64;
        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_cnt > 0 { tp as f64 / (tp + fn_cnt) as f64 } else { 0.0 };
        let f1_score = if precision + recall > 0.0 {
            2.0 * (precision * recall) / (precision + recall)
        } else {
            0.0
        };

        let roc_auc = Self::compute_roc_auc(y_true, y_probs);
        let pr_auc = Self::compute_pr_auc(y_true, y_probs);
        let positive_rate = (tp + fn_cnt) as f64 / n as f64;

        ModelMetrics {
            model_name: model_name.to_string(),
            true_positives: tp,
            false_positives: fp,
            true_negatives: tn,
            false_negatives: fn_cnt,
            accuracy,
            precision,
            recall,
            f1_score,
            roc_auc,
            pr_auc,
            total_samples: n,
            positive_rate,
        }
    }

    /// Calcula el Área bajo la Curva ROC (ROC-AUC) utilizando la estadística U de Mann-Whitney / Wilcoxon.
    /// Formula: AUC = (R_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    pub fn compute_roc_auc(y_true: &[u8], y_probs: &[f64]) -> f64 {
        let n = y_true.len();
        if n == 0 {
            return 0.5;
        }

        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| y_probs[a].partial_cmp(&y_probs[b]).unwrap_or(std::cmp::Ordering::Equal));

        let mut n_pos = 0usize;
        let mut n_neg = 0usize;
        let mut rank_sum_pos = 0.0f64;

        // Asignación de rangos con manejo de empates
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j < n && (y_probs[indices[j]] - y_probs[indices[i]]).abs() < 1e-9 {
                j += 1;
            }
            let avg_rank = (i + j + 1) as f64 / 2.0; // 1-based indexing rank
            for k in i..j {
                let idx = indices[k];
                if y_true[idx] == 1 {
                    n_pos += 1;
                    rank_sum_pos += avg_rank;
                } else {
                    n_neg += 1;
                }
            }
            i = j;
        }

        if n_pos == 0 || n_neg == 0 {
            return 0.5;
        }

        let u_pos = rank_sum_pos - (n_pos as f64 * (n_pos as f64 + 1.0)) / 2.0;
        (u_pos / (n_pos as f64 * n_neg as f64)).clamp(0.0, 1.0)
    }

    /// Calcula el Área bajo la Curva Precision-Recall (PR-AUC / Average Precision).
    pub fn compute_pr_auc(y_true: &[u8], y_probs: &[f64]) -> f64 {
        let n = y_true.len();
        if n == 0 {
            return 0.0;
        }

        let mut items: Vec<(f64, u8)> = y_probs.iter().cloned().zip(y_true.iter().cloned()).collect();
        // Orden descendente por probabilidad
        items.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let total_positives: usize = y_true.iter().map(|&y| y as usize).sum();
        if total_positives == 0 {
            return 0.0;
        }

        let mut cum_tp = 0;
        let mut cum_fp = 0;
        let mut prev_recall = 0.0f64;
        let mut pr_auc = 0.0f64;

        for (_, y) in items {
            if y == 1 {
                cum_tp += 1;
            } else {
                cum_fp += 1;
            }

            let precision = cum_tp as f64 / (cum_tp + cum_fp) as f64;
            let recall = cum_tp as f64 / total_positives as f64;

            let delta_recall = recall - prev_recall;
            if delta_recall > 0.0 {
                pr_auc += precision * delta_recall;
                prev_recall = recall;
            }
        }

        pr_auc.clamp(0.0, 1.0)
    }
}
