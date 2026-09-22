use crate::domain::traits::{AppResult, ModelPredictor, ModelTrainer};
use crate::ml::decision_tree::DecisionTree;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Random Forest: Ensemble de bagging con particionado aleatorio de features (Breiman, 2001).
/// Votación / Promedio: \hat{P}(Y=1|X) = \frac{1}{B} \sum_{b=1}^B T_b(X)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestModel {
    pub trees: Vec<DecisionTree>,
    pub feature_importances: Vec<f64>,
}

impl ModelPredictor for RandomForestModel {
    fn name(&self) -> &'static str {
        "Random Forest Classifier"
    }

    fn predict_probability(&self, features: &[f64]) -> f64 {
        if self.trees.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.trees.iter().map(|t| t.predict_probability(features)).sum();
        sum / self.trees.len() as f64
    }

    fn feature_importances(&self) -> Vec<f64> {
        self.feature_importances.clone()
    }
}

pub struct RandomForestTrainer {
    pub n_estimators: usize,
    pub max_depth: usize,
    pub min_samples_split: usize,
}

impl Default for RandomForestTrainer {
    fn default() -> Self {
        Self {
            n_estimators: 25,
            max_depth: 7,
            min_samples_split: 10,
        }
    }
}

impl ModelTrainer for RandomForestTrainer {
    type Model = RandomForestModel;

    fn train(&self, x: &[Vec<f64>], y: &[u8]) -> AppResult<Self::Model> {
        let n_samples = x.len();
        if n_samples == 0 {
            return Err("Sin muestras para Random Forest".into());
        }
        let n_features = x[0].len();
        let max_features = ((n_features as f64).sqrt().round() as usize).max(2).min(n_features);

        // Class weights para contrarrestar el desbalance
        let n_pos = y.iter().filter(|&&v| v == 1).count().max(1);
        let n_neg = (n_samples - n_pos).max(1);
        let weight_pos = (n_samples as f64) / (2.0 * n_pos as f64);
        let weight_neg = (n_samples as f64) / (2.0 * n_neg as f64);
        let sample_weights: Vec<f64> = y
            .iter()
            .map(|&label| if label == 1 { weight_pos } else { weight_neg })
            .collect();

        // Entrenamiento paralelo de árboles con Rayon
        let trees: Vec<DecisionTree> = (0..self.n_estimators)
            .into_par_iter()
            .map(|seed_idx| {
                let mut rng = thread_rng();
                // 1. Bootstrap sampling (muestreo con reemplazo)
                let mut bootstrap_indices = Vec::with_capacity(n_samples);
                for _ in 0..n_samples {
                    let rand_idx = rand::Rng::gen_range(&mut rng, 0..n_samples);
                    bootstrap_indices.push(rand_idx);
                }

                // 2. Subsample aleatorio de features (\sqrt{D})
                let mut all_features: Vec<usize> = (0..n_features).collect();
                all_features.shuffle(&mut rng);
                let feature_subset: Vec<usize> = all_features.into_iter().take(max_features).collect();

                let sub_x: Vec<Vec<f64>> = bootstrap_indices.iter().map(|&i| x[i].clone()).collect();
                let sub_y: Vec<u8> = bootstrap_indices.iter().map(|&i| y[i]).collect();
                let sub_w: Vec<f64> = bootstrap_indices.iter().map(|&i| sample_weights[i]).collect();

                DecisionTree::fit(
                    &sub_x,
                    &sub_y,
                    self.max_depth,
                    self.min_samples_split,
                    Some(&feature_subset),
                    Some(&sub_w),
                )
            })
            .collect();

        // 3. Estimación de Importancia de Variables por MDI / frecuencia en splits
        let mut importances = vec![0.0f64; n_features];
        for tree in &trees {
            accumulate_tree_importance(&tree.root, &mut importances);
        }
        let total_imp: f64 = importances.iter().sum();
        let normalized_importances = if total_imp > 0.0 {
            importances.iter().map(|&v| v / total_imp).collect()
        } else {
            vec![1.0 / n_features as f64; n_features]
        };

        Ok(RandomForestModel {
            trees,
            feature_importances: normalized_importances,
        })
    }
}

fn accumulate_tree_importance(node: &crate::ml::decision_tree::TreeNode, importances: &mut [f64]) {
    use crate::ml::decision_tree::TreeNode;
    match node {
        TreeNode::Leaf { .. } => {}
        TreeNode::Split {
            feature_index,
            left,
            right,
            ..
        } => {
            if *feature_index < importances.len() {
                importances[*feature_index] += 1.0;
            }
            accumulate_tree_importance(left, importances);
            accumulate_tree_importance(right, importances);
        }
    }
}
