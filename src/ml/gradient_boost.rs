use crate::domain::traits::{AppResult, ModelPredictor, ModelTrainer};
use crate::ml::decision_tree::DecisionTree;
use serde::{Deserialize, Serialize};

/// Gradient Boosted Decision Trees (GBDT) para clasificación binaria.
/// Optimiza la pérdida Negative Log-Likelihood (Deviance) mediante boosting aditivo de pseudorresiduos:
/// r_{im} = - \left[ \frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \right] = y_i - \sigma(F(x_i))
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientBoostingModel {
    pub base_log_odds: f64,
    pub learning_rate: f64,
    pub trees: Vec<DecisionTree>,
    pub feature_importances: Vec<f64>,
}

impl ModelPredictor for GradientBoostingModel {
    fn name(&self) -> &'static str {
        "Gradient Boosted Decision Trees (GBDT)"
    }

    fn predict_probability(&self, features: &[f64]) -> f64 {
        let mut raw_score = self.base_log_odds;
        for tree in &self.trees {
            // El árbol predice una corrección de probabilidad/residual escalada
            let pred = tree.predict_probability(features);
            raw_score += self.learning_rate * (pred - 0.5) * 2.0;
        }

        if raw_score >= 35.0 {
            1.0
        } else if raw_score <= -35.0 {
            0.0
        } else {
            1.0 / (1.0 + (-raw_score).exp())
        }
    }

    fn feature_importances(&self) -> Vec<f64> {
        self.feature_importances.clone()
    }
}

pub struct GradientBoostingTrainer {
    pub n_estimators: usize,
    pub learning_rate: f64,
    pub max_depth: usize,
    pub min_samples_split: usize,
}

impl Default for GradientBoostingTrainer {
    fn default() -> Self {
        Self {
            n_estimators: 20,
            learning_rate: 0.1,
            max_depth: 4,
            min_samples_split: 15,
        }
    }
}

impl ModelTrainer for GradientBoostingTrainer {
    type Model = GradientBoostingModel;

    fn train(&self, x: &[Vec<f64>], y: &[u8]) -> AppResult<Self::Model> {
        let n_samples = x.len();
        if n_samples == 0 {
            return Err("Sin datos para GBDT".into());
        }
        let n_features = x[0].len();

        let n_pos = y.iter().filter(|&&v| v == 1).count().max(1);
        let n_neg = (n_samples - n_pos).max(1);
        let base_log_odds = ((n_pos as f64) / (n_neg as f64)).ln();

        let mut current_predictions = vec![base_log_odds; n_samples];
        let mut trees = Vec::with_capacity(self.n_estimators);
        let mut importances = vec![0.0f64; n_features];

        for _ in 0..self.n_estimators {
            // 1. Calcular pseudorresiduos r_i = y_i - p_i
            let mut pseudo_residuals = Vec::with_capacity(n_samples);
            for i in 0..n_samples {
                let log_odds = current_predictions[i];
                let p = 1.0 / (1.0 + (-log_odds).exp());
                let r = y[i] as f64 - p;
                pseudo_residuals.push(r);
            }

            // 2. Mapear pseudorresiduos a target binario aproximado para el árbol
            let residual_target: Vec<u8> = pseudo_residuals.iter().map(|&r| if r >= 0.0 { 1 } else { 0 }).collect();
            let weights: Vec<f64> = pseudo_residuals.iter().map(|&r| r.abs().max(0.01)).collect();

            let tree = DecisionTree::fit(
                x,
                &residual_target,
                self.max_depth,
                self.min_samples_split,
                None,
                Some(&weights),
            );

            // 3. Actualizar puntuaciones acumuladas
            for i in 0..n_samples {
                let tree_pred = tree.predict_probability(&x[i]);
                current_predictions[i] += self.learning_rate * (tree_pred - 0.5) * 2.0;
            }

            trees.push(tree);
        }

        let total_trees = trees.len() as f64;
        let uniform_imp = 1.0 / n_features as f64;
        let feature_importances = vec![uniform_imp; n_features];

        Ok(GradientBoostingModel {
            base_log_odds,
            learning_rate: self.learning_rate,
            trees,
            feature_importances,
        })
    }
}
