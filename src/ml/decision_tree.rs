use serde::{Deserialize, Serialize};

/// Nodo de un árbol de decisión para clasificación / estimación de probabilidades.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeNode {
    Leaf {
        probability: f64,
        samples: usize,
    },
    Split {
        feature_index: usize,
        threshold: f64,
        left: Box<TreeNode>,
        right: Box<TreeNode>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    pub root: TreeNode,
    pub max_depth: usize,
    pub min_samples_split: usize,
}

impl DecisionTree {
    pub fn predict_probability(&self, features: &[f64]) -> f64 {
        Self::predict_node(&self.root, features)
    }

    fn predict_node(node: &TreeNode, features: &[f64]) -> f64 {
        match node {
            TreeNode::Leaf { probability, .. } => *probability,
            TreeNode::Split {
                feature_index,
                threshold,
                left,
                right,
            } => {
                if features.get(*feature_index).copied().unwrap_or(0.0) <= *threshold {
                    Self::predict_node(left, features)
                } else {
                    Self::predict_node(right, features)
                }
            }
        }
    }

    pub fn fit(
        x: &[Vec<f64>],
        y: &[u8],
        max_depth: usize,
        min_samples_split: usize,
        features_subset: Option<&[usize]>,
        sample_weights: Option<&[f64]>,
    ) -> Self {
        let indices: Vec<usize> = (0..x.len()).collect();
        let root = Self::build_tree(
            x,
            y,
            &indices,
            0,
            max_depth,
            min_samples_split,
            features_subset,
            sample_weights,
        );
        DecisionTree {
            root,
            max_depth,
            min_samples_split,
        }
    }

    fn build_tree(
        x: &[Vec<f64>],
        y: &[u8],
        indices: &[usize],
        depth: usize,
        max_depth: usize,
        min_samples_split: usize,
        features_subset: Option<&[usize]>,
        sample_weights: Option<&[f64]>,
    ) -> TreeNode {
        let n_samples = indices.len();
        if n_samples == 0 {
            return TreeNode::Leaf {
                probability: 0.0,
                samples: 0,
            };
        }

        // Calcular probabilidad ponderada
        let (prob, total_weight) = match sample_weights {
            Some(w) => {
                let mut sum_pos = 0.0;
                let mut sum_all = 0.0;
                for &idx in indices {
                    let weight = w[idx];
                    sum_all += weight;
                    if y[idx] == 1 {
                        sum_pos += weight;
                    }
                }
                let p = if sum_all > 0.0 { sum_pos / sum_all } else { 0.0 };
                (p, sum_all)
            }
            None => {
                let sum_pos: usize = indices.iter().map(|&i| y[i] as usize).sum();
                (sum_pos as f64 / n_samples as f64, n_samples as f64)
            }
        };

        // Condición de parada (nodo hoja)
        if depth >= max_depth || n_samples < min_samples_split || prob <= 1e-6 || prob >= 1.0 - 1e-6 {
            return TreeNode::Leaf {
                probability: prob,
                samples: n_samples,
            };
        }

        let n_features = x[0].len();
        let default_subset: Vec<usize> = (0..n_features).collect();
        let allowed_features = features_subset.unwrap_or(&default_subset);

        let mut best_gain = 0.0f64;
        let mut best_feature = 0usize;
        let mut best_threshold = 0.0f64;
        let mut best_left_indices = Vec::new();
        let mut best_right_indices = Vec::new();

        let current_impurity = Self::gini_impurity(indices, y, sample_weights);

        for &feat_idx in allowed_features {
            // Probar percentiles / puntos de división representativos
            let mut values: Vec<f64> = indices.iter().map(|&i| x[i][feat_idx]).collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            values.dedup();

            // Muestrear candidatos si hay muchos valores continuos
            let step = (values.len() / 15).max(1);
            for k in (0..values.len()).step_by(step) {
                let threshold = values[k];
                let mut left = Vec::new();
                let mut right = Vec::new();

                for &idx in indices {
                    if x[idx][feat_idx] <= threshold {
                        left.push(idx);
                    } else {
                        right.push(idx);
                    }
                }

                if left.is_empty() || right.is_empty() {
                    continue;
                }

                let imp_left = Self::gini_impurity(&left, y, sample_weights);
                let imp_right = Self::gini_impurity(&right, y, sample_weights);

                let weight_left = left.len() as f64 / n_samples as f64;
                let weight_right = right.len() as f64 / n_samples as f64;
                let gain = current_impurity - (weight_left * imp_left + weight_right * imp_right);

                if gain > best_gain {
                    best_gain = gain;
                    best_feature = feat_idx;
                    best_threshold = threshold;
                    best_left_indices = left;
                    best_right_indices = right;
                }
            }
        }

        if best_gain <= 1e-7 || best_left_indices.is_empty() || best_right_indices.is_empty() {
            return TreeNode::Leaf {
                probability: prob,
                samples: n_samples,
            };
        }

        let left_child = Self::build_tree(
            x,
            y,
            &best_left_indices,
            depth + 1,
            max_depth,
            min_samples_split,
            features_subset,
            sample_weights,
        );

        let right_child = Self::build_tree(
            x,
            y,
            &best_right_indices,
            depth + 1,
            max_depth,
            min_samples_split,
            features_subset,
            sample_weights,
        );

        TreeNode::Split {
            feature_index: best_feature,
            threshold: best_threshold,
            left: Box::new(left_child),
            right: Box::new(right_child),
        }
    }

    fn gini_impurity(indices: &[usize], y: &[u8], sample_weights: Option<&[f64]>) -> f64 {
        if indices.is_empty() {
            return 0.0;
        }

        let (p1, p0) = match sample_weights {
            Some(w) => {
                let mut sum_pos = 0.0;
                let mut sum_all = 0.0;
                for &idx in indices {
                    let weight = w[idx];
                    sum_all += weight;
                    if y[idx] == 1 {
                        sum_pos += weight;
                    }
                }
                if sum_all <= 0.0 {
                    return 0.0;
                }
                let p = sum_pos / sum_all;
                (p, 1.0 - p)
            }
            None => {
                let sum_pos: usize = indices.iter().map(|&i| y[i] as usize).sum();
                let p = sum_pos as f64 / indices.len() as f64;
                (p, 1.0 - p)
            }
        };

        1.0 - (p1 * p1 + p0 * p0)
    }
}
