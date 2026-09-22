use crate::domain::entities::{ApplicationRecord, CustomerFeatureVector, CustomerId};
use crate::domain::risk_score::ScorecardResult;
use crate::engine::scoring_engine::ScoringEngine;
use crate::ml::metrics::ModelMetrics;
use std::collections::HashMap;
use std::sync::Arc;

pub struct AppState {
    pub scoring_engine: Arc<ScoringEngine>,
    pub applications: Arc<HashMap<CustomerId, ApplicationRecord>>,
    pub feature_dataset: Arc<Vec<CustomerFeatureVector>>,
    pub customer_id_map: Arc<HashMap<CustomerId, usize>>,
    pub model_metrics: Arc<Vec<ModelMetrics>>,
    pub active_model_name: String,
}
