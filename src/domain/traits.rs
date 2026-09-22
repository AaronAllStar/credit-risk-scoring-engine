use crate::domain::entities::{ApplicationRecord, CustomerFeatureVector, CustomerId};
use crate::domain::risk_score::ScorecardResult;
use std::error::Error;

pub type AppResult<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// Interfaz para transformación y preparación de datasets (Principio Abierto/Cerrado OCP).
pub trait PipelineStep<I, O> {
    fn execute(&self, input: I) -> AppResult<O>;
}

/// Contrato para modelos predictivos de Machine Learning.
pub trait ModelPredictor: Send + Sync {
    fn name(&self) -> &'static str;
    fn predict_probability(&self, features: &[f64]) -> f64;
    fn predict(&self, features: &[f64], threshold: f64) -> u8 {
        if self.predict_probability(features) >= threshold {
            1
        } else {
            0
        }
    }
    fn feature_importances(&self) -> Vec<f64>;
}

/// Contrato para entrenamiento de modelos de Machine Learning.
pub trait ModelTrainer {
    type Model: ModelPredictor;
    fn train(&self, x: &[Vec<f64>], y: &[u8]) -> AppResult<Self::Model>;
}

/// Repositorio de consulta para clientes y sus scores.
pub trait CustomerRepository: Send + Sync {
    fn get_by_id(&self, id: CustomerId) -> AppResult<Option<ApplicationRecord>>;
    fn get_feature_vector(&self, id: CustomerId) -> AppResult<Option<CustomerFeatureVector>>;
    fn list_recent(&self, limit: usize, offset: usize) -> AppResult<Vec<CustomerFeatureVector>>;
    fn total_count(&self) -> usize;
}
