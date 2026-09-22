use crate::domain::entities::{ApplicationRecord, CustomerFeatureVector, CustomerId};
use serde::{Deserialize, Serialize};

/// Resultado estandarizado de calificación de riesgo crediticio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorecardResult {
    pub customer_id: CustomerId,
    pub probability_of_default: f64,
    pub credit_score: u32,
    pub risk_band: RiskBand,
    pub decision: CreditDecision,
    pub factor_contributions: Vec<FactorImpact>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskBand {
    Excellent, // 750 - 850 (Riesgo mínimo)
    Good,      // 670 - 749 (Bajo riesgo)
    Fair,      // 580 - 669 (Riesgo moderado)
    Poor,      // 300 - 579 (Alto riesgo / Default probable)
}

impl RiskBand {
    pub fn from_score(score: u32) -> Self {
        match score {
            750..=850 => RiskBand::Excellent,
            670..=749 => RiskBand::Good,
            580..=669 => RiskBand::Fair,
            _ => RiskBand::Poor,
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            RiskBand::Excellent => "Excelente",
            RiskBand::Good => "Bueno",
            RiskBand::Fair => "Moderado",
            RiskBand::Poor => "Alto Riesgo",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CreditDecision {
    Approved,
    ManualReview,
    Declined,
}

impl CreditDecision {
    pub fn from_band(band: RiskBand) -> Self {
        match band {
            RiskBand::Excellent => CreditDecision::Approved,
            RiskBand::Good => CreditDecision::Approved,
            RiskBand::Fair => CreditDecision::ManualReview,
            RiskBand::Poor => CreditDecision::Declined,
        }
    }
}

/// Contribución explicable de una variable a la calificación (Waterfall / Explainability).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorImpact {
    pub feature_name: String,
    pub display_name: String,
    pub feature_value: f64,
    pub impact_direction: String, // "Positivo" (reduce riesgo) o "Negativo" (aumenta riesgo)
    pub relative_importance: f64,
    pub description: String,
}

/// Calibrador de Scorecard bajo el estándar bancario PDO (Points to Double Odds).
/// Formula: Score = Offset + Factor * ln(Odds)
/// donde Odds = (1 - P(Y=1)) / P(Y=1)
pub struct ScorecardCalibrator {
    factor: f64,
    offset: f64,
}

impl ScorecardCalibrator {
    pub fn new(base_score: f64, base_odds: f64, pdo: f64) -> Self {
        let factor = pdo / (2.0f64).ln();
        let offset = base_score - factor * base_odds.ln();
        Self { factor, offset }
    }

    /// Calcula el puntaje de crédito (FICO scale 300 - 850) dada la probabilidad de default.
    pub fn calculate_score(&self, p_default: f64) -> u32 {
        let p_clamped = p_default.clamp(0.0001, 0.9999);
        let odds = (1.0 - p_clamped) / p_clamped;
        let score = self.offset + self.factor * odds.ln();
        (score.round() as u32).clamp(300, 850)
    }
}

impl Default for ScorecardCalibrator {
    fn default() -> Self {
        // Base score = 660 a Odds de 50:1 (tasa de default ~1.96% del portafolio) con PDO = 35 puntos
        Self::new(660.0, 50.0, 35.0)
    }
}
