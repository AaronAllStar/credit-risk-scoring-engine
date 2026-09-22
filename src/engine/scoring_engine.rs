use crate::domain::entities::{CustomerFeatureVector, CustomerId};
use crate::domain::risk_score::{CreditDecision, FactorImpact, RiskBand, ScorecardCalibrator, ScorecardResult};
use crate::domain::traits::ModelPredictor;
use std::sync::Arc;

/// Motor central de inferencia, calibración PDO y explicabilidad de variables.
pub struct ScoringEngine {
    model: Arc<dyn ModelPredictor>,
    calibrator: ScorecardCalibrator,
    feature_means: Vec<f64>,
}

impl ScoringEngine {
    pub fn new(model: Arc<dyn ModelPredictor>, feature_means: Vec<f64>) -> Self {
        Self {
            model,
            calibrator: ScorecardCalibrator::default(),
            feature_means,
        }
    }

    pub fn evaluate_customer(&self, customer_id: CustomerId, vector: &CustomerFeatureVector) -> ScorecardResult {
        let features = vector.to_array();
        let p_default = self.model.predict_probability(&features);
        let credit_score = self.calibrator.calculate_score(p_default);
        let risk_band = RiskBand::from_score(credit_score);
        let decision = CreditDecision::from_band(risk_band);

        // Explicabilidad de variables (Waterfall / Factor Impact)
        let importances = self.model.feature_importances();
        let names = CustomerFeatureVector::feature_names();
        let mut factor_contributions = Vec::new();

        for (i, &name) in names.iter().enumerate() {
            let val = features.get(i).copied().unwrap_or(0.0);
            let mean = self.feature_means.get(i).copied().unwrap_or(val);
            let imp = importances.get(i).copied().unwrap_or(0.05);

            let (display_name, impact_dir, desc) = match name {
                "age_years" => (
                    "Edad del Cliente",
                    if val >= mean { "Positivo" } else { "Negativo" },
                    format!("{:.1} años (Promedio: {:.1} años)", val, mean),
                ),
                "employment_years" => (
                    "Años de Empleo",
                    if val >= mean { "Positivo" } else { "Negativo" },
                    format!("{:.1} años de estabilidad laboral", val),
                ),
                "annual_income" => (
                    "Ingreso Total Anual",
                    if val >= mean { "Positivo" } else { "Negativo" },
                    format!("${:.0} ingresos declarados", val),
                ),
                "income_per_family_member" => (
                    "Ingreso por Dependiente",
                    if val >= mean { "Positivo" } else { "Negativo" },
                    format!("${:.0} / miembro", val),
                ),
                "credit_history_length_months" => (
                    "Madurez del Historial",
                    if val >= 12.0 { "Positivo" } else { "Negativo" },
                    format!("{:.0} meses observados en buró", val),
                ),
                "historical_delinquency_rate" => (
                    "Tasa de Mora Histórica",
                    if val <= 0.05 { "Positivo" } else { "Negativo" },
                    format!("{:.1}% meses con mora registrada", val * 100.0),
                ),
                "max_past_due_severity" => (
                    "Atraso Máximo Registrado",
                    if val <= 1.0 { "Positivo" } else { "Negativo" },
                    format!("Nivel de severidad {:.0} en buró", val),
                ),
                "delinquency_rate_last_6m" => (
                    "Mora Reciente (Últimos 6m)",
                    if val == 0.0 { "Positivo" } else { "Negativo" },
                    format!("{:.1}% en últimos meses", val * 100.0),
                ),
                _ => (
                    name,
                    if val > 0.0 { "Positivo" } else { "Neutro" },
                    format!("Valor: {:.2}", val),
                ),
            };

            factor_contributions.push(FactorImpact {
                feature_name: name.to_string(),
                display_name: display_name.to_string(),
                feature_value: val,
                impact_direction: impact_dir.to_string(),
                relative_importance: imp,
                description: desc,
            });
        }

        // Ordenar factores por mayor impacto relativo
        factor_contributions.sort_by(|a, b| {
            b.relative_importance
                .partial_cmp(&a.relative_importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        ScorecardResult {
            customer_id,
            probability_of_default: p_default,
            credit_score,
            risk_band,
            decision,
            factor_contributions,
        }
    }
}
