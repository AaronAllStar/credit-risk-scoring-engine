#[cfg(test)]
mod tests {
    use credit_risk_engine::domain::entities::{ApplicationRecord, CustomerFeatureVector, MonthStatus};
    use credit_risk_engine::domain::risk_score::{RiskBand, ScorecardCalibrator};
    use credit_risk_engine::ml::metrics::MetricsCalculator;
    use credit_risk_engine::ml::naive_bayes::{GaussianNaiveBayesTrainer};
    use credit_risk_engine::domain::traits::{ModelPredictor, ModelTrainer};

    #[test]
    fn test_month_status_severity() {
        assert_eq!(MonthStatus::PaidOff.severity_score(), 0.0);
        assert_eq!(MonthStatus::NoLoan.severity_score(), 0.0);
        assert_eq!(MonthStatus::PastDue1To29.severity_score(), 1.0);
        assert_eq!(MonthStatus::PastDue60To89.severity_score(), 3.0);
        assert!(MonthStatus::PastDue60To89.is_severe_delinquency());
        assert!(!MonthStatus::PastDue1To29.is_severe_delinquency());
    }

    #[test]
    fn test_scorecard_calibrator() {
        let calibrator = ScorecardCalibrator::default();
        // Probabilidad muy baja de default -> Score alto (FICO excelente > 750)
        let score_safe = calibrator.calculate_score(0.001);
        assert!(score_safe >= 750, "Score seguro esperado >= 750, obtenido {}", score_safe);

        // Probabilidad muy alta de default -> Score bajo (< 580)
        let score_risky = calibrator.calculate_score(0.85);
        assert!(score_risky <= 580, "Score riesgoso esperado <= 580, obtenido {}", score_risky);
    }

    #[test]
    fn test_roc_auc_perfect_separation() {
        let y_true = vec![0, 0, 0, 1, 1, 1];
        let y_probs = vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9];
        let auc = MetricsCalculator::compute_roc_auc(&y_true, &y_probs);
        assert!((auc - 1.0).abs() < 1e-6, "ROC-AUC debería ser 1.0 para separación perfecta");
    }

    #[test]
    fn test_risk_band_mapping() {
        assert_eq!(RiskBand::from_score(800), RiskBand::Excellent);
        assert_eq!(RiskBand::from_score(700), RiskBand::Good);
        assert_eq!(RiskBand::from_score(620), RiskBand::Fair);
        assert_eq!(RiskBand::from_score(500), RiskBand::Poor);
    }
}
