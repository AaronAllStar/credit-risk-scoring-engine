use crate::domain::entities::{ApplicationRecord, CreditHistoryRecord, CreditSummary, CustomerFeatureVector, CustomerId};
use std::collections::HashMap;

/// Motor de ingeniería de variables y agregaciones temporales.
pub struct FeatureEngine;

impl FeatureEngine {
    /// Agrega el historial mensual de un cliente para calcular variables de comportamiento y definir la etiqueta de Default.
    pub fn summarize_credit_history(
        customer_id: CustomerId,
        history: &[CreditHistoryRecord],
    ) -> CreditSummary {
        if history.is_empty() {
            return CreditSummary {
                customer_id,
                total_months_observed: 0,
                min_month_balance: 0,
                max_month_balance: 0,
                severe_delinquency_count: 0,
                any_delinquency_count: 0,
                is_bad_customer: false,
                delinquency_rate: 0.0,
                max_severity: 0.0,
                recent_delinquency_rate_6m: 0.0,
                months_since_last_delinquency: None,
            };
        }

        let total_months_observed = history.len();
        let mut min_month_balance = 0;
        let mut max_month_balance = i32::MIN;
        let mut severe_delinquency_count = 0;
        let mut any_delinquency_count = 0;
        let mut max_severity = 0.0f64;
        let mut recent_delinquencies_6m = 0;
        let mut recent_months_6m = 0;
        let mut last_delinquent_month: Option<i32> = None;

        for rec in history {
            if rec.months_balance < min_month_balance {
                min_month_balance = rec.months_balance;
            }
            if rec.months_balance > max_month_balance {
                max_month_balance = rec.months_balance;
            }

            let sev = rec.status.severity_score();
            if sev > max_severity {
                max_severity = sev;
            }

            if rec.status.is_severe_delinquency() {
                severe_delinquency_count += 1;
            }

            if sev >= 1.0 {
                any_delinquency_count += 1;
                match last_delinquent_month {
                    Some(m) if rec.months_balance > m => last_delinquent_month = Some(rec.months_balance),
                    None => last_delinquent_month = Some(rec.months_balance),
                    _ => {}
                }
            }

            // Tendencia reciente en los últimos 6 meses (months_balance >= -5)
            if rec.months_balance >= -5 {
                recent_months_6m += 1;
                if sev >= 1.0 {
                    recent_delinquencies_6m += 1;
                }
            }
        }

        let delinquency_rate = any_delinquency_count as f64 / total_months_observed as f64;
        let recent_delinquency_rate_6m = if recent_months_6m > 0 {
            recent_delinquencies_6m as f64 / recent_months_6m as f64
        } else {
            0.0
        };

        // Regla estándar de Default (Basilea II: mora >= 60 días)
        let is_bad_customer = severe_delinquency_count > 0;

        let months_since_last_delinquency = last_delinquent_month.map(|m| 0 - m);

        CreditSummary {
            customer_id,
            total_months_observed,
            min_month_balance,
            max_month_balance,
            severe_delinquency_count,
            any_delinquency_count,
            is_bad_customer,
            delinquency_rate,
            max_severity,
            recent_delinquency_rate_6m,
            months_since_last_delinquency,
        }
    }

    /// Limpia anomalías demográficas y fusiona con el resumen crediticio para generar el vector final de features.
    pub fn build_feature_vector(
        app: &ApplicationRecord,
        credit_summary: Option<&CreditSummary>,
    ) -> CustomerFeatureVector {
        // Limpieza de edad: DAYS_BIRTH es negativo (e.g. -12005)
        let age_years = (-app.days_birth as f64) / 365.25;

        // Limpieza de empleo: 365243 es un código especial del dataset que indica desempleado / jubilado
        let (employment_years, is_unemployed_or_retired) = if app.days_employed > 0 {
            (0.0, 1.0)
        } else {
            let years = (-app.days_employed as f64) / 365.25;
            (years, 0.0)
        };

        let family_members = if app.family_members_count > 0.0 {
            app.family_members_count
        } else {
            1.0
        };

        let income_per_family_member = app.total_income / family_members;

        let (history_length, hist_delinq_rate, max_sev, recent_delinq_6m, is_default) = match credit_summary {
            Some(sum) => (
                sum.total_months_observed as f64,
                sum.delinquency_rate,
                sum.max_severity,
                sum.recent_delinquency_rate_6m,
                Some(if sum.is_bad_customer { 1.0 } else { 0.0 }),
            ),
            None => (0.0, 0.0, 0.0, 0.0, None),
        };

        CustomerFeatureVector {
            customer_id: app.id,
            age_years,
            employment_years,
            is_unemployed_or_retired,
            annual_income: app.total_income,
            income_per_family_member,
            children_count: app.children_count as f64,
            family_members_count: family_members,
            own_car: if app.own_car { 1.0 } else { 0.0 },
            own_realty: if app.own_realty { 1.0 } else { 0.0 },
            has_work_phone: if app.has_work_phone { 1.0 } else { 0.0 },
            has_email: if app.has_email { 1.0 } else { 0.0 },
            credit_history_length_months: history_length,
            historical_delinquency_rate: hist_delinq_rate,
            max_past_due_severity: max_sev,
            delinquency_rate_last_6m: recent_delinq_6m,
            is_default,
        }
    }

    /// Transforma el catálogo de aplicaciones y crédito en una lista de features unificada.
    pub fn build_dataset(
        applications: &HashMap<CustomerId, ApplicationRecord>,
        credit_history: &HashMap<CustomerId, Vec<CreditHistoryRecord>>,
    ) -> Vec<CustomerFeatureVector> {
        let mut summaries: HashMap<CustomerId, CreditSummary> = HashMap::with_capacity(credit_history.len());
        for (id, records) in credit_history {
            summaries.insert(*id, Self::summarize_credit_history(*id, records));
        }

        let mut dataset = Vec::with_capacity(summaries.len());

        // Hacemos el inner-join entre clientes con historial y datos de solicitud para entrenamiento
        for (id, summary) in &summaries {
            if let Some(app) = applications.get(id) {
                dataset.push(Self::build_feature_vector(app, Some(summary)));
            }
        }

        dataset
    }
}
