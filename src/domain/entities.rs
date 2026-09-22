use serde::{Deserialize, Serialize};

/// Identificador de cliente único en la cartera bancaria.
pub type CustomerId = i64;

/// Registro de solicitud demográfico y socioeconómico de un cliente.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplicationRecord {
    pub id: CustomerId,
    pub gender: String,
    pub own_car: bool,
    pub own_realty: bool,
    pub children_count: i32,
    pub total_income: f64,
    pub income_type: String,
    pub education_type: String,
    pub family_status: String,
    pub housing_type: String,
    pub days_birth: i32,
    pub days_employed: i32,
    pub has_mobile: bool,
    pub has_work_phone: bool,
    pub has_phone: bool,
    pub has_email: bool,
    pub occupation_type: Option<String>,
    pub family_members_count: f64,
}

/// Estado mensual de cumplimiento crediticio según el buró de crédito.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MonthStatus {
    PaidOff,         // 'C'
    NoLoan,          // 'X'
    PastDue1To29,    // '0'
    PastDue30To59,   // '1'
    PastDue60To89,   // '2' (Default Threshold)
    PastDue90To119,  // '3'
    PastDue120To149, // '4'
    PastDue150Plus,  // '5'
}

impl MonthStatus {
    pub fn parse(s: &str) -> Self {
        match s.trim() {
            "C" => MonthStatus::PaidOff,
            "X" => MonthStatus::NoLoan,
            "0" => MonthStatus::PastDue1To29,
            "1" => MonthStatus::PastDue30To59,
            "2" => MonthStatus::PastDue60To89,
            "3" => MonthStatus::PastDue90To119,
            "4" => MonthStatus::PastDue120To149,
            "5" => MonthStatus::PastDue150Plus,
            _ => MonthStatus::PastDue1To29,
        }
    }

    /// Determina si el estado corresponde a un evento de mora severa (Basilea II: >= 60 días).
    pub fn is_severe_delinquency(&self) -> bool {
        matches!(
            self,
            MonthStatus::PastDue60To89
                | MonthStatus::PastDue90To119
                | MonthStatus::PastDue120To149
                | MonthStatus::PastDue150Plus
        )
    }

    /// Severidad numérica de atraso para modelado de tasas y medias móviles.
    pub fn severity_score(&self) -> f64 {
        match self {
            MonthStatus::PaidOff | MonthStatus::NoLoan => 0.0,
            MonthStatus::PastDue1To29 => 1.0,
            MonthStatus::PastDue30To59 => 2.0,
            MonthStatus::PastDue60To89 => 3.0,
            MonthStatus::PastDue90To119 => 4.0,
            MonthStatus::PastDue120To149 => 5.0,
            MonthStatus::PastDue150Plus => 6.0,
        }
    }
}

/// Registro mensual del historial crediticio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditHistoryRecord {
    pub id: CustomerId,
    pub months_balance: i32, // 0 = mes actual, -1 = mes pasado...
    pub status: MonthStatus,
}

/// Agregación longitudinal del comportamiento crediticio por cliente.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditSummary {
    pub customer_id: CustomerId,
    pub total_months_observed: usize,
    pub min_month_balance: i32,
    pub max_month_balance: i32,
    pub severe_delinquency_count: usize,
    pub any_delinquency_count: usize,
    pub is_bad_customer: bool, // Target Y in {0, 1}
    pub delinquency_rate: f64,
    pub max_severity: f64,
    pub recent_delinquency_rate_6m: f64,
    pub months_since_last_delinquency: Option<i32>,
}

/// Vector de features procesadas listo para inferencia o entrenamiento.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomerFeatureVector {
    pub customer_id: CustomerId,
    pub age_years: f64,
    pub employment_years: f64,
    pub is_unemployed_or_retired: f64,
    pub annual_income: f64,
    pub income_per_family_member: f64,
    pub children_count: f64,
    pub family_members_count: f64,
    pub own_car: f64,
    pub own_realty: f64,
    pub has_work_phone: f64,
    pub has_email: f64,
    // Features de comportamiento temporal
    pub credit_history_length_months: f64,
    pub historical_delinquency_rate: f64,
    pub max_past_due_severity: f64,
    pub delinquency_rate_last_6m: f64,
    // Target bancario
    pub is_default: Option<f64>,
}

impl CustomerFeatureVector {
    pub fn feature_names() -> Vec<&'static str> {
        vec![
            "age_years",
            "employment_years",
            "is_unemployed_or_retired",
            "annual_income",
            "income_per_family_member",
            "children_count",
            "family_members_count",
            "own_car",
            "own_realty",
            "has_work_phone",
            "has_email",
            "credit_history_length_months",
            "historical_delinquency_rate",
            "max_past_due_severity",
            "delinquency_rate_last_6m",
        ]
    }

    pub fn to_array(&self) -> Vec<f64> {
        vec![
            self.age_years,
            self.employment_years,
            self.is_unemployed_or_retired,
            self.annual_income,
            self.income_per_family_member,
            self.children_count,
            self.family_members_count,
            self.own_car,
            self.own_realty,
            self.has_work_phone,
            self.has_email,
            self.credit_history_length_months,
            self.historical_delinquency_rate,
            self.max_past_due_severity,
            self.delinquency_rate_last_6m,
        ]
    }
}
