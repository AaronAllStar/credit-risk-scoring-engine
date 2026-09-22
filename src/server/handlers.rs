use crate::domain::entities::{ApplicationRecord, CustomerFeatureVector, CustomerId};
use crate::domain::risk_score::ScorecardResult;
use crate::server::state::AppState;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub service: &'static str,
    pub total_customers: usize,
    pub active_model: String,
}

pub async fn health_handler(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        service: "Credit Risk Scoring Engine (Rust)",
        total_customers: state.feature_dataset.len(),
        active_model: state.active_model_name.clone(),
    })
}

#[derive(Serialize)]
pub struct ModelInfoResponse {
    pub active_model: String,
    pub evaluated_models: Vec<crate::ml::metrics::ModelMetrics>,
    pub feature_names: Vec<&'static str>,
}

pub async fn model_info_handler(State(state): State<Arc<AppState>>) -> Json<ModelInfoResponse> {
    Json(ModelInfoResponse {
        active_model: state.active_model_name.clone(),
        evaluated_models: (*state.model_metrics).clone(),
        feature_names: CustomerFeatureVector::feature_names(),
    })
}

#[derive(Deserialize)]
pub struct PaginationQuery {
    pub page: Option<usize>,
    pub limit: Option<usize>,
    pub search: Option<String>,
}

#[derive(Serialize)]
pub struct CustomerItemDto {
    pub customer_id: CustomerId,
    pub age: f64,
    pub annual_income: f64,
    pub delinquency_rate: f64,
    pub credit_score: u32,
    pub risk_band: &'static str,
    pub decision: &'static str,
}

#[derive(Serialize)]
pub struct CustomersListResponse {
    pub total: usize,
    pub page: usize,
    pub limit: usize,
    pub items: Vec<CustomerItemDto>,
}

pub async fn list_customers_handler(
    State(state): State<Arc<AppState>>,
    Query(params): Query<PaginationQuery>,
) -> Json<CustomersListResponse> {
    let page = params.page.unwrap_or(1).max(1);
    let limit = params.limit.unwrap_or(20).clamp(1, 100);

    let search_id: Option<CustomerId> = params.search.as_deref().and_then(|s| s.trim().parse().ok());

    let filtered: Vec<&CustomerFeatureVector> = state
        .feature_dataset
        .iter()
        .filter(|c| {
            if let Some(target_id) = search_id {
                c.customer_id.to_string().contains(&target_id.to_string())
            } else {
                true
            }
        })
        .collect();

    let total = filtered.len();
    let start = (page - 1) * limit;
    let end = (start + limit).min(total);

    let mut items = Vec::new();
    if start < total {
        for &vec in &filtered[start..end] {
            let res = state.scoring_engine.evaluate_customer(vec.customer_id, vec);
            items.push(CustomerItemDto {
                customer_id: vec.customer_id,
                age: vec.age_years,
                annual_income: vec.annual_income,
                delinquency_rate: vec.historical_delinquency_rate,
                credit_score: res.credit_score,
                risk_band: res.risk_band.to_str(),
                decision: match res.decision {
                    crate::domain::risk_score::CreditDecision::Approved => "Aprobado",
                    crate::domain::risk_score::CreditDecision::ManualReview => "Revisión Manual",
                    crate::domain::risk_score::CreditDecision::Declined => "Rechazado",
                },
            });
        }
    }

    Json(CustomersListResponse {
        total,
        page,
        limit,
        items,
    })
}

pub async fn get_customer_handler(
    State(state): State<Arc<AppState>>,
    Path(id): Path<CustomerId>,
) -> Result<Json<ScorecardResult>, StatusCode> {
    if let Some(&idx) = state.customer_id_map.get(&id) {
        let vec = &state.feature_dataset[idx];
        let result = state.scoring_engine.evaluate_customer(id, vec);
        Ok(Json(result))
    } else {
        Err(StatusCode::NOT_FOUND)
    }
}

#[derive(Deserialize)]
pub struct ScoreCustomerRequest {
    pub customer_id: Option<CustomerId>,
    pub age_years: f64,
    pub employment_years: f64,
    pub is_unemployed_or_retired: Option<f64>,
    pub annual_income: f64,
    pub income_per_family_member: Option<f64>,
    pub children_count: f64,
    pub family_members_count: f64,
    pub own_car: f64,
    pub own_realty: f64,
    pub has_work_phone: f64,
    pub has_email: f64,
    pub credit_history_length_months: f64,
    pub historical_delinquency_rate: f64,
    pub max_past_due_severity: f64,
    pub delinquency_rate_last_6m: f64,
}

pub async fn score_customer_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<ScoreCustomerRequest>,
) -> Json<ScorecardResult> {
    let customer_id = payload.customer_id.unwrap_or(9999999);
    let fam_members = if payload.family_members_count > 0.0 { payload.family_members_count } else { 1.0 };
    let inc_per_fam = payload
        .income_per_family_member
        .unwrap_or(payload.annual_income / fam_members);
    let unemployed = payload.is_unemployed_or_retired.unwrap_or(if payload.employment_years <= 0.0 { 1.0 } else { 0.0 });

    let vector = CustomerFeatureVector {
        customer_id,
        age_years: payload.age_years,
        employment_years: payload.employment_years,
        is_unemployed_or_retired: unemployed,
        annual_income: payload.annual_income,
        income_per_family_member: inc_per_fam,
        children_count: payload.children_count,
        family_members_count: fam_members,
        own_car: payload.own_car,
        own_realty: payload.own_realty,
        has_work_phone: payload.has_work_phone,
        has_email: payload.has_email,
        credit_history_length_months: payload.credit_history_length_months,
        historical_delinquency_rate: payload.historical_delinquency_rate,
        max_past_due_severity: payload.max_past_due_severity,
        delinquency_rate_last_6m: payload.delinquency_rate_last_6m,
        is_default: None,
    };

    let result = state.scoring_engine.evaluate_customer(customer_id, &vector);
    Json(result)
}

#[derive(Serialize)]
pub struct BatchScoreResponse {
    pub scored_count: usize,
    pub results: Vec<ScorecardResult>,
}

pub async fn batch_score_handler(
    State(state): State<Arc<AppState>>,
    Json(requests): Json<Vec<ScoreCustomerRequest>>,
) -> Json<BatchScoreResponse> {
    use rayon::prelude::*;

    let results: Vec<ScorecardResult> = requests
        .into_par_iter()
        .map(|req| {
            let customer_id = req.customer_id.unwrap_or(9999999);
            let fam_members = if req.family_members_count > 0.0 { req.family_members_count } else { 1.0 };
            let inc_per_fam = req
                .income_per_family_member
                .unwrap_or(req.annual_income / fam_members);
            let unemployed = req.is_unemployed_or_retired.unwrap_or(if req.employment_years <= 0.0 { 1.0 } else { 0.0 });

            let vec = CustomerFeatureVector {
                customer_id,
                age_years: req.age_years,
                employment_years: req.employment_years,
                is_unemployed_or_retired: unemployed,
                annual_income: req.annual_income,
                income_per_family_member: inc_per_fam,
                children_count: req.children_count,
                family_members_count: fam_members,
                own_car: req.own_car,
                own_realty: req.own_realty,
                has_work_phone: req.has_work_phone,
                has_email: req.has_email,
                credit_history_length_months: req.credit_history_length_months,
                historical_delinquency_rate: req.historical_delinquency_rate,
                max_past_due_severity: req.max_past_due_severity,
                delinquency_rate_last_6m: req.delinquency_rate_last_6m,
                is_default: None,
            };

            state.scoring_engine.evaluate_customer(customer_id, &vec)
        })
        .collect();

    Json(BatchScoreResponse {
        scored_count: results.len(),
        results,
    })
}
