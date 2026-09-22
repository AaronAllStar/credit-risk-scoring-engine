use crate::server::handlers::{
    batch_score_handler, get_customer_handler, health_handler, list_customers_handler,
    model_info_handler, score_customer_handler,
};
use crate::server::state::AppState;
use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::services::ServeDir;

pub fn create_router(state: Arc<AppState>) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(health_handler))
        .route("/model/info", get(model_info_handler))
        .route("/customers", get(list_customers_handler))
        .route("/customers/{id}", get(get_customer_handler))
        .route("/score", post(score_customer_handler))
        .route("/batch-score", post(batch_score_handler))
        .nest_service("/dashboard", ServeDir::new("dashboard"))
        .layer(cors)
        .with_state(state)
}
