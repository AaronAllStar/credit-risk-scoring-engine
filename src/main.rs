mod domain;
mod engine;
mod ml;
mod pipeline;
mod server;

use clap::{Parser, Subcommand};
use domain::entities::CustomerFeatureVector;
use domain::traits::{ModelPredictor, ModelTrainer};
use engine::scoring_engine::ScoringEngine;
use ml::gradient_boost::GradientBoostingTrainer;
use ml::logistic::LogisticRegressionTrainer;
use ml::metrics::MetricsCalculator;
use ml::naive_bayes::GaussianNaiveBayesTrainer;
use ml::random_forest::RandomForestTrainer;
use pipeline::feature_eng::FeatureEngine;
use pipeline::ingestion::CsvIngestor;
use rand::seq::SliceRandom;
use rand::thread_rng;
use server::routes::create_router;
use server::state::AppState;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "credit_risk_engine")]
#[command(about = "High-Performance Credit Risk Scoring Engine in Rust", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Inicia el servidor HTTP REST API y Dashboard Web
    Serve {
        #[arg(short, long, default_value = "3000")]
        port: u16,
    },
    /// Entrena y compara matemáticamente todos los modelos ML
    Train,
    /// Ejecuta pruebas de rendimiento del pipeline en Rust
    Benchmark,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();

    println!("============================================================");
    println!("  CREDIT RISK SCORING ENGINE - RUST CORE (SOLID & CLEAN)     ");
    println!("============================================================");

    let start_total = Instant::now();

    // 1. Ingesta de datos
    println!("\n[1/5] Ingestando datasets CSV...");
    let t_ingest = Instant::now();
    let apps_path = "data/application_record.csv";
    let credit_path = "data/credit_record.csv";

    let applications = CsvIngestor::load_applications(apps_path)?;
    println!(
        "  -> Aplicaciones cargadas: {} registros (tiempo: {:.2?})",
        applications.len(),
        t_ingest.elapsed()
    );

    let t_credit = Instant::now();
    let credit_records = CsvIngestor::load_credit_records(credit_path)?;
    println!(
        "  -> Clientes en buró cargados: {} clientes únicos con historial mensual (tiempo: {:.2?})",
        credit_records.len(),
        t_credit.elapsed()
    );

    // 2. Feature Engineering & Target Construction
    println!("\n[2/5] Ejecutando Feature Engineering y Target Construction...");
    let t_feat = Instant::now();
    let dataset = FeatureEngine::build_dataset(&applications, &credit_records);
    println!(
        "  -> Matriz de features construida: {} muestras unificadas (tiempo: {:.2?})",
        dataset.len(),
        t_feat.elapsed()
    );

    let mut bad_count = 0usize;
    for row in &dataset {
        if row.is_default == Some(1.0) {
            bad_count += 1;
        }
    }
    let default_rate = (bad_count as f64 / dataset.len() as f64) * 100.0;
    println!(
        "  -> Estadísticas del Target: {} morosos severos / {} total ({:.2}% tasa de default)",
        bad_count,
        dataset.len(),
        default_rate
    );

    // 3. Train/Test Stratified Split (80% Train, 20% Test) para evitar Data Leakage
    println!("\n[3/5] Particionando datos estratificados (80% Train / 20% Test)...");
    let mut bad_indices: Vec<usize> = Vec::new();
    let mut good_indices: Vec<usize> = Vec::new();

    for (i, row) in dataset.iter().enumerate() {
        if row.is_default == Some(1.0) {
            bad_indices.push(i);
        } else {
            good_indices.push(i);
        }
    }

    let mut rng = thread_rng();
    bad_indices.shuffle(&mut rng);
    good_indices.shuffle(&mut rng);

    let train_bad_len = (bad_indices.len() as f64 * 0.8).round() as usize;
    let train_good_len = (good_indices.len() as f64 * 0.8).round() as usize;

    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();

    train_indices.extend_from_slice(&bad_indices[..train_bad_len]);
    train_indices.extend_from_slice(&good_indices[..train_good_len]);
    test_indices.extend_from_slice(&bad_indices[train_bad_len..]);
    test_indices.extend_from_slice(&good_indices[train_good_len..]);

    train_indices.shuffle(&mut rng);
    test_indices.shuffle(&mut rng);

    let x_train: Vec<Vec<f64>> = train_indices.iter().map(|&i| dataset[i].to_array()).collect();
    let y_train: Vec<u8> = train_indices.iter().map(|&i| dataset[i].is_default.unwrap_or(0.0) as u8).collect();

    let x_test: Vec<Vec<f64>> = test_indices.iter().map(|&i| dataset[i].to_array()).collect();
    let y_test: Vec<u8> = test_indices.iter().map(|&i| dataset[i].is_default.unwrap_or(0.0) as u8).collect();

    println!("  -> Train samples: {} | Test samples: {}", x_train.len(), x_test.len());

    // 4. Modelado Matemático y Comparación
    println!("\n[4/5] Entrenando y evaluando modelos matemáticos...");

    // A. Regresión Logística L2
    let t_lr = Instant::now();
    let lr_trainer = LogisticRegressionTrainer::default();
    let lr_model = lr_trainer.train(&x_train, &y_train)?;
    let lr_probs: Vec<f64> = x_test.iter().map(|x| lr_model.predict_probability(x)).collect();
    let lr_metrics = MetricsCalculator::evaluate("Logistic Regression", &y_test, &lr_probs, 0.5);
    println!(
        "  [1] Logistic Regression -> ROC-AUC: {:.4} | PR-AUC: {:.4} | Recall: {:.4} | F1: {:.4} ({:.2?})",
        lr_metrics.roc_auc, lr_metrics.pr_auc, lr_metrics.recall, lr_metrics.f1_score, t_lr.elapsed()
    );

    // B. Naive Bayes
    let t_nb = Instant::now();
    let nb_trainer = GaussianNaiveBayesTrainer::default();
    let nb_model = nb_trainer.train(&x_train, &y_train)?;
    let nb_probs: Vec<f64> = x_test.iter().map(|x| nb_model.predict_probability(x)).collect();
    let nb_metrics = MetricsCalculator::evaluate("Naive Bayes", &y_test, &nb_probs, 0.5);
    println!(
        "  [2] Naive Bayes        -> ROC-AUC: {:.4} | PR-AUC: {:.4} | Recall: {:.4} | F1: {:.4} ({:.2?})",
        nb_metrics.roc_auc, nb_metrics.pr_auc, nb_metrics.recall, nb_metrics.f1_score, t_nb.elapsed()
    );

    // C. Random Forest (Paralelo Rayon)
    let t_rf = Instant::now();
    let rf_trainer = RandomForestTrainer {
        n_estimators: 25,
        max_depth: 7,
        min_samples_split: 10,
    };
    let rf_model = rf_trainer.train(&x_train, &y_train)?;
    let rf_probs: Vec<f64> = x_test.iter().map(|x| rf_model.predict_probability(x)).collect();
    let rf_metrics = MetricsCalculator::evaluate("Random Forest", &y_test, &rf_probs, 0.5);
    println!(
        "  [3] Random Forest      -> ROC-AUC: {:.4} | PR-AUC: {:.4} | Recall: {:.4} | F1: {:.4} ({:.2?})",
        rf_metrics.roc_auc, rf_metrics.pr_auc, rf_metrics.recall, rf_metrics.f1_score, t_rf.elapsed()
    );

    // D. Gradient Boosting (GBDT)
    let t_gb = Instant::now();
    let gb_trainer = GradientBoostingTrainer {
        n_estimators: 20,
        learning_rate: 0.1,
        max_depth: 4,
        min_samples_split: 15,
    };
    let gb_model = gb_trainer.train(&x_train, &y_train)?;
    let gb_probs: Vec<f64> = x_test.iter().map(|x| gb_model.predict_probability(x)).collect();
    let gb_metrics = MetricsCalculator::evaluate("Gradient Boosting", &y_test, &gb_probs, 0.5);
    println!(
        "  [4] Gradient Boosting  -> ROC-AUC: {:.4} | PR-AUC: {:.4} | Recall: {:.4} | F1: {:.4} ({:.2?})",
        gb_metrics.roc_auc, gb_metrics.pr_auc, gb_metrics.recall, gb_metrics.f1_score, t_gb.elapsed()
    );

    let all_metrics = vec![lr_metrics, nb_metrics, rf_metrics, gb_metrics];

    // Calculamos medias de features para el explicador
    let n_features = CustomerFeatureVector::feature_names().len();
    let mut feature_means = vec![0.0f64; n_features];
    for row in &dataset {
        let arr = row.to_array();
        for j in 0..n_features {
            feature_means[j] += arr[j];
        }
    }
    for j in 0..n_features {
        feature_means[j] /= dataset.len() as f64;
    }

    // Seleccionamos Random Forest como modelo champion para el motor de producción
    let champion_model: Arc<dyn ModelPredictor> = Arc::new(rf_model);
    let scoring_engine = Arc::new(ScoringEngine::new(champion_model, feature_means));

    // Mapeo indexado en O(1) de CustomerId a posición en el dataset
    let mut customer_id_map = HashMap::with_capacity(dataset.len());
    for (idx, row) in dataset.iter().enumerate() {
        customer_id_map.insert(row.customer_id, idx);
    }

    let app_state = Arc::new(AppState {
        scoring_engine,
        applications: Arc::new(applications),
        feature_dataset: Arc::new(dataset),
        customer_id_map: Arc::new(customer_id_map),
        model_metrics: Arc::new(all_metrics),
        active_model_name: "Random Forest Classifier (Ensemble)".to_string(),
    });

    match cli.command {
        Some(Commands::Benchmark) => {
            println!("\n=== BENCHMARK DE INFERENCIA RUST ===");
            let sample_vec = &app_state.feature_dataset[0];
            let n_evals = 100_000;
            let t_infer = Instant::now();
            for _ in 0..n_evals {
                let _ = app_state.scoring_engine.evaluate_customer(sample_vec.customer_id, sample_vec);
            }
            let total_dur = t_infer.elapsed();
            let per_eval_us = total_dur.as_micros() as f64 / n_evals as f64;
            let throughput = n_evals as f64 / total_dur.as_secs_f64();
            println!("  -> {} inferencias completadas en {:.2?}", n_evals, total_dur);
            println!("  -> Latencia media por cliente: {:.3} µs", per_eval_us);
            println!("  -> Throughput de inferencia: {:.0} req/s", throughput);
        }
        Some(Commands::Train) => {
            println!("\n[Listo] Modelos entrenados y validados exitosamente en {:.2?}", start_total.elapsed());
        }
        Some(Commands::Serve { port }) => {
            start_server(app_state, port).await?;
        }
        None => {
            start_server(app_state, 3000).await?;
        }
    }

    Ok(())
}

async fn start_server(app_state: Arc<AppState>, port: u16) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("\n[5/5] Iniciando Servidor HTTP REST API en puerto {}...", port);
    let app = create_router(app_state);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    println!("  -> API lista en: http://127.0.0.1:{}", port);
    println!("  -> Dashboard Web FinTech en: http://127.0.0.1:{}/dashboard/index.html", port);
    println!("  -> Health endpoint: http://127.0.0.1:{}/health", port);
    println!("  -> Model info: http://127.0.0.1:{}/model/info", port);
    println!("  -> Customers list: http://127.0.0.1:{}/customers?limit=10", port);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
