-- Schema de Base de Datos para Análisis de Riesgo Crediticio (Credit Risk Analytics)
-- Optimizado para analistas de datos, BI, consultas analíticas y Data Science

CREATE SCHEMA IF NOT EXISTS credit_analytics;
SET search_path TO credit_analytics, public;

-- 1. Tabla de Solicitudes y Demografía
CREATE TABLE IF NOT EXISTS applications (
    customer_id BIGINT PRIMARY KEY,
    gender VARCHAR(2) NOT NULL,
    own_car BOOLEAN NOT NULL,
    own_realty BOOLEAN NOT NULL,
    children_count INT NOT NULL,
    annual_income NUMERIC(14, 2) NOT NULL,
    income_type VARCHAR(64) NOT NULL,
    education_type VARCHAR(64) NOT NULL,
    family_status VARCHAR(64) NOT NULL,
    housing_type VARCHAR(64) NOT NULL,
    age_years NUMERIC(5, 2) NOT NULL,
    employment_years NUMERIC(5, 2) NOT NULL,
    is_unemployed_or_retired BOOLEAN NOT NULL,
    has_mobile BOOLEAN NOT NULL,
    has_work_phone BOOLEAN NOT NULL,
    has_phone BOOLEAN NOT NULL,
    has_email BOOLEAN NOT NULL,
    occupation_type VARCHAR(64),
    family_members_count NUMERIC(4, 1) NOT NULL,
    income_per_family_member NUMERIC(14, 2) NOT NULL
);

-- 2. Tabla Longitudinal de Registros Mensuales del Buró
CREATE TABLE IF NOT EXISTS credit_bureau_records (
    record_id BIGSERIAL PRIMARY KEY,
    customer_id BIGINT NOT NULL,
    months_balance INT NOT NULL,
    status_code VARCHAR(2) NOT NULL,
    severity_level INT NOT NULL,
    is_past_due BOOLEAN NOT NULL,
    is_severe_delinquency BOOLEAN NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bureau_cust_months ON credit_bureau_records(customer_id, months_balance);
CREATE INDEX IF NOT EXISTS idx_bureau_status ON credit_bureau_records(status_code);

-- 3. Tabla Resumen de Comportamiento Crediticio (Feature Store)
CREATE TABLE IF NOT EXISTS customer_credit_summary (
    customer_id BIGINT PRIMARY KEY,
    total_months_observed INT NOT NULL,
    delinquent_months_count INT NOT NULL,
    severe_delinquency_count INT NOT NULL,
    historical_delinquency_rate NUMERIC(6, 4) NOT NULL,
    max_past_due_severity INT NOT NULL,
    delinquency_rate_last_6m NUMERIC(6, 4) NOT NULL,
    is_default INT NOT NULL, -- Target Basilea II: 1 = Default, 0 = Solvente
    CONSTRAINT fk_summary_app FOREIGN KEY (customer_id) REFERENCES applications(customer_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_summary_default ON customer_credit_summary(is_default);

-- 4. Tabla de Calificaciones y Scores (Audit & Scoring Log)
CREATE TABLE IF NOT EXISTS credit_scores (
    score_id BIGSERIAL PRIMARY KEY,
    customer_id BIGINT NOT NULL,
    credit_score INT NOT NULL CHECK (credit_score BETWEEN 300 AND 850),
    probability_of_default NUMERIC(7, 5) NOT NULL,
    risk_band VARCHAR(32) NOT NULL,
    decision VARCHAR(32) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    scored_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_scores_customer ON credit_scores(customer_id);
CREATE INDEX IF NOT EXISTS idx_scores_band ON credit_scores(risk_band);

-- 5. Vistas Analíticas para Data Analysts y Business Intelligence (BI)

-- Vista 360 del Cliente y su Perfil de Riesgo
CREATE OR REPLACE VIEW v_customer_risk_profile AS
SELECT 
    a.customer_id,
    a.gender,
    a.age_years,
    a.employment_years,
    a.annual_income,
    a.income_per_family_member,
    a.income_type,
    a.education_type,
    a.housing_type,
    a.family_status,
    a.own_car,
    a.own_realty,
    s.total_months_observed,
    s.historical_delinquency_rate,
    s.max_past_due_severity,
    s.delinquency_rate_last_6m,
    s.is_default,
    sc.credit_score,
    sc.probability_of_default,
    sc.risk_band,
    sc.decision
FROM applications a
JOIN customer_credit_summary s ON a.customer_id = s.customer_id
LEFT JOIN LATERAL (
    SELECT credit_score, probability_of_default, risk_band, decision
    FROM credit_scores cs
    WHERE cs.customer_id = a.customer_id
    ORDER BY scored_at DESC
    LIMIT 1
) sc ON true;

-- Vista Agregada de Métricas de Cartera por Banda de Riesgo
CREATE OR REPLACE VIEW v_portfolio_risk_kpis AS
SELECT 
    sc.risk_band,
    sc.decision,
    COUNT(*) AS total_customers,
    ROUND(AVG(a.annual_income), 2) AS avg_annual_income,
    ROUND(AVG(a.age_years), 1) AS avg_age,
    ROUND(AVG(s.historical_delinquency_rate) * 100, 2) AS avg_delinquency_rate_pct,
    ROUND(AVG(sc.credit_score), 0) AS avg_credit_score,
    ROUND(AVG(s.is_default) * 100, 2) AS empirical_default_rate_pct
FROM applications a
JOIN customer_credit_summary s ON a.customer_id = s.customer_id
JOIN credit_scores sc ON a.customer_id = sc.customer_id
GROUP BY sc.risk_band, sc.decision
ORDER BY avg_credit_score DESC;
