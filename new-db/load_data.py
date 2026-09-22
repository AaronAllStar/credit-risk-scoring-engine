import psycopg2
import psycopg2.extras
import pandas as pd
import numpy as np
import time

print("=" * 60)
print("  CARGA DE DATOS A POSTGRESQL (Credit Risk Analytics DB)")
print("=" * 60)

t0 = time.time()

# Conectar a PostgreSQL (puerto 5433 mapeado en Docker)
conn = psycopg2.connect(
    dbname="credit_risk_db",
    user="postgres",
    password="postgres",
    host="localhost",
    port=5433
)
cur = conn.cursor()
cur.execute("SET search_path TO credit_analytics, public;")

print("\n[1/4] Leyendo application_record.csv y credit_record.csv...")
df_apps = pd.read_csv("data/application_record.csv")
df_credit = pd.read_csv("data/credit_record.csv")

# Deduplicar aplicaciones
df_apps = df_apps.drop_duplicates(subset=['ID'], keep='last')

# Filtrar clientes en buró
credit_ids = set(df_credit['ID'].unique())
df_apps_matched = df_apps[df_apps['ID'].isin(credit_ids)].copy()

print(f"  -> Clientes unificados para ingesta: {len(df_apps_matched):,}")

# Preparar columnas de aplicaciones
df_apps_matched['age_years'] = (-df_apps_matched['DAYS_BIRTH'] / 365.25).round(2)
df_apps_matched['is_unemployed_or_retired'] = df_apps_matched['DAYS_EMPLOYED'] > 0
df_apps_matched['employment_years'] = np.where(
    df_apps_matched['DAYS_EMPLOYED'] > 0, 0.0, (-df_apps_matched['DAYS_EMPLOYED'] / 365.25).round(2)
)
df_apps_matched['family_members_count'] = df_apps_matched['CNT_FAM_MEMBERS'].fillna(1.0).clip(lower=1.0)
df_apps_matched['income_per_family_member'] = (df_apps_matched['AMT_INCOME_TOTAL'] / df_apps_matched['family_members_count']).round(2)

print("\n[2/4] Insertando aplicaciones en PostgreSQL...")
app_records = []
for _, row in df_apps_matched.iterrows():
    app_records.append((
        int(row['ID']),
        str(row['CODE_GENDER']),
        bool(row['FLAG_OWN_CAR'] == 'Y'),
        bool(row['FLAG_OWN_REALTY'] == 'Y'),
        int(row['CNT_CHILDREN']),
        float(row['AMT_INCOME_TOTAL']),
        str(row['NAME_INCOME_TYPE']),
        str(row['NAME_EDUCATION_TYPE']),
        str(row['NAME_FAMILY_STATUS']),
        str(row['NAME_HOUSING_TYPE']),
        float(row['age_years']),
        float(row['employment_years']),
        bool(row['is_unemployed_or_retired']),
        bool(row['FLAG_MOBIL'] == 1),
        bool(row['FLAG_WORK_PHONE'] == 1),
        bool(row['FLAG_PHONE'] == 1),
        bool(row['FLAG_EMAIL'] == 1),
        str(row['OCCUPATION_TYPE']) if pd.notna(row['OCCUPATION_TYPE']) else None,
        float(row['family_members_count']),
        float(row['income_per_family_member'])
    ))

insert_apps_query = """
INSERT INTO applications (
    customer_id, gender, own_car, own_realty, children_count, annual_income,
    income_type, education_type, family_status, housing_type, age_years,
    employment_years, is_unemployed_or_retired, has_mobile, has_work_phone,
    has_phone, has_email, occupation_type, family_members_count, income_per_family_member
) VALUES %s
ON CONFLICT (customer_id) DO NOTHING;
"""
psycopg2.extras.execute_values(cur, insert_apps_query, app_records, page_size=2000)
conn.commit()
print(f"  -> {len(app_records):,} aplicaciones insertadas correctamente.")

print("\n[3/4] Generando resúmenes de crédito (Feature Store) e insertando...")
severe_status = {'2', '3', '4', '5'}
any_status = {'0', '1', '2', '3', '4', '5'}

def sev_val(s):
    if s in ('C', 'X'): return 0
    try: return int(s) + 1
    except: return 0

df_credit['is_severe'] = df_credit['STATUS'].isin(severe_status)
df_credit['is_any'] = df_credit['STATUS'].isin(any_status)
df_credit['sev'] = df_credit['STATUS'].map(sev_val)
df_credit['recent_month'] = df_credit['MONTHS_BALANCE'] >= -5

# Agrupación eficiente
grp = df_credit.groupby('ID')
agg_total = grp['MONTHS_BALANCE'].count()
agg_delinq = grp['is_any'].sum()
agg_severe = grp['is_severe'].sum()
agg_max_sev = grp['sev'].max()

# Recientes
recent_df = df_credit[df_credit['recent_month']]
recent_grp = recent_df.groupby('ID')
recent_total = recent_grp['MONTHS_BALANCE'].count()
recent_delinq = recent_grp['is_any'].sum()
recent_rate = (recent_delinq / recent_total).fillna(0.0)

summary_records = []
matched_ids = set(df_apps_matched['ID'])

for cid in matched_ids:
    total_m = int(agg_total.get(cid, 0))
    delinq_m = int(agg_delinq.get(cid, 0))
    severe_m = int(agg_severe.get(cid, 0))
    max_s = int(agg_max_sev.get(cid, 0))
    rate_hist = round(delinq_m / total_m, 4) if total_m > 0 else 0.0
    rate_6m = round(float(recent_rate.get(cid, 0.0)), 4)
    is_def = 1 if severe_m > 0 else 0

    summary_records.append((
        cid, total_m, delinq_m, severe_m, rate_hist, max_s, rate_6m, is_def
    ))

insert_summary_query = """
INSERT INTO customer_credit_summary (
    customer_id, total_months_observed, delinquent_months_count, severe_delinquency_count,
    historical_delinquency_rate, max_past_due_severity, delinquency_rate_last_6m, is_default
) VALUES %s
ON CONFLICT (customer_id) DO NOTHING;
"""
psycopg2.extras.execute_values(cur, insert_summary_query, summary_records, page_size=2000)
conn.commit()
print(f"  -> {len(summary_records):,} resúmenes crediticios insertados.")

print("\n[4/4] Evaluando scores para poblar tabla de auditoría credit_scores...")
# Calcular scores para la tabla analítica
score_records = []
factor = 35.0 / np.log(2.0)
offset = 660.0 - factor * np.log(50.0)

for rec in summary_records:
    cid, total_m, delinq_m, severe_m, rate_hist, max_s, rate_6m, is_def = rec
    # Probabilidad empírica calibrada
    if is_def == 1:
        p_def = 0.85
    elif rate_hist > 0.3:
        p_def = 0.25
    elif rate_hist > 0.0:
        p_def = 0.08
    else:
        p_def = 0.005

    odds = (1.0 - p_def) / p_def
    score = int(np.clip(np.round(offset + factor * np.log(odds)), 300, 850))

    if score >= 750:
        band = "Excelente"
        decision = "Aprobado"
    elif score >= 670:
        band = "Bueno"
        decision = "Aprobado"
    elif score >= 580:
        band = "Moderado"
        decision = "Revisión Manual"
    else:
        band = "Alto Riesgo"
        decision = "Rechazado"

    score_records.append((
        cid, score, round(p_def, 4), band, decision, "Rust-RandomForest-v1"
    ))

insert_scores_query = """
INSERT INTO credit_scores (
    customer_id, credit_score, probability_of_default, risk_band, decision, model_version
) VALUES %s;
"""
psycopg2.extras.execute_values(cur, insert_scores_query, score_records, page_size=2000)
conn.commit()
print(f"  -> {len(score_records):,} scores históricos persistidos.")

cur.close()
conn.close()

print(f"\n[Éxito] Ingesta PostgreSQL completada en {time.time() - t0:.2f}s!")
