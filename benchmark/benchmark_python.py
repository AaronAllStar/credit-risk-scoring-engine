import time
import os
import sys
import psutil
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

def get_process_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

print("=" * 60)
print("  CREDIT RISK PIPELINE - PYTHON BENCHMARK (Pandas + Scikit-Learn)")
print("=" * 60)

start_mem = get_process_memory_mb()
t0 = time.time()

# 1. Ingesta
print("\n[1/5] Ingestando datasets CSV...")
t_ingest = time.time()
df_apps = pd.read_csv("data/application_record.csv")
df_credit = pd.read_csv("data/credit_record.csv")
dur_ingest = time.time() - t_ingest
print(f"  -> Aplicaciones: {len(df_apps)} | Buró: {len(df_credit)} ({dur_ingest:.2f}s)")

# 2. Feature Engineering & Target
print("\n[2/5] Feature Engineering y Target Construction...")
t_feat = time.time()

# Target Basilea II: mora >= 60 días (STATUS in '2','3','4','5')
severe_status = {'2', '3', '4', '5'}
any_status = {'0', '1', '2', '3', '4', '5'}

def summarize_credit(group):
    total_months = len(group)
    severe_count = group['STATUS'].isin(severe_status).sum()
    any_count = group['STATUS'].isin(any_status).sum()
    
    # Reciente (últimos 6 meses)
    recent = group[group['MONTHS_BALANCE'] >= -5]
    recent_delinq_rate = (recent['STATUS'].isin(any_status).sum() / len(recent)) if len(recent) > 0 else 0.0
    
    # Severidad máxima
    def sev_val(s):
        if s in ('C', 'X'): return 0
        try: return int(s) + 1
        except: return 0
    max_sev = group['STATUS'].map(sev_val).max()
    
    return pd.Series({
        'credit_history_length_months': float(total_months),
        'historical_delinquency_rate': float(any_count / total_months) if total_months > 0 else 0.0,
        'max_past_due_severity': float(max_sev),
        'delinquency_rate_last_6m': float(recent_delinq_rate),
        'is_default': 1.0 if severe_count > 0 else 0.0
    })

credit_summary = df_credit.groupby('ID').apply(summarize_credit).reset_index()

# Limpieza demográfica
df_apps_clean = df_apps.drop_duplicates(subset=['ID'], keep='last').copy()
df_apps_clean['age_years'] = -df_apps_clean['DAYS_BIRTH'] / 365.25
df_apps_clean['is_unemployed_or_retired'] = (df_apps_clean['DAYS_EMPLOYED'] > 0).astype(float)
df_apps_clean['employment_years'] = np.where(df_apps_clean['DAYS_EMPLOYED'] > 0, 0.0, -df_apps_clean['DAYS_EMPLOYED'] / 365.25)
df_apps_clean['family_members_count'] = df_apps_clean['CNT_FAM_MEMBERS'].fillna(1.0).clip(lower=1.0)
df_apps_clean['income_per_family_member'] = df_apps_clean['AMT_INCOME_TOTAL'] / df_apps_clean['family_members_count']
df_apps_clean['annual_income'] = df_apps_clean['AMT_INCOME_TOTAL']
df_apps_clean['children_count'] = df_apps_clean['CNT_CHILDREN'].astype(float)
df_apps_clean['own_car'] = (df_apps_clean['FLAG_OWN_CAR'] == 'Y').astype(float)
df_apps_clean['own_realty'] = (df_apps_clean['FLAG_OWN_REALTY'] == 'Y').astype(float)
df_apps_clean['has_work_phone'] = df_apps_clean['FLAG_WORK_PHONE'].astype(float)
df_apps_clean['has_email'] = df_apps_clean['FLAG_EMAIL'].astype(float)

# Merge
df_merged = pd.merge(df_apps_clean, credit_summary, on='ID', how='inner')
dur_feat = time.time() - t_feat
print(f"  -> Matriz final: {len(df_merged)} filas ({dur_feat:.2f}s)")
print(f"  -> Tasa de default: {df_merged['is_default'].mean()*100:.2f}%")

feature_cols = [
    'age_years', 'employment_years', 'is_unemployed_or_retired', 'annual_income',
    'income_per_family_member', 'children_count', 'family_members_count', 'own_car',
    'own_realty', 'has_work_phone', 'has_email', 'credit_history_length_months',
    'historical_delinquency_rate', 'max_past_due_severity', 'delinquency_rate_last_6m'
]

X = df_merged[feature_cols].values
y = df_merged['is_default'].values.astype(int)

# 3. Train/Test Stratified Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Modelado
print("\n[3/5] Entrenando modelos Scikit-Learn...")

# A. Logistic Regression
t_lr = time.time()
lr = LogisticRegression(class_weight='balanced', max_iter=200, random_state=42)
lr.fit(X_train, y_train)
lr_probs = lr.predict_proba(X_test)[:, 1]
dur_lr = time.time() - t_lr
print(f"  [1] Logistic Regression -> ROC-AUC: {roc_auc_score(y_test, lr_probs):.4f} ({dur_lr:.2f}s)")

# B. Naive Bayes
t_nb = time.time()
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_probs = nb.predict_proba(X_test)[:, 1]
dur_nb = time.time() - t_nb
print(f"  [2] Naive Bayes        -> ROC-AUC: {roc_auc_score(y_test, nb_probs):.4f} ({dur_nb:.2f}s)")

# C. Random Forest
t_rf = time.time()
rf = RandomForestClassifier(n_estimators=25, max_depth=7, min_samples_split=10, class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_test)[:, 1]
dur_rf = time.time() - t_rf
print(f"  [3] Random Forest      -> ROC-AUC: {roc_auc_score(y_test, rf_probs):.4f} ({dur_rf:.2f}s)")

# D. Gradient Boosting
t_gb = time.time()
gb = GradientBoostingClassifier(n_estimators=20, max_depth=4, learning_rate=0.1, random_state=42)
gb.fit(X_train, y_train)
gb_probs = gb.predict_proba(X_test)[:, 1]
dur_gb = time.time() - t_gb
print(f"  [4] Gradient Boosting  -> ROC-AUC: {roc_auc_score(y_test, gb_probs):.4f} ({dur_gb:.2f}s)")

# 5. Benchmark Inferencia
print("\n=== BENCHMARK DE INFERENCIA PYTHON ===")
sample = X_test[0:1]
n_evals = 100000
t_infer = time.time()
for _ in range(n_evals):
    _ = rf.predict_proba(sample)[0, 1]
dur_infer = time.time() - t_infer
per_eval_us = (dur_infer / n_evals) * 1_000_000
throughput = n_evals / dur_infer
print(f"  -> {n_evals} inferencias en {dur_infer:.2f}s")
print(f"  -> Latencia media por cliente: {per_eval_us:.3f} µs")
print(f"  -> Throughput de inferencia: {throughput:.0f} req/s")

peak_mem = get_process_memory_mb()
print(f"\nMemoria consumida: ~{peak_mem - start_mem:.1f} MB (Total proceso: {peak_mem:.1f} MB)")
