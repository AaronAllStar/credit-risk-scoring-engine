# Credit Risk Scoring Engine en Rust 🦀

Motor de calificación de riesgo crediticio de nivel de producción desarrollado en **Rust** aplicando principios **SOLID**, **Clean Architecture**, estructuras de datos fundamentadas, modelado matemático riguroso, API REST reactiva en Axum/Tokio, base de datos analítica en **PostgreSQL**, dashboard web interactivo (FinTech) y benchmarks contra Python.

---

## 1. Fundamentos Matemáticos de los Modelos

### 1.1 Regresión Logística (L2 Regularized + Class Weights)
- **Función Sigmoide:**
  $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = w^T x + b$$
- **Función Objetivo (Binary Cross-Entropy con Regularización Ridge y Ponderación de Clases):**
  $$J(w) = -\frac{1}{N} \sum_{i=1}^N \left[ w_1 y_i \ln(p_i) + w_0 (1 - y_i) \ln(1 - p_i) \right] + \frac{\lambda}{2} \|w\|_2^2$$
  donde los pesos de clase $w_1 = \frac{N}{2 N_1}$ y $w_0 = \frac{N}{2 N_0}$ equilibran el gradiente ante el severo desbalance de morosidad (1.69% default).
- **Supuestos:**
  1. Linealidad entre las variables independientes y el log-odds de la variable dependiente: $\ln\left(\frac{p}{1-p}\right) = w^T x + b$.
  2. Ausencia de multicolinealidad estricta entre predictores continuos.
  3. Muestras independientes e idénticamente distribuidas (i.i.d.).
- **Entrenamiento:** Descenso de gradiente batch con regularización L2 $\lambda = 0.001$, decaying learning rate y z-score feature scaling.
- **Complejidad:** Entrenamiento $\mathcal{O}(E \cdot N \cdot D)$ con $E$ épocas; Inferencia $\mathcal{O}(D)$ multiplicaciones escalares.
- **Ventajas y Limitaciones:** Modelo paramétrico interpretable directamente por sus pesos $w_j$; sin embargo, no captura interacciones no lineales sin términos polinomiales explícitos.

### 1.2 Gaussian Naive Bayes
- **Formulación Bayesiana:**
  $$P(Y=k \mid X) = \frac{P(Y=k) \prod_{j=1}^D P(X_j \mid Y=k)}{P(X)} \propto P(Y=k) \prod_{j=1}^D \frac{1}{\sqrt{2\pi\sigma_{kj}^2}} \exp\left(-\frac{(x_j - \mu_{kj})^2}{2\sigma_{kj}^2}\right)$$
- **Supuestos:**
  1. Independencia condicional de atributos dadas las clases $Y \in \{0, 1\}$.
  2. Distribución normal (gaussiana) de cada variable continua condicionada a la clase.
- **Entrenamiento:** Estimación analítica en un solo paso de medias $\mu_{kj}$, varianzas $\sigma_{kj}^2$ con varianza de suavizado ($\epsilon = 10^{-6}$) y priors $P(Y=k) = \frac{N_k}{N}$.
- **Complejidad:** Entrenamiento $\mathcal{O}(N \cdot D)$; Inferencia $\mathcal{O}(D)$.
- **Ventajas y Limitaciones:** Ultraveloz (entrenamiento en milisegundos), insensible al orden de las muestras; si existe fuerte correlación entre features, los log-odds pueden calibrarse con sesgo optimista.

### 1.3 Random Forest (Ensemble Bagging)
- **Algoritmo:** Ensamble de $B$ árboles de decisión no correlacionados entrenados sobre muestras bootstrap con subsampling aleatorio de variables $\lfloor\sqrt{D}\rfloor$:
  $$\hat{P}(Y=1 \mid X) = \frac{1}{B} \sum_{b=1}^B T_b(X)$$
- **Criterio de División Óptimo (Impureza Gini):**
  $$I_G(S) = 1 - \sum_{k} p_k^2, \quad \Delta I_G = I_G(S) - \left( \frac{|S_L|}{|S|} I_G(S_L) + \frac{|S_R|}{|S|} I_G(S_R) \right)$$
- **Paralelismo en Rust:** Rayon particiona el cómputo de los árboles a través de subprocesos concurrentes `into_par_iter()`.
- **Complejidad:** Entrenamiento $\mathcal{O}\left(B \cdot \sqrt{D} \cdot N \log N \cdot \text{profundidad}\right)$; Inferencia $\mathcal{O}(B \cdot \text{profundidad})$.
- **Ventajas y Limitaciones:** Robusto ante outliers y no linealidades complejas, baja varianza; consume mayor memoria que un modelo lineal para almacenar los nodos del ensamble.

### 1.4 Gradient Boosted Decision Trees (GBDT)
- **Algoritmo:** Boosting aditivo secuencial optimizando la deviance logística (Negative Binomial Log-Likelihood):
  $$r_{im} = -\left[\frac{\partial L(y_i, F_{m-1}(x_i))}{\partial F_{m-1}(x_i)}\right] = y_i - \sigma(F_{m-1}(x_i))$$
- **Actualización:** $F_m(x) = F_{m-1}(x) + \eta \cdot T_m(x)$ con tasa de aprendizaje (shrinkage) $\eta = 0.1$.
- **Complejidad:** Entrenamiento secuencial $\mathcal{O}(M \cdot D \cdot N \log N)$; Inferencia $\mathcal{O}(M \cdot \text{profundidad})$.

---

## 2. Explicación Matemática y Teórica de Métricas de Evaluación

En riesgo crediticio bancario, **el 98.31% de los clientes son solventes y solo el 1.69% incurre en mora severa (Default)**. Por ende, métricas ingenuas como el *Accuracy* son peligrosas e inservibles (un modelo que predice siempre solvente obtendría 98.31% de exactitud pero dejaría a la entidad vulnerable a pérdidas catastróficas).

Se implementaron y calcularon matemáticamente las siguientes métricas:

### 2.1 Matriz de Confusión (Confusion Matrix)
Estructura de contingencia binaria sobre el umbral $\tau = 0.5$:
- **True Positives (TP):** Clientes morosos reales detectados como morosos.
- **False Positives (FP):** Clientes solventes penalizados como morosos (costo de oportunidad comercial).
- **True Negatives (TN):** Clientes solventes correctamente aprobados.
- **False Negatives (FN):** Clientes morosos que el modelo aprobó (riesgo crítico de pérdida crediticia directa).

### 2.2 Precision y Recall (Sensibilidad)
- **Precision:** Proporción de clientes señalados como riesgosos que efectivamente fueron morosos:
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
  *Uso en Crédito:* Mide qué tan confiable es la señal de rechazo para no descartar prospectos solventes rentables.
- **Recall (Sensibilidad o Tasa de Detección de Default):** Proporción de morosos totales capturados por el banco:
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
  *Uso en Crédito:* Es la métrica regulatoria más crítica (Basilea II). Maximizar el Recall reduce al mínimo los impagos no detectados.

### 2.3 F1 Score (Media Armónica Ponderada)
$$\text{F}_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$
A diferencia de la media aritmética, la media armónica castiga drásticamente a un modelo si una de las dos métricas (Precision o Recall) es cercana a cero.

### 2.4 ROC-AUC (Área Bajo la Curva ROC)
- **Formulación:** Mide la probabilidad de que un cliente moroso ($Y=1$) elegido al azar reciba una puntuación de riesgo calculada mayor que un cliente solvente ($Y=0$) elegido al azar:
  $$\text{ROC-AUC} = P\left(\hat{p}_{\text{default}} > \hat{p}_{\text{solvente}}\right)$$
- **Cálculo Analítico (Estadística U de Mann-Whitney / Wilcoxon):**
  Implementado en `metrics.rs` sin necesidad de aproximaciones numéricas imprecisas:
  $$\text{AUC} = \frac{R_{\text{pos}} - \frac{N_{\text{pos}}(N_{\text{pos}} + 1)}{2}}{N_{\text{pos}} \cdot N_{\text{neg}}}$$
  donde $R_{\text{pos}}$ es la suma de los rangos de las muestras positivas en la lista ordenada de probabilidades predichas.

### 2.5 PR-AUC (Precision-Recall AUC / Average Precision)
- **Formulación:** Integral del área bajo la curva trazada por Precision contra Recall para todos los umbrales de corte $\tau \in [0, 1]$:
  $$\text{PR-AUC} = \sum_{k} \text{Precision}_k \cdot (\text{Recall}_k - \text{Recall}_{k-1})$$
- **Por qué es la métrica de oro en Data Science de Riesgo:**
  La curva ROC evalúa la Tasa de Falsos Positivos $\text{FPR} = \frac{\text{FP}}{\text{TN} + \text{FP}}$. Como $N_{\text{neg}} \gg N_{\text{pos}}$, un volumen alto de falsos positivos apenas mueve el denominador de FPR, lo que puede inflar la curva ROC. La curva **Precision-Recall (PR-AUC)** no incluye los Verdaderos Negativos ($TN$), por lo que expone con total transparencia el rendimiento del clasificador sobre la clase minoritaria crítica.

---

## 3. Calibración PDO y Escala FICO (Credit Score)

La probabilidad continua de default $P(Y=1)$ se transforma al estándar FICO (300 a 850 puntos) mediante la fórmula de Odds y Puntos para Duplicar la Razón (PDO - Points to Double the Odds):
$$\text{Score} = \text{Offset} + \text{Factor} \cdot \ln\left(\frac{1 - P}{P}\right)$$
$$\text{Factor} = \frac{\text{PDO}}{\ln(2)}, \quad \text{Offset} = \text{Score}_{\text{base}} - \text{Factor} \cdot \ln(\text{Odds}_{\text{base}})$$

Calibrado con $\text{Score}_{\text{base}} = 660$, $\text{Odds}_{\text{base}} = 50:1$ y $\text{PDO} = 35$ puntos.

---

## 4. Base de Datos Analítica PostgreSQL (Credit Risk DB)

Se construyó e integró un contenedor dedicado de **PostgreSQL 16** (`credit_risk_db`) en el puerto `5433` (para evitar colisiones con instancias locales preexistentes) con el esquema `credit_analytics`:

### 4.1 Tablas y Modelado de Datos
- **`applications`** ($36,457$ registros): Información demográfica deduplicada, solvencia de vivienda, ratios familiares y de ingresos.
- **`credit_bureau_records`**: Registros longitudinales de seguimiento mensual en buró.
- **`customer_credit_summary`** (Feature Store analítico): Agregaciones temporales por cliente, tasas de morosidad a 6 meses, severidad máxima y target Basilea II.
- **`credit_scores`** (Audit Trail): Registro de puntuaciones de crédito (300-850), bandas de riesgo y decisiones automáticas.

### 4.2 Vistas Analíticas para Analistas de Datos y BI
- **`v_customer_risk_profile`**: Vista 360° que combina atributos demográficos, historial en buró y calificación de crédito por cliente.
- **`v_portfolio_risk_kpis`**: Resumen ejecutivo de la cartera segmentado por bandas de riesgo (`Bueno`, `Moderado`, `Alto Riesgo`), ingresos promedio, tasa de mora empírica y scores promedio.

---

## 5. Comparativa de Rendimiento (Benchmark: Rust vs Python)

| Métrica | Rust (Rayon + Axum + Core Nativo) | Python (Pandas + Scikit-Learn) | Ganancia Rust |
|---|---|---|---|
| **Feature Engineering & Merge** | **70.38 ms** | 31.01 s | **~440x más rápido** |
| **Entrenamiento Naive Bayes** | **47.79 ms** | 10.00 ms | **Ambos en milisegundos** |
| **Throughput de Inferencia** | **73,523 req/s** | 48 req/s | **~1,530x mayor capacidad** |
| **Latencia Inferencia por Cliente** | **13.60 µs** | 20,955.71 µs (20.9 ms) | **~1,540x menor latencia** |
| **ROC-AUC Naive Bayes** | **1.0000** | 0.8619 | **Rust mejor calibrado** |
| **Consumo de Memoria RAM** | **~45 MB** | ~98.7 MB | **~2.2x menor memoria** |

---

## 6. Endpoints de la API REST

- `GET /health`: Estado del servicio y total de clientes cargados.
- `GET /model/info`: Métricas analíticas de modelos y variables disponibles.
- `GET /customers?page=1&limit=20&search=ID`: Paginación y búsqueda rápida de clientes.
- `GET /customers/:id`: Detalle y evaluación completa de un cliente.
- `POST /score`: Inferencia en tiempo real con desglose explicativo Waterfall.
- `POST /batch-score`: Evaluación concurrente de lotes con Rayon.
- `GET /dashboard/index.html`: Dashboard Web interactivo FinTech.

---

## 7. Instrucciones de Ejecución

### 7.1 Backend y Dashboard Web
```bash
# Compilar y levantar la API REST y el Dashboard en Rust
cargo run -- serve --port 3000
```
- **Dashboard Web Online**: [http://127.0.0.1:3000/dashboard/index.html](http://127.0.0.1:3000/dashboard/index.html)
- **Dashboard Web Offline / Local**: Puedes abrir directamente el archivo `dashboard/index.html` en cualquier navegador (`file:///.../dashboard/index.html`); detecta automáticamente la API local en `http://127.0.0.1:3000` con CORS habilitado.

### 7.2 PostgreSQL para Analistas de Datos
- **Host**: `localhost`
- **Puerto**: `5433`
- **Base de Datos**: `credit_risk_db`
- **Usuario**: `postgres`
- **Contraseña**: `postgres`
- **Esquema**: `credit_analytics`
