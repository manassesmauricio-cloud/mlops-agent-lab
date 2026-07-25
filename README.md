# Laboratorio: Agente MLOps para Automatizar el Ciclo de Vida del Modelo

## Contexto

La empresa tiene un modelo de predicción de churn que se ejecuta diariamente.
Durante varios meses fue estable. Ahora comienzan a detectarse cambios en la
distribución de los datos y degradación del rendimiento.

Actualmente un ingeniero revisa los dashboards cada día y decide:
- Esperar
- Generar una alerta
- Volver a entrenar
- Desplegar un nuevo modelo

**Tu misión**: automatizar esa decisión con un Agente de IA.

---

## Arquitectura

```
monitoring/latest_metrics.json
         │
         ▼
┌─────────────────────────────────────────┐
│           MLOps AI Agent                │
│   1. Lee métricas del JSON              │
│   2. Evalúa drift / performance / lat.  │
│   3. Genera diagnóstico (LLM o reglas)  │
│   4. Decide: esperar / alertar / retrain│
│   5. Dispara Airflow DAG (si retrain)   │
│   6. Compara modelos en MLflow          │
│   7. Genera informe                     │
│   8. Registra en agent_log.csv          │
└─────────────────────────────────────────┘
         │                │
         ▼                ▼
    Airflow DAG        MLflow
  (reentrenamiento)  (comparación)
```

El agente aparece **encima** del pipeline, no dentro del entrenamiento.

---

## Estructura del Proyecto

```
mlops-agent-lab/
│
├── agent/
│   └── mlops_agent.py         ← EL AGENTE (archivo principal del lab)
│
├── dags/
│   └── training_pipeline.py   ← DAG de Airflow (ya desarrollado)
│
├── scripts/
│   ├── train.py               ← Entrenamiento con MLflow
│   ├── evaluate.py            ← Evaluación de métricas
│   └── monitoring.py          ← Cálculo de PSI y métricas
│
├── monitoring/
│   └── latest_metrics.json    ← Entrada del agente (PSI, AUC, latencia...)
│
├── registry/
│   └── registry.csv           ← Historial de modelos promovidos
│
├── policies/
│   └── policies.yaml          ← Reglas de gobernanza (Human in the Loop)
│
├── models/                    ← Modelos entrenados
├── mlruns/                    ← Artefactos de MLflow
├── logs/
│   ├── agent_log.csv          ← Historial de decisiones del agente
│   └── reports/               ← Informes generados por el agente
│
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Setup Inicial

### Opción A — Solo el Agente (sin Docker, modo demo)

Ideal para empezar rápido y entender el flujo sin infraestructura.

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Copiar y configurar variables de entorno
cp .env.example .env
# Editar .env: configurar LLM_PROVIDER y la API key del proveedor elegido

# 3. Correr el agente en modo demo (sin Airflow ni MLflow)
python agent/mlops_agent.py --no-llm
```

### Opción B — Stack Completo (con Docker)

```bash
# 1. Construir la imagen
docker compose build

# 2. Inicializar Airflow
docker compose up airflow-init

# 3. Levantar todos los servicios
docker compose up -d

# 4. Verificar que todo esté corriendo
docker compose ps
```

Interfaces web:
- **Airflow**: http://localhost:8080 (admin / admin)
- **MLflow**: http://localhost:5000

---

## Simulador de Escenarios

Antes de correr el agente, puedes simular distintos niveles de drift:

```bash
# Escenario estable → agente decide: ESPERAR
python scripts/monitoring.py --scenario stable

# Escenario con drift moderado → agente decide: GENERAR ALERTA
python scripts/monitoring.py --scenario warning

# Escenario con drift alto (el default) → agente decide: REENTRENAR
python scripts/monitoring.py --scenario drift

# Escenario crítico → agente decide: REENTRENAR (urgente)
python scripts/monitoring.py --scenario critical
```

Luego ejecuta el agente:

```bash
python agent/mlops_agent.py --no-llm              # Solo con reglas (sin API key)
python agent/mlops_agent.py                        # Con LLM (proveedor por defecto)
python agent/mlops_agent.py --provider openai      # Con OpenAI
python agent/mlops_agent.py --provider groq        # Con Groq (Llama 3.3)
python agent/mlops_agent.py --provider google      # Con Gemini
python agent/mlops_agent.py --provider mistral     # Con Mistral
```

---

## Las 7 Partes del Laboratorio

### Parte 1 — Evaluación del Estado

Ubica las tres funciones en `agent/mlops_agent.py`:

```python
def evaluate_drift(metrics, thresholds) -> str:      # "OK" | "WARNING" | "CRITICAL"
def evaluate_performance(metrics, thresholds) -> str:
def evaluate_latency(metrics, thresholds) -> str:
```

Cada función lee una métrica del JSON y retorna un estado según los umbrales
definidos en `policies/policies.yaml`.

**Ejercicio**: modifica los umbrales en el YAML y observa cómo cambia el estado.

---

### Parte 2 — Diagnóstico

El agente puede usar dos métodos:

**Con LLM** (`generate_diagnosis_llm`):
```bash
# Proveedor configurado en LLM_PROVIDER del .env (default: anthropic)
python agent/mlops_agent.py

# O elige el proveedor desde el CLI
python agent/mlops_agent.py --provider openai
python agent/mlops_agent.py --provider groq
```

**Con reglas** (`generate_diagnosis_rules`):
```bash
python agent/mlops_agent.py --no-llm
```

El LLM actúa como un MLOps Engineer senior: interpreta las métricas y genera
una recomendación en lenguaje natural. El proveedor es intercambiable; todos
reciben el mismo prompt y retornan texto plano.

**Ejercicio**: prueba ambos modos y compara los diagnósticos generados.
Prueba también con distintos proveedores LLM y compara las respuestas.

---

### Parte 3 — Árbol de Decisión

La lógica de decisión en `decide_action()`:

```python
if psi < 0.10:
    action = "wait"
elif psi < 0.20:
    action = "alert"
else:
    action = "retrain"

# Escalados adicionales
if perf_s == "CRITICAL" and action == "alert":
    action = "retrain"
if latency_s == "CRITICAL" and action == "wait":
    action = "alert"
```

Esta lógica vive en el **agente**, no en Airflow. Eso es lo que lo hace un
agente MLOps: toma decisiones sobre el pipeline.

**Ejercicio**: agrega una regla nueva. Por ejemplo: si `error_rate > 5%`,
escalar siempre a `retrain`.

---

### Parte 4 — Trigger de Airflow

Cuando el agente decide reentrenar, dispara el DAG via REST API:

```python
POST /api/v1/dags/training_pipeline/dagRuns
Authorization: Basic admin:admin
Content-Type: application/json

{"conf": {"triggered_by": "mlops_agent"}}
```

Función en el agente: `trigger_airflow_dag(dag_id)`.

También puedes dispararlo manualmente:

```bash
# Desde CLI de Airflow (dentro del contenedor)
docker compose exec airflow-scheduler \
    airflow dags trigger training_pipeline

# Desde curl
curl -X POST http://localhost:8080/api/v1/dags/training_pipeline/dagRuns \
     -H "Content-Type: application/json" \
     -u admin:admin \
     -d '{"conf": {"triggered_by": "test"}}'
```

---

### Parte 5 — Comparación de Modelos en MLflow

Después del entrenamiento, el agente consulta MLflow y compara:

| Métrica | Modelo actual | Modelo nuevo | Δ      |
|---------|--------------|--------------|--------|
| AUC     | 0.8100       | 0.8700       | +0.060 |
| F1      | 0.7800       | 0.8300       | +0.050 |
| Recall  | 0.7400       | 0.7900       | +0.050 |

Si el nuevo modelo no mejora → se **descarta**.
Si mejora → se evalúa la política de despliegue.

---

### Parte 6 — Informe

El agente genera un informe en `logs/reports/report_YYYYMMDD_HHMMSS.txt`:

```
===============================================
           MLOps Agent Report
===============================================

Fecha:   2026-07-25 14:32:11
Modelo:  churn_v11
Acción:  REENTRENAR

─── Diagnóstico ───────────────────────────────
Drift crítico detectado (PSI=0.29). La distribución
actual difiere significativamente de la referencia.
AUC cayó 6.9% (0.87 → 0.81). Recomendación: REENTRENAR.

─── Acciones ejecutadas ───────────────────────
  ✓ Airflow DAG 'training_pipeline' disparado
  ✓ Nuevo modelo entrenado (AUC=0.8700)
  ⚠ Deploy bloqueado por política: auto_deploy=false

─── Comparación de Modelos ────────────────────
  Métrica      Actual    Nuevo         Δ
  ────────────────────────────────────────
  AUC          0.8100   0.8700   +0.0600
  F1           0.7800   0.8300   +0.0500
  Recall       0.7400   0.7900   +0.0500

─── Decisión Final ────────────────────────────
  Nuevo modelo recomendado.
  Deployment pendiente de aprobación.

===============================================
```

---

### Parte 7 — Registro de Actividad (agent_log.csv)

El agente registra cada decisión en `logs/agent_log.csv`:

| fecha       | psi  | auc  | decision | resultado                  |
|-------------|------|------|----------|----------------------------|
| 01/07 08:00 | 0.06 | 0.91 | wait     | Estable                    |
| 08/07 08:00 | 0.11 | 0.90 | alert    | Alerta generada            |
| 15/07 08:00 | 0.18 | 0.87 | alert    | Alerta generada            |
| 25/07 08:00 | 0.29 | 0.81 | retrain  | Retraining completado      |

**Concepto clave**: debemos monitorear al agente igual que monitoreamos el modelo.

---

## Desafío — Human in the Loop

Abre `policies/policies.yaml` y modifica las políticas:

```yaml
approval:
  auto_deploy: false            # El agente NO puede desplegar sin aprobación
  min_auc_improvement: 0.02     # El nuevo modelo debe mejorar al menos 2% de AUC
  max_cost_training: 30         # Costo máximo de entrenamiento
  allow_weekend_training: false # No entrenar los fines de semana
```

Prueba este escenario:
1. Simula drift crítico: `python scripts/monitoring.py --scenario critical`
2. Ejecuta el agente un sábado o cambia temporalmente el día en el código
3. Observa cómo el agente **genera una solicitud de aprobación** en lugar de reentrenar

**Aprendizaje**: un agente no es solo "llamar a un LLM". Es operar dentro de
un marco de políticas, restricciones y gobernanza.

---

## Referencia Rápida de Comandos

```bash
# Simular escenario y correr agente (todo en modo demo)
python scripts/monitoring.py --scenario drift
python agent/mlops_agent.py --no-llm

# Correr agente con LLM (proveedor configurado en .env)
python agent/mlops_agent.py

# Correr agente con proveedor específico
python agent/mlops_agent.py --provider groq

# Entrenar modelo manualmente (XGB / LGBM / CatBoost)
python scripts/train.py

# Entrenar con búsqueda de hiperparámetros
python scripts/train.py --hpo

# Calcular PSI desde datos reales
python scripts/monitoring.py --ref-data data/ref.csv --actual-data data/actual.csv

# Ver historial de decisiones del agente
cat logs/agent_log.csv

# Ver último informe generado
ls -t logs/reports/ | head -1 | xargs -I{} cat logs/reports/{}

# Con Docker — trigger manual del DAG
docker compose exec airflow-scheduler \
    airflow dags trigger training_pipeline

# Con Docker — ver logs del agente dentro del contenedor
docker compose exec airflow-scheduler \
    python /opt/airflow/agent/mlops_agent.py --no-llm
```

---

## Archivo de Métricas (monitoring/latest_metrics.json)

El agente lee este JSON en cada ejecución:

```json
{
  "date": "2026-07-25",
  "model_version": "churn_v11",
  "psi": 0.29,
  "auc": 0.81,
  "auc_previous": 0.87,
  "latency": 145,
  "error_rate": 0.03,
  "samples": 15320
}
```

En producción, este archivo sería generado por el pipeline de monitoreo
(Airflow DAG de inferencia + cálculo de PSI).

---

## Stack Tecnológico

| Componente   | Tecnología                        | Puerto |
|--------------|-----------------------------------|--------|
| Orquestador  | Apache Airflow                    | 8080   |
| Experimentos | MLflow                            | 5000   |
| Base de datos| PostgreSQL                        | 5432   |
| LLM Agent    | Configurable (multi-proveedor)    | API    |
| Modelos ML   | XGBoost / LightGBM / CatBoost     | —      |
| Monitoreo    | PSI / KL divergence               | —      |

### Proveedores LLM soportados

| Proveedor   | Variable de entorno | Modelo por defecto          |
|-------------|---------------------|-----------------------------|
| anthropic   | `ANTHROPIC_API_KEY` | claude-sonnet-4-6           |
| openai      | `OPENAI_API_KEY`    | gpt-4o                      |
| google      | `GOOGLE_API_KEY`    | gemini-2.0-flash            |
| groq        | `GROQ_API_KEY`      | llama-3.3-70b-versatile     |
| mistral     | `MISTRAL_API_KEY`   | mistral-large-latest        |

Configura `LLM_PROVIDER` en tu `.env` o usa `--provider` en el CLI.
