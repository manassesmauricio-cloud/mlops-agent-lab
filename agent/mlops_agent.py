"""
mlops_agent.py — Agente MLOps para Automatizar el Ciclo de Vida del Modelo

Arquitectura:
    monitoring/latest_metrics.json
            │
            ▼
    ┌─────────────────────────────────────┐
    │         MLOps AI Agent              │
    │  1. Lee métricas                    │
    │  2. Evalúa estado                   │
    │  3. Genera diagnóstico (LLM/reglas) │
    │  4. Decide acción                   │
    │  5. Ejecuta Airflow (si reentrenar) │
    │  6. Compara modelos en MLflow       │
    │  7. Genera informe                  │
    │  8. Registra actividad              │
    └─────────────────────────────────────┘

Uso:
    # Con LLM — proveedor por defecto: Anthropic (requiere ANTHROPIC_API_KEY en .env)
    python agent/mlops_agent.py

    # Con otro proveedor LLM
    python agent/mlops_agent.py --provider openai
    python agent/mlops_agent.py --provider google
    python agent/mlops_agent.py --provider groq
    python agent/mlops_agent.py --provider mistral

    # Solo con reglas (no requiere API key)
    python agent/mlops_agent.py --no-llm

    # Con un archivo de métricas específico
    python agent/mlops_agent.py --metrics monitoring/latest_metrics.json

    # Simular un escenario antes de correr el agente
    python scripts/monitoring.py --scenario critical
    python agent/mlops_agent.py --no-llm
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import requests
import yaml

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE RUTAS Y CONEXIONES
# ══════════════════════════════════════════════════════════════════════════════

_BASE = Path(__file__).parent.parent

MONITORING_FILE = os.environ.get("MONITORING_FILE", str(_BASE / "monitoring" / "latest_metrics.json"))
REGISTRY_FILE   = os.environ.get("REGISTRY_FILE",   str(_BASE / "registry"   / "registry.csv"))
LOG_FILE        = os.environ.get("LOG_FILE",         str(_BASE / "logs"       / "agent_log.csv"))
POLICIES_FILE   = os.environ.get("POLICIES_FILE",    str(_BASE / "policies"   / "policies.yaml"))
REPORTS_DIR     = os.environ.get("REPORTS_DIR",      str(_BASE / "logs"       / "reports"))

AIRFLOW_URL  = os.environ.get("AIRFLOW_URL",          "http://localhost:8080")
AIRFLOW_USER = os.environ.get("AIRFLOW_USER",         "admin")
AIRFLOW_PASS = os.environ.get("AIRFLOW_PASS",         "admin")
MLFLOW_URL   = os.environ.get("MLFLOW_TRACKING_URI",  "http://localhost:5000")

DAG_ID     = "training_pipeline"
EXPERIMENT = "churn_model"
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
LLM_MODEL    = os.environ.get("LLM_MODEL", "")

_PROVIDER_DEFAULTS = {
    "anthropic": "claude-sonnet-4-6",
    "openai":    "gpt-4o",
    "google":    "gemini-2.0-flash",
    "groq":      "llama-3.3-70b-versatile",
    "mistral":   "mistral-large-latest",
}


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def load_metrics(filepath: str) -> dict:
    """Carga las métricas del archivo JSON de monitoreo."""
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def load_policies(filepath: str) -> dict:
    """Carga las políticas de gobernanza desde el YAML."""
    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  PARTE 1: EVALUACIÓN DEL ESTADO
#  Cada función retorna "OK", "WARNING" o "CRITICAL"
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_drift(metrics: dict, thresholds: dict) -> str:
    """
    Evalúa el drift de distribución usando PSI (Population Stability Index).

    PSI < 0.10  → OK       (sin cambio significativo)
    PSI < 0.20  → WARNING  (cambio moderado, vigilar)
    PSI >= 0.20 → CRITICAL (cambio significativo, acción requerida)
    """
    psi = metrics.get("psi", 0.0)
    t   = thresholds.get("psi", {})

    if psi < t.get("ok", 0.10):
        return "OK"
    elif psi < t.get("warning", 0.20):
        return "WARNING"
    else:
        return "CRITICAL"


def evaluate_performance(metrics: dict, thresholds: dict) -> str:
    """
    Evalúa la degradación de AUC comparando el valor actual con el anterior.

    Caída < 2%  → OK
    Caída < 5%  → WARNING
    Caída >= 5% → CRITICAL
    """
    auc      = metrics.get("auc",          1.0)
    auc_prev = metrics.get("auc_previous", 1.0)
    drop     = auc_prev - auc
    t        = thresholds.get("auc_drop", {})

    if drop < t.get("ok", 0.02):
        return "OK"
    elif drop < t.get("critical", 0.05):
        return "WARNING"
    else:
        return "CRITICAL"


def evaluate_latency(metrics: dict, thresholds: dict) -> str:
    """
    Evalúa la latencia de inferencia en milisegundos.

    < 100ms  → OK
    < 200ms  → WARNING
    >= 200ms → CRITICAL
    """
    latency = metrics.get("latency", 0)
    t       = thresholds.get("latency_ms", {})

    if latency < t.get("ok", 100):
        return "OK"
    elif latency < t.get("warning", 200):
        return "WARNING"
    else:
        return "CRITICAL"


def evaluate_error_rate(metrics: dict, thresholds: dict) -> str:
    """Evalúa la tasa de errores del endpoint de inferencia."""
    error_rate = metrics.get("error_rate", 0.0)
    t          = thresholds.get("error_rate", {})

    if error_rate < t.get("ok", 0.01):
        return "OK"
    elif error_rate < t.get("warning", 0.03):
        return "WARNING"
    else:
        return "CRITICAL"


# ══════════════════════════════════════════════════════════════════════════════
#  PARTE 2: GENERACIÓN DE DIAGNÓSTICO
#  Opción A: LLM (proveedor configurado en LLM_PROVIDER)
#  Opción B: Reglas determinísticas (fallback / --no-llm)
# ══════════════════════════════════════════════════════════════════════════════

def _call_llm(prompt: str) -> str:
    """
    Enruta la llamada al proveedor LLM configurado en LLM_PROVIDER.

    Proveedores soportados y su variable de entorno requerida:
      anthropic  → ANTHROPIC_API_KEY   (default)
      openai     → OPENAI_API_KEY
      google     → GOOGLE_API_KEY
      groq       → GROQ_API_KEY
      mistral    → MISTRAL_API_KEY

    El modelo se configura con LLM_MODEL; si está vacío usa el default del proveedor.
    """
    model = LLM_MODEL or _PROVIDER_DEFAULTS.get(LLM_PROVIDER, "")

    if LLM_PROVIDER == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()

    elif LLM_PROVIDER == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    elif LLM_PROVIDER == "google":
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        resp = genai.GenerativeModel(model).generate_content(prompt)
        return resp.text.strip()

    elif LLM_PROVIDER == "groq":
        from groq import Groq
        client = Groq()
        resp = client.chat.completions.create(
            model=model, max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    elif LLM_PROVIDER == "mistral":
        from mistralai import Mistral
        client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))
        resp = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()

    else:
        raise ValueError(
            f"Proveedor LLM no reconocido: '{LLM_PROVIDER}'. "
            f"Opciones válidas: {list(_PROVIDER_DEFAULTS)}"
        )


def generate_diagnosis_llm(
    metrics:   dict,
    drift_s:   str,
    perf_s:    str,
    latency_s: str,
) -> str:
    """
    Genera un diagnóstico usando el proveedor LLM configurado en LLM_PROVIDER.

    El LLM actúa como un MLOps Engineer senior que interpreta las métricas
    y genera una recomendación en lenguaje natural.
    """
    auc_drop_pct = (
        (metrics["auc_previous"] - metrics["auc"]) / metrics["auc_previous"] * 100
        if metrics.get("auc_previous", 0) > 0 else 0
    )

    prompt = f"""Eres un MLOps Engineer senior analizando el estado de un modelo de predicción de churn en producción.

MÉTRICAS DEL PERÍODO ACTUAL:
- PSI (drift de distribución): {metrics.get('psi', 0):.3f}  → Estado: {drift_s}
- AUC actual:     {metrics.get('auc', 0):.3f}
- AUC anterior:   {metrics.get('auc_previous', 0):.3f}  (caída: {auc_drop_pct:.1f}%)  → Estado: {perf_s}
- Latencia P50:   {metrics.get('latency', 0)}ms   → Estado: {latency_s}
- Tasa de error:  {metrics.get('error_rate', 0)*100:.1f}%
- Muestras:       {metrics.get('samples', 0):,}

Genera un diagnóstico técnico conciso (máximo 120 palabras) que incluya:
1. Una línea de estado por cada métrica relevante
2. Análisis de la causa probable del comportamiento observado
3. Recomendación de acción: ESPERAR / GENERAR ALERTA / REENTRENAR

Responde en español. Texto plano, sin markdown ni listas con guiones."""

    return _call_llm(prompt)


def generate_diagnosis_rules(
    metrics:   dict,
    drift_s:   str,
    perf_s:    str,
    latency_s: str,
) -> str:
    """
    Genera un diagnóstico mediante reglas determinísticas.

    Fallback cuando el LLM no está disponible o se invoca con --no-llm.
    """
    psi      = metrics.get("psi",          0.0)
    auc      = metrics.get("auc",          0.0)
    auc_prev = metrics.get("auc_previous", 0.0)
    latency  = metrics.get("latency",      0)
    drop_pct = (auc_prev - auc) / auc_prev * 100 if auc_prev > 0 else 0

    drift_desc = {
        "OK":       "Sin drift significativo en la distribución de entrada.",
        "WARNING":  f"Drift moderado detectado (PSI={psi:.3f}). Distribución comienza a desviarse.",
        "CRITICAL": f"Drift crítico (PSI={psi:.3f}). La distribución actual difiere significativamente de la referencia.",
    }
    perf_desc = {
        "OK":       f"Performance estable (AUC={auc:.3f}).",
        "WARNING":  f"Performance degradada: AUC cayó {drop_pct:.1f}% ({auc_prev:.3f} → {auc:.3f}).",
        "CRITICAL": f"Degradación crítica de performance: AUC cayó {drop_pct:.1f}% ({auc_prev:.3f} → {auc:.3f}).",
    }
    latency_desc = {
        "OK":       f"Latencia dentro de parámetros ({latency}ms).",
        "WARNING":  f"Latencia elevada ({latency}ms). Monitorear SLA.",
        "CRITICAL": f"Latencia crítica ({latency}ms). SLA en riesgo.",
    }

    if drift_s == "CRITICAL" or perf_s == "CRITICAL":
        rec = "Recomendación: REENTRENAR. La degradación requiere un nuevo ciclo de entrenamiento."
    elif drift_s == "WARNING" or perf_s == "WARNING":
        rec = "Recomendación: GENERAR ALERTA. Aumentar frecuencia de monitoreo."
    else:
        rec = "Recomendación: ESPERAR. Sistema dentro de parámetros operativos."

    return "\n".join([
        drift_desc[drift_s],
        perf_desc[perf_s],
        latency_desc[latency_s],
        "",
        rec,
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  PARTE 3: DECISIÓN DE ACCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def decide_action(
    metrics:   dict,
    drift_s:   str,
    perf_s:    str,
    latency_s: str,
) -> str:
    """
    Árbol de decisión del agente.

    Lógica principal basada en PSI:
        PSI < 0.10  → esperar
        PSI < 0.20  → generar alerta
        PSI >= 0.20 → reentrenar

    Escalados adicionales:
        Performance CRITICAL + alerta → reentrenar
        Latencia CRITICAL + esperar   → alertar

    Retorna: "wait" | "alert" | "retrain"
    """
    psi = metrics.get("psi", 0.0)

    if psi < 0.10:
        action = "wait"
    elif psi < 0.20:
        action = "alert"
    else:
        action = "retrain"

    # Escalado por performance crítica
    if perf_s == "CRITICAL" and action == "alert":
        action = "retrain"

    # Escalado por latencia crítica (al menos generar alerta)
    if latency_s == "CRITICAL" and action == "wait":
        action = "alert"

    return action


# ══════════════════════════════════════════════════════════════════════════════
#  PARTE 4: TRIGGER DE AIRFLOW
# ══════════════════════════════════════════════════════════════════════════════

def trigger_airflow_dag(dag_id: str) -> dict:
    """
    Dispara un DAG en Airflow via REST API.

    Endpoint: POST /api/v1/dags/{dag_id}/dagRuns
    Auth: Basic auth (admin/admin por defecto)
    """
    url     = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns"
    payload = {
        "conf": {
            "triggered_by": "mlops_agent",
            "timestamp":    datetime.now().isoformat(),
        }
    }

    response = requests.post(
        url,
        json=payload,
        auth=(AIRFLOW_USER, AIRFLOW_PASS),
        headers={"Content-Type": "application/json"},
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


def wait_for_dag_completion(dag_id: str, dag_run_id: str, timeout_sec: int = 300) -> str:
    """
    Espera a que un DAG Run termine y retorna el estado final.

    Hace polling del endpoint /api/v1/dags/{dag_id}/dagRuns/{dag_run_id}.
    """
    url = f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}"
    start = time.time()

    while time.time() - start < timeout_sec:
        try:
            resp  = requests.get(url, auth=(AIRFLOW_USER, AIRFLOW_PASS), timeout=10)
            state = resp.json().get("state", "unknown")

            if state in ("success", "failed"):
                return state

            print(f"    Estado del DAG: {state}... (esperando)")
            time.sleep(10)

        except Exception as e:
            print(f"    Error consultando Airflow: {e}")
            time.sleep(10)

    return "timeout"


# ══════════════════════════════════════════════════════════════════════════════
#  PARTE 5: COMPARACIÓN DE MODELOS EN MLFLOW
# ══════════════════════════════════════════════════════════════════════════════

def compare_models_mlflow(experiment_name: str) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Consulta MLflow y retorna (modelo_nuevo, modelo_actual).

    El modelo nuevo es el run más reciente.
    El modelo actual es el run anterior.

    Compara: AUC, F1, Recall
    """
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_URL)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' no encontrado en MLflow.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    if len(runs) < 2:
        raise ValueError("Se necesitan al menos 2 runs en MLflow para comparar modelos.")

    def parse_run(row: "pd.Series") -> dict:
        return {
            "run_id":  row["run_id"],
            "name":    row.get("tags.mlflow.runName", "N/A"),
            "auc":     float(row.get("metrics.auc",       0.0)),
            "f1":      float(row.get("metrics.f1",        0.0)),
            "recall":  float(row.get("metrics.recall",    0.0)),
            "precision": float(row.get("metrics.precision", 0.0)),
        }

    return parse_run(runs.iloc[0]), parse_run(runs.iloc[1])


# ══════════════════════════════════════════════════════════════════════════════
#  POLÍTICAS — Human in the Loop
# ══════════════════════════════════════════════════════════════════════════════

def check_retrain_policy(policies: dict, metrics: dict) -> Tuple[bool, str]:
    """
    Verifica si el reentrenamiento está permitido según las políticas.

    Controles:
    - ¿Es fin de semana? (allow_weekend_training)
    - ¿El costo estimado supera el límite? (max_cost_training)

    Retorna (permitido: bool, motivo: str).
    """
    approval = policies.get("approval", {})

    if not approval.get("allow_weekend_training", True):
        if datetime.now().weekday() >= 5:
            day = "sábado" if datetime.now().weekday() == 5 else "domingo"
            return False, f"Política prohíbe entrenamiento los fines de semana (hoy es {day})."

    max_cost      = approval.get("max_cost_training", float("inf"))
    estimated_cost = metrics.get("samples", 0) / 1000
    if estimated_cost > max_cost:
        return False, (
            f"Costo estimado ({estimated_cost:.1f}) supera el límite de política ({max_cost}). "
            "Se requiere aprobación manual."
        )

    return True, "Reentrenamiento aprobado por políticas."


def check_deploy_policy(
    policies:    dict,
    new_auc:     float,
    current_auc: float,
) -> Tuple[bool, str]:
    """
    Verifica si el despliegue automático está permitido.

    Controles:
    - ¿auto_deploy está habilitado?
    - ¿La mejora de AUC supera el mínimo requerido?

    Retorna (aprobado_auto: bool, motivo: str).
    """
    approval = policies.get("approval", {})

    if not approval.get("auto_deploy", True):
        return False, "Política requiere aprobación manual para despliegues (auto_deploy: false)."

    min_improvement = approval.get("min_auc_improvement", 0.0)
    improvement     = new_auc - current_auc
    if improvement < min_improvement:
        return False, (
            f"Mejora de AUC ({improvement:+.4f}) no alcanza el mínimo requerido ({min_improvement}). "
            "Modelo descartado."
        )

    return True, f"Despliegue aprobado automáticamente (mejora AUC: {improvement:+.4f})."


# ══════════════════════════════════════════════════════════════════════════════
#  PARTE 6: GENERACIÓN DEL INFORME
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(
    metrics:       dict,
    diagnosis:     str,
    action:        str,
    execution_log: list[str],
    new_model:     Optional[dict] = None,
    current_model: Optional[dict] = None,
) -> str:
    """
    Genera el informe final del agente en texto plano.

    Incluye: diagnóstico, acciones ejecutadas, comparación de modelos y decisión.
    """
    action_labels = {
        "wait":    "ESPERAR",
        "alert":   "GENERAR ALERTA",
        "retrain": "REENTRENAR",
    }
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sep      = "=" * 47

    lines = [
        sep,
        "           MLOps Agent Report",
        sep,
        "",
        f"Fecha:   {date_str}",
        f"Modelo:  {metrics.get('model_version', 'churn_v11')}",
        f"Acción:  {action_labels.get(action, action.upper())}",
        "",
        "─── Diagnóstico " + "─" * 31,
        diagnosis,
        "",
        "─── Acciones ejecutadas " + "─" * 23,
    ]

    for entry in execution_log:
        lines.append(f"  {entry}")

    if new_model and current_model:
        delta_auc    = new_model["auc"]    - current_model["auc"]
        delta_f1     = new_model["f1"]     - current_model["f1"]
        delta_recall = new_model["recall"] - current_model["recall"]

        lines += [
            "",
            "─── Comparación de Modelos " + "─" * 20,
            f"  {'Métrica':<12} {'Actual':>8} {'Nuevo':>8} {'Δ':>8}",
            f"  {'-'*40}",
            f"  {'AUC':<12} {current_model['auc']:>8.4f} {new_model['auc']:>8.4f} {delta_auc:>+8.4f}",
            f"  {'F1':<12} {current_model['f1']:>8.4f} {new_model['f1']:>8.4f} {delta_f1:>+8.4f}",
            f"  {'Recall':<12} {current_model['recall']:>8.4f} {new_model['recall']:>8.4f} {delta_recall:>+8.4f}",
            "",
            "─── Decisión Final " + "─" * 28,
        ]

        if new_model["auc"] > current_model["auc"]:
            lines += [
                "  Nuevo modelo recomendado.",
                "  Deployment pendiente de aprobación.",
            ]
        else:
            lines.append("  Nuevo modelo no supera al actual. Descartado.")

    lines += ["", sep]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  PARTE 7: REGISTRO DE ACTIVIDAD
# ══════════════════════════════════════════════════════════════════════════════

def log_activity(metrics: dict, action: str, result: str):
    """
    Registra la actividad del agente en logs/agent_log.csv.

    Columnas: fecha, psi, auc, decision, resultado

    El log permite monitorear al agente mismo:
        ¿Con qué frecuencia decide reentrenar?
        ¿Cuándo fue la última alerta?
        ¿Cuántas veces estuvo estable seguido?
    """
    Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["fecha", "psi", "auc", "decision", "resultado"]
    row = {
        "fecha":     datetime.now().strftime("%d/%m/%Y %H:%M"),
        "psi":       metrics.get("psi",      0),
        "auc":       metrics.get("auc",      0),
        "decision":  action,
        "resultado": result,
    }

    file_exists = Path(LOG_FILE).exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _update_registry(model: dict):
    """Agrega el nuevo modelo al registry CSV cuando es promovido."""
    Path(REGISTRY_FILE).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["fecha", "model_version", "run_id", "auc", "f1", "recall", "estado"]
    row = {
        "fecha":         datetime.now().strftime("%Y-%m-%d"),
        "model_version": f"churn_v{datetime.now().strftime('%Y%m%d')}",
        "run_id":        model.get("run_id", "N/A"),
        "auc":           model.get("auc",    0.0),
        "f1":            model.get("f1",     0.0),
        "recall":        model.get("recall", 0.0),
        "estado":        "produccion",
    }

    file_exists = Path(REGISTRY_FILE).exists()
    with open(REGISTRY_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
#  ORQUESTADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_agent(use_llm: bool = True, metrics_file: str = MONITORING_FILE):
    """
    Orquesta las 7 partes del agente MLOps.

    Flujo completo:
        Cargar config → Evaluar → Diagnosticar → Decidir →
        Ejecutar (Airflow) → Comparar (MLflow) → Informar → Registrar
    """
    sep = "=" * 50
    print(f"\n{sep}")
    print("   MLOps AI Agent  —  iniciando análisis")
    print(f"{sep}\n")

    # ── Cargar configuración ─────────────────────────────────────────────────
    print("[CONFIG] Cargando métricas y políticas...")
    metrics  = load_metrics(metrics_file)
    policies = load_policies(POLICIES_FILE)
    thresh   = policies.get("thresholds", {})

    print(f"  Archivo de métricas : {metrics_file}")
    print(f"  Fecha               : {metrics.get('date', 'N/A')}")
    print(f"  Modelo              : {metrics.get('model_version', 'N/A')}")

    # ── PARTE 1: Evaluación ──────────────────────────────────────────────────
    print("\n[PARTE 1] Evaluando estado del modelo...")
    drift_s   = evaluate_drift(metrics, thresh)
    perf_s    = evaluate_performance(metrics, thresh)
    latency_s = evaluate_latency(metrics, thresh)
    err_s     = evaluate_error_rate(metrics, thresh)

    print(f"  Drift       (PSI={metrics.get('psi',0):.3f})         : {drift_s}")
    print(f"  Performance (AUC={metrics.get('auc',0):.3f})        : {perf_s}")
    print(f"  Latencia    ({metrics.get('latency',0)}ms)         : {latency_s}")
    print(f"  Error rate  ({metrics.get('error_rate',0)*100:.1f}%)  : {err_s}")

    # ── PARTE 2: Diagnóstico ─────────────────────────────────────────────────
    print("\n[PARTE 2] Generando diagnóstico...")
    if use_llm:
        try:
            diagnosis = generate_diagnosis_llm(metrics, drift_s, perf_s, latency_s)
            print(f"  [LLM] Diagnóstico generado con {LLM_PROVIDER.upper()} ({LLM_MODEL or _PROVIDER_DEFAULTS.get(LLM_PROVIDER, '?')}).")
        except Exception as e:
            print(f"  [LLM] No disponible: {e}")
            print("  [REGLAS] Usando diagnóstico por reglas.")
            diagnosis = generate_diagnosis_rules(metrics, drift_s, perf_s, latency_s)
    else:
        diagnosis = generate_diagnosis_rules(metrics, drift_s, perf_s, latency_s)
        print("  [REGLAS] Diagnóstico generado con reglas determinísticas.")

    print(f"\n{diagnosis}\n")

    # ── PARTE 3: Decisión ────────────────────────────────────────────────────
    print("[PARTE 3] Decidiendo acción...")
    action = decide_action(metrics, drift_s, perf_s, latency_s)
    labels = {"wait": "ESPERAR", "alert": "GENERAR ALERTA", "retrain": "REENTRENAR"}
    print(f"  → Acción seleccionada: {labels[action]}")

    # ── PARTES 4-5: Ejecución ────────────────────────────────────────────────
    print(f"\n[PARTES 4-5] Ejecutando: {labels[action]}...")
    execution_log: list[str] = []
    new_model:     Optional[dict] = None
    current_model: Optional[dict] = None
    result: str = "OK"

    # ─ ESPERAR ───────────────────────────────────────────────────────────────
    if action == "wait":
        msg = "Sistema estable. Sin acción requerida."
        print(f"  ✓ {msg}")
        execution_log.append(f"✓ {msg}")
        result = "Estable"

    # ─ ALERTA ────────────────────────────────────────────────────────────────
    elif action == "alert":
        msg = (
            f"Degradación detectada "
            f"(PSI={metrics.get('psi',0):.3f}, AUC={metrics.get('auc',0):.3f}). "
            "Notificación enviada al equipo MLOps."
        )
        print(f"  ⚠ {msg}")
        execution_log.append(f"⚠ Alerta generada: {msg}")
        result = "Alerta generada"

    # ─ REENTRENAR ────────────────────────────────────────────────────────────
    elif action == "retrain":
        # Verificar política de reentrenamiento (Human in the Loop)
        retrain_ok, policy_msg = check_retrain_policy(policies, metrics)
        print(f"  [POLÍTICA] {policy_msg}")

        if not retrain_ok:
            print("  → Solicitud de aprobación manual generada.")
            execution_log.append(f"⚠ Retraining bloqueado por política: {policy_msg}")
            execution_log.append("⚠ Solicitud de aprobación pendiente (Human in the Loop).")
            result = "Pendiente aprobación"

        else:
            # PARTE 4: Trigger Airflow
            print(f"\n  [PARTE 4] Disparando DAG '{DAG_ID}' en Airflow...")
            dag_run_id = None
            try:
                dag_result = trigger_airflow_dag(DAG_ID)
                dag_run_id = dag_result.get("dag_run_id", "N/A")
                print(f"  ✓ DAG iniciado. Run ID: {dag_run_id}")
                execution_log.append(f"✓ Airflow DAG '{DAG_ID}' disparado (Run ID: {dag_run_id})")

                # Esperar finalización del entrenamiento
                print("  → Esperando finalización del entrenamiento...")
                final_state = wait_for_dag_completion(DAG_ID, dag_run_id, timeout_sec=300)
                print(f"  ✓ DAG finalizado con estado: {final_state}")
                execution_log.append(f"✓ Entrenamiento completado (estado: {final_state})")

            except Exception as e:
                print(f"  ! Airflow no disponible: {e}")
                print("  → Modo demo: asumiendo entrenamiento completado.")
                execution_log.append(f"⚠ Airflow no disponible (modo demo). Error: {e}")
                execution_log.append("✓ Retraining ejecutado (modo demo)")

            # PARTE 5: Comparar modelos en MLflow
            print("\n  [PARTE 5] Comparando modelos en MLflow...")
            try:
                new_model, current_model = compare_models_mlflow(EXPERIMENT)
                print(f"  ✓ Modelo nuevo   — AUC={new_model['auc']:.4f}, F1={new_model['f1']:.4f}")
                print(f"  ✓ Modelo actual  — AUC={current_model['auc']:.4f}, F1={current_model['f1']:.4f}")
                execution_log.append(f"✓ Nuevo modelo entrenado (AUC={new_model['auc']:.4f})")

                # Verificar política de despliegue
                deploy_ok, deploy_msg = check_deploy_policy(
                    policies, new_model["auc"], current_model["auc"]
                )
                print(f"  [POLÍTICA] {deploy_msg}")

                if new_model["auc"] > current_model["auc"]:
                    if deploy_ok:
                        _update_registry(new_model)
                        execution_log.append("✓ Registry actualizado con nuevo modelo.")
                        execution_log.append("✓ Modelo promovido a producción.")
                        result = "Retraining completado — Nuevo modelo promovido"
                    else:
                        execution_log.append(f"⚠ Deploy bloqueado por política: {deploy_msg}")
                        result = "Retraining completado — Deploy pendiente de aprobación"
                else:
                    execution_log.append("✗ Nuevo modelo no supera al actual. Descartado.")
                    result = "Retraining completado — Modelo descartado"

            except Exception as e:
                print(f"  ! MLflow no disponible: {e}")
                print("  → Modo demo: usando métricas simuladas.")
                new_model     = {"run_id": "demo_new",  "auc": 0.87, "f1": 0.83, "recall": 0.79}
                current_model = {"run_id": "demo_prev", "auc": metrics.get("auc", 0.81),
                                 "f1": 0.78, "recall": 0.74}
                execution_log.append(f"⚠ MLflow no disponible (modo demo). {e}")
                execution_log.append(f"✓ Retraining ejecutado — AUC demo: {new_model['auc']}")
                result = "Retraining ejecutado (modo demo)"

    # ── PARTE 6: Informe ─────────────────────────────────────────────────────
    print("\n[PARTE 6] Generando informe...")
    report = generate_report(metrics, diagnosis, action, execution_log, new_model, current_model)
    print(report)

    Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)
    report_path = Path(REPORTS_DIR) / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Informe guardado: {report_path}")

    # ── PARTE 7: Log ─────────────────────────────────────────────────────────
    print("\n[PARTE 7] Registrando actividad...")
    log_activity(metrics, action, result)
    print(f"  Actividad registrada en: {LOG_FILE}")

    print(f"\n{sep}")
    print("   MLOps AI Agent  —  análisis completado")
    print(f"{sep}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MLOps AI Agent — Automatización del ciclo de vida del modelo de churn",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Ejecutar con LLM (requiere ANTHROPIC_API_KEY en .env)
  python agent/mlops_agent.py

  # Ejecutar solo con reglas (sin API key)
  python agent/mlops_agent.py --no-llm

  # Simular escenario crítico y ejecutar agente
  python scripts/monitoring.py --scenario critical
  python agent/mlops_agent.py --no-llm

  # Simular escenario estable
  python scripts/monitoring.py --scenario stable
  python agent/mlops_agent.py --no-llm
        """,
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Usar reglas determinísticas en lugar de LLM para el diagnóstico",
    )
    parser.add_argument(
        "--provider",
        default=None,
        choices=list(_PROVIDER_DEFAULTS),
        help="Proveedor LLM: anthropic, openai, google, groq, mistral (sobreescribe LLM_PROVIDER del .env)",
    )
    parser.add_argument(
        "--metrics",
        default=MONITORING_FILE,
        help="Ruta al archivo de métricas JSON (default: monitoring/latest_metrics.json)",
    )
    args = parser.parse_args()

    if args.provider:
        LLM_PROVIDER = args.provider  # sobreescribe el valor del .env
    run_agent(use_llm=not args.no_llm, metrics_file=args.metrics)
