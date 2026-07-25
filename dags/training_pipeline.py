"""
training_pipeline.py — DAG de Airflow para reentrenamiento del modelo de churn.

Este DAG es disparado por el Agente MLOps cuando detecta drift o degradación
de performance. No corre en schedule; solo se ejecuta bajo demanda.

Flujo:
    train_model → evaluate_model → update_monitoring_metrics

El agente puede dispararlo via:
    POST /api/v1/dags/training_pipeline/dagRuns
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

SCRIPTS_DIR = Path("/opt/airflow/scripts")

default_args = {
    "owner":            "mlops-agent",
    "retries":          1,
    "retry_delay":      timedelta(minutes=3),
    "start_date":       datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry":   False,
}


def _run_script(script_name: str, args: list[str] | None = None) -> None:
    """Ejecuta un script Python y falla el task si el script retorna error."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name)] + (args or [])
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError(f"{script_name} terminó con código {result.returncode}")


def task_train_model(**context):
    """
    Tarea 1: Entrenar el modelo con los datos más recientes.

    Genera datos sintéticos, entrena GradientBoostingClassifier
    y registra el experimento en MLflow.
    """
    conf = context.get("dag_run", {}).conf or {}
    triggered_by = conf.get("triggered_by", "manual")
    print(f"  Entrenamiento disparado por: {triggered_by}")
    print(f"  Timestamp: {conf.get('timestamp', 'N/A')}")
    _run_script("train.py")


def task_evaluate_model(**context):
    """
    Tarea 2: Evaluar el modelo recién entrenado consultando MLflow.

    Compara el nuevo modelo contra el anterior y muestra el delta de AUC.
    """
    _run_script("evaluate.py")


def task_update_monitoring(**context):
    """
    Tarea 3: Actualizar el archivo de métricas de monitoreo.

    Simula la ejecución del pipeline de monitoreo post-entrenamiento.
    En producción, esto correría el pipeline completo de inference + PSI.
    """
    # Después del reentrenamiento, el nuevo modelo suele tener mejor performance
    # → simulamos el escenario "warning" (aún hay drift, pero AUC mejoró)
    _run_script("monitoring.py", ["--scenario", "warning"])


with DAG(
    dag_id="training_pipeline",
    default_args=default_args,
    schedule_interval=None,   # Solo se ejecuta bajo demanda (triggered by agent)
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "churn", "retraining"],
    doc_md="""
## Training Pipeline

DAG disparado por el Agente MLOps cuando detecta degradación del modelo.

### Flujo
1. **train_model** — Entrena el modelo y registra en MLflow
2. **evaluate_model** — Compara métricas del modelo nuevo vs anterior
3. **update_monitoring** — Actualiza `monitoring/latest_metrics.json`

### Trigger manual (CLI)
```bash
docker compose exec airflow-scheduler \\
    airflow dags trigger training_pipeline
```

### Trigger via API (lo hace el Agente)
```bash
curl -X POST http://localhost:8080/api/v1/dags/training_pipeline/dagRuns \\
     -H "Content-Type: application/json" \\
     -u admin:admin \\
     -d '{"conf": {"triggered_by": "mlops_agent"}}'
```
""",
) as dag:

    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
    )

    evaluate = PythonOperator(
        task_id="evaluate_model",
        python_callable=task_evaluate_model,
    )

    update_monitoring = PythonOperator(
        task_id="update_monitoring_metrics",
        python_callable=task_update_monitoring,
    )

    train >> evaluate >> update_monitoring
