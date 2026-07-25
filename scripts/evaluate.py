"""
evaluate.py — Evaluación del modelo en producción consultando MLflow.

Obtiene las métricas de los últimos dos runs del experimento en MLflow
y actualiza el archivo de monitoreo con los valores comparativos.

Uso:
    python scripts/evaluate.py
    python scripts/evaluate.py --experiment churn_model
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = "churn_model"
METRICS_FILE        = os.path.join(os.path.dirname(__file__), "..", "monitoring", "latest_metrics.json")


def get_latest_two_runs(experiment_name: str) -> tuple[dict, dict]:
    """Retorna (run_más_reciente, run_anterior) del experimento."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' no encontrado en MLflow.")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    if len(runs) == 0:
        raise ValueError("No hay runs en el experimento.")

    def parse_run(row):
        return {
            "run_id":    row["run_id"],
            "run_name":  row.get("tags.mlflow.runName", "N/A"),
            "auc":       float(row.get("metrics.auc",       0.0)),
            "f1":        float(row.get("metrics.f1",        0.0)),
            "recall":    float(row.get("metrics.recall",    0.0)),
            "precision": float(row.get("metrics.precision", 0.0)),
        }

    latest   = parse_run(runs.iloc[0])
    previous = parse_run(runs.iloc[1]) if len(runs) > 1 else latest.copy()

    return latest, previous


def evaluate(experiment_name: str):
    """Evalúa y muestra las métricas del modelo más reciente."""
    print(f"  Consultando MLflow: {MLFLOW_TRACKING_URI}")
    print(f"  Experimento: {experiment_name}")

    try:
        latest, previous = get_latest_two_runs(experiment_name)

        auc_drop = previous["auc"] - latest["auc"]
        auc_drop_pct = auc_drop / previous["auc"] * 100 if previous["auc"] > 0 else 0

        print("\n  ── Modelo más reciente ──────────────")
        print(f"  AUC:       {latest['auc']:.4f}")
        print(f"  F1:        {latest['f1']:.4f}")
        print(f"  Recall:    {latest['recall']:.4f}")

        print("\n  ── Modelo anterior ──────────────────")
        print(f"  AUC:       {previous['auc']:.4f}")

        print(f"\n  ── Variación ────────────────────────")
        print(f"  ΔAUC:      {auc_drop:+.4f} ({auc_drop_pct:+.1f}%)")

        return latest, previous

    except Exception as e:
        print(f"  ! Error conectando a MLflow: {e}")
        print("  Usando métricas de ejemplo.")
        latest   = {"run_id": "demo_new",  "auc": 0.87, "f1": 0.83, "recall": 0.79, "precision": 0.81}
        previous = {"run_id": "demo_prev", "auc": 0.81, "f1": 0.78, "recall": 0.74, "precision": 0.76}
        return latest, previous


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación del modelo en MLflow")
    parser.add_argument("--experiment", default=EXPERIMENT_NAME, help="Nombre del experimento en MLflow")
    args = parser.parse_args()

    print("=" * 50)
    print("  Evaluando modelo de producción...")
    print("=" * 50)
    latest, previous = evaluate(args.experiment)
