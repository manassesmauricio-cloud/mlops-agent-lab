"""
monitoring.py — Cálculo de métricas de monitoreo y actualización del JSON.

Integra las funciones de drift de monitoring_utils.py para el cálculo de PSI.
Soporta dos modos:
  - Simulación (--scenario): para el lab sin datos reales.
  - Real (--ref-data / --actual-data): calcula PSI desde archivos CSV.

Uso:
    # Modo simulación (no requiere datos reales)
    python scripts/monitoring.py --scenario stable
    python scripts/monitoring.py --scenario warning
    python scripts/monitoring.py --scenario drift
    python scripts/monitoring.py --scenario critical

    # Modo real (calcula PSI desde archivos CSV)
    python scripts/monitoring.py --ref-data data/ref.csv --actual-data data/actual.csv
    python scripts/monitoring.py --ref-data ref.csv --actual-data actual.csv --quantils 20
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

METRICS_FILE = os.environ.get(
    "MONITORING_FILE",
    str(Path(__file__).parent.parent / "monitoring" / "latest_metrics.json")
)

SCENARIOS = {
    "stable": {
        "psi": 0.06, "auc": 0.91, "auc_previous": 0.91,
        "latency": 82,  "error_rate": 0.008, "samples": 14800,
        "model_version": "churn_v11",
    },
    "warning": {
        "psi": 0.17, "auc": 0.86, "auc_previous": 0.91,
        "latency": 130, "error_rate": 0.022, "samples": 15100,
        "model_version": "churn_v11",
    },
    "drift": {
        "psi": 0.29, "auc": 0.81, "auc_previous": 0.87,
        "latency": 145, "error_rate": 0.031, "samples": 15320,
        "model_version": "churn_v11",
    },
    "critical": {
        "psi": 0.41, "auc": 0.73, "auc_previous": 0.87,
        "latency": 290, "error_rate": 0.072, "samples": 15800,
        "model_version": "churn_v11",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES DE DRIFT — base: monitoring_utils.py
#  calculate_drift / psi_report / estado_psi
# ══════════════════════════════════════════════════════════════════════════════

def calculate_drift(dd_metric: str, df_ref: np.ndarray, df_actual: np.ndarray, quantils: int) -> float:
    """
    Calcula el índice de drift (PSI o KL) entre dos arrays de valores.

    Usa binning basado en quantiles de la distribución de referencia,
    más robusto que bins fijos para variables con distribuciones no uniformes.

    dd_metric: 'PSI' | 'KL'
    quantils:  número de bins (recomendado: 10-20)
    """
    breakpoints = np.percentile(df_ref, np.linspace(0, 100, quantils + 1))
    breakpoints[0]  = -np.inf
    breakpoints[-1] =  np.inf

    ref_counts    = np.histogram(df_ref,    bins=breakpoints)[0] / len(df_ref)
    actual_counts = np.histogram(df_actual, bins=breakpoints)[0] / len(df_actual)

    ref_counts    = np.where(ref_counts    == 0, 1e-6, ref_counts)
    actual_counts = np.where(actual_counts == 0, 1e-6, actual_counts)

    if dd_metric == 'PSI':
        dd_values = (ref_counts - actual_counts) * np.log(ref_counts / actual_counts)
    elif dd_metric == 'KL':
        dd_values = ref_counts * np.log(ref_counts / actual_counts)
    else:
        raise ValueError(f"Métrica desconocida: '{dd_metric}'. Usa 'PSI' o 'KL'.")

    return float(np.sum(dd_values))


def psi_report(df_actual: pd.DataFrame, df_ref: pd.DataFrame, quantils: int = 10) -> pd.DataFrame:
    """
    PSI por columna numérica común entre df_actual y df_ref.

    Retorna un DataFrame con columna 'psi', ordenado de mayor a menor.
    Solo incluye columnas con al menos 10 valores no nulos en cada dataset.
    """
    cols = [
        c for c in df_actual.columns
        if c in df_ref.columns
        and pd.api.types.is_numeric_dtype(df_actual[c])
        and pd.api.types.is_numeric_dtype(df_ref[c])
        and df_actual[c].notna().sum() > 10
        and df_ref[c].notna().sum() > 10
    ]
    rows = []
    for col in cols:
        psi_val = calculate_drift(
            'PSI',
            df_ref[col].dropna().astype(float).values,
            df_actual[col].dropna().astype(float).values,
            quantils,
        )
        rows.append({'feature': col, 'psi': round(psi_val, 5)})

    return pd.DataFrame(rows).set_index('feature').sort_values('psi', ascending=False)


def estado_psi(psi_val: float) -> str:
    """Clasifica el PSI en OK / WARN / ALARM según umbrales estándar."""
    if psi_val < 0.10:
        return 'OK'
    if psi_val < 0.25:
        return 'WARN'
    return 'ALARM'


# ══════════════════════════════════════════════════════════════════════════════
#  MODO SIMULACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def simulate_scenario(name: str, noise: float = 0.01) -> dict:
    """Genera métricas con pequeño ruido aleatorio para el escenario dado."""
    if name not in SCENARIOS:
        raise ValueError(f"Escenario desconocido: '{name}'. Opciones: {list(SCENARIOS.keys())}")

    rng  = np.random.default_rng()
    base = SCENARIOS[name].copy()

    return {
        "date":          datetime.now().strftime("%Y-%m-%d"),
        "model_version": base["model_version"],
        "psi":           round(max(0.0, base["psi"]        + rng.normal(0, noise)), 3),
        "auc":           round(max(0.0, base["auc"]        + rng.normal(0, noise)), 3),
        "auc_previous":  round(base["auc_previous"], 3),
        "latency":       int(base["latency"]  + rng.integers(-8, 9)),
        "error_rate":    round(max(0.0, base["error_rate"] + rng.normal(0, noise * 0.5)), 4),
        "samples":       int(base["samples"]  + rng.integers(-200, 201)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODO REAL — desde archivos CSV
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics_from_data(
    ref_path:      str,
    actual_path:   str,
    model_version: str = "churn_model",
    quantils:      int = 10,
) -> dict:
    """
    Calcula PSI medio desde archivos CSV de referencia y período actual.

    El PSI medio se calcula como promedio del PSI por columna (psi_report).
    Las métricas AUC, latencia y error_rate deben provenir del pipeline de
    inferencia; aquí se dejan en None para que el agente las complete.
    """
    df_ref    = pd.read_csv(ref_path)
    df_actual = pd.read_csv(actual_path)

    report  = psi_report(df_actual, df_ref, quantils)
    psi_med = float(report['psi'].mean()) if len(report) > 0 else 0.0

    print(f"\n  PSI por variable (top 10):")
    print(report.head(10).to_string())
    print(f"\n  PSI medio: {psi_med:.5f}  → {estado_psi(psi_med)}")

    existing = {}
    if Path(METRICS_FILE).exists():
        with open(METRICS_FILE, encoding="utf-8") as f:
            existing = json.load(f)

    return {
        "date":          datetime.now().strftime("%Y-%m-%d"),
        "model_version": model_version or existing.get("model_version", "churn_model"),
        "psi":           round(psi_med, 4),
        "auc":           existing.get("auc"),
        "auc_previous":  existing.get("auc_previous"),
        "latency":       existing.get("latency"),
        "error_rate":    existing.get("error_rate"),
        "samples":       len(df_actual),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ESCRITURA DEL JSON
# ══════════════════════════════════════════════════════════════════════════════

def update_metrics_file(metrics: dict, output_file: str):
    """Escribe las métricas en el archivo JSON de monitoreo."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Monitoreo del modelo de churn — cálculo de PSI y actualización de métricas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Simular escenario de drift
  python scripts/monitoring.py --scenario drift

  # Calcular PSI desde datos reales
  python scripts/monitoring.py --ref-data data/ref.csv --actual-data data/actual.csv

  # Calcular PSI con más bins para mayor precisión
  python scripts/monitoring.py --ref-data ref.csv --actual-data actual.csv --quantils 20
        """,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        help="Escenario de simulación: stable / warning / drift / critical",
    )
    group.add_argument(
        "--ref-data",
        metavar="REF_CSV",
        help="CSV de referencia para cálculo de PSI real",
    )

    parser.add_argument("--actual-data",  metavar="ACTUAL_CSV", help="CSV del período actual")
    parser.add_argument("--quantils",     type=int, default=10,  help="Número de bins PSI (default: 10)")
    parser.add_argument("--model",        default=None,          help="Nombre del modelo (modo real)")
    parser.add_argument("--output",       default=METRICS_FILE,  help="Archivo de salida JSON")
    args = parser.parse_args()

    print("=" * 50)

    if args.scenario:
        print(f"  Simulando escenario: {args.scenario.upper()}")
        print("=" * 50)
        metrics = simulate_scenario(args.scenario)

    else:
        if not args.actual_data:
            parser.error("--actual-data es requerido cuando se usa --ref-data")
        print(f"  Calculando PSI desde datos reales")
        print("=" * 50)
        metrics = compute_metrics_from_data(
            ref_path=args.ref_data,
            actual_path=args.actual_data,
            model_version=args.model,
            quantils=args.quantils,
        )

    update_metrics_file(metrics, args.output)

    print(f"\n{json.dumps(metrics, indent=2, ensure_ascii=False)}")
    print(f"\n  Métricas guardadas en: {args.output}")
