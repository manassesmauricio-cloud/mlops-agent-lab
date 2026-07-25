"""
train.py — Entrenamiento del modelo de churn con registro en MLflow.

Entrena XGBoost, LightGBM y CatBoost, selecciona el campeón
(mejor AUC test con decay < 10%) y registra en MLflow Tracking.

Modos:
  - Default: datos sintéticos, parámetros por defecto
  - --drift:       simula drift de distribución en los datos
  - --hpo:         búsqueda de hiperparámetros con RandomizedSearchCV
  - --train-data / --test-data: datos reales desde CSV

Uso:
    python scripts/train.py
    python scripts/train.py --drift
    python scripts/train.py --hpo
    python scripts/train.py --train-data data/train.csv --test-data data/test.csv
    python scripts/train.py --train-data data/train.csv --test-data data/test.csv --hpo
"""

from __future__ import annotations

import os
import sys
import argparse
import subprocess
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import xgboost as xgb
import lightgbm as lgb

try:
    import catboost as catb
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'catboost>=1.2.0', '--quiet'])
    import catboost as catb

from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint, uniform
from datetime import datetime

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME     = "churn_model"


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos: bool→int, enteros, floats a 4 decimales."""
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
        elif df[col].dtype in ('int16', 'int32', 'int64'):
            df[col] = df[col].astype(int)
        elif df[col].dtype in ('float16', 'float32', 'float64'):
            df[col] = df[col].astype(float).round(4)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN DE DATOS SINTÉTICOS (modo demo)
# ══════════════════════════════════════════════════════════════════════════════

def generate_churn_data(n_samples: int = 12000, drift: bool = False, seed: int = 42):
    """
    Genera datos sintéticos de clientes con etiqueta de churn.

    Si drift=True, desplaza las distribuciones para simular data shift
    (como ocurre en producción tras varios meses).
    """
    rng = np.random.default_rng(seed)

    edad             = rng.normal(45, 15, n_samples).clip(18, 80)
    cargo_mensual    = rng.normal(65, 30, n_samples).clip(10, 200)
    antiguedad_meses = rng.exponential(24, n_samples).clip(1, 120)
    num_productos    = rng.integers(1, 5, n_samples)
    llamadas_soporte = rng.poisson(1.5, n_samples)

    if drift:
        cargo_mensual    = (cargo_mensual * 1.35 + 25).clip(10, 250)
        antiguedad_meses = (antiguedad_meses * 0.65).clip(1, 120)
        llamadas_soporte = rng.poisson(2.8, n_samples)

    X = pd.DataFrame({
        "edad":             edad,
        "cargo_mensual":    cargo_mensual,
        "antiguedad_meses": antiguedad_meses,
        "num_productos":    num_productos,
        "llamadas_soporte": llamadas_soporte,
    })

    prob_churn = (
        0.25 * (cargo_mensual    > 85).astype(float) +
        0.35 * (antiguedad_meses < 12).astype(float) +
        0.20 * (num_productos    == 1).astype(float) +
        0.15 * (llamadas_soporte >  3).astype(float) +
        rng.normal(0, 0.08, n_samples)
    )
    y = (prob_churn > 0.45).astype(int)
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
#  ESPACIOS DE BÚSQUEDA HPO — base: hpo_training.py
# ══════════════════════════════════════════════════════════════════════════════

HPO_GRIDS: dict = {
    'xgb': {
        'n_estimators':    randint(100, 500),
        'learning_rate':   uniform(0.01, 0.2),
        'max_depth':       randint(3, 10),
        'subsample':       uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
    },
    'lgbm': {
        'n_estimators':  randint(100, 500),
        'learning_rate': uniform(0.01, 0.2),
        'num_leaves':    randint(20, 50),
        'max_depth':     randint(3, 10),
        'subsample':     uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.7, 0.3),
    },
    'catb': {
        'iterations':    randint(100, 500),
        'learning_rate': uniform(0.01, 0.2),
        'depth':         randint(3, 10),
        'subsample':     uniform(0.7, 0.3),
        'l2_leaf_reg':   uniform(1, 10),
    },
}


def _build_base_models() -> dict:
    return {
        'xgb':  xgb.XGBClassifier(eval_metric='logloss', random_state=42),
        'lgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
        'catb': catb.CatBoostClassifier(verbose=0, random_state=42),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO + SELECCIÓN DE CAMPEÓN + MLFLOW
# ══════════════════════════════════════════════════════════════════════════════

def train_and_log(
    X_train, X_test, y_train, y_test,
    drift: bool = False,
    hpo:   bool = False,
) -> tuple[dict, str]:
    """
    Entrena XGBoost, LightGBM y CatBoost.

    Selecciona el campeón: mejor AUC test con decay < 10%.
    Si ninguno cumple el criterio de decay, elige el de mayor AUC test.
    Registra el campeón en MLflow.

    Retorna (metrics_dict, run_id).
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    mode     = ("hpo" if hpo else "default") + ("_drift" if drift else "_stable")
    run_name = f"training_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    base_models     = _build_base_models()
    results: dict   = {}
    best_ml_name    = None
    best_auc_test   = -np.inf

    print(f"  Modo: {'HPO (RandomizedSearchCV)' if hpo else 'parámetros por defecto'}")

    for ml_name, base_model in base_models.items():
        print(f"\n  [{ml_name.upper()}] Entrenando...")

        if hpo:
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=HPO_GRIDS[ml_name],
                n_iter=15, cv=3, scoring='roc_auc',
                n_jobs=-1, random_state=42, verbose=0,
            )
            search.fit(X_train, y_train)
            model       = search.best_estimator_
            best_params = search.best_params_
            print(f"    Mejor AUC en CV: {search.best_score_:.4f}")
        else:
            model = base_model
            if ml_name == 'catb':
                model.fit(X_train, y_train,
                          eval_set=(X_test, y_test),
                          early_stopping_rounds=10, verbose=0)
            else:
                model.fit(X_train, y_train)
            best_params = model.get_params()

        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_proba       = model.predict_proba(X_test)[:, 1]
        y_pred        = model.predict(X_test)

        auc_train = roc_auc_score(y_train, y_train_proba)
        auc_test  = roc_auc_score(y_test,  y_proba)
        decay     = (auc_train - auc_test) / auc_train * 100 if auc_train > 0 else np.inf

        metrics = {
            "auc":       round(auc_test,                             4),
            "f1":        round(f1_score(y_test, y_pred),             4),
            "recall":    round(recall_score(y_test, y_pred),         4),
            "precision": round(precision_score(y_test, y_pred),      4),
        }

        print(f"    AUC train={auc_train:.4f} | AUC test={auc_test:.4f} | decay={decay:.2f}%")
        results[ml_name] = {
            "model":  model,
            "metrics": metrics,
            "params":  best_params,
            "decay":   decay,
        }

        if auc_test > best_auc_test and decay < 10:
            best_auc_test = auc_test
            best_ml_name  = ml_name

    if best_ml_name is None:
        best_ml_name = max(results, key=lambda k: results[k]["metrics"]["auc"])
        print(f"\n  Ningún modelo cumplió decay < 10%. Campeón por mayor AUC: {best_ml_name.upper()}")

    champion = results[best_ml_name]
    print(f"\n  Campeón: {best_ml_name.upper()} (AUC={champion['metrics']['auc']:.4f}, decay={champion['decay']:.2f}%)")

    loggable_params = {
        k: v for k, v in champion["params"].items()
        if isinstance(v, (int, float, str, bool))
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("model_type",       best_ml_name)
        mlflow.log_param("hpo",              hpo)
        mlflow.log_param("drift_simulation", drift)
        mlflow.log_param("train_samples",    len(X_train))
        mlflow.log_params({f"best_{k}": v for k, v in loggable_params.items()})
        mlflow.log_metrics(champion["metrics"])
        mlflow.sklearn.log_model(champion["model"], "model")
        run_id = run.info.run_id

    print(f"  Run ID: {run_id}")
    return champion["metrics"], run_id


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Entrenamiento del modelo de churn (XGB / LGBM / CatBoost)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Entrenamiento básico (datos sintéticos, parámetros por defecto)
  python scripts/train.py

  # Con drift de distribución simulado
  python scripts/train.py --drift

  # Con búsqueda de hiperparámetros (15 iter, CV=3 — más lento)
  python scripts/train.py --hpo

  # Datos reales desde CSV (primera columna = target)
  python scripts/train.py --train-data data/train.csv --test-data data/test.csv --hpo
        """,
    )
    parser.add_argument("--drift",       action="store_true", help="Simular drift de distribución")
    parser.add_argument("--hpo",         action="store_true", help="Búsqueda de hiperparámetros (más lento)")
    parser.add_argument("--train-data",  default=None, metavar="CSV", help="CSV de entrenamiento (primera columna = target)")
    parser.add_argument("--test-data",   default=None, metavar="CSV", help="CSV de test")
    args = parser.parse_args()

    print("=" * 50)
    print("  Entrenando modelo de churn...")
    print("=" * 50)

    if args.train_data and args.test_data:
        print(f"  Datos reales: {args.train_data}")
        train_df = preprocess_dataframe(pd.read_csv(args.train_data))
        test_df  = preprocess_dataframe(pd.read_csv(args.test_data))
        X_train  = train_df.iloc[:, 1:]
        y_train  = train_df.iloc[:, 0]
        X_test   = test_df.iloc[:, 1:]
        y_test   = test_df.iloc[:, 0]

        # Alinear columnas entre train y test
        for c in set(X_train.columns) - set(X_test.columns):
            X_test[c] = 0
        for c in set(X_test.columns) - set(X_train.columns):
            X_train[c] = 0
        X_test = X_test[X_train.columns]
    else:
        print("  Datos sintéticos (modo demo)...")
        X, y = generate_churn_data(drift=args.drift)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    metrics, run_id = train_and_log(
        X_train, X_test, y_train, y_test,
        drift=args.drift,
        hpo=args.hpo,
    )

    print("\n  Modelo registrado en MLflow exitosamente.")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
