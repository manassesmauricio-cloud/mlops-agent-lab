"""
app.py — Dashboard del Agente MLOps

Interfaz web local para monitorear y ejecutar el agente de churn.

Uso:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

# ── Page config — debe ir antes de cualquier otro st.* ──────────────────────
st.set_page_config(
    page_title="MLOps Agent",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

import agent.mlops_agent as mlops
from scripts.monitoring import SCENARIOS, simulate_scenario, update_metrics_file

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES DE PRESENTACIÓN
# ══════════════════════════════════════════════════════════════════════════════

STATUS_COLOR = {"OK": "#48BB78", "WARNING": "#ECC94B", "CRITICAL": "#FC8181"}
STATUS_BG    = {"OK": "#0F2A1A", "WARNING": "#2A2208", "CRITICAL": "#2A0F0F"}
STATUS_ICON  = {"OK": "●", "WARNING": "▲", "CRITICAL": "✕"}

ACTION_COLOR = {"wait": "#48BB78", "alert": "#ECC94B", "retrain": "#FC8181"}
ACTION_BG    = {"wait": "#0F2A1A", "alert": "#2A2208",  "retrain": "#2A0F0F"}
ACTION_LABEL = {"wait": "ESPERAR", "alert": "GENERAR ALERTA", "retrain": "REENTRENAR"}
ACTION_DESC  = {
    "wait":    "Sistema estable. Sin acción requerida.",
    "alert":   "Degradación detectada. El equipo MLOps ha sido notificado.",
    "retrain": "Se requiere un nuevo ciclo de entrenamiento.",
}

SCENARIO_LABEL = {
    "stable":   "🟢  Estable",
    "warning":  "🟡  Advertencia",
    "drift":    "🔴  Drift",
    "critical": "🚨  Crítico",
}

PROVIDER_LABEL = {
    "anthropic": "Anthropic",
    "openai":    "OpenAI",
    "google":    "Google",
    "groq":      "Groq",
    "mistral":   "Mistral",
}

# ══════════════════════════════════════════════════════════════════════════════
#  CSS GLOBAL
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
  --bg:       #0A0E1A;
  --surf:     #141929;
  --surf2:    #1C2436;
  --border:   #2A3350;
  --text:     #E2E8F0;
  --muted:    #6B82A0;
  --accent:   #00D4A4;
  --ok:       #48BB78;
  --warn:     #ECC94B;
  --crit:     #FC8181;
  --mono:     'IBM Plex Mono', 'Courier New', monospace;
  --sans:     'IBM Plex Sans', system-ui, sans-serif;
}

/* ── Base ── */
html, body, .stApp { background: var(--bg) !important; }
* { font-family: var(--sans); }
p, li, span { color: var(--text); }
hr { border-color: var(--border) !important; margin: 1rem 0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surf) !important;
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text); }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stToggle label { color: var(--muted) !important; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }

/* ── Buttons ── */
button[kind="primary"] {
  background: var(--accent) !important;
  color: #071210 !important;
  font-family: var(--mono) !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em;
  border: none !important;
  border-radius: 6px !important;
}
button[kind="primary"]:hover { background: #00b892 !important; }

/* ── Cards ── */
.card {
  background: var(--surf);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 1.25rem 1.5rem;
  height: 100%;
}

/* ── Metric card ── */
.metric-lbl {
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin-bottom: 0.6rem;
}
.metric-val {
  font-family: var(--mono);
  font-size: 2.2rem;
  font-weight: 600;
  line-height: 1;
  color: var(--text);
  font-variant-numeric: tabular-nums;
}
.metric-unit {
  font-family: var(--mono);
  font-size: 0.85rem;
  color: var(--muted);
  margin-left: 3px;
}
.bar-track {
  height: 3px;
  background: var(--surf2);
  border-radius: 2px;
  margin: 0.65rem 0 0.5rem;
  overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 2px; }
.status-pill {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  padding: 0.15rem 0.6rem;
  border-radius: 99px;
}

/* ── Decision banner ── */
.decision {
  border-radius: 10px;
  padding: 1.5rem 1.75rem;
}
.decision-icon { font-family: var(--mono); font-size: 1.8rem; line-height: 1; }
.decision-action {
  font-family: var(--mono);
  font-size: 1.35rem;
  font-weight: 600;
  letter-spacing: 0.03em;
  margin: 0.35rem 0 0.2rem;
}
.decision-desc { font-size: 0.83rem; opacity: 0.75; }

/* ── Section label ── */
.sec {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-size: 0.67rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--muted);
  margin: 1.6rem 0 0.8rem;
}
.sec::after { content: ''; flex: 1; height: 1px; background: var(--border); }

/* ── Log entry ── */
.log { font-family: var(--mono); font-size: 0.8rem; padding: 0.4rem 0; border-bottom: 1px solid var(--border); color: var(--text); display: flex; gap: 0.5rem; }
.log:last-child { border-bottom: none; }
.log-ts { color: var(--muted); flex-shrink: 0; }

/* ── Diagnosis ── */
.diag-text { font-size: 0.9rem; line-height: 1.75; white-space: pre-wrap; color: var(--text); }
.pill {
  display: inline-block; font-family: var(--mono); font-size: 0.67rem; font-weight: 600;
  letter-spacing: 0.04em; background: var(--surf2); border: 1px solid var(--border);
  border-radius: 99px; padding: 0.2rem 0.7rem; color: var(--muted); margin-bottom: 0.75rem;
}

/* ── History table ── */
.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px; }

/* ── Empty state ── */
.empty {
  text-align: center;
  padding: 5rem 2rem;
  color: var(--muted);
}
.empty-icon { font-size: 3.5rem; margin-bottom: 1rem; line-height: 1; }
.empty-title { font-family: var(--mono); font-size: 1.1rem; font-weight: 600; color: var(--text); margin-bottom: 0.5rem; }
.empty-sub { font-size: 0.85rem; line-height: 1.6; }

/* ── AUC trend ── */
.trend { font-family: var(--mono); font-size: 0.72rem; color: var(--crit); }
.trend-ok { color: var(--ok); }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS HTML
# ══════════════════════════════════════════════════════════════════════════════

def metric_card(label: str, value: str, unit: str, status: str, bar_pct: float, sub: str = "") -> str:
    c = STATUS_COLOR[status]
    bg = STATUS_BG[status]
    return f"""
    <div class="card">
      <div class="metric-lbl">{label}</div>
      <div class="metric-val">{value}<span class="metric-unit">{unit}</span></div>
      <div class="bar-track">
        <div class="bar-fill" style="width:{bar_pct:.1f}%;background:{c}"></div>
      </div>
      <span class="status-pill" style="color:{c};background:{bg}">
        {STATUS_ICON[status]}&nbsp;{status}
      </span>
      {"<div style='font-size:0.72rem;color:var(--muted);margin-top:0.35rem'>" + sub + "</div>" if sub else ""}
    </div>"""


def decision_block(action: str) -> str:
    c  = ACTION_COLOR[action]
    bg = ACTION_BG[action]
    icon_map = {"wait": "◉", "alert": "▲", "retrain": "⟳"}
    return f"""
    <div class="decision" style="background:{bg};border:1px solid {c}33">
      <div class="decision-icon" style="color:{c}">{icon_map[action]}</div>
      <div class="decision-action" style="color:{c}">{ACTION_LABEL[action]}</div>
      <div class="decision-desc">{ACTION_DESC[action]}</div>
    </div>"""


def section(title: str) -> None:
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — CONTROLES
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div style="font-family:\'IBM Plex Mono\',monospace;font-size:1.05rem;'
        'font-weight:600;color:#00D4A4;padding:0.25rem 0 1.75rem;letter-spacing:0.02em">'
        '⬡ MLOps Agent</div>',
        unsafe_allow_html=True,
    )

    scenario = st.selectbox(
        "Escenario",
        options=list(SCENARIOS.keys()),
        index=2,
        format_func=lambda x: SCENARIO_LABEL[x],
    )

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    use_llm  = st.toggle("Diagnóstico con LLM", value=False)
    provider = None

    if use_llm:
        provider = st.selectbox(
            "Proveedor LLM",
            options=list(mlops._PROVIDER_DEFAULTS.keys()),
            format_func=lambda x: PROVIDER_LABEL[x],
        )
        model = mlops._PROVIDER_DEFAULTS.get(provider, "")
        st.caption(f"Modelo: `{model}`")

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    run_btn = st.button("▶  Ejecutar Agente", type="primary", use_container_width=True)

    if "last_run_ts" in st.session_state:
        st.markdown(
            f'<div style="font-size:0.72rem;color:#6B82A0;text-align:center;margin-top:0.75rem">'
            f'Última ejecución {st.session_state.last_run_ts}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.72rem;color:#6B82A0;line-height:1.6">'
        '<strong style="color:#E2E8F0">Escenarios</strong><br>'
        '🟢 Estable — PSI bajo, AUC óptimo<br>'
        '🟡 Advertencia — Drift moderado<br>'
        '🔴 Drift — PSI alto, reentrenar<br>'
        '🚨 Crítico — Degradación severa'
        '</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LÓGICA DE EJECUCIÓN
# ══════════════════════════════════════════════════════════════════════════════

if run_btn:
    if provider:
        mlops.LLM_PROVIDER = provider

    with st.spinner("Ejecutando agente..."):
        metrics  = simulate_scenario(scenario)
        update_metrics_file(metrics, str(ROOT / "monitoring" / "latest_metrics.json"))

        policies = mlops.load_policies(str(ROOT / "policies" / "policies.yaml"))
        thresh   = policies.get("thresholds", {})

        drift_s   = mlops.evaluate_drift(metrics, thresh)
        perf_s    = mlops.evaluate_performance(metrics, thresh)
        latency_s = mlops.evaluate_latency(metrics, thresh)
        err_s     = mlops.evaluate_error_rate(metrics, thresh)

        diag_mode = "Reglas determinísticas"
        if use_llm and provider:
            try:
                diagnosis = mlops.generate_diagnosis_llm(metrics, drift_s, perf_s, latency_s)
                diag_mode = f"LLM · {PROVIDER_LABEL[provider]}"
            except Exception as exc:
                diagnosis = mlops.generate_diagnosis_rules(metrics, drift_s, perf_s, latency_s)
                diag_mode = f"Reglas (LLM no disponible: {exc})"
        else:
            diagnosis = mlops.generate_diagnosis_rules(metrics, drift_s, perf_s, latency_s)

        action = mlops.decide_action(metrics, drift_s, perf_s, latency_s)

        retrain_ok, policy_msg = mlops.check_retrain_policy(policies, metrics)
        policy_blocked = (action == "retrain" and not retrain_ok)

        execution_log = []
        if action == "wait":
            execution_log.append(("✓", "Sistema estable. Sin acción requerida."))
        elif action == "alert":
            execution_log.append(("⚠", f"Alerta generada — PSI={metrics['psi']:.3f}, AUC={metrics['auc']:.3f}"))
        elif action == "retrain":
            if policy_blocked:
                execution_log.append(("⚠", f"Retraining bloqueado: {policy_msg}"))
                execution_log.append(("⚠", "Solicitud de aprobación manual generada (Human in the Loop)."))
            else:
                execution_log.append(("✓", "Airflow DAG 'training_pipeline' disparado."))
                execution_log.append(("…", "En modo demo — Airflow no está corriendo localmente."))

        mlops.log_activity(metrics, action, "dashboard")

        st.session_state.result = {
            "metrics": metrics, "scenario": scenario,
            "drift_s": drift_s, "perf_s": perf_s,
            "latency_s": latency_s, "err_s": err_s,
            "diagnosis": diagnosis, "diag_mode": diag_mode,
            "action": action, "policy_blocked": policy_blocked,
            "policy_msg": policy_msg if policy_blocked else "",
            "execution_log": execution_log,
        }
        st.session_state.last_run_ts = datetime.now().strftime("%H:%M:%S")


# ══════════════════════════════════════════════════════════════════════════════
#  PANEL PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

if "result" not in st.session_state:
    st.markdown("""
    <div class="empty">
      <div class="empty-icon">⬡</div>
      <div class="empty-title">MLOps Agent Dashboard</div>
      <div class="empty-sub">
        Selecciona un escenario en el panel izquierdo<br>
        y pulsa <strong>Ejecutar Agente</strong> para iniciar el análisis.
      </div>
    </div>
    """, unsafe_allow_html=True)

else:
    r = st.session_state.result
    m = r["metrics"]

    # ── Header ──────────────────────────────────────────────────────────────
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown(
            f'<h2 style="font-family:\'IBM Plex Mono\',monospace;font-size:1.25rem;'
            f'font-weight:600;color:#E2E8F0;margin:0.25rem 0 0.1rem">'
            f'{m.get("model_version","churn_model")}</h2>'
            f'<div style="font-size:0.78rem;color:#6B82A0">'
            f'Fecha: {m.get("date","—")} &nbsp;·&nbsp; '
            f'Escenario: {SCENARIO_LABEL[r["scenario"]]} &nbsp;·&nbsp; '
            f'Muestras: {m.get("samples",0):,}</div>',
            unsafe_allow_html=True,
        )
    with col_h2:
        action = r["action"]
        c = ACTION_COLOR[action]
        st.markdown(
            f'<div style="text-align:right;padding-top:0.4rem">'
            f'<span style="font-family:\'IBM Plex Mono\',monospace;font-size:0.72rem;'
            f'font-weight:700;letter-spacing:0.06em;color:{c};background:{ACTION_BG[action]};'
            f'border:1px solid {c}44;border-radius:6px;padding:0.3rem 0.8rem">'
            f'{ACTION_LABEL[action]}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)

    # ── Metric cards ─────────────────────────────────────────────────────────
    section("Métricas del período")

    c1, c2, c3, c4 = st.columns(4)

    auc_drop = m.get("auc_previous", 0) - m.get("auc", 0)
    auc_drop_pct = auc_drop / m.get("auc_previous", 1) * 100 if m.get("auc_previous") else 0

    with c1:
        pct = min(m["psi"] / 0.5, 1.0) * 100
        st.markdown(metric_card(
            "PSI — Drift de distribución",
            f'{m["psi"]:.3f}', "",
            r["drift_s"], pct,
            "< 0.10 OK · < 0.20 WARN · ≥ 0.20 CRIT",
        ), unsafe_allow_html=True)

    with c2:
        pct = m.get("auc", 0) * 100
        trend = f"▼ {auc_drop_pct:.1f}% vs anterior" if auc_drop_pct > 0 else "— sin caída"
        st.markdown(metric_card(
            "AUC — Performance del modelo",
            f'{m["auc"]:.3f}', "",
            r["perf_s"], pct,
            trend,
        ), unsafe_allow_html=True)

    with c3:
        pct = min(m.get("latency", 0) / 300, 1.0) * 100
        st.markdown(metric_card(
            "Latencia P50 — Inferencia",
            f'{m.get("latency",0)}', "ms",
            r["latency_s"], pct,
            "< 100ms OK · < 200ms WARN",
        ), unsafe_allow_html=True)

    with c4:
        er = m.get("error_rate", 0)
        pct = min(er / 0.10, 1.0) * 100
        st.markdown(metric_card(
            "Tasa de error — Endpoint",
            f'{er*100:.1f}', "%",
            r["err_s"], pct,
            "< 1% OK · < 3% WARN",
        ), unsafe_allow_html=True)

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    # ── Diagnosis + Decision ─────────────────────────────────────────────────
    section("Diagnóstico y decisión")

    col_diag, col_dec = st.columns([3, 2])

    with col_diag:
        st.markdown(
            f'<div class="card">'
            f'<span class="pill">{r["diag_mode"]}</span>'
            f'<div class="diag-text">{r["diagnosis"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_dec:
        st.markdown(decision_block(action), unsafe_allow_html=True)

        if r["policy_blocked"]:
            st.markdown(
                f'<div style="background:#2A2208;border:1px solid #ECC94B44;border-radius:8px;'
                f'padding:0.85rem 1rem;margin-top:0.6rem">'
                f'<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;'
                f'color:#ECC94B;margin-bottom:0.3rem">⊘ HUMAN IN THE LOOP</div>'
                f'<div style="font-size:0.8rem;color:#E2E8F0;line-height:1.55">'
                f'{r["policy_msg"]}</div></div>',
                unsafe_allow_html=True,
            )

    # ── Execution log ────────────────────────────────────────────────────────
    section("Log de ejecución")

    ts = st.session_state.get("last_run_ts", "—")
    log_html = '<div class="card">'
    for icon, msg in r["execution_log"]:
        color = {"✓": "var(--ok)", "⚠": "var(--warn)", "…": "var(--muted)", "✕": "var(--crit)"}.get(icon, "var(--text)")
        log_html += (
            f'<div class="log">'
            f'<span class="log-ts">{ts}</span>'
            f'<span style="color:{color};flex-shrink:0">{icon}</span>'
            f'<span>{msg}</span></div>'
        )
    log_html += "</div>"
    st.markdown(log_html, unsafe_allow_html=True)

    # ── History ──────────────────────────────────────────────────────────────
    log_path = ROOT / "logs" / "agent_log.csv"
    if log_path.exists():
        section("Historial de decisiones")
        try:
            df = pd.read_csv(log_path)
            df.columns = [c.capitalize() for c in df.columns]
            df = df.iloc[::-1].reset_index(drop=True)

            color_map = {"wait": "🟢", "alert": "🟡", "retrain": "🔴"}
            if "Decision" in df.columns:
                df["Decision"] = df["Decision"].map(lambda x: f"{color_map.get(x,'●')} {x}")

            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception:
            st.caption("No se pudo cargar el historial.")
