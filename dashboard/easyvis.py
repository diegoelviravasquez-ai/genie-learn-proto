"""
dashboard/easyvis.py — Dashboard Streamlit inspirado en EasyVis (Mohseni 2025)
===============================================================================
Tres pestañas: Vista Clase, Vista Estudiante, Configuración Docente.
Datos desde analytics/bridge.py; si no hay datos, 20 estudiantes sintéticos.
Estilo: plotly_white, #2E4057, #048A81.

Ejecutar: streamlit run dashboard/easyvis.py --server.port 8502
"""

import sys
from pathlib import Path

# Asegurar raíz del proyecto en path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_BLUE = "#2E4057"
COLOR_TEAL = "#048A81"

st.set_page_config(
    page_title="EasyVis — GENIE Learn",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
    .kpi-card {{
        background: linear-gradient(135deg, {COLOR_BLUE} 0%, #1a2634 100%);
        color: white;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }}
    .kpi-card h3 {{ margin: 0; font-size: 0.85rem; opacity: 0.9; }}
    .kpi-card p {{ margin: 0.25rem 0 0; font-size: 1.5rem; font-weight: 600; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _synthetic_profiles(n: int = 20) -> pd.DataFrame:
    """Genera n estudiantes sintéticos con columnas de get_student_profiles."""
    np.random.seed(42)
    return pd.DataFrame({
        "student_id": [f"est_{i:02d}" for i in range(1, n + 1)],
        "total_prompts": np.random.randint(8, 120, n),
        "bloom_mean": np.random.uniform(1.5, 5.5, n),
        "bloom_max": np.random.randint(2, 7, n),
        "autonomy_score": np.random.uniform(0.2, 0.95, n),
        "pct_socratic": np.random.uniform(0.1, 0.9, n),
        "kappa_p": np.random.uniform(0.2, 0.9, n),
    })


def _synthetic_interactions(student_ids: list, weeks: int = 8) -> pd.DataFrame:
    """Interacciones sintéticas por estudiante y semana para heatmap/tiempo."""
    np.random.seed(43)
    rows = []
    for sid in student_ids:
        n_inter = np.random.randint(3, 25)
        for _ in range(n_inter):
            w = np.random.randint(0, weeks)
            rows.append({
                "student_id": sid,
                "bloom_level": float(np.random.randint(1, 7)),
                "scaffolding_mode": np.random.choice(
                    ["socratic", "hints", "examples", "direct"], p=[0.4, 0.3, 0.2, 0.1]
                ),
                "timestamp": pd.Timestamp.now() - pd.to_timedelta((weeks - w) * 7, unit="D"),
            })
    return pd.DataFrame(rows)


@st.cache_data(ttl=120)
def _load_data(course_id=None):
    """Carga perfiles + clusters + dropout; si vacío, datos sintéticos."""
    try:
        from analytics.bridge import get_student_profiles, get_clusters, get_dropout_risk
        profiles = get_student_profiles(course_id=course_id)
        if profiles.empty or len(profiles) < 2:
            profiles = _synthetic_profiles(20)
            profiles["kappa_p"] = (
                (profiles["bloom_mean"] - 1) / 5 * 0.4
                + profiles["autonomy_score"] * 0.4
                + profiles["pct_socratic"] * 0.2
            )
        profiles = get_clusters(profiles)
        profiles = get_dropout_risk(profiles)
        return profiles, False
    except Exception:
        profiles = _synthetic_profiles(20)
        profiles["kappa_p"] = (
            (profiles["bloom_mean"] - 1) / 5 * 0.4
            + profiles["autonomy_score"] * 0.4
            + profiles["pct_socratic"] * 0.2
        )
        try:
            from analytics.bridge import get_clusters, get_dropout_risk
            profiles = get_clusters(profiles)
            profiles = get_dropout_risk(profiles)
        except Exception:
            profiles["cluster"] = 0
            profiles["cluster_label"] = "Moderado"
            profiles["pca_x"] = np.random.randn(len(profiles)) * 0.5
            profiles["pca_y"] = np.random.randn(len(profiles)) * 0.5
            profiles["dropout_risk"] = np.clip(np.random.rand(len(profiles)), 0, 1)
        return profiles, True


def _interactions_for_heatmap_and_student(course_id=None):
    """DataFrame de interacciones; si vacío, sintético."""
    try:
        from data.database import get_interactions_df
        df = get_interactions_df(course_id=course_id)
        if df.empty:
            return None
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        if df.empty:
            return None
        return df
    except Exception:
        return None


# ─── Carga global ─────────────────────────────────────────────────────────

course_id = st.sidebar.text_input("Course ID (opcional)", value="", key="easyvis_course")
profiles, use_synthetic = _load_data(course_id=None if not course_id else course_id)

# Interacciones crudas para heatmap y vista estudiante
raw_interactions = _interactions_for_heatmap_and_student(course_id=None if not course_id else course_id)
if raw_interactions is None and not profiles.empty:
    raw_interactions = _synthetic_interactions(profiles["student_id"].tolist())

# ─── Tab 1: Vista Clase ───────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["Vista Clase", "Vista Estudiante", "Configuración Docente"])

with tab1:
    st.subheader("Vista Clase")

    if profiles.empty:
        st.info("No hay datos de perfiles. Usa datos sintéticos en Configuración o ingesta interacciones.")
    else:
        # KPI cards
        n_students = len(profiles)
        bloom_medio = float(profiles["bloom_mean"].mean())
        alertas_cp = int((profiles["autonomy_score"] < 0.3).sum())

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f'<div class="kpi-card"><h3>Total estudiantes</h3><p>{n_students}</p></div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f'<div class="kpi-card"><h3>Bloom medio del grupo</h3><p>{bloom_medio:.2f}</p></div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f'<div class="kpi-card"><h3>Alertas copy-paste (autonomy &lt; 0.3)</h3><p>{alertas_cp}</p></div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Scatter: pca_x vs pca_y, color=cluster_label, size=total_prompts
        if "pca_x" in profiles.columns and "pca_y" in profiles.columns:
            fig_scatter = px.scatter(
                profiles,
                x="pca_x",
                y="pca_y",
                color="cluster_label",
                size="total_prompts",
                hover_data=["student_id", "kappa_p"],
                template="plotly_white",
                color_discrete_sequence=[COLOR_BLUE, COLOR_TEAL, "#7BA098", "#1a2634"],
            )
            fig_scatter.update_layout(
                title="Clusters en espacio PCA (color=cluster, size=total_prompts)",
                xaxis_title="PCA 1",
                yaxis_title="PCA 2",
                height=450,
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.caption("Scatter no disponible: faltan pca_x/pca_y.")

        # Heatmap: student_id × semana, valor = bloom_level
        st.markdown("**Heatmap: Bloom por estudiante y semana**")
        if raw_interactions is not None and not raw_interactions.empty:
            raw_interactions = raw_interactions.copy()
            raw_interactions["week_iso"] = pd.to_datetime(raw_interactions["timestamp"]).dt.isocalendar().year.astype(str) + "-W" + pd.to_datetime(raw_interactions["timestamp"]).dt.isocalendar().week.astype(str).str.zfill(2)
            heat = raw_interactions.groupby(["student_id", "week_iso"], as_index=False)["bloom_level"].mean()
            pivot = heat.pivot(index="student_id", columns="week_iso", values="bloom_level").fillna(0)
            fig_heat = go.Figure(
                data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    colorscale=[[0, "#e8f4f3"], [0.5, COLOR_TEAL], [1, COLOR_BLUE]],
                    hovertemplate="%{y} · %{x}<br>Bloom medio: %{z:.2f}<extra></extra>",
                )
            )
            fig_heat.update_layout(
                template="plotly_white",
                xaxis_title="Semana (ISO)",
                yaxis_title="Estudiante",
                height=400,
            )
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.caption("Sin interacciones para heatmap.")

# ─── Tab 2: Vista Estudiante ──────────────────────────────────────────────

with tab2:
    st.subheader("Vista Estudiante")

    if profiles.empty:
        st.info("No hay perfiles. Carga datos o usa sintéticos.")
    else:
        student_ids = profiles["student_id"].tolist()
        sel_id = st.selectbox("Estudiante", student_ids, key="easyvis_student")

        row = profiles[profiles["student_id"] == sel_id].iloc[0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bloom medio", f"{row['bloom_mean']:.2f}")
        m2.metric("Autonomy score", f"{row['autonomy_score']:.2f}")
        m3.metric("Kappa P", f"{row['kappa_p']:.2f}")
        m4.metric("Riesgo abandono", f"{row['dropout_risk']:.2f}")

        st.markdown("---")

        if raw_interactions is not None and not raw_interactions.empty:
            one = raw_interactions[raw_interactions["student_id"] == sel_id]
            if not one.empty:
                # Bar: distribución scaffolding_mode
                mode_counts = one["scaffolding_mode"].value_counts()
                fig_bar = go.Figure(
                    data=[go.Bar(x=mode_counts.index.astype(str), y=mode_counts.values, marker_color=COLOR_TEAL)]
                )
                fig_bar.update_layout(
                    title="Uso de modo de scaffolding",
                    template="plotly_white",
                    xaxis_title="Modo",
                    yaxis_title="Nº interacciones",
                    height=320,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Línea temporal: bloom_level en el tiempo
                one = one.sort_values("timestamp")
                fig_line = go.Figure(
                    data=[go.Scatter(x=one["timestamp"], y=one["bloom_level"], mode="lines+markers", line=dict(color=COLOR_BLUE), marker=dict(size=6))]
                )
                fig_line.update_layout(
                    title="Bloom level a lo largo del tiempo",
                    template="plotly_white",
                    xaxis_title="Fecha",
                    yaxis_title="Bloom level",
                    height=320,
                )
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.caption("Sin interacciones para este estudiante.")
        else:
            st.caption("Sin datos de interacciones para gráficos individuales.")

# ─── Tab 3: Configuración Docente ─────────────────────────────────────────

with tab3:
    st.subheader("Configuración Docente")

    socratic = st.toggle(
        "Modo socrático (priorizar preguntas guía)",
        value=True,
        key="easyvis_toggle_socratic",
    )
    max_prompts = st.slider(
        "Máximo de consultas diarias por estudiante",
        min_value=1,
        max_value=50,
        value=20,
        key="easyvis_slider_prompts",
    )
    hallucination_pct = st.slider(
        "Tasa de alucinación pedagógica (%)",
        min_value=0,
        max_value=30,
        value=0,
        step=5,
        format="%d",
        key="easyvis_slider_hallucination",
    )

    if st.button("Aplicar configuración", type="primary", key="easyvis_apply"):
        st.success(
            f"Configuración aplicada: socrático={socratic}, "
            f"máx prompts={max_prompts}, alucinación={hallucination_pct}%."
        )

    st.caption("Estos controles replican el panel de app.py. La aplicación principal no se modifica desde aquí.")
