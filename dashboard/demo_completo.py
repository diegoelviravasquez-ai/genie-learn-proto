"""
dashboard/demo_completo.py — Dashboard Streamlit de demostración académica
===========================================================================
Datos sintéticos de 8 estudiantes. Loop: chatbot → analytics → decisión docente.
Basado en EasyVis (Mohseni et al., 2025).
"""

import sys
from pathlib import Path

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
COLOR_ORANGE = "#F4A261"

# ─── Datos sintéticos fijos (8 estudiantes) ───────────────────────────────

STUDENTS = [
    {"name": "Ana García", "bloom_mean": 4.2, "autonomy": 0.8, "prompts": 23, "mode": "socratic", "risk": 0.1, "cluster": "Explorador"},
    {"name": "Carlos López", "bloom_mean": 2.1, "autonomy": 0.3, "prompts": 45, "mode": "direct", "risk": 0.7, "cluster": "Delegador"},
    {"name": "María Ruiz", "bloom_mean": 3.8, "autonomy": 0.7, "prompts": 18, "mode": "hints", "risk": 0.2, "cluster": "Verificador"},
    {"name": "David Chen", "bloom_mean": 1.9, "autonomy": 0.2, "prompts": 52, "mode": "direct", "risk": 0.8, "cluster": "Delegador"},
    {"name": "Sara Kim", "bloom_mean": 5.1, "autonomy": 0.9, "prompts": 12, "mode": "socratic", "risk": 0.05, "cluster": "Explorador"},
    {"name": "Javier Mora", "bloom_mean": 3.2, "autonomy": 0.5, "prompts": 31, "mode": "progressive", "risk": 0.4, "cluster": "Moderado"},
    {"name": "Laura Sanz", "bloom_mean": 4.5, "autonomy": 0.75, "prompts": 20, "mode": "hints", "risk": 0.15, "cluster": "Explorador"},
    {"name": "Ahmed Al-Rashid", "bloom_mean": 2.8, "autonomy": 0.4, "prompts": 38, "mode": "challenge", "risk": 0.55, "cluster": "Moderado"},
]

PCT_SOCRATIC = {"socratic": 1.0, "hints": 0.6, "progressive": 0.5, "challenge": 0.4, "direct": 0.0}


def _build_df():
    rows = []
    for s in STUDENTS:
        bloom_norm = (s["bloom_mean"] - 1) / 5.0
        pct = PCT_SOCRATIC.get(s["mode"], 0.3)
        kappa_p = bloom_norm * 0.4 + s["autonomy"] * 0.4 + pct * 0.2
        rows.append({
            **s,
            "kappa_p": round(kappa_p, 3),
            "autonomy_score": s["autonomy"],
            "total_prompts": s["prompts"],
        })
    return pd.DataFrame(rows)


def _bloom_by_week(name: str, base_bloom: float, trend: float = 0.08) -> list:
    """Simula bloom por semana (6 semanas) con ligera tendencia (determinista por nombre)."""
    seed = hash(name) % 1000
    np.random.seed(seed)
    return [round(base_bloom + trend * w + np.random.uniform(-0.15, 0.15), 2) for w in range(6)]


# ─── Page config y estilos ────────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="GENIE Learn Analytics")

st.markdown(
    f"""
    <style>
    .main-title {{
        background: linear-gradient(90deg, {COLOR_BLUE} 0%, {COLOR_TEAL} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }}
    .subtitle {{ color: #666; font-size: 1rem; margin-bottom: 1.5rem; }}
    .kpi-card {{ padding: 1rem 1.25rem; border-radius: 12px; color: white; margin-bottom: 0.5rem; }}
    .kpi-green {{ background: {COLOR_TEAL}; }}
    .kpi-red {{ background: #c0392b; }}
    .kpi-blue {{ background: {COLOR_BLUE}; }}
    .kpi-violet {{ background: #6c5ce7; }}
    .footer {{ margin-top: 2rem; font-size: 0.8rem; color: #888; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-title">GENIE Learn · Visual Analytics Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Middleware pedagógico → datos → decisión docente</p>', unsafe_allow_html=True)

df = _build_df()

# ─── TAB 1 — Visión de Clase ──────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs(["Visión de Clase", "Comparar Estudiantes", "Estudiante Individual", "Panel Docente"])

with tab1:
    st.markdown("**Curso: Programación Python · 8 estudiantes · Semana 6**")
    st.markdown("---")

    bloom_medio = float(df["bloom_mean"].mean())
    en_riesgo = int((df["risk"] > 0.5).sum())
    autonomia_pct = int(df["autonomy_score"].mean() * 100)
    modo_mas_usado = df["mode"].mode().iloc[0] if not df["mode"].empty else "socratic"
    modo_label = modo_mas_usado.capitalize()

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="kpi-card kpi-green"><strong>Bloom medio del grupo</strong><br><span style="font-size:1.8rem;">{bloom_medio:.1f}</span></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="kpi-card kpi-red"><strong>⚠ Estudiantes en riesgo</strong><br><span style="font-size:1.8rem;">{en_riesgo}</span></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="kpi-card kpi-blue"><strong>Autonomía media</strong><br><span style="font-size:1.8rem;">{autonomia_pct}%</span></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="kpi-card kpi-violet"><strong>Modo más usado</strong><br><span style="font-size:1.2rem;">{modo_label}</span></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Scatter: X=autonomy, Y=bloom_mean, size=prompts, color=cluster
    fig = go.Figure()
    clusters = df["cluster"].unique()
    colors = {c: px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)] for i, c in enumerate(clusters)}
    for cl in clusters:
        sub = df[df["cluster"] == cl]
        fig.add_trace(go.Scatter(
            x=sub["autonomy_score"],
            y=sub["bloom_mean"],
            mode="markers+text",
            text=sub["name"],
            textposition="top center",
            marker=dict(size=sub["total_prompts"] * 2.5, color=colors[cl], line=dict(width=1, color="white")),
            name=cl,
            hovertemplate="<b>%{text}</b><br>κ(P)=%{customdata[0]:.2f}<br>Riesgo=%{customdata[1]:.2f}<extra></extra>",
            customdata=sub[["kappa_p", "risk"]].values,
        ))
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray", opacity=0.7)
    fig.add_hline(y=3, line_dash="dot", line_color="gray", opacity=0.7)
    fig.add_annotation(x=0.9, y=5, text="Alto Bloom / Alta Autonomía", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0.1, y=5, text="Alto Bloom / Baja Autonomía", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0.9, y=1.5, text="Bajo Bloom / Alta Autonomía", showarrow=False, font=dict(size=10))
    fig.add_annotation(x=0.1, y=1.5, text="ZONA DE RIESGO", showarrow=False, font=dict(size=11, color="darkred"))
    fig.add_shape(type="rect", x0=0, y0=0, x1=0.5, y1=3, line=dict(width=0), fillcolor="rgba(200,50,50,0.08)")
    fig.update_layout(
        title="Autonomía vs Bloom (tamaño = nº prompts, color = cluster)",
        xaxis_title="Autonomy score",
        yaxis_title="Bloom medio",
        template="plotly_white",
        height=500,
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Tabla comparativa con Riesgo coloreado
    display = df[["name", "bloom_mean", "autonomy_score", "kappa_p", "mode", "risk"]].copy()
    display.columns = ["Nombre", "Bloom", "Autonomía", "κ(P)", "Modo scaffolding", "Riesgo"]

    def color_risk(val):
        if isinstance(val, (int, float)) and 0 <= val <= 1:
            if val < 0.3:
                return "background-color: #27ae60; color: white; padding: 4px; border-radius: 4px;"
            if val < 0.6:
                return f"background-color: {COLOR_ORANGE}; color: white; padding: 4px; border-radius: 4px;"
            return "background-color: #c0392b; color: white; padding: 4px; border-radius: 4px;"
        return ""

    st.markdown("**Tabla comparativa**")
    styled = (
        display.style.format({"Bloom": "{:.2f}", "Autonomía": "{:.2f}", "κ(P)": "{:.2f}", "Riesgo": "{:.2f}"})
        .apply(lambda s: s.map(color_risk), subset=["Riesgo"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ─── TAB 2 — Comparar Estudiantes ─────────────────────────────────────────

with tab2:
    selected = st.multiselect("Elige 2 a 4 estudiantes para comparar", df["name"].tolist(), default=df["name"].iloc[:2].tolist(), max_selections=4)
    if len(selected) < 2:
        st.info("Selecciona al menos 2 estudiantes.")
    else:
        sub = df[df["name"].isin(selected)]
        # Radar: Bloom, Autonomía, Engagement, κ(P), Regularidad
        categories = ["Bloom", "Autonomía", "Engagement", "κ(P)", "Regularidad"]
        colors_radar = [COLOR_BLUE, COLOR_TEAL, COLOR_ORANGE, "#6c5ce7"]
        fig_radar = go.Figure()
        for i, (_, row) in enumerate(sub.iterrows()):
            engagement = (row["bloom_mean"] / 6 + row["autonomy_score"]) / 2
            regularidad = 1 - row["risk"]
            values = [
                row["bloom_mean"] / 6,
                row["autonomy_score"],
                engagement,
                row["kappa_p"],
                regularidad,
            ]
            fig_radar.add_trace(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill="toself", name=row["name"], line=dict(color=colors_radar[i % len(colors_radar)])))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, title="Radar comparativo", template="plotly_white", height=450)
        st.plotly_chart(fig_radar, use_container_width=True)

        # Bar: bloom por semana (6 semanas simuladas)
        weeks = [f"Semana {w+1}" for w in range(6)]
        fig_bar = go.Figure()
        for i, (_, row) in enumerate(sub.iterrows()):
            vals = _bloom_by_week(row["name"], row["bloom_mean"])
            fig_bar.add_trace(go.Bar(name=row["name"], x=weeks, y=vals, marker_color=colors_radar[i % len(colors_radar)]))
        fig_bar.update_layout(barmode="group", title="Bloom por semana (simulado)", template="plotly_white", height=350)
        st.plotly_chart(fig_bar, use_container_width=True)

        st.dataframe(sub[["name", "bloom_mean", "autonomy_score", "kappa_p", "total_prompts", "mode", "risk"]].style.format("{:.2f}", subset=pd.IndexSlice[:, ["bloom_mean", "autonomy_score", "kappa_p", "risk"]]), use_container_width=True, hide_index=True)

# ─── TAB 3 — Estudiante Individual ───────────────────────────────────────

with tab3:
    sel = st.selectbox("Estudiante", df["name"].tolist(), key="demo_sel")
    row = df[df["name"] == sel].iloc[0]
    risk_badge = "🟢" if row["risk"] < 0.3 else "🟡" if row["risk"] < 0.6 else "🔴"
    st.markdown(f"### {row['name']} · {row['cluster']} {risk_badge}")

    # Timeline Bloom 6 semanas
    weeks = list(range(1, 7))
    bloom_vals = _bloom_by_week(row["name"], row["bloom_mean"])
    fig_tl = go.Figure(go.Scatter(x=weeks, y=bloom_vals, mode="lines+markers", line=dict(color=COLOR_BLUE), marker=dict(size=10)))
    fig_tl.update_layout(title="Bloom por semana", xaxis_title="Semana", yaxis_title="Bloom", template="plotly_white", height=300)
    st.plotly_chart(fig_tl, use_container_width=True)

    # Donut: tipos de preguntas
    labels = ["Recordar", "Comprender", "Aplicar", "Analizar"]
    values = [20, 30, 35, 15]
    fig_donut = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.6, marker_colors=[COLOR_BLUE, COLOR_TEAL, COLOR_ORANGE, "#6c5ce7"])])
    fig_donut.update_layout(title="Distribución de tipos de preguntas", showlegend=True, template="plotly_white", height=320)
    st.plotly_chart(fig_donut, use_container_width=True)

    # Recomendación docente
    st.markdown("**Recomendación docente**")
    if row["risk"] > 0.5:
        rec = "⚠ Este estudiante muestra señales de abandono. Recomendado: cambiar a modo Socrático + reducir complejidad"
    elif row["bloom_mean"] < 2.5:
        rec = "📚 Nivel Bloom bajo sostenido. Recomendado: activar modo Ejemplos + revisar prerequisites"
    elif row["autonomy_score"] < 0.3:
        rec = "🔄 Alta dependencia del chatbot. Recomendado: activar límite de prompts diarios"
    else:
        rec = "✅ Progreso adecuado. Mantener configuración actual."
    st.info(rec)

# ─── TAB 4 — Panel Docente ───────────────────────────────────────────────

with tab4:
    st.markdown("**Configuración Pedagógica Activa**")
    col_left, col_right = st.columns(2)
    with col_left:
        modo = st.selectbox("Modo scaffolding", ["socratic", "hints", "examples", "progressive", "direct"], index=0, key="demo_modo")
        limite = st.slider("Límite de prompts diarios", 5, 50, 20, key="demo_limite")
        aluc = st.slider("Tasa de alucinación pedagógica (%)", 0, 30, 0, key="demo_aluc")
    with col_right:
        st.markdown("**Efecto esperado**")
        effects = []
        if modo == "socratic":
            effects.append("Los estudiantes desarrollarán razonamiento autónomo.")
        if limite <= 15:
            effects.append("Fomenta reflexión antes de preguntar.")
        if aluc > 0:
            effects.append("Activa lectura crítica y verificación.")
        if not effects:
            effects.append("Configuración estándar. Ajusta los controles para ver efectos.")
        for e in effects:
            st.markdown(f"- {e}")
    if st.button("Exportar informe del grupo", type="primary", key="demo_export"):
        st.success("Informe generado: grupo_python_semana6.pdf ✓")

st.markdown('<p class="footer">Basado en la metodología EasyVis (Mohseni et al., 2025)</p>', unsafe_allow_html=True)
