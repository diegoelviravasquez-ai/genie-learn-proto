"""
student_learning_module.py — Módulo de aprendizaje del investigador
====================================================================
Streamlit autónomo: el investigador usa el chatbot GENIE para aprender
las tecnologías del proyecto (Pandas, Plotly, K-Means, RAG, etc.).
Sus interacciones generan datos reales para el dashboard.

Ejecutar: streamlit run student_learning_module.py --server.port 8504
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Imports del proyecto (mismo core que app.py)
from middleware import PedagogicalConfig
from rag_pipeline import get_rag_pipeline, SAMPLE_COURSE_CONTENT
from llm_client import get_llm_client
from ecosystem_orchestrator import EcosystemOrchestrator


def _LLMAdapter(llm):
    class Adapter:
        def __init__(self, inner):
            self._llm = inner
            self.model_name = getattr(inner, "model_name", "unknown")

        def generate(self, system_prompt, user_prompt, context="", conversation_history=None):
            r = self._llm.chat(system_prompt=system_prompt, user_prompt=user_prompt, context=context)
            return r["response"] if isinstance(r, dict) else str(r)
    return Adapter(llm)


# ─── System prompts por tema (contexto Learning Analytics) ─────────────────

TOPIC_PROMPTS = {
    "Python básico": (
        "Eres un tutor de Python básico para un investigador de Learning Analytics. "
        "El estudiante aprende Python para implementar análisis de datos educativos. "
        "Usa ejemplos relacionados con listas de estudiantes, diccionarios de métricas y scripts de análisis."
    ),
    "Pandas": (
        "Eres un tutor de Pandas para un investigador de Learning Analytics. "
        "El estudiante aprende Pandas para analizar logs de chatbot educativo. "
        "Usa ejemplos con DataFrames de interacciones estudiante-chatbot."
    ),
    "Plotly": (
        "Eres un tutor de Plotly para un investigador de LA. "
        "El estudiante aprende visualizaciones para dashboards de analytics educativos. "
        "Usa ejemplos con gráficos de Bloom, autonomía y clusters de estudiantes."
    ),
    "Streamlit": (
        "Eres un tutor de Streamlit para un investigador de Learning Analytics. "
        "El estudiante aprende Streamlit para construir dashboards de investigación. "
        "Usa ejemplos con métricas, tablas y gráficos interactivos de datos educativos."
    ),
    "scikit-learn / K-Means": (
        "Eres un tutor de K-Means para un investigador de LA. "
        "El estudiante aprende clustering para agrupar perfiles de estudiantes. "
        "Usa ejemplos con datos de bloom_level y autonomy_score."
    ),
    "RAG + ChromaDB": (
        "Eres un tutor de RAG y ChromaDB para un investigador de LA. "
        "El estudiante aprende retrieval aumentado para chatbots educativos. "
        "Usa ejemplos con documentos de curso, embeddings y contexto para el LLM."
    ),
    "PostgreSQL": (
        "Eres un tutor de PostgreSQL para un investigador de LA. "
        "El estudiante aprende a almacenar interacciones, perfiles y logs de chatbot. "
        "Usa ejemplos con tablas de estudiantes, interacciones y métricas."
    ),
    "FastAPI": (
        "Eres un tutor de FastAPI para un investigador de LA. "
        "El estudiante aprende a exponer APIs del chatbot educativo. "
        "Usa ejemplos con endpoints de interacción, analytics y configuración pedagógica."
    ),
    "Git": (
        "Eres un tutor de Git para un investigador de LA. "
        "El estudiante aprende control de versiones para proyectos de investigación en código. "
        "Usa ejemplos con repos de análisis, dashboards y pipelines de datos."
    ),
}

TOPICS = list(TOPIC_PROMPTS.keys())
MODE_OPTIONS = ["Socrático", "Ejemplos", "Pistas", "Directo"]
MODE_TO_CONFIG = {"Socrático": "socratic", "Ejemplos": "examples", "Pistas": "hints", "Directo": "direct"}

BLOOM_LABELS = {1: "Recordar", 2: "Comprender", 3: "Aplicar", 4: "Analizar", 5: "Evaluar", 6: "Crear"}
BLOOM_COLORS = {1: "#9E9E9E", 2: "#42A5F5", 3: "#66BB6A", 4: "#FFA726", 5: "#EF5350", 6: "#AB47BC"}


def _detect_question_type(prompt: str) -> tuple[int, str]:
    """Detecta tipo de pregunta por keywords → Bloom 1-6 y etiqueta."""
    p = (prompt or "").strip().lower()
    if any(x in p for x in ["diseña", "crea", "propón", "implementa un"]):
        return 6, "Crear"
    if any(x in p for x in ["mejor forma", "evalúa", "qué opinas", "valorar"]):
        return 5, "Evaluar"
    if any(x in p for x in ["por qué", "diferencia entre", "analiza", "relación"]):
        return 4, "Analizar"
    if any(x in p for x in ["cómo se hace", "ejemplo", "pasos para", "cómo puedo"]):
        return 3, "Aplicar"
    if any(x in p for x in ["cómo funciona", "explica", "qué significa"]):
        return 2, "Comprender"
    if any(x in p for x in ["qué es", "define", "cuántos", "cuál es la definición"]):
        return 1, "Recordar"
    return 3, "Aplicar"  # default


# ─── Page config y estilos ────────────────────────────────────────────────

st.set_page_config(
    page_title="GENIE Learn — Módulo de Aprendizaje",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #0a0a0a; color: #ffffff; }
    .stApp { background: linear-gradient(180deg, #0a0a0a 0%, #0d0d0d 100%); }
    .main-header {
        font-size: 2rem; font-weight: 300; color: #ffffff;
        padding: 24px 0; margin-bottom: 8px; letter-spacing: -0.02em;
    }
    .main-header-accent { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .subtitle { color: #b0b0b0; font-size: 1rem; margin-bottom: 24px; }
    .footer-note { margin-top: 32px; font-size: 0.8rem; color: #666; }
    .scaffolding-indicator { background: rgba(102, 126, 234, 0.12); border-radius: 12px; padding: 8px 12px; margin-top: 8px; font-size: 0.85rem; color: #888; }
    [data-testid="stSidebar"] { background: #0d0d0d; border-right: 1px solid rgba(255,255,255,0.06); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="main-header">🎓 Módulo de Aprendizaje — GENIE Learn</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">El investigador aprende las tecnologías del proyecto usando el sistema que está construyendo</p>',
    unsafe_allow_html=True,
)

# ─── Inicialización del orchestrator (session_state) ───────────────────────

if "slm_config" not in st.session_state:
    st.session_state.slm_config = PedagogicalConfig()
if "slm_rag" not in st.session_state:
    st.session_state.slm_rag = get_rag_pipeline(use_openai=True)
    st.session_state.slm_rag.ingest_text(SAMPLE_COURSE_CONTENT, "Fundamentos_Programacion.pdf")
if "slm_llm" not in st.session_state:
    st.session_state.slm_llm = get_llm_client()
if "slm_orchestrator" not in st.session_state:
    adapter = _LLMAdapter(st.session_state.slm_llm)
    st.session_state.slm_orchestrator = EcosystemOrchestrator(
        st.session_state.slm_config,
        rag_pipeline=st.session_state.slm_rag,
        llm_client=adapter,
    )
if "slm_chat_history" not in st.session_state:
    st.session_state.slm_chat_history = []
if "slm_records" not in st.session_state:
    st.session_state.slm_records = []  # [{topic, bloom, mode, bloom_detected}]


# ─── Sidebar: API Key (antes del selector de tema) ─────────────────────────

with st.sidebar:
    api_key = st.text_input(
        "🔑 API Key Anthropic",
        type="password",
        placeholder="sk-ant-...",
    )
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key


# ─── SECCIÓN 1 — Mi sesión de aprendizaje ─────────────────────────────────

st.markdown("---")
st.markdown("### 1. Mi sesión de aprendizaje")

col_tema, col_modo = st.columns(2)
with col_tema:
    tema = st.selectbox("Tema", TOPICS, key="slm_tema")
with col_modo:
    modo_label = st.selectbox("Modo", MODE_OPTIONS, key="slm_modo")

# Inyectar system prompt del tema y modo en config
config = st.session_state.slm_config
config.role_play = TOPIC_PROMPTS[tema]
config.scaffolding_mode = MODE_TO_CONFIG[modo_label]
orch = st.session_state.slm_orchestrator
student_id = "investigador"

# Chat
for msg in st.session_state.slm_chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("meta"):
            m = msg["meta"]
            if m.get("bloom_level") is not None:
                b = m["bloom_level"]
                lbl = BLOOM_LABELS.get(b, "?")
                st.markdown(f'<div class="scaffolding-indicator">Bloom: {lbl} (N{b})</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Pregunta sobre el tema seleccionado..."):
    st.session_state.slm_chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            result = orch.process_interaction(student_id, prompt)
        st.markdown(result.response_text)
        bloom_num = getattr(result, "bloom_level", 3)
        bloom_label = BLOOM_LABELS.get(bloom_num, "Aplicar")
        st.markdown(f'<div class="scaffolding-indicator">Bloom: {bloom_label} (N{bloom_num})</div>', unsafe_allow_html=True)

        st.session_state.slm_chat_history.append({
            "role": "assistant",
            "content": result.response_text,
            "meta": {"bloom_level": bloom_num},
        })
        detected_bloom, detected_label = _detect_question_type(prompt)
        st.session_state.slm_records.append({
            "topic": tema,
            "bloom": bloom_num,
            "mode": modo_label,
            "bloom_detected": detected_bloom,
            "label_detected": detected_label,
        })
        # Log al middleware para que aparezca en dashboard
        if not result.was_blocked:
            pre = {
                "processed_prompt": prompt,
                "detected_topics": getattr(result, "detected_topics", [tema]),
                "scaffolding_level": getattr(result, "scaffolding_level", 0),
                "copy_paste_score": getattr(result, "copy_paste_score", 0),
                "allowed": True,
                "block_reason": "",
            }
            post = {"response": result.response_text, "hallucination_injected": getattr(result, "hallucination_injected", False)}
            orch.middleware.log_interaction(student_id=student_id, prompt_raw=prompt, pre_result=pre, response_raw=result.response_text, post_result=post, response_time_ms=getattr(result, "processing_time_ms", 0))
    st.rerun()


# ─── SECCIÓN 2 — Mi progreso en tiempo real ───────────────────────────────

st.markdown("---")
st.markdown("### 2. Mi progreso en tiempo real")

n = len(st.session_state.slm_records)
st.metric("Preguntas en esta sesión", n)

if n > 0:
    last = st.session_state.slm_records[-1]
    bloom_num = last["bloom"]
    label_detected = last["label_detected"]
    color = BLOOM_COLORS.get(bloom_num, "#888")
    st.markdown(f'**Bloom estimado última pregunta:** <span style="background:{color};color:white;padding:4px 10px;border-radius:8px;">N{bloom_num} — {BLOOM_LABELS.get(bloom_num, "?")}</span>', unsafe_allow_html=True)
    st.markdown(f'**Tipo de pregunta detectado:** {label_detected} (Bloom {last["bloom_detected"]})')

    # Mini heatmap: tema × sesión (cuántas preguntas por tema)
    recs = st.session_state.slm_records
    topic_counts = pd.Series([r["topic"] for r in recs]).value_counts()
    if not topic_counts.empty:
        st.markdown("**Preguntas por tema (esta sesión)**")
        fig = go.Figure(go.Bar(x=topic_counts.index, y=topic_counts.values, marker_color="#667eea"))
        fig.update_layout(template="plotly_white", height=220, margin=dict(t=20, b=40), xaxis_tickangle=-45)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#ccc"))
        st.plotly_chart(fig, use_container_width=True)


# ─── SECCIÓN 3 — Exportar mi perfil al dashboard ──────────────────────────

st.markdown("---")
st.markdown("### 3. Exportar mi perfil al dashboard")

if st.session_state.slm_records:
    recs = st.session_state.slm_records
    df_rec = pd.DataFrame(recs)
    resumen = df_rec.groupby("topic").agg(
        n_preguntas=("topic", "count"),
        bloom_medio=("bloom", "mean"),
        modo_usado=("mode", lambda s: s.mode().iloc[0] if len(s) else ""),
    ).reset_index()
    resumen.columns = ["Tema", "N preguntas", "Bloom medio", "Modo usado"]
    st.dataframe(resumen.style.format({"Bloom medio": "{:.2f}"}), use_container_width=True, hide_index=True)

    bloom_medio_global = df_rec["bloom"].mean()
    modos = df_rec["mode"].tolist()
    modo_mas_usado = max(set(modos), key=modos.count)
    pct_directo = modos.count("Directo") / len(modos)
    autonomy = 1.0 - pct_directo
    total_prompts = len(recs)
    risk = 0.1 if bloom_medio_global > 3 else 0.4
    cluster = "Explorador" if bloom_medio_global > 4 else "Moderado"

    if st.button("Añadir mi perfil a demo_completo.py", type="primary", key="slm_export"):
        code = f'''# Añade esto a STUDENTS en dashboard/demo_completo.py:
{{
  "name": "Diego V. (tú)",
  "bloom_mean": {bloom_medio_global:.2f},
  "autonomy": {autonomy:.2f},
  "prompts": {total_prompts},
  "mode": "{MODE_TO_CONFIG.get(modo_mas_usado, 'socratic')}",
  "risk": {risk:.2f},
  "cluster": "{cluster}"
}}'''
        st.code(code, language="python")
else:
    st.caption("Haz al menos una pregunta para ver el resumen y exportar tu perfil.")

st.markdown(
    '<p class="footer-note">Tus interacciones demuestran el sistema funcionando con datos reales · CP25/152 GSIC/EMIC</p>',
    unsafe_allow_html=True,
)
