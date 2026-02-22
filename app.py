"""
GENIE LEARN — Prototipo de Chatbot Educativo con IA Generativa
===============================================================
Réplica funcional mínima del sistema descrito en:
  - TFG Pablo de Arriba Mendizábal (UVa, 2025)
  - Paper LAK 2026 (Ortega-Arranz et al.)

Tres vistas:
  1. Interfaz del Estudiante — Chat con el chatbot pedagógico
  2. Panel del Docente — Configuraciones pedagógicas + Analytics
  3. Dashboard Analytics — Visualización de datos de interacción

Ejecución: streamlit run app.py
Autor: Diego Elvira Vásquez — Prototipo para entrevista GSIC/EMIC CP25/152
"""

import streamlit as st
import time
import json
import random
from datetime import datetime, timedelta

from middleware import PedagogicalConfig, InteractionLog
from rag_pipeline import get_rag_pipeline, SAMPLE_COURSE_CONTENT
from llm_client import get_llm_client
from ecosystem_orchestrator import EcosystemOrchestrator
from cognitive_analyzer import CognitiveAnalyzer, EngagementProfiler, BLOOM_LEVELS
from trust_dynamics import TrustDynamicsAnalyzer


def _LLMAdapterForOrchestrator(llm):
    """Adapta llm_client.chat() (retorna dict) a la interfaz generate() (retorna str) del orquestador."""

    class Adapter:
        def __init__(self, inner):
            self._llm = inner
            self.model_name = getattr(inner, "model_name", "unknown")

        def generate(self, system_prompt, user_prompt, context="", conversation_history=None):
            r = self._llm.chat(
                system_prompt=system_prompt, user_prompt=user_prompt, context=context
            )
            return r["response"] if isinstance(r, dict) else str(r)

    return Adapter(llm)


# ──────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="GENIE Learn — Chatbot Pedagógico GenAI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS — Estilo Apple / Minimalista
# ──────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background-color: #0a0a0a;
        color: #ffffff;
    }

    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #0d0d0d 100%);
    }

    .main-header {
        font-size: 2rem;
        font-weight: 300;
        color: #ffffff;
        padding: 24px 0;
        margin-bottom: 32px;
        letter-spacing: -0.02em;
    }

    .main-header-accent {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .config-card {
        background: #1a1a1a;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.06);
    }

    .scaffolding-indicator {
        background: rgba(102, 126, 234, 0.12);
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 12px;
        font-size: 0.85rem;
        color: #888888;
    }

    .alert-copypaste {
        background: rgba(255,100,100,0.12);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
        color: #cc8888;
    }

    .stChatMessage { max-width: 85%; }

    [data-testid="stSidebar"] {
        background: #0d0d0d;
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdown"] {
        color: #888888;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #ffffff;
    }

    [data-testid="stSidebar"] .stRadio label:hover {
        color: #667eea;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff;
    }

    [data-testid="stMetricLabel"] {
        color: #888888;
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
    }

    .stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# INICIALIZACIÓN DE ESTADO
# ──────────────────────────────────────────────

def init_state():
    """Inicializa el estado de sesión de Streamlit."""
    if "config" not in st.session_state:
        st.session_state.config = PedagogicalConfig()
    if "rag" not in st.session_state:
        st.session_state.rag = get_rag_pipeline(use_openai=True)
        n = st.session_state.rag.ingest_text(
            SAMPLE_COURSE_CONTENT, "Fundamentos_Programacion.pdf"
        )
        st.session_state.rag_loaded = True
        st.session_state.rag_chunks = n
    if "llm" not in st.session_state:
        st.session_state.llm = get_llm_client()
    if "orchestrator" not in st.session_state:
        llm_adapted = _LLMAdapterForOrchestrator(st.session_state.llm)
        st.session_state.orchestrator = EcosystemOrchestrator(
            st.session_state.config,
            rag_pipeline=st.session_state.rag,
            llm_client=llm_adapted,
        )
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "student_id" not in st.session_state:
        st.session_state.student_id = "estudiante_01"
    if "demo_logs" not in st.session_state:
        st.session_state.demo_logs = generate_demo_data()
    if "cognitive_analyzer" not in st.session_state:
        st.session_state.cognitive_analyzer = CognitiveAnalyzer()
    if "trust_analyzer" not in st.session_state:
        st.session_state.trust_analyzer = TrustDynamicsAnalyzer()
    if "cognitive_history" not in st.session_state:
        st.session_state.cognitive_history = []  # CognitiveAnalysis per interaction


def generate_demo_data() -> list[dict]:
    """Genera datos de demo para el dashboard de analytics."""
    students = [f"est_{i:02d}" for i in range(1, 16)]
    topics = ["variables", "bucles", "funciones", "arrays", "recursión",
              "entrada/salida", "depuración", "conceptual", "ejercicio"]
    logs = []
    base_time = datetime.now() - timedelta(days=7)

    for i in range(120):
        t = base_time + timedelta(
            hours=random.randint(0, 168),
            minutes=random.randint(0, 59)
        )
        logs.append({
            "timestamp": t.isoformat(),
            "hour": t.hour,
            "day": t.strftime("%A"),
            "student_id": random.choice(students),
            "topics": random.sample(topics, k=random.randint(1, 3)),
            "scaffolding_level": random.choices([0, 1, 2, 3], weights=[40, 30, 20, 10])[0],
            "copy_paste_score": round(random.random() * 0.3 + (0.6 if random.random() < 0.15 else 0), 2),
            "response_time_ms": random.randint(300, 3000),
            "was_blocked": random.random() < 0.05,
            "hallucination_injected": random.random() < 0.08,
        })
    return logs


init_state()


# ──────────────────────────────────────────────
# SIDEBAR — NAVEGACIÓN
# ──────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Escudo_de_la_Universidad_de_Valladolid.svg/200px-Escudo_de_la_Universidad_de_Valladolid.svg.png", width=64)
    st.markdown("### GENIE Learn")
    st.markdown("*Chatbot Pedagógico GenAI*")
    st.markdown("---")

    view = st.radio(
        "Vista activa",
        ["Estudiante", "Docente — Configuración", "Docente — Analytics", "Investigador"],
        index=0,
    )

    st.markdown("---")

    orch = st.session_state.orchestrator
    model_name = orch.llm.model_name if orch.llm else "mock-demo"
    rag_stats = st.session_state.rag.get_stats()
    scaffolding_labels = {0: "Socrático", 1: "Pista", 2: "Ejemplo", 3: "Explicación"}
    current_state = orch.middleware.conversation_states.get(
        st.session_state.student_id, {"level": 0}
    )
    current_level = current_state.get("level", 0)
    scaff_label = scaffolding_labels.get(current_level, "?")

    LIGHTNING_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>'
    DOC_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6zm-1 2l5 5h-5V4z"/></svg>'
    BRAIN_SVG = '<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2a4 4 0 0 1 4 4c0 1.5-.8 2.8-2 3.5.5 1.2.5 2.5 0 3.8 1.2.7 2 2 2 3.5a4 4 0 0 1-8 0c0-1.5.8-2.8 2-3.5-.5-1.2-.5-2.5 0-3.8C8.8 8.8 8 7.5 8 6a4 4 0 0 1 4-4z"/></svg>'

    st.markdown(
        f"""
        <div style="display: flex; flex-direction: column; gap: 8px; margin-bottom: 16px;">
            <div style="background: #1a1a1a; border-radius: 12px; padding: 12px 16px;">
                <div style="font-size: 12px; color: #888; margin-bottom: 4px;">Estado del sistema</div>
                <div style="display: flex; flex-direction: column; gap: 8px;">
                    <div style="background: {"#3d3a00" if model_name == "mock-demo" else "#001a3d"}; color: {"#ffd700" if model_name == "mock-demo" else "#66b3ff"}; border-radius: 8px; padding: 8px 12px; display: flex; align-items: center; gap: 8px;">
                        <span style="display: flex; align-items: center;">{LIGHTNING_SVG}</span>
                        <div>
                            <div style="font-size: 12px; opacity: 0.9;">Modo</div>
                            <div style="font-size: 14px; font-weight: bold;">{"Modo demo (sin API key)" if model_name == "mock-demo" else model_name}</div>
                        </div>
                    </div>
                    <div style="background: #003d00; color: #00ff88; border-radius: 8px; padding: 8px 12px; display: flex; align-items: center; gap: 8px;">
                        <span style="display: flex; align-items: center;">{DOC_SVG}</span>
                        <div>
                            <div style="font-size: 12px; opacity: 0.9;">RAG</div>
                            <div style="font-size: 14px; font-weight: bold;">{rag_stats["total_chunks"]} fragmentos indexados</div>
                        </div>
                    </div>
                    <div style="background: #001a3d; color: #66b3ff; border-radius: 8px; padding: 8px 12px; display: flex; align-items: center; gap: 8px;">
                        <span style="display: flex; align-items: center;">{BRAIN_SVG}</span>
                        <div>
                            <div style="font-size: 12px; opacity: 0.9;">Scaffolding actual</div>
                            <div style="font-size: 14px; font-weight: bold;">{scaff_label}</div>
                        </div>
                    </div>
                </div>
            </div>
            <div style="font-size: 11px; color: #888; opacity: 0.5; margin-top: 8px; text-align: center;">
                Prototipo para entrevista CP25/152<br>Diego Elvira Vásquez · Feb 2026
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# VISTA 1: INTERFAZ DEL ESTUDIANTE
# ──────────────────────────────────────────────

if view == "Estudiante":
    st.markdown('<div class="main-header">Asistente de Fundamentos de Programación</div>', unsafe_allow_html=True)

    orch = st.session_state.orchestrator
    config = st.session_state.config
    scaffolding_labels = {0: "Socrático", 1: "Pista", 2: "Ejemplo", 3: "Explicación"}

    # Info contextual
    col1, col2, col3 = st.columns(3)
    with col1:
        remaining = config.max_daily_prompts - orch.middleware.daily_prompt_counts.get(
            st.session_state.student_id, {}
        ).get(datetime.now().strftime("%Y-%m-%d"), 0)
        st.metric("Consultas restantes hoy", f"{remaining}/{config.max_daily_prompts}")
    with col2:
        level = orch.middleware.conversation_states.get(
            st.session_state.student_id, {"level": 0}
        ).get("level", 0)
        st.metric("Nivel de ayuda", scaffolding_labels[level])
    with col3:
        mode = "RAG" if config.use_rag else "General"
        st.metric("Modo", f"[RAG] {mode}" if config.use_rag else mode)

    st.markdown("---")

    # Mostrar historial de chat
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                if meta.get("scaffolding_level") is not None:
                    label = scaffolding_labels.get(meta["scaffolding_level"], "")
                    st.markdown(
                        f'<div class="scaffolding-indicator">Scaffolding: {label}</div>',
                        unsafe_allow_html=True,
                    )

    # Input del estudiante
    if prompt := st.chat_input("Escribe tu pregunta sobre la asignatura..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                result = orch.process_interaction(st.session_state.student_id, prompt)

                # Log para analytics (middleware.interaction_logs)
                if not result.was_blocked:
                    pre_fake = {
                        "processed_prompt": prompt,
                        "detected_topics": result.detected_topics,
                        "scaffolding_level": result.scaffolding_level,
                        "copy_paste_score": result.copy_paste_score,
                        "allowed": True,
                        "block_reason": "",
                    }
                    post_fake = {
                        "response": result.response_text,
                        "hallucination_injected": result.hallucination_injected,
                    }
                    orch.middleware.log_interaction(
                        student_id=st.session_state.student_id,
                        prompt_raw=prompt,
                        pre_result=pre_fake,
                        response_raw=result.response_text,
                        post_result=post_fake,
                        response_time_ms=result.processing_time_ms,
                    )

                if result.was_blocked:
                    response_text = f"[BLOQUEADO] {result.block_reason}"
                    st.warning(response_text)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "meta": {"blocked": True},
                    })
                else:
                    st.markdown(result.response_text)

                    level = result.scaffolding_level
                    label = scaffolding_labels.get(level, "")
                    bloom_label = f"{result.bloom_estimate} (N{result.bloom_level})"

                    # Trust analysis (opcional, usando cognitive_analyzer/trust si existen)
                    trust_signal = st.session_state.trust_analyzer.analyze_prompt(
                        st.session_state.student_id, prompt, ""
                    )
                    trust_label = (
                        "OK"
                        if abs(trust_signal.trust_direction) < 0.2
                        else ("?" if trust_signal.trust_direction > 0 else "!")
                    )
                    st.markdown(
                        f'<div class="scaffolding-indicator">'
                        f'Scaffolding: {label} &nbsp;|&nbsp; Bloom: <b>{bloom_label}</b> '
                        f'&nbsp;|&nbsp; Confianza: {trust_label} ({trust_signal.signal_type}) '
                        f'&nbsp;|&nbsp; Topics: {", ".join(result.detected_topics)} '
                        f'&nbsp;|&nbsp; {result.processing_time_ms}ms'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    if result.copy_paste_score > 0.5:
                        st.markdown(
                            f'<div class="alert-copypaste">Sospecha de copy-paste detectada '
                            f'(score: {result.copy_paste_score:.0%}). El profesor sera notificado.</div>',
                            unsafe_allow_html=True,
                        )

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.response_text,
                        "meta": {
                            "scaffolding_level": level,
                            "topics": result.detected_topics,
                            "copy_paste": result.copy_paste_score,
                            "hallucination": result.hallucination_injected,
                        },
                    })

    # Botones de acción
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Nuevo tema (reset scaffolding)"):
            orch.middleware.reset_student(st.session_state.student_id)
            st.rerun()
    with col2:
        if st.button("Limpiar chat"):
            st.session_state.chat_history = []
            orch.middleware.reset_student(st.session_state.student_id)
            st.rerun()
    with col3:
        st.session_state.student_id = st.text_input(
            "ID estudiante", value=st.session_state.student_id
        )


# ──────────────────────────────────────────────
# VISTA 2: PANEL DEL DOCENTE — CONFIGURACIÓN
# ──────────────────────────────────────────────

elif view == "Docente — Configuración":
    st.markdown('<div class="main-header">Panel de Configuración Pedagógica</div>', unsafe_allow_html=True)

    st.markdown(
        "Configure el comportamiento del chatbot según sus objetivos pedagógicos. "
        "Cada configuración modifica cómo el sistema responde a los estudiantes."
    )

    config = st.session_state.config

    # --- Columna izquierda: Configuraciones ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### Configuraciones Pedagógicas")
        st.markdown("*(Paper LAK 2026, Ortega-Arranz et al., Sección 3)*")

        st.markdown('<div class="config-card">', unsafe_allow_html=True)
        st.markdown("**Modelo de IA**")
        config.model_name = st.selectbox(
            "Seleccionar modelo",
            ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"],
            index=0,
            help="El modelo determina la calidad y coste de las respuestas."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="config-card">', unsafe_allow_html=True)
        st.markdown("**Límites de uso**")
        config.max_daily_prompts = st.slider(
            "Máximo de consultas diarias por estudiante",
            min_value=1, max_value=50, value=config.max_daily_prompts,
            help="Limitar prompts fomenta la reflexión antes de preguntar (desirable difficulties, Bjork 1994)"
        )
        col_min, col_max = st.columns(2)
        with col_min:
            config.min_response_length = st.number_input(
                "Longitud mínima respuesta (chars)", value=config.min_response_length, min_value=0
            )
        with col_max:
            config.max_response_length = st.number_input(
                "Longitud máxima respuesta (chars)", value=config.max_response_length, min_value=100
            )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="config-card">', unsafe_allow_html=True)
        st.markdown("**Comportamiento pedagógico**")
        config.scaffolding_mode = st.selectbox(
            "Modo de scaffolding",
            ["socratic", "hints", "direct"],
            format_func=lambda x: {
                "socratic": "Socrático (preguntas, pistas, ejemplos, explicación)",
                "hints": "Pistas progresivas",
                "direct": "Respuesta directa",
            }[x],
            help="Socrático: 4 niveles de ayuda progresiva (Wood, Bruner & Ross, scaffolding)"
        )
        config.block_direct_solutions = st.toggle(
            "Bloquear soluciones directas",
            value=config.block_direct_solutions,
            help="Si el estudiante pide 'resuélveme esto', el chatbot guía en vez de resolver"
        )
        pct_val = st.slider(
            "% de alucinaciones pedagógicas forzadas (0-30)",
            min_value=0, max_value=30, value=int(config.forced_hallucination_pct * 100),
            step=5, format="%d",
            help="Inyecta errores intencionales para fomentar lectura crítica (valorado al 50% por profesores en LAK 2026)"
        )
        config.forced_hallucination_pct = pct_val / 100.0
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="config-card">', unsafe_allow_html=True)
        st.markdown("**Contextualización RAG**")
        config.use_rag = st.toggle(
            "Usar materiales del curso como contexto",
            value=config.use_rag,
            help="RAG: las respuestas se basan en los documentos del curso, no en conocimiento general"
        )
        config.no_context_behavior = st.radio(
            "Si la pregunta no tiene contexto en los materiales:",
            ["refuse", "general"],
            format_func=lambda x: {
                "refuse": "Rechazar: Consulta a tu profesor",
                "general": "Responder con conocimiento general",
            }[x],
            horizontal=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="config-card">', unsafe_allow_html=True)
        st.markdown("**Add-ons invisibles**")
        config.role_play = st.text_area(
            "Personalidad del chatbot (invisible al estudiante)",
            value=config.role_play,
            placeholder="Ej: 'Eres un tutor paciente que usa analogías del mundo real para explicar programación'",
            help="Se inyecta como system prompt. El estudiante no lo ve."
        )
        config.system_addon = st.text_area(
            "Instrucciones adicionales ocultas",
            value=config.system_addon,
            placeholder="Ej: 'Cuando el estudiante pregunte sobre recursión, siempre menciona el caso base primero'",
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # Aplicar configuración
        if st.button("Aplicar configuración", type="primary", use_container_width=True):
            st.session_state.config = config
            llm_adapted = _LLMAdapterForOrchestrator(st.session_state.llm)
            st.session_state.orchestrator = EcosystemOrchestrator(
                config,
                rag_pipeline=st.session_state.rag,
                llm_client=llm_adapted,
            )
            st.success("Configuración aplicada. Los cambios afectan a las próximas interacciones.")

    with col_right:
        st.markdown("### Materiales del curso")

        # Subir PDF
        uploaded = st.file_uploader(
            "Subir PDF de la asignatura",
            type=["pdf", "txt"],
            help="El contenido se indexará como contexto RAG"
        )
        if uploaded:
            if uploaded.type == "application/pdf":
                import tempfile, os
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded.read())
                    temp_path = f.name
                n = st.session_state.rag.ingest_pdf(temp_path)
                os.unlink(temp_path)
            else:
                content = uploaded.read().decode("utf-8")
                n = st.session_state.rag.ingest_text(content, uploaded.name)
            st.success(f"{n} fragmentos indexados de {uploaded.name}")

        # Stats RAG
        stats = st.session_state.rag.get_stats()
        st.markdown(f"**Fragmentos indexados:** {stats['total_chunks']}")
        st.markdown(f"**Fuentes:** {', '.join(stats['sources'])}")
        st.markdown(f"**Tamaño medio:** {stats['avg_chunk_length']} chars")

        st.markdown("---")
        st.markdown("### Resumen de config activa")
        st.json({
            "modelo": config.model_name,
            "max_prompts_dia": config.max_daily_prompts,
            "scaffolding": config.scaffolding_mode,
            "bloqueo_soluciones": config.block_direct_solutions,
            "alucinaciones_%": f"{config.forced_hallucination_pct:.0%}",
            "RAG_activo": config.use_rag,
            "sin_contexto": config.no_context_behavior,
        })


# ──────────────────────────────────────────────
# VISTA 3: DASHBOARD ANALYTICS
# ──────────────────────────────────────────────

elif view == "Docente — Analytics":
    st.markdown('<div class="main-header">GenAI Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("*Monitorización en tiempo real de las interacciones estudiante-chatbot (paper LAK 2026)*")

    # Datos: combinamos logs reales del middleware + datos de demo
    real_logs = st.session_state.orchestrator.middleware.interaction_logs
    demo_logs = st.session_state.demo_logs

    # --- Métricas principales ---
    st.markdown("### Métricas globales")
    m1, m2, m3, m4, m5 = st.columns(5)

    total = len(demo_logs) + len(real_logs)
    students = len(set(l["student_id"] for l in demo_logs)) + len(set(l.student_id for l in real_logs))
    copy_paste_alerts = len([l for l in demo_logs if l["copy_paste_score"] > 0.5])
    blocked = len([l for l in demo_logs if l["was_blocked"]])
    avg_time = sum(l["response_time_ms"] for l in demo_logs) / max(len(demo_logs), 1)

    m1.metric("Total interacciones", total)
    m2.metric("Estudiantes únicos", students)
    m3.metric("Alertas copy-paste", copy_paste_alerts, delta=None)
    m4.metric("Consultas bloqueadas", blocked)
    m5.metric("Tiempo respuesta medio", f"{avg_time:.0f}ms")

    st.markdown("---")

    # --- Gráficos ---
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        import pandas as pd

        df = pd.DataFrame(demo_logs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            st.markdown("### Distribución de consultas por hora")
            fig_hours = px.histogram(
                df, x="hour", nbins=24,
                labels={"hour": "Hora del día", "count": "Nº consultas"},
                color_discrete_sequence=["#2E75B6"]
            )
            fig_hours.update_layout(bargap=0.1, height=350)
            st.plotly_chart(fig_hours, use_container_width=True)

        with col_chart2:
            st.markdown("### Distribución de topics")
            all_topics = [t for log in demo_logs for t in log["topics"]]
            topic_counts = pd.Series(all_topics).value_counts()
            fig_topics = px.bar(
                x=topic_counts.index, y=topic_counts.values,
                labels={"x": "Topic", "y": "Frecuencia"},
                color=topic_counts.values,
                color_continuous_scale="Blues"
            )
            fig_topics.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_topics, use_container_width=True)

        col_chart3, col_chart4 = st.columns(2)

        with col_chart3:
            st.markdown("### Niveles de scaffolding")
            scaff_counts = df["scaffolding_level"].value_counts().sort_index()
            labels_map = {0: "Socrático", 1: "Pista", 2: "Ejemplo", 3: "Explicación"}
            fig_scaff = px.pie(
                names=[labels_map.get(k, k) for k in scaff_counts.index],
                values=scaff_counts.values,
                color_discrete_sequence=["#4CAF50", "#8BC34A", "#FFC107", "#FF9800"]
            )
            fig_scaff.update_layout(height=350)
            st.plotly_chart(fig_scaff, use_container_width=True)

        with col_chart4:
            st.markdown("### Alertas de copy-paste por estudiante")
            cp_df = df[df["copy_paste_score"] > 0.5]
            if not cp_df.empty:
                cp_counts = cp_df["student_id"].value_counts().head(10)
                fig_cp = px.bar(
                    x=cp_counts.index, y=cp_counts.values,
                    labels={"x": "Estudiante", "y": "Nº alertas"},
                    color_discrete_sequence=["#E53935"]
                )
                fig_cp.update_layout(height=350)
                st.plotly_chart(fig_cp, use_container_width=True)
            else:
                st.info("Sin alertas de copy-paste en el periodo.")

        # --- Tabla de interacciones recientes ---
        st.markdown("---")
        st.markdown("### Últimas interacciones (datos reales de la sesión)")
        if real_logs:
            real_df = pd.DataFrame([{
                "Hora": l.timestamp[-8:],
                "Estudiante": l.student_id,
                "Topics": ", ".join(l.detected_topics),
                "Scaffolding": labels_map.get(l.scaffolding_level, "?"),
                "Copy-paste": f"{l.copy_paste_score:.0%}",
                "Bloqueado": "Sí" if l.was_blocked else "No",
                "Tiempo": f"{l.response_time_ms}ms",
            } for l in real_logs[-20:]])
            st.dataframe(real_df, use_container_width=True, hide_index=True)
        else:
            st.info("Aún no hay interacciones reales. Usa la vista de estudiante para generar datos.")

    except ImportError:
        st.error("Instala plotly y pandas: `pip install plotly pandas`")

        # Fallback sin plotly
        st.markdown("### Analytics (modo texto)")
        summary = st.session_state.orchestrator.middleware.get_analytics_summary()
        st.json(summary)


# ──────────────────────────────────────────────
# VISTA 4: INVESTIGADOR
# ──────────────────────────────────────────────

elif view == "Investigador":
    from researcher_view import render_researcher_view
    render_researcher_view()
