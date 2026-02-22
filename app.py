"""
GENIE LEARN — Prototipo de Chatbot Educativo con IA Generativa
===============================================================
Réplica funcional mínima del sistema descrito en:
  - TFG Pablo de Arriba Mendizábal (UVa, 2025)
  - Paper LAK 2026 (Ortega-Arranz et al.)

Tres vistas:
  1. 🎓 Interfaz del Estudiante — Chat con el chatbot pedagógico
  2. 🧑‍🏫 Panel del Docente — Configuraciones pedagógicas + Analytics
  3. 📊 Dashboard Analytics — Visualización de datos de interacción

Ejecución: streamlit run app.py
Autor: Diego Elvira Vásquez — Prototipo para entrevista GSIC/EMIC CP25/152
"""

import streamlit as st
import time
import json
import random
from datetime import datetime, timedelta

from middleware import PedagogicalMiddleware, PedagogicalConfig, InteractionLog
from rag_pipeline import get_rag_pipeline, SAMPLE_COURSE_CONTENT
from llm_client import get_llm_client
from cognitive_analyzer import CognitiveAnalyzer, EngagementProfiler, BLOOM_LEVELS
from trust_dynamics import TrustDynamicsAnalyzer

# ──────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="GENIE Learn — Chatbot Pedagógico GenAI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CSS PERSONALIZADO
# ──────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1B3A5C;
        border-bottom: 3px solid #2E75B6;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
    .config-card {
        background: #F7F9FC;
        border: 1px solid #E0E7EF;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .metric-card {
        background: linear-gradient(135deg, #F0F7FF, #FFFFFF);
        border: 1px solid #D0E3F7;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .alert-copypaste {
        background: #FFF3CD;
        border: 1px solid #FFC107;
        border-radius: 6px;
        padding: 10px;
        margin: 5px 0;
    }
    .scaffolding-indicator {
        background: #E8F5E9;
        border-left: 4px solid #4CAF50;
        padding: 8px 12px;
        border-radius: 0 6px 6px 0;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    .stChatMessage { max-width: 85%; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# INICIALIZACIÓN DE ESTADO
# ──────────────────────────────────────────────

def init_state():
    """Inicializa el estado de sesión de Streamlit."""
    if "config" not in st.session_state:
        st.session_state.config = PedagogicalConfig()
    if "middleware" not in st.session_state:
        st.session_state.middleware = PedagogicalMiddleware(st.session_state.config)
    if "rag" not in st.session_state:
        st.session_state.rag = get_rag_pipeline(use_openai=True)
        # Cargar contenido de ejemplo
        n = st.session_state.rag.ingest_text(SAMPLE_COURSE_CONTENT, "Fundamentos_Programacion.pdf")
        st.session_state.rag_loaded = True
        st.session_state.rag_chunks = n
    if "llm" not in st.session_state:
        st.session_state.llm = get_llm_client()
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
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/af/Escudo_de_la_Universidad_de_Valladolid.svg/200px-Escudo_de_la_Universidad_de_Valladolid.svg.png", width=80)
    st.markdown("### 🧬 GENIE Learn")
    st.markdown("*Chatbot Pedagógico GenAI*")
    st.markdown("---")

    view = st.radio(
        "Vista activa",
        ["🎓 Estudiante", "🧑‍🏫 Docente — Configuración", "📊 Docente — Analytics", "🔬 Investigador"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Estado del sistema:**")

    model_name = st.session_state.llm.model_name
    if model_name == "mock-demo":
        st.warning("⚡ Modo demo (sin API key)")
        st.caption("Configura `OPENAI_API_KEY` o `ANTHROPIC_API_KEY` para respuestas reales")
    else:
        st.success(f"✅ Modelo: `{model_name}`")

    rag_stats = st.session_state.rag.get_stats()
    st.info(f"📚 RAG: {rag_stats['total_chunks']} fragmentos indexados")

    scaffolding_labels = {0: "Socrático 🤔", 1: "Pista 💡", 2: "Ejemplo 📝", 3: "Explicación ✅"}
    current_state = st.session_state.middleware.conversation_states.get(
        st.session_state.student_id, {"level": 0}
    )
    current_level = current_state.get("level", 0)
    st.markdown(f"**Scaffolding actual:** {scaffolding_labels.get(current_level, '?')}")

    st.markdown("---")
    st.caption("Prototipo para entrevista CP25/152")
    st.caption("Diego Elvira Vásquez · Feb 2026")


# ──────────────────────────────────────────────
# VISTA 1: INTERFAZ DEL ESTUDIANTE
# ──────────────────────────────────────────────

if view == "🎓 Estudiante":
    st.markdown('<div class="main-header">🎓 Asistente de Fundamentos de Programación</div>', unsafe_allow_html=True)

    # Info contextual
    col1, col2, col3 = st.columns(3)
    with col1:
        config = st.session_state.config
        remaining = config.max_daily_prompts - st.session_state.middleware.daily_prompt_counts.get(
            st.session_state.student_id, {}
        ).get(
            datetime.now().strftime("%Y-%m-%d"), 0
        )
        st.metric("Consultas restantes hoy", f"{remaining}/{config.max_daily_prompts}")
    with col2:
        scaffolding_labels = {0: "Socrático 🤔", 1: "Pista 💡", 2: "Ejemplo 📝", 3: "Explicación ✅"}
        level = st.session_state.middleware.conversation_states.get(
            st.session_state.student_id, {"level": 0}
        ).get("level", 0)
        st.metric("Nivel de ayuda", scaffolding_labels[level])
    with col3:
        mode = "RAG" if config.use_rag else "General"
        st.metric("Modo", f"📚 {mode}")

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
                        f'<div class="scaffolding-indicator">📐 Nivel scaffolding: {label}</div>',
                        unsafe_allow_html=True,
                    )

    # Input del estudiante
    if prompt := st.chat_input("Escribe tu pregunta sobre la asignatura..."):
        # Mostrar mensaje del estudiante
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # === FLUJO COMPLETO DEL MIDDLEWARE ===
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):

                # 1. PRE-PROCESO
                pre = st.session_state.middleware.pre_process(
                    st.session_state.student_id, prompt
                )

                if not pre["allowed"]:
                    # Bloqueado por middleware
                    response_text = f"⛔ {pre['block_reason']}"
                    st.warning(response_text)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": response_text,
                        "meta": {"blocked": True}
                    })
                else:
                    # 2. RETRIEVAL RAG
                    context = ""
                    if st.session_state.config.use_rag:
                        context = st.session_state.rag.build_context(prompt, top_k=3)

                    # 3. LLAMADA AL LLM
                    start_time = time.time()
                    llm_result = st.session_state.llm.chat(
                        system_prompt=pre["system_prompt"],
                        user_prompt=prompt,
                        context=context,
                    )
                    response_time = int((time.time() - start_time) * 1000)

                    # 4. POST-PROCESO
                    post = st.session_state.middleware.post_process(
                        st.session_state.student_id,
                        llm_result["response"],
                    )

                    # 5. LOG
                    st.session_state.middleware.log_interaction(
                        student_id=st.session_state.student_id,
                        prompt_raw=prompt,
                        pre_result=pre,
                        response_raw=llm_result["response"],
                        post_result=post,
                        response_time_ms=llm_result.get("response_time_ms", response_time),
                    )

                    # 6. ANÁLISIS COGNITIVO (Bloom)
                    cognitive = st.session_state.cognitive_analyzer.analyze(prompt)
                    st.session_state.cognitive_history.append(cognitive)

                    # 7. ANÁLISIS DE CONFIANZA
                    trust_signal = st.session_state.trust_analyzer.analyze_prompt(
                        st.session_state.student_id, prompt, ""
                    )

                    # Mostrar respuesta
                    st.markdown(post["response"])

                    # Metadata visual
                    level = pre.get("scaffolding_level", 0)
                    label = scaffolding_labels.get(level, "")
                    bloom_label = f"{cognitive.bloom_name} (N{cognitive.bloom_level})"
                    trust_icon = "🟢" if abs(trust_signal.trust_direction) < 0.2 else ("🟡" if trust_signal.trust_direction > 0 else "🔴")
                    st.markdown(
                        f'<div class="scaffolding-indicator">'
                        f'📐 Scaffolding: {label} '
                        f'&nbsp;|&nbsp; 🧠 Bloom: <b>{bloom_label}</b> '
                        f'&nbsp;|&nbsp; {trust_icon} Confianza: {trust_signal.signal_type} '
                        f'&nbsp;|&nbsp; Topics: {", ".join(pre["detected_topics"])} '
                        f'&nbsp;|&nbsp; ⏱ {llm_result.get("response_time_ms", "?")}ms'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    if pre["copy_paste_score"] > 0.5:
                        st.markdown(
                            '<div class="alert-copypaste">⚠️ Sospecha de copy-paste detectada '
                            '(score: {:.0%}). El profesor será notificado.</div>'.format(
                                pre["copy_paste_score"]
                            ),
                            unsafe_allow_html=True,
                        )

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": post["response"],
                        "meta": {
                            "scaffolding_level": level,
                            "topics": pre["detected_topics"],
                            "copy_paste": pre["copy_paste_score"],
                            "hallucination": post.get("hallucination_injected", False),
                        }
                    })

    # Botones de acción
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Nuevo tema (reset scaffolding)"):
            st.session_state.middleware.reset_student(st.session_state.student_id)
            st.rerun()
    with col2:
        if st.button("🗑️ Limpiar chat"):
            st.session_state.chat_history = []
            st.session_state.middleware.reset_student(st.session_state.student_id)
            st.rerun()
    with col3:
        st.session_state.student_id = st.text_input(
            "ID estudiante", value=st.session_state.student_id
        )


# ──────────────────────────────────────────────
# VISTA 2: PANEL DEL DOCENTE — CONFIGURACIÓN
# ──────────────────────────────────────────────

elif view == "🧑‍🏫 Docente — Configuración":
    st.markdown('<div class="main-header">🧑‍🏫 Panel de Configuración Pedagógica</div>', unsafe_allow_html=True)

    st.markdown(
        "Configure el comportamiento del chatbot según sus objetivos pedagógicos. "
        "Cada configuración modifica cómo el sistema responde a los estudiantes."
    )

    config = st.session_state.config

    # --- Columna izquierda: Configuraciones ---
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown("### ⚙️ Configuraciones Pedagógicas")
        st.markdown("*(Paper LAK 2026, Ortega-Arranz et al., Sección 3)*")

        st.markdown('<div class="config-card">', unsafe_allow_html=True)
        st.markdown("**🤖 Modelo de IA**")
        config.model_name = st.selectbox(
            "Seleccionar modelo",
            ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-20250514"],
            index=0,
            help="El modelo determina la calidad y coste de las respuestas."
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="config-card">', unsafe_allow_html=True)
        st.markdown("**📏 Límites de uso**")
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
        st.markdown("**🎓 Comportamiento pedagógico**")
        config.scaffolding_mode = st.selectbox(
            "Modo de scaffolding",
            ["socratic", "hints", "direct"],
            format_func=lambda x: {
                "socratic": "🤔 Socrático (preguntas → pistas → ejemplos → explicación)",
                "hints": "💡 Pistas progresivas",
                "direct": "📖 Respuesta directa",
            }[x],
            help="Socrático: 4 niveles de ayuda progresiva (Wood, Bruner & Ross, scaffolding)"
        )
        config.block_direct_solutions = st.toggle(
            "Bloquear soluciones directas",
            value=config.block_direct_solutions,
            help="Si el estudiante pide 'resuélveme esto', el chatbot guía en vez de resolver"
        )
        config.forced_hallucination_pct = st.slider(
            "% de alucinaciones pedagógicas forzadas",
            min_value=0.0, max_value=0.3, value=config.forced_hallucination_pct,
            step=0.05, format="%.0%%",
            help="Inyecta errores intencionales para fomentar lectura crítica (valorado al 50% por profesores en LAK 2026)"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="config-card">', unsafe_allow_html=True)
        st.markdown("**📚 Contextualización RAG**")
        config.use_rag = st.toggle(
            "Usar materiales del curso como contexto",
            value=config.use_rag,
            help="RAG: las respuestas se basan en los documentos del curso, no en conocimiento general"
        )
        config.no_context_behavior = st.radio(
            "Si la pregunta no tiene contexto en los materiales:",
            ["refuse", "general"],
            format_func=lambda x: {
                "refuse": "❌ Rechazar: 'Consulta a tu profesor'",
                "general": "🌐 Responder con conocimiento general",
            }[x],
            horizontal=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="config-card">', unsafe_allow_html=True)
        st.markdown("**🎭 Add-ons invisibles**")
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
        if st.button("✅ Aplicar configuración", type="primary", use_container_width=True):
            st.session_state.config = config
            st.session_state.middleware = PedagogicalMiddleware(config)
            st.success("Configuración aplicada. Los cambios afectan a las próximas interacciones.")

    with col_right:
        st.markdown("### 📄 Materiales del curso")

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
            st.success(f"✅ {n} fragmentos indexados de {uploaded.name}")

        # Stats RAG
        stats = st.session_state.rag.get_stats()
        st.markdown(f"**Fragmentos indexados:** {stats['total_chunks']}")
        st.markdown(f"**Fuentes:** {', '.join(stats['sources'])}")
        st.markdown(f"**Tamaño medio:** {stats['avg_chunk_length']} chars")

        st.markdown("---")
        st.markdown("### 📋 Resumen de config activa")
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

elif view == "📊 Docente — Analytics":
    st.markdown('<div class="main-header">📊 GenAI Analytics Dashboard</div>', unsafe_allow_html=True)
    st.markdown("*Monitorización en tiempo real de las interacciones estudiante-chatbot (paper LAK 2026)*")

    # Datos: combinamos logs reales del middleware + datos de demo
    real_logs = st.session_state.middleware.interaction_logs
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
            st.markdown("### 📈 Distribución de consultas por hora")
            fig_hours = px.histogram(
                df, x="hour", nbins=24,
                labels={"hour": "Hora del día", "count": "Nº consultas"},
                color_discrete_sequence=["#2E75B6"]
            )
            fig_hours.update_layout(bargap=0.1, height=350)
            st.plotly_chart(fig_hours, use_container_width=True)

        with col_chart2:
            st.markdown("### 🏷️ Distribución de topics")
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
            st.markdown("### 📐 Niveles de scaffolding")
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
            st.markdown("### ⚠️ Alertas de copy-paste por estudiante")
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
        st.markdown("### 📋 Últimas interacciones (datos reales de la sesión)")
        if real_logs:
            real_df = pd.DataFrame([{
                "Hora": l.timestamp[-8:],
                "Estudiante": l.student_id,
                "Topics": ", ".join(l.detected_topics),
                "Scaffolding": labels_map.get(l.scaffolding_level, "?"),
                "Copy-paste": f"{l.copy_paste_score:.0%}",
                "Bloqueado": "⛔" if l.was_blocked else "✅",
                "Tiempo": f"{l.response_time_ms}ms",
            } for l in real_logs[-20:]])
            st.dataframe(real_df, use_container_width=True, hide_index=True)
        else:
            st.info("Aún no hay interacciones reales. Usa la vista de estudiante para generar datos.")

    except ImportError:
        st.error("Instala plotly y pandas: `pip install plotly pandas`")

        # Fallback sin plotly
        st.markdown("### Analytics (modo texto)")
        summary = st.session_state.middleware.get_analytics_summary()
        st.json(summary)


# ──────────────────────────────────────────────
# VISTA 4: INVESTIGADOR
# ──────────────────────────────────────────────

elif view == "🔬 Investigador":
    from researcher_view import render_researcher_view
    render_researcher_view()
