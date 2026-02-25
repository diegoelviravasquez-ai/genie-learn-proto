"""
ubun_builder.py — Minijuego Streamlit: Construye UBUN.IA
==========================================================
Robot educativo que se monta pieza a pieza. Cada pieza = una tecnología.
El jugador demuestra comprensión respondiendo preguntas → la pieza se instala.

Ejecutar: streamlit run ubun_builder.py --server.port 8505
"""

import streamlit as st
import re

st.set_page_config(
    page_title="Construye UBUN.IA",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Definición de las 10 piezas ──────────────────────────────────────────

PIECES = [
    {
        "id": "python",
        "name": "Esqueleto",
        "emoji": "🦴",
        "tech": "Python",
        "analogy": "El esqueleto define la estructura. Sin él no hay robot.",
        "code": '''# app.py — punto de entrada
# from ecosystem_orchestrator import EcosystemOrchestrator
# from middleware import PedagogicalConfig
#
# config = PedagogicalConfig()
# orch = EcosystemOrchestrator(config, rag_pipeline=rag, llm_client=llm)
# result = orch.process_interaction("est_01", "¿Qué es recursión?")

# Mock para funcionar sin LLM
def mock_response(prompt):
    return f"[Demo] Procesado: {prompt[:40]}"
''',
        "q1": "¿Por qué Python y no JavaScript para UBUN.IA?",
        "q2": "Escribe una función que cuente prompts por estudiante (en pseudocódigo).",
        "q3": "¿Por qué el orchestrator usa _safe_import en vez de import directo?",
        "keywords": ["python", "estructura", "orchestrator", "middleware", "config", "función", "import"],
        "in_action": "Flujo: app.py → EcosystemOrchestrator → middleware.pre_process → LLM → middleware.post_process",
        "ficha": "Sospechoso desde 1991. Sin coartada para la desaparición del JavaScript en proyectos de ML. Conocido por frecuentar entornos científicos.",
    },
    {
        "id": "middleware",
        "name": "Corazón",
        "emoji": "❤️",
        "tech": "Middleware",
        "analogy": "El corazón bombea las reglas pedagógicas a cada respuesta.",
        "code": '''# middleware.py — reglas antes y después del LLM
def pre_process(self, student_id: str, raw_prompt: str) -> dict:
    # Límite diario, copy-paste, scaffolding level
    return {"allowed": True, "system_prompt": ..., "processed_prompt": ...}

def post_process(self, student_id: str, response: str) -> dict:
    # Truncado, alucinación pedagógica
    return {"response": response}''',
        "q1": "¿Qué hace el middleware ANTES de llamar al LLM?",
        "q2": "¿Cómo activarías el modo socrático en el config?",
        "q3": "¿Qué pasaría si el middleware no existiera?",
        "keywords": ["middleware", "pre_process", "post_process", "scaffolding", "socrático", "reglas"],
        "in_action": "Los 8 modos de scaffolding: socratic, hints, examples, analogies, direct, challenge, rubber_duck, progressive",
        "ficha": "No tiene cara propia. Intercepta toda comunicación antes de que llegue al destinatario. La policía lo considera 'de interés' en todos los casos.",
    },
    {
        "id": "llm",
        "name": "Cerebro",
        "emoji": "🧠",
        "tech": "LLM API",
        "analogy": "El cerebro genera lenguaje. Pero no decide — obedece al corazón.",
        "code": '''# llm_client.py — Anthropic / OpenAI
def chat(self, system_prompt: str, user_prompt: str, context: str = "") -> dict:
    # system_prompt lleva las instrucciones pedagógicas del middleware
    response = self.client.messages.create(
        model="claude-sonnet",
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt + "\\n\\nContexto: " + context}]
    )
    return {"response": response.content[0].text}''',
        "q1": "¿Qué recibe el LLM además del mensaje del estudiante?",
        "q2": "¿Cómo se pasa el modo socrático al modelo?",
        "q3": "¿Por qué el LLM no debe decidir solo el nivel de ayuda?",
        "keywords": ["llm", "system_prompt", "contexto", "anthropic", "openai", "modelo", "pedagógico"],
        "in_action": "Llamada a Anthropic con system_prompt que incluye nivel de scaffolding y rol de tutor",
        "ficha": "Extranjero. Ha leído todo internet pero jura que no recuerda nada específico. Responde preguntas con una fluidez que resulta sospechosa.",
    },
    {
        "id": "rag",
        "name": "Memoria",
        "emoji": "💾",
        "tech": "ChromaDB/RAG",
        "analogy": "La memoria guarda los apuntes del curso. El cerebro solo recuerda lo que le pasas.",
        "code": '''# rag_pipeline.py — retrieval
def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
    # Embeddings del query vs chunks del curso
    results = self.collection.query(query_texts=[query], n_results=top_k)
    return [{"text": doc, "source": meta["source"]} for doc, meta in ...]

# El contexto se inyecta en el prompt al LLM
context = "\\n\\n".join(chunk["text"] for chunk in chunks)''',
        "q1": "¿Qué es RAG y por qué lo usa GENIE?",
        "q2": "¿Cómo se usa el resultado de retrieve() en la llamada al LLM?",
        "q3": "¿Por qué top_k=3 y no 10 chunks?",
        "keywords": ["rag", "retrieve", "chunks", "contexto", "embedding", "chroma", "curso"],
        "in_action": "rag.retrieve(prompt, top_k=3) devuelve los 3 fragmentos más relevantes del material del curso",
        "ficha": "Especialista en recuperar lo que otros olvidaron. Opera a través de ChromaDB. No deja huellas —solo vectores de 1536 dimensiones.",
    },
    {
        "id": "pandas",
        "name": "Músculos",
        "emoji": "💪",
        "tech": "Pandas",
        "analogy": "Los músculos procesan los datos. Sin Pandas el dashboard no podría agrupar ni calcular.",
        "code": '''# analytics/bridge.py — perfiles por estudiante
def get_student_profiles(course_id=None) -> pd.DataFrame:
    df = get_interactions_df(course_id=course_id)
    agg = df.groupby("student_id").agg(
        total_prompts=("id", "count"),
        bloom_mean=("bloom_level", "mean"),
        autonomy_score=("copy_paste_score", lambda s: 1 - s.mean()),
    ).reset_index()
    return agg''',
        "q1": "¿Qué es un DataFrame y por qué lo necesita el dashboard?",
        "q2": "¿Cómo calcularías el bloom_mean por student_id?",
        "q3": "¿Por qué groupby es mejor que un bucle for aquí?",
        "keywords": ["pandas", "dataframe", "groupby", "bloom", "estudiante", "agregar"],
        "in_action": "get_student_profiles() agrupa por student_id y devuelve DataFrame con bloom_mean, autonomy_score",
        "ficha": "Nombre en clave: DataFrame. Agrupa, filtra y calcula sin mostrar emoción. Visto en compañía de NumPy en horas intempestivas.",
    },
    {
        "id": "plotly",
        "name": "Ojos",
        "emoji": "👁️",
        "tech": "Plotly",
        "analogy": "Los ojos visualizan el aprendizaje. Scatter, heatmaps y radares son la vista del robot.",
        "code": '''# dashboard/demo_completo.py — scatter de cuadrantes
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["autonomy_score"], y=df["bloom_mean"],
    mode="markers", marker=dict(size=df["total_prompts"]*2),
    text=df["name"]
))
fig.add_vline(x=0.5); fig.add_hline(y=3)  # cuadrantes
st.plotly_chart(fig)''',
        "q1": "¿Para qué sirve Plotly en un dashboard de LA?",
        "q2": "¿Cómo representarías autonomía vs Bloom en un solo gráfico?",
        "q3": "¿Por qué añadir líneas en x=0.5 e y=3 en el scatter?",
        "keywords": ["plotly", "scatter", "gráfico", "visualización", "bloom", "autonomía", "cuadrante"],
        "in_action": "Scatter autonomy vs bloom_mean en demo_completo.py con cuadrantes y ZONA DE RIESGO",
        "ficha": "Hace que los datos actúen. Testigos describen gráficos que 'responden' cuando los tocas. Antecedentes: Florence Nightingale, 1858.",
    },
    {
        "id": "streamlit",
        "name": "Manos",
        "emoji": "🤝",
        "tech": "Streamlit",
        "analogy": "Las manos son la interfaz. El docente toca sliders, el estudiante escribe en el chat.",
        "code": '''# app.py — vistas
view = st.radio("Vista", ["Estudiante", "Docente — Config", "Docente — Analytics",
                          "Mapa Epistémico", "Demo en Vivo", "Investigador"])

if view == "Estudiante":
    prompt = st.chat_input("Escribe tu pregunta...")
    if prompt:
        result = orch.process_interaction(student_id, prompt)
        st.chat_message("assistant").markdown(result.response_text)''',
        "q1": "¿Qué es Streamlit y por qué se usa en GENIE?",
        "q2": "¿Cómo se cambia de vista Estudiante a Docente en app.py?",
        "q3": "¿Por qué usar st.session_state para el historial de chat?",
        "keywords": ["streamlit", "vista", "chat", "st.", "session_state", "interfaz"],
        "in_action": "Las 7 vistas de app.py: Estudiante, Práctica Guiada, Docente Config, Docente Analytics, Mapa Epistémico, Demo, Investigador",
        "ficha": "Construye interfaces sin JavaScript. Los puristas del frontend lo consideran una afrenta. Los científicos de datos, una salvación.",
    },
    {
        "id": "postgresql",
        "name": "Estómago",
        "emoji": "🗄️",
        "tech": "PostgreSQL",
        "analogy": "El estómago guarda todo. Interacciones, estudiantes y configs viven en tablas.",
        "code": '''# data/models.py — SQLAlchemy
class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    prompt = Column(Text); response = Column(Text)
    bloom_level = Column(Integer)
    copy_paste_score = Column(Float)
    scaffolding_mode = Column(String(64))
    timestamp = Column(DateTime)''',
        "q1": "¿Por qué guardar las interacciones en una base de datos?",
        "q2": "¿Qué columnas tiene la tabla Interaction en el proyecto?",
        "q3": "¿Cuándo se llama a log_interaction en el orchestrator?",
        "keywords": ["postgresql", "tabla", "interaction", "student", "bloom", "log_interaction"],
        "in_action": "Tabla Interaction: student_id, prompt, response, bloom_level, copy_paste_score, scaffolding_mode, timestamp",
        "ficha": "El archivero. Recuerda todo. Para siempre. Sin opinión sobre el contenido. Trabaja en SQL, un idioma que nadie eligió pero todo el mundo usa.",
    },
    {
        "id": "fastapi",
        "name": "Sistema nervioso",
        "emoji": "🦷",
        "tech": "FastAPI",
        "analogy": "El sistema nervioso conecta los módulos. La API expone el chatbot a Moodle o a otros clientes.",
        "code": '''# api.py — endpoint de chat
@app.post("/chat")
def chat(request: ChatRequest):
    pre = middleware.pre_process(request.student_id, request.prompt)
    if not pre["allowed"]:
        raise HTTPException(403, pre["block_reason"])
    response = llm.chat(pre["system_prompt"], pre["processed_prompt"], context)
    return {"response": response, "scaffolding_level": pre["scaffolding_level"]}''',
        "q1": "¿Para qué sirve FastAPI en GENIE?",
        "q2": "¿Qué devuelve el endpoint POST /chat?",
        "q3": "¿Por qué el middleware se llama antes del LLM en la API?",
        "keywords": ["fastapi", "api", "endpoint", "post", "chat", "middleware"],
        "in_action": "Endpoint POST /chat recibe student_id y prompt, devuelve response y scaffolding_level",
        "ficha": "Intérprete entre sistemas que no se hablan. Transforma peticiones HTTP en acciones Python. Completamente ajeno a las consecuencias.",
    },
    {
        "id": "git",
        "name": "ADN",
        "emoji": "📦",
        "tech": "Git",
        "analogy": "El ADN es la memoria del proyecto. Cada commit guarda una versión; el PR revisa el cambio.",
        "code": '''# Flujo típico en GENIE Learn
git checkout -b feature/nueva-vista
# ... editar app.py, añadir vista ...
git add app.py
git commit -m "feat: vista Práctica Guiada con mini-retos"
git push origin feature/nueva-vista
# Abrir Pull Request → revisión → merge a main''',
        "q1": "¿Por qué usar Git en un proyecto de investigación con código?",
        "q2": "¿Qué es un branch y cuándo crearlo?",
        "q3": "¿Por qué hacer commit de app.py y middleware por separado si cambian juntos?",
        "keywords": ["git", "commit", "branch", "merge", "versión", "pr", "pull request"],
        "in_action": "Flujo: branch → commit de módulos → push → Pull Request → merge a main",
        "ficha": "Testigo perfecto. Recuerda qué, cuándo y quién. El por qué depende del mensaje de commit, que rara vez es suficientemente explicativo.",
    },
]

COLOR_INSTALLED = "#048A81"
COLOR_CURRENT = "#E87722"
COLOR_LOCKED = "#444"

# ─── Estilos ───────────────────────────────────────────────────────────────

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; background: #0a0a0a; color: #eee; }}
    .stApp {{ background: linear-gradient(180deg, #0a0a0a 0%, #0d0d0d 100%); }}
    .ubun-title {{ font-size: 2rem; font-weight: 600; margin-bottom: 4px; }}
    .ubun-sub {{ color: #888; margin-bottom: 24px; }}
    .ubun-robot {{ font-family: 'Consolas', monospace; font-size: 14px; line-height: 1.4; padding: 16px; background: #111; border-radius: 12px; white-space: pre; }}
    .ubun-piece-ok {{ color: {COLOR_INSTALLED}; }}
    .ubun-piece-current {{ color: {COLOR_CURRENT}; animation: pulse 1.5s ease-in-out infinite; }}
    @keyframes pulse {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.6; }} }}
    .ubun-piece-locked {{ color: {COLOR_LOCKED}; }}
    .ubun-badge {{ display: inline-block; padding: 6px 12px; border-radius: 8px; font-size: 0.9rem; margin-top: 8px; }}
    .ubun-card {{ background: #1a1a1a; border-radius: 12px; padding: 16px; margin-top: 16px; border-left: 4px solid {COLOR_INSTALLED}; }}
    .footer-ubun {{ margin-top: 32px; font-size: 0.8rem; color: #555; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Session state ────────────────────────────────────────────────────────

if "ubun_pieces_installed" not in st.session_state:
    st.session_state.ubun_pieces_installed = []
if "ubun_current_piece" not in st.session_state:
    st.session_state.ubun_current_piece = 0
if "ubun_questions_answered" not in st.session_state:
    st.session_state.ubun_questions_answered = {}  # piece_id -> [bool, bool, bool]
if "ubun_answers" not in st.session_state:
    st.session_state.ubun_answers = {}  # piece_id -> [str, str, str]
if "ubun_bloom_history" not in st.session_state:
    st.session_state.ubun_bloom_history = []
if "ubun_just_installed" not in st.session_state:
    st.session_state.ubun_just_installed = None  # piece_id or None
if "ubun_tutorial_done" not in st.session_state:
    st.session_state.ubun_tutorial_done = False

for p in PIECES:
    if p["id"] not in st.session_state.ubun_questions_answered:
        st.session_state.ubun_questions_answered[p["id"]] = [False, False, False]
    if p["id"] not in st.session_state.ubun_answers:
        st.session_state.ubun_answers[p["id"]] = ["", "", ""]

# ─── Pantalla de bienvenida (tutorial inicial) ─────────────────────────────

if not st.session_state.ubun_tutorial_done:
    st.markdown("""
    <div style="background:#0d0d0d; border:2px solid #C4922A;
                border-radius:12px; padding:32px; max-width:700px;
                margin:40px auto; font-family:monospace;">
      <div style="font-size:2rem; text-align:center; margin-bottom:16px">🔍</div>
      <h2 style="color:#C4922A; text-align:center; margin-bottom:8px">
        EL CASO DE UBUN.IA
      </h2>
      <p style="color:#9A8C78; text-align:center; font-style:italic; margin-bottom:24px">
        Diez piezas desaparecidas. Diez tecnologías sospechosas.<br>
        Un detective con Python.
      </p>
      <div style="color:#D4CAB8; line-height:1.8; margin-bottom:24px">
        <p>🤖 <strong style="color:#C4922A">UBUN.IA</strong> es un robot educativo
        desmontado. Cada pieza es una tecnología del proyecto GENIE Learn.</p>
        <p>🔍 Tu misión: <strong style="color:#C4922A">fichar a los 10 sospechosos</strong>
        demostrando que los entiendes.</p>
        <p>📋 <strong style="color:#C4922A">Cómo funciona:</strong></p>
        <ul style="color:#9A8C78; margin-left:20px">
          <li>La columna izquierda muestra el pipeline del robot</li>
          <li>La columna derecha tiene 3 preguntas por pieza</li>
          <li>Las preguntas escalan: Comprender → Aplicar → Analizar</li>
          <li>Responde con más de 20 palabras usando el concepto clave</li>
          <li>Cuando las 3 estén aceptadas, aparece el botón "FICHAR SOSPECHOSO"</li>
        </ul>
        <p>🎖️ Tu rango sube conforme fichás más tecnologías.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        if st.button("🔍 COMENZAR LA INVESTIGACIÓN", type="primary", use_container_width=True):
            st.session_state.ubun_tutorial_done = True
            st.rerun()
    st.stop()

def word_count(text):
    return len((text or "").split())


def answer_acceptable(text: str, keywords: list) -> tuple[bool, str]:
    if word_count(text) < 20:
        return False, "Escribe al menos 20 palabras."
    lower = (text or "").lower()
    for k in keywords:
        if k.lower() in lower:
            return True, "✅ Respuesta aceptada."
    return False, f"🔄 Intenta mencionar alguna de: {', '.join(keywords[:4])}..."


# Orden del pipeline vertical (flujo del sistema)
PIPELINE_ORDER = [
    ("rag", "💾", "ChromaDB/RAG", "La memoria guarda los apuntes"),
    ("llm", "🧠", "LLM API", "El cerebro genera texto"),
    ("middleware", "❤️", "Middleware", "El corazón aplica reglas"),
    ("python", "🦴", "Python", "El esqueleto conecta todo"),
    ("pandas", "💪", "Pandas", "Los músculos procesan datos"),
    ("plotly", "👁️", "Plotly", "Los ojos visualizan el aprendizaje"),
    ("streamlit", "🤝", "Streamlit", "Las manos son la interfaz"),
    ("postgresql", "🗄️", "PostgreSQL", "El estómago guarda todo"),
    ("fastapi", "🦷", "FastAPI", "El sistema nervioso conecta los módulos"),
    ("git", "📦", "Git", "El ADN versiona el proyecto"),
]
ARROW_TEXTS = [
    "busca en apuntes",
    "prompt + contexto RAG",
    "reglas pedagógicas aplicadas",
    "logs de interacciones",
    "DataFrame de métricas",
    "gráficos interactivos",
    "persistencia",
    "datos del dashboard",
    "versiona el pipeline",
]


def render_robot(installed, current):
    """Dibuja el robot como pipeline vertical con flechas entre piezas."""
    installed = set(installed)

    def piece_box(pid, emoji, name, role):
        if pid in installed:
            bg = "#0a2a20"
            border = "#048A81"
            op = "1"
            extra = ""
        elif pid == current:
            bg = "#2a1500"
            border = "#E87722"
            op = "1"
            extra = '<div style="font-size:0.55rem;color:#E87722;margin-top:2px;">⚡ INSTALANDO</div>'
        else:
            bg = "#111"
            border = "#333"
            op = "0.4"
            extra = ""
        return (
            f'<div style="background:{bg};border:2px solid {border};border-radius:8px;'
            f'padding:10px 14px;margin:0 auto;opacity:{op};'
            f'font-family:monospace;text-align:center;max-width:280px;">'
            f'<div style="font-size:1.8rem">{emoji}</div>'
            f'<div style="font-size:0.75rem;color:#eee">{name}</div>'
            f'<div style="font-size:0.6rem;color:#aaa;margin-top:2px;">{role}</div>'
            f'{extra}</div>'
        )

    def arrow_line(text, from_installed, to_installed):
        if from_installed and to_installed:
            color = "#048A81"
            style = "solid"
        else:
            color = "#555"
            style = "dashed"
        return (
            f'<div style="text-align:center;margin:4px 0;">'
            f'<div style="border-left:2px {style} {color};height:20px;margin:0 auto;width:0;"></div>'
            f'<div style="font-size:0.6rem;color:{color};margin:2px 0;">↓ {text}</div>'
            f'<div style="border-left:2px {style} {color};height:12px;margin:0 auto;width:0;"></div>'
            f'</div>'
        )

    parts = []
    for i, (pid, emoji, name, role) in enumerate(PIPELINE_ORDER):
        parts.append(piece_box(pid, emoji, name, role))
        if i < len(ARROW_TEXTS):
            from_ok = pid in installed
            next_pid = PIPELINE_ORDER[i + 1][0]
            to_ok = next_pid in installed
            parts.append(arrow_line(ARROW_TEXTS[i], from_ok, to_ok))

    html = f"""
    <div style="background:#0d0d0d;border-radius:12px;padding:20px;
                border:1px solid #1a1a2e;">
      <div style="display:flex;flex-direction:column;align-items:center;">
        {"".join(parts)}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ─── Título ────────────────────────────────────────────────────────────────

st.markdown('<p class="ubun-title">🔍 EL CASO DE UBUN.IA</p>', unsafe_allow_html=True)
st.markdown('<p class="ubun-sub">Diez piezas desaparecidas. Diez tecnologías sospechosas.<br>Un detective con Python.</p>', unsafe_allow_html=True)

n_installed = len(st.session_state.ubun_pieces_installed)
idx = min(st.session_state.ubun_current_piece, len(PIECES) - 1)
piece = PIECES[idx]

# ─── Card de celebración al completar todas las piezas ──────────────────────

if n_installed == 10:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a0a04, #2a1a08);
                border: 2px solid #C4922A; border-radius:12px;
                padding:24px; text-align:center; margin-bottom:20px">
      <div style="font-size:3rem">🎖️</div>
      <h2 style="color:#C4922A">EL CASO ESTÁ CERRADO</h2>
      <p style="color:#D4CAB8; font-style:italic">
        Los diez sospechosos han sido fichados.<br>
        UBUN.IA opera al completo.<br>
        El detective puede descansar.
      </p>
      <p style="color:#9A8C78; font-size:0.8rem">
        CP25/152 · GSIC/EMIC · Universidad de Valladolid
      </p>
    </div>
    """, unsafe_allow_html=True)

# ─── Dos columnas ─────────────────────────────────────────────────────────

col_left, col_right = st.columns([0.4, 0.6])

with col_left:
    st.markdown("#### 🤖 UBUN.IA — Pipeline del sistema")
    with st.expander("❓ ¿Qué estoy viendo aquí?"):
        st.markdown("""
        <div style="color:#D4CAB8; font-family:monospace; font-size:0.85rem;
                    line-height:1.8; padding:8px">
        <p>Este es el <strong style="color:#C4922A">pipeline de GENIE Learn</strong>
        — el camino que sigue cada pregunta de un estudiante.</p>
        <p>Las piezas se leen de <strong>arriba a abajo</strong>:</p>
        <p>🧠 <strong style="color:#C4922A">LLM</strong> genera la respuesta<br>
        ↓ recibe instrucciones del...<br>
        ❤️ <strong style="color:#C4922A">Middleware</strong> que aplica las reglas del docente<br>
        ↓ que se nutre de...<br>
        💾 <strong style="color:#C4922A">ChromaDB</strong> con los apuntes del curso<br>
        ↓ todo corre sobre...<br>
        🦴 <strong style="color:#C4922A">Python</strong> que conecta los módulos<br>
        ↓ que procesa datos con...<br>
        💪 <strong style="color:#C4922A">Pandas</strong> y los visualiza con 👁️ <strong style="color:#C4922A">Plotly</strong></p>
        <p><strong style="color:#048A81">Verde</strong> = instalado ·
        <strong style="color:#E87722">Naranja</strong> = en progreso ·
        <strong style="color:#444">Gris</strong> = bloqueado</p>
        </div>
        """, unsafe_allow_html=True)
    render_robot(set(st.session_state.ubun_pieces_installed), piece["id"])
    st.progress(n_installed / 10.0)
    st.caption(f"{n_installed}/10 piezas instaladas")

    if n_installed <= 2:
        badge = "Inspector Novato 🔎"
    elif n_installed <= 5:
        badge = "Detective Senior 🕵️"
    elif n_installed <= 8:
        badge = "Comisario 🎖️"
    else:
        badge = "El Caso Está Cerrado ✅"
    st.markdown(f'<span class="ubun-badge" style="background:{COLOR_CURRENT};color:#000;">{badge}</span>', unsafe_allow_html=True)

with col_right:
    st.markdown(f"#### ⚡ Instalando: **{piece['name']}** ({piece['tech']})")
    with st.expander("❓ ¿Cómo responder correctamente?"):
        st.markdown("""
        <div style="color:#D4CAB8; font-family:monospace; font-size:0.85rem;
                    line-height:1.8; padding:8px">
        <p>Cada pieza tiene <strong style="color:#C4922A">3 preguntas</strong>
        que escalan en dificultad:</p>
        <p>📘 <strong>Pregunta 1 (Bloom 2 — Comprender)</strong><br>
        &nbsp;&nbsp;¿Qué es y para qué sirve? Explícalo con tus palabras.</p>
        <p>📗 <strong>Pregunta 2 (Bloom 3 — Aplicar)</strong><br>
        &nbsp;&nbsp;¿Cómo lo usarías en el proyecto GENIE?</p>
        <p>📕 <strong>Pregunta 3 (Bloom 4 — Analizar)</strong><br>
        &nbsp;&nbsp;¿Por qué esta solución y no otra?</p>
        <p>✅ Una respuesta se acepta si:<br>
        &nbsp;&nbsp;• Tiene más de 20 palabras<br>
        &nbsp;&nbsp;• Menciona algún concepto clave de la tecnología<br>
        &nbsp;&nbsp;• (El hint aparece si no pasa la validación)</p>
        </div>
        """, unsafe_allow_html=True)
    st.caption(piece["analogy"])
    st.markdown(f"""
<div style="background:#1a0a04; border:1px solid #4A3520;
            border-radius:6px; padding:12px; margin:8px 0;
            font-family:monospace; font-size:0.8rem;">
  <span style="color:#C4922A;">📋 FICHA POLICIAL</span><br>
  <span style="color:#9A8C78; font-style:italic;">
    {piece['ficha']}
  </span>
</div>
""", unsafe_allow_html=True)
    st.code(piece["code"], language="python")

    qs = [piece["q1"], piece["q2"], piece["q3"]]
    keywords = piece["keywords"]
    q_ok = st.session_state.ubun_questions_answered[piece["id"]]
    answers = st.session_state.ubun_answers[piece["id"]]

    bloom_captions = [
        "💡 Bloom 2 · Demuestra que entiendes qué es y para qué existe",
        "💡 Bloom 3 · Demuestra que sabes usarlo en el proyecto real",
        "💡 Bloom 4 · Demuestra que entiendes por qué esta solución y no otra",
    ]
    for i in range(3):
        st.markdown(f"**Pregunta {i+1}** (Bloom {i+2})")
        st.caption(bloom_captions[i])
        ans = st.text_area(f"Respuesta {i+1}", value=answers[i], key=f"ubun_a_{piece['id']}_{i}", height=80)
        if ans != answers[i]:
            st.session_state.ubun_answers[piece["id"]][i] = ans
            ok, msg = answer_acceptable(ans, keywords)
            if ok:
                st.session_state.ubun_questions_answered[piece["id"]][i] = True
                st.success(msg)
            else:
                nw = len(ans.split())
                st.warning(f"""
**Respuesta demasiado corta o sin el concepto clave.**

Para que se acepte necesitas:
- Más de 20 palabras ({'✅' if nw >= 20 else f'❌ tienes {nw}'})
- Mencionar alguna de estas palabras: `{', '.join(piece['keywords'][:3])}`

💡 *Pista: Incluye alguno de los conceptos clave y desarrolla la idea con más de 20 palabras.*
""")
        elif q_ok[i]:
            st.success("✅ Respuesta aceptada.")

    if all(q_ok) and piece["id"] not in st.session_state.ubun_pieces_installed:
        if st.button("🔍 FICHAR SOSPECHOSO", type="primary", key="ubun_install"):
            st.session_state.ubun_pieces_installed.append(piece["id"])
            st.session_state.ubun_current_piece = min(idx + 1, len(PIECES) - 1)
            st.session_state.ubun_just_installed = piece["id"]
            mensajes = {
                "python": "🔍 Python arrestado e integrado. El esqueleto sostiene.",
                "middleware": "🔍 Middleware identificado. El corazón late. Las reglas se aplican.",
                "llm": "🔍 LLM detenido. El cerebro habla. Pero obedece al corazón.",
                "pandas": "🔍 Pandas incorporado. Los músculos procesan. Los datos fluyen.",
                "plotly": "🔍 Plotly capturado. Los ojos ven. El docente comprende.",
                "rag": "🔍 ChromaDB localizado. La memoria recuerda. Sin inventar.",
                "postgresql": "🔍 PostgreSQL fichado. El archivero guarda. Para siempre.",
                "fastapi": "🔍 FastAPI interceptado. Los módulos se comunican.",
                "streamlit": "🔍 Streamlit detenido. La interfaz responde. El humano toca.",
                "git": "🔍 Git arrestado. El caso queda documentado. Todo el caso.",
            }
            st.success(mensajes.get(piece["id"], "🔍 Pieza instalada."))
            st.balloons()
            st.rerun()

# ─── Sección inferior: Esta pieza en acción ───────────────────────────────

if st.session_state.ubun_just_installed or n_installed > 0:
    show_id = st.session_state.ubun_just_installed or st.session_state.ubun_pieces_installed[-1]
    p_show = next(x for x in PIECES if x["id"] == show_id)
    st.markdown("---")
    st.markdown("#### Lo que GENIE Learn hace con esto")
    st.markdown(f'<div class="ubun-card"><strong>{p_show["name"]} ({p_show["tech"]}) en acción en el dashboard:</strong><br>{p_show["in_action"]}</div>', unsafe_allow_html=True)
    if st.session_state.ubun_just_installed:
        st.session_state.ubun_just_installed = None

st.markdown('<p class="footer-ubun">UBUN.IA · Ganador Hack4Edu 2025 · CP25/152 GSIC/EMIC</p>', unsafe_allow_html=True)
