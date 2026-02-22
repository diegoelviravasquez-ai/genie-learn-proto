# PROMPT INICIAL PARA CURSOR COMPOSER
# ====================================
# Copia TODO este texto y pégalo en el Composer de Cursor (Cmd+Shift+I)
# Cursor generará el proyecto entero de una vez.

Crea un proyecto Python completo llamado `genie-chatbot-proto` para un chatbot educativo con IA generativa. El proyecto implementa la arquitectura de un chatbot pedagógico con RAG (Retrieval-Augmented Generation) y un middleware que aplica configuraciones pedagógicas configurables por el docente.

## Estructura de archivos

```
genie-chatbot-proto/
├── .env.example
├── .gitignore
├── .cursorrules
├── README.md
├── requirements.txt
├── docs/
│   └── ejemplo/
│       └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   ├── retriever.py
│   │   └── prompts.py
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── configs.py
│   │   ├── guardrails.py
│   │   └── scaffolding.py
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── metrics.py
│   │   └── copypaste.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── client.py
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       ├── routes_student.py
│       └── routes_teacher.py
├── frontend/
│   ├── app.py
│   ├── student_view.py
│   └── teacher_view.py
├── tests/
│   ├── test_rag.py
│   ├── test_middleware.py
│   └── test_analytics.py
└── data/
    ├── chroma_db/.gitkeep
    ├── logs/.gitkeep
    └── configs/.gitkeep
```

## Dependencias (requirements.txt)
```
fastapi>=0.109.0
uvicorn>=0.27.0
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-community>=0.0.10
chromadb>=0.4.22
openai>=1.10.0
anthropic>=0.18.0
streamlit>=1.31.0
plotly>=5.18.0
python-dotenv>=1.0.0
pydantic>=2.5.0
pypdf>=4.0.0
loguru>=0.7.0
httpx>=0.26.0
pytest>=8.0.0
```

## Implementación completa de cada archivo:

### src/config.py
Carga variables de entorno con python-dotenv. Expone: OPENAI_API_KEY, ANTHROPIC_API_KEY, LLM_PROVIDER (default "openai"), LLM_MODEL (default "gpt-4o-mini"), CHROMA_PATH (default "./data/chroma_db"), LOGS_PATH (default "./data/logs"), CONFIGS_PATH (default "./data/configs").

### src/rag/ingest.py
- Función `ingest_documents(docs_path: str, collection_name: str = "course_docs")`:
  - Recorre todos los PDFs en docs_path usando PyPDFLoader de langchain_community
  - Split con RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  - Embeddings con OpenAIEmbeddings(model="text-embedding-3-small")
  - Almacena en ChromaDB con persist_directory del config
  - Metadata por chunk: source (filename), page (número)
- CLI: `python -m src.rag.ingest --docs-path ./docs/ejemplo`

### src/rag/retriever.py
- Clase `RAGRetriever`:
  - Conecta al ChromaDB existente
  - `retrieve(query: str, k: int = 4) -> list[dict]` con similarity_search_with_score
  - `format_context(docs) -> str` formatea chunks como: "[Fuente: {source}, pág. {page}]\n{content}\n"

### src/rag/prompts.py
- SYSTEM_PROMPT_TEMPLATE con placeholders: {course_name}, {pedagogical_rules}
  - Rol de asistente educativo
  - Instrucción de usar SOLO contexto proporcionado
  - Placeholder para reglas pedagógicas del middleware
- USER_PROMPT_TEMPLATE con placeholders: {context}, {history}, {question}
- Función `build_prompt(course_name, pedagogical_rules, context, history, question) -> tuple[str, str]` que retorna (system_prompt, user_prompt)

### src/middleware/configs.py
- `PedagogicalConfig(BaseModel)`:
  - course_name: str
  - max_prompts_per_day: int = 20
  - min_response_length: int = 50
  - max_response_length: int = 500
  - hallucination_rate: float = 0.0
  - direct_solution_behavior: Literal["give", "refuse", "socratic"] = "socratic"
  - scaffolding_level: int = 3
  - custom_addon: str = ""
  - allowed_topics: list[str] = []
  - no_context_behavior: Literal["general", "decline"] = "decline"
- `ConfigManager`:
  - `load_config(course_id: str) -> PedagogicalConfig`
  - `save_config(course_id: str, config: PedagogicalConfig)`
  - Persiste como JSON en CONFIGS_PATH
  - Incluye 2 configs de ejemplo: "fundamentos_prog" (estricta) y "intro_ia" (permisiva)

### src/middleware/engine.py
- `MiddlewareResult(BaseModel)`:
  - allowed: bool
  - modified_system_prompt: str
  - modified_user_query: str
  - post_flags: dict = {}
  - rejection_reason: str | None = None
- `PedagogicalMiddleware`:
  - `pre_process(query: str, config: PedagogicalConfig, session: dict) -> MiddlewareResult`:
    - Verifica límite prompts diarios (session["prompt_count"])
    - Detecta solicitud de solución directa (patterns: "dame la solución", "resuelve", "dime la respuesta", "solución completa")
    - Detecta copy-paste (delega a CopyPasteDetector)
    - Genera pedagogical_rules string según config
    - Si direct_solution_behavior == "socratic": añade regla socrática
    - Si direct_solution_behavior == "refuse": añade regla de rechazo
    - Si custom_addon: lo inyecta
    - Retorna MiddlewareResult con prompt modificado
  - `post_process(response: str, config: PedagogicalConfig) -> str`:
    - Trunca si > max_response_length palabras
    - Si hallucination_rate > 0 y random() < hallucination_rate: añade warning "[⚠️ Esta respuesta puede contener errores intencionales. Verifica con tus apuntes.]"
    - Retorna respuesta procesada

### src/middleware/scaffolding.py
- `ScaffoldingLevel(IntEnum)`: QUESTION=1, HINT=2, EXAMPLE=3, PARTIAL=4, FULL=5
- `SocraticScaffolder`:
  - `get_instruction(level: ScaffoldingLevel) -> str` — instrucción para el LLM según nivel
  - `should_escalate(session: dict) -> bool` — True si el estudiante repite pregunta similar o dice "no entiendo"
  - `escalate(session: dict) -> ScaffoldingLevel` — sube un nivel, máximo FULL

### src/middleware/guardrails.py
- `GuardrailsFilter`:
  - `check_input(text: str) -> tuple[bool, str | None]` — detecta contenido inapropiado, inyección de prompt
  - `check_output(text: str) -> str` — sanitiza la respuesta del LLM

### src/analytics/logger.py
- `InteractionLogger`:
  - `log(interaction: dict)` — escribe JSON line en LOGS_PATH/{date}.jsonl
  - Campos: timestamp, student_id, course_id, prompt, response, tokens_used, response_time_ms, topics_detected, was_copy_paste, scaffolding_level, was_rejected, rejection_reason

### src/analytics/metrics.py
- `EngagementMetrics`:
  - `load_logs(course_id: str, days: int = 7) -> list[dict]` — lee los jsonl
  - `prompts_per_student(logs) -> dict[str, int]`
  - `prompts_by_hour(logs) -> dict[int, int]`
  - `avg_prompt_length(logs) -> float`
  - `copy_paste_rate(logs) -> float`
  - `topic_distribution(logs) -> dict[str, int]`

### src/analytics/copypaste.py
- `CopyPasteDetector`:
  - `detect(text: str) -> bool`
  - Heurísticas: len(text) > 200 AND no "?" AND (ratio código > 0.3 OR matches enunciado patterns)

### src/llm/client.py
- `LLMClient`:
  - `__init__(provider: str = None, model: str = None)` — usa config defaults
  - `async chat(system_prompt: str, user_prompt: str, history: list = []) -> str`
  - Soporta OpenAI y Anthropic según provider
  - Manejo de errores con retry

### src/api/main.py
- FastAPI app con CORS middleware (allow all origins para desarrollo)
- Include routers de student y teacher

### src/api/routes_student.py
- `POST /api/chat` — Body: ChatRequest(student_id, course_id, message, session_id)
  - Flow completo: middleware.pre → retriever → prompts.build → llm.chat → middleware.post → logger.log
  - Response: ChatResponse(response, sources, was_modified, scaffolding_level, prompts_remaining)
- `GET /api/session/{session_id}/history`

### src/api/routes_teacher.py
- `GET /api/config/{course_id}`
- `PUT /api/config/{course_id}` — body PedagogicalConfig
- `GET /api/analytics/{course_id}/summary`
- `POST /api/ingest/{course_id}` — file upload

### frontend/app.py
- Streamlit app con page_config(title="GENIE Learn Proto", layout="wide")
- Tab1: student_view, Tab2: teacher_view
- Sidebar con info del proyecto

### frontend/student_view.py
- Selectbox de curso
- Input student_id
- Chat con st.chat_message/st.chat_input
- Llama a la API /api/chat
- Muestra fuentes expandibles
- Indicador prompts restantes

### frontend/teacher_view.py
- Selectbox de curso
- Formulario con sliders/selects para PedagogicalConfig
- Botón guardar config
- 4 gráficos Plotly (prompts por hora, topics, por estudiante, copy-paste rate)
- Upload de PDFs

### tests/
- test_rag.py: test ingesta y retrieval con fixture de PDF
- test_middleware.py: test límite prompts, detección solución directa, scaffolding, copy-paste
- test_analytics.py: test logging y métricas

## .env.example
```
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
```

## .gitignore
```
.env
__pycache__/
*.pyc
venv/
data/chroma_db/*
!data/chroma_db/.gitkeep
data/logs/*
!data/logs/.gitkeep
.streamlit/
```

Genera TODOS los archivos con implementación COMPLETA y funcional. Cada archivo debe tener imports, type hints, docstrings, y manejo de errores. El código debe poder ejecutarse tras `pip install -r requirements.txt` y configurar .env.
