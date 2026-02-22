# GENIE Learn — Prototipo Funcional de Chatbot Pedagógico
## CP25/152 · GSIC/EMIC · Universidad de Valladolid

**Diego Elvira Vásquez · Febrero 2026**

---

## Ejecución Rápida

```bash
# 1. Instalar dependencias mínimas
pip install streamlit plotly pandas

# 2. Ejecutar (modo demo, sin API key)
streamlit run app.py

# 3. Con LLM real (opcional)
export OPENAI_API_KEY="sk-..."   # o ANTHROPIC_API_KEY
streamlit run app.py

# 4. Verificar que todo funciona
python test_integration.py
```

## Qué se ha construido

### El problema que resuelve el middleware

El **middleware pedagógico** (`middleware.py`) es la pieza diferencial del sistema. No es un wrapper de API — es la capa que **materializa las decisiones pedagógicas del docente** sobre el comportamiento del LLM.

El docente configura; el middleware transforma esas configuraciones en instrucciones concretas que modifican el system prompt, bloquean respuestas, inyectan errores pedagógicos o escalan el scaffolding. El estudiante interactúa con un chatbot cuyo comportamiento refleja la intención pedagógica del docente, sin ser consciente de la mecánica subyacente.

### Módulo faltante creado: `cognitive_analyzer.py`

El `app.py` importaba desde `cognitive_analyzer` — un módulo que no existía en el repositorio. El motor real es `cognitive_engine.py` (con `CognitiveEngine`). Se creó `cognitive_analyzer.py` como bridge module que expone la interfaz que `app.py` espera (`CognitiveAnalyzer`, `EngagementProfiler`, `BLOOM_LEVELS`) delegando a la implementación real.

### Verificación de las 5 configuraciones pedagógicas

| Configuración | Test | Estado |
|---|---|---|
| `max_daily_prompts` | Bloquea al superar límite | ✅ Test 4 |
| `scaffolding_mode` (socrático) | 4 niveles progresivos, system prompt distinto | ✅ Tests 2, 3 |
| `block_direct_solutions` | Inyecta instrucción en system prompt | ✅ Test 9 |
| `forced_hallucination_pct` | Añade aviso pedagógico controlado | ✅ Test 5 |
| `response_length` min/max | Trunca respuestas excedentes | ✅ Código en post_process |

### Pipeline end-to-end verificado

```
Prompt estudiante
    → middleware.pre_process()     [scaffolding, límites, copy-paste, topics]
    → rag.build_context()          [retrieval sobre materiales del curso]
    → llm.chat()                   [OpenAI/Anthropic/Mock]
    → middleware.post_process()    [truncado, alucinación, escalación]
    → cognitive_analyzer.analyze() [Bloom, ICAP, engagement]
    → trust_dynamics.analyze()     [calibración de confianza]
    → log_interaction()            [analytics para dashboard]
```

## Arquitectura de archivos (núcleo funcional)

```
genie-learn-proto/
├── app.py                  # Streamlit: 4 vistas (estudiante, docente, analytics, investigador)
├── middleware.py            # Motor de reglas pedagógicas (LA pieza diferencial)
├── rag_pipeline.py          # RAG: chunking + embeddings (ChromaDB/fallback keyword)
├── llm_client.py            # Wrapper OpenAI/Anthropic/Mock
├── cognitive_analyzer.py    # Bridge → cognitive_engine.py (Bloom + ICAP)
├── cognitive_engine.py      # Motor de análisis cognitivo real
├── trust_dynamics.py        # Dinámicas de confianza (Lee & See, 2004)
├── researcher_view.py       # Vista del investigador (integra ND, ACH, autonomía)
├── test_integration.py      # 11 tests end-to-end: 11/11 ✅
└── requirements.txt         # Dependencias (3 niveles)
```

## Test de Integración: 11/11

```
TEST 1:  RAG Pipeline                          ✅
TEST 2:  Scaffolding Socrático — Escalación    ✅
TEST 3:  Modo directo vs Socrático (distintos) ✅
TEST 4:  Límite diario de prompts              ✅
TEST 5:  Alucinación pedagógica controlada     ✅
TEST 6:  Detección de copy-paste               ✅
TEST 7:  Análisis cognitivo (Bloom + ICAP)     ✅
TEST 8:  Dinámicas de confianza                ✅
TEST 9:  Bloqueo de soluciones directas        ✅
TEST 10: Detección de topics                   ✅
TEST 11: Pipeline completo end-to-end          ✅
```
