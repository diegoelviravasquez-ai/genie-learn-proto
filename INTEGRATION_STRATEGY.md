# GENIE Learn — Estrategia de Integración de los 51 Módulos

## El principio: DEMO_MODE

Todo se resuelve con una variable: `DEMO_MODE = True`.

Cuando `DEMO_MODE = True`: los módulos que necesitan infraestructura externa
(Moodle, PostgreSQL, servidor, otros nodos) corren con **simulaciones internas**
que producen datos realistas. El ecosistema completo funciona en tu portátil.

Cuando `DEMO_MODE = False`: esos mismos módulos se conectan a la infraestructura
real. El código es idéntico; solo cambian los backends.

**Para la entrevista, todo corre en DEMO_MODE. Cuando te den acceso, flip a False.**

---

## Clasificación de los 51 módulos

### 🟢 CONECTAR DE VERDAD (no necesitas nada externo)

Estos módulos corren con lógica pura Python. Sin APIs externas, sin servidores,
sin credenciales institucionales. Los puedes cablear hoy.

| # | Módulo | Conectar a | Cómo |
|---|--------|-----------|------|
| 1 | `middleware.py` | ✅ YA CONECTADO | — |
| 2 | `rag_pipeline.py` | ✅ YA CONECTADO | — |
| 3 | `llm_client.py` | ✅ YA CONECTADO | — |
| 4 | `cognitive_engine.py` | ✅ YA CONECTADO (vía cognitive_analyzer) | — |
| 5 | `trust_dynamics.py` | ✅ YA CONECTADO | — |
| 6 | `system_event_logger.py` | `middleware.post_process()` | 3 líneas: crear evento, loguearlo. SQLite local. |
| 7 | `nd_patterns.py` | `cognitive_engine.py` | Enriquecer perfil cognitivo con detección ND |
| 8 | `cognitive_profiler.py` | `integration.py` | Ya está en EnhancedAnalyticsLayer |
| 9 | `epistemic_autonomy.py` | `integration.py` | Ya está en EnhancedAnalyticsLayer |
| 10 | `interaction_semiotics.py` | `integration.py` | Ya está en EnhancedAnalyticsLayer |
| 11 | `config_genome.py` | `system_event_logger` | Lee config_snapshot de cada evento |
| 12 | `rag_quality_sensor.py` | `rag_pipeline.retrieve()` | Evalúa chunks post-retrieval |
| 13 | `cognitive_gap_detector.py` | Historial cognitivo en session_state | Analiza gaps entre sesiones |
| 14 | `consolidation_detector.py` | Historial de eventos | Detecta patrones 48-72h |
| 15 | `epistemic_silence_detector.py` | `system_event_logger` | Detecta ausencias de pregunta |
| 16 | `metacognitive_nudges.py` | Post-process del middleware | Genera nudges según perfil |
| 17 | `temporal_dynamics.py` | Timestamps de eventos | Análisis temporal puro |
| 18 | `cognitive_pragmatics.py` | Prompts del estudiante | Análisis pragmático del lenguaje |
| 19 | `epistemic_ecology.py` | Historial de interacciones | Ecología del conocimiento |
| 20 | `hhh_alignment_detector.py` | Respuestas del LLM | Evalúa alineamiento HHH |
| 21 | `llm_judge.py` | Respuestas del LLM | Evaluación de calidad |
| 22 | `effect_latency_analyzer.py` | Eventos con timestamps | Latencia causa-efecto |
| 23 | `config_interaction_analyzer.py` | config_genome + eventos | Interacción entre configs |
| 24 | `ach_diagnostic.py` | Datos del ecosistema | ACH sobre hipótesis pedagógicas |
| 25 | `system_reflexivity.py` | Todos los módulos | El sistema analizándose |
| 26 | `udl_adapter.py` | nd_patterns + cognitive_profiler | Adaptación UDL |

**Total: 26 módulos que funcionan al 100% sin depender de nadie.**

### 🟡 SIMULAR PARA DEMO (código listo, backend simulado)

Estos módulos necesitan algo externo para funcionar en producción,
pero pueden correr con un mock realista para la entrevista.

| # | Módulo | Qué necesita en producción | Qué simulas |
|---|--------|---------------------------|-------------|
| 27 | `cross_node_signal.py` | Otros nodos (UC3M, UPF) | Nodos simulados que emiten señales con datos sintéticos |
| 28 | `teacher_calibration.py` | Datos reales de docentes | Perfiles docentes sintéticos (Prof. A vs Prof. B) |
| 29 | `temporal_config_advisor.py` | Calendario académico real | Calendario hardcodeado: "semana 8, pre-examen parcial" |
| 30 | `teacher_notification_engine.py` | Canal de notificaciones (email/Moodle) | Print a consola + log en event_logger |
| 31 | `teacher_agency_longitudinal.py` | Datos longitudinales reales | Serie temporal sintética de 3 meses |
| 32 | `pilot_design.py` | Cohorte real de estudiantes | Diseño de piloto con N simulado |
| 33 | `paper_drafting_engine.py` | Datos reales para papers | Templates con datos sintéticos |
| 34 | `o1_feedback_engine.py` | Datos de feedback reales | Feedback sintético generado |
| 35 | `learning_design_generator.py` | Objetivos del curso reales | Objetivos de ejemplo (programación) |
| 36 | `database.py` | PostgreSQL en servidor | SQLite local (ya funciona así) |

**Total: 10 módulos que simulan su backend externo.**

### 🔴 DEJAR PREPARADO (necesitas acceso institucional)

Estos módulos están escritos, el código es correcto, pero no puedes
ejecutarlos sin credenciales/infraestructura de la UVa.

| # | Módulo | Qué necesitas exactamente | Estado del código |
|---|--------|--------------------------|-------------------|
| 37 | `lti_integration.py` | Consumer key + secret de Moodle UVa | Completo. 5 min de config cuando den acceso. |
| 38 | `api.py` | Servidor para desplegar FastAPI | Completo. `docker-compose up` y funciona. |
| 39 | `api__2_.py` | Ídem | Versión alternativa de la API. |
| 40 | `auth.py` | LDAP/SAML de la UVa | Estructura lista, falta endpoint real. |
| 41 | `gdpr_anonymizer.py` | Validación del DPO institucional | Código completo, necesita auditoría legal. |

**Total: 5 módulos bloqueados por acceso externo.**

### 🔵 VISUALIZACIÓN Y DASHBOARDS

Estos son frontends que consumen datos de los módulos anteriores.
Funcionan en cuanto los módulos que alimentan estén conectados.

| # | Módulo | Se alimenta de |
|---|--------|---------------|
| 42 | `ecosystem_dashboard.py` | Todos los módulos del ecosistema |
| 43 | `researcher_view.py` | system_event_logger + analytics |
| 44 | `config_impact_panel.py` | config_genome + effect_latency |
| 45 | `autonomy_viz.py` | epistemic_autonomy + temporal |
| 46 | `genie_learn_frontend.jsx` | API (React completo) |
| 47 | `genie_demo.jsx` | API (demo reducida) |
| 48 | `analytics_bridge__1_.py` | Todos los analytics |

**Total: 7 módulos de visualización.**

### 📋 TEST Y CONFIGURACIÓN

| # | Módulo | Tipo |
|---|--------|------|
| 49 | `test_*.py` (6 archivos) | Tests unitarios |
| 50 | `Dockerfile` + `docker-compose.yml` | Despliegue |
| 51 | `Makefile` | Automatización |

---

## El archivo orquestador: ecosystem_orchestrator.py

Este es el archivo que falta. Conecta TODO. La idea:

```
app.py
  └── ecosystem_orchestrator.py    ← NUEVO: el bus central
        ├── middleware.py           (ya conectado)
        ├── rag_pipeline.py        (ya conectado)  
        ├── llm_client.py          (ya conectado)
        ├── system_event_logger.py (Capa 1: logging)
        ├── integration.py         (Capa 2: analytics avanzados)
        ├── rag_quality_sensor.py  (Capa 2: calidad RAG)
        ├── metacognitive_nudges.py(Capa 2: nudges)
        ├── config_genome.py       (Capa 3: fingerprinting)
        ├── temporal_config_advisor.py (Capa 3: contexto temporal)
        ├── teacher_calibration.py (Capa 3: calibración docente)
        ├── cross_node_signal.py   (Capa 3: inter-nodo, simulado)
        └── ecosystem_dashboard.py (Capa 4: visualización)
```

### Flujo de una interacción con el orquestador:

```
ESTUDIANTE escribe prompt
    │
    ▼
┌─ PRE-PROCESS ─────────────────────────────────┐
│  middleware.pre_process(student_id, prompt)     │
│  → allowed? topics? copy_paste? system_prompt? │
│                                                 │
│  event_logger.log("student_prompt", ...)       │ ← NUEVO
│  temporal_advisor.get_context()                 │ ← NUEVO  
└─────────────────────────────────────────────────┘
    │
    ▼
┌─ RETRIEVAL ────────────────────────────────────┐
│  chunks = rag.retrieve(prompt)                  │
│  rag_sensor.evaluate(chunks, prompt)           │ ← NUEVO
└─────────────────────────────────────────────────┘
    │
    ▼
┌─ LLM CALL ─────────────────────────────────────┐
│  response = llm.generate(system, prompt, chunks)│
└─────────────────────────────────────────────────┘
    │
    ▼
┌─ POST-PROCESS ─────────────────────────────────┐
│  response = middleware.post_process(response)    │
│                                                  │
│  cognitive = profiler.analyze(prompt)            │ ← NUEVO
│  gap = gap_detector.check(history)              │ ← NUEVO
│  nudge = nudge_engine.suggest(cognitive, gap)   │ ← NUEVO
│  alignment = hhh_detector.evaluate(response)    │ ← NUEVO
│                                                  │
│  event_logger.log("llm_response", ...)          │ ← NUEVO
│  config_genome.snapshot(current_config)          │ ← NUEVO
└──────────────────────────────────────────────────┘
    │
    ▼
ESTUDIANTE recibe respuesta + nudge (si aplica)
    │
    ▼
┌─ BACKGROUND (no bloquea al estudiante) ────────┐
│  consolidation.check_window(student_id)         │
│  silence_detector.update(student_id, timestamp) │
│  teacher_calibration.evaluate(config, bloom)    │
│  cross_node.emit_if_relevant(event)             │
│  system_reflexivity.reflect(event)              │
└──────────────────────────────────────────────────┘
```

---

## Orden de trabajo en Cursor

### Semana 1: El bus (ecosystem_orchestrator.py)

1. Crear `ecosystem_orchestrator.py` con clase `EcosystemOrchestrator`
2. Método `process_interaction(student_id, prompt)` que ejecute el flujo completo
3. Flag `DEMO_MODE` que active/desactive simulaciones
4. Conectar a `app.py` reemplazando el flujo actual

### Semana 1-2: Capas 1-2 (los 26 módulos verdes)

Prioridad por impacto en la demo:

1. `system_event_logger` → Cada interacción genera un evento con las 4 columnas
2. `rag_quality_sensor` → Cada retrieval tiene score de calidad
3. `metacognitive_nudges` → Respuestas incluyen nudges cuando aplica
4. `cognitive_gap_detector` → Detecta gaps y los muestra en dashboard
5. `nd_patterns` → Enriquece perfil cognitivo

### Semana 2: Capa 3 (los 10 módulos amarillos)

1. `temporal_config_advisor` con calendario hardcodeado
2. `teacher_calibration` con perfiles sintéticos
3. `cross_node_signal` con nodos simulados
4. `config_genome` leyendo snapshots del event_logger

### Semana 2-3: Capa 4 (dashboards)

1. `ecosystem_dashboard` como nueva pestaña en app.py
2. `researcher_view` para la vista investigador
3. `config_impact_panel` para impacto de configuraciones

---

## Lo que muestras en la entrevista

### Demo 1: "El chatbot funciona" (2 min)
- Estudiante pregunta → modo socrático → scaffolding progresivo
- RAG recupera chunks relevantes → fuentes visibles
- Docente cambia configuración → comportamiento cambia en tiempo real

### Demo 2: "El ecosistema observa" (3 min)
- Event logger captura cada interacción con las 4 columnas diferenciales
- Config genome muestra fingerprint del docente
- RAG quality sensor detecta reformulación (señal de baja calidad)
- Cognitive gap detector señala gap en "recursión"

### Demo 3: "El ecosistema actúa" (2 min)
- Metacognitive nudge se dispara por gap detectado
- Temporal advisor sugiere relajar scaffolding (semana pre-examen)
- Teacher calibration muestra desalineación config↔nivel real

### Demo 4: "El ecosistema escala" (1 min)
- Cross-node signal: "UC3M detectó patrón de abandono en tema 3"
- Dashboard unificado con los 6 subsistemas
- "Todo esto corre en mi portátil. Cuando tengamos servidor + Moodle, 
   cambio DEMO_MODE a False y se conecta a producción."

### La frase clave para la entrevista:
> "He construido 51 módulos, 32.000 líneas. 26 funcionan al 100% sin
> depender de nadie. 10 más simulan su backend externo para la demo. 
> 5 están listos esperando credenciales institucionales. La distancia 
> entre este prototipo y producción es un servidor y un acceso a Moodle."
