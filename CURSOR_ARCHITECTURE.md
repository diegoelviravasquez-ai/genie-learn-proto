# 🧬 GENIE Learn — Arquitectura del Ecosistema
## Documento de Estado para Cursor Composer

**Fecha:** Febrero 2026  
**Proyecto:** CP25/152 GSIC/EMIC-UVa  
**Total:** 53 módulos · 36,081 líneas de código Python

---

## 📊 RESUMEN EJECUTIVO

| Categoría | Módulos | Líneas | Estado | Prioridad |
|-----------|---------|--------|--------|-----------|
| **CORE** | 7 | 2,500 | 🟢 95% | CRÍTICO |
| **ANALYTICS** | 5 | 2,402 | 🟢 90% | ALTA |
| **TEMPORAL** | 4 | 3,046 | 🟢 85% | ALTA |
| **DETECTION** | 4 | 3,696 | 🟢 80% | MEDIA |
| **TEACHER** | 3 | 2,424 | 🟡 75% | ALTA |
| **RESEARCH** | 4 | 4,516 | 🟡 75% | MEDIA |
| **ADAPTATION** | 3 | 2,462 | 🟡 70% | MEDIA |
| **VIZ** | 3 | 2,106 | 🟡 65% | BAJA |
| **INFRASTRUCTURE** | 6 | 3,500 | 🟢 80% | CRÍTICO |
| **META** | 4 | 2,687 | 🟡 70% | BAJA |

**Estado global: ~78% completado**

---

## 🏗️ ARQUITECTURA DE CAPAS

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         CAPA DE PRESENTACIÓN                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │   app.py    │  │researcher   │  │ ecosystem   │  │ autonomy    │    │
│  │ (Streamlit) │  │  _view.py   │  │ _dashboard  │  │   _viz.py   │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                         CAPA DE ORQUESTACIÓN                            │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │   middleware.py │  │ config_genome   │  │ system_event_logger.py  │  │
│  │  (5 reglas ped) │  │ (fingerprints)  │  │   (audit trail)        │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────┘  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                      CAPA DE ANÁLISIS COGNITIVO                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │
│  │ cognitive     │  │ nd_patterns   │  │ epistemic     │               │
│  │ _profiler.py  │  │ (ADHD, ASD)   │  │ _autonomy.py  │               │
│  │ (Bloom+ICAP)  │  └───────────────┘  │ (dependencia) │               │
│  └───────────────┘                     └───────────────┘               │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐               │
│  │ trust         │  │ epistemic     │  │ cognitive     │               │
│  │ _dynamics.py  │  │ _ecology.py   │  │ _gap_detector │               │
│  │ (Lee & See)   │  │ (colectivo)   │  │ (Dunning-K)   │               │
│  └───────────────┘  └───────────────┘  └───────────────┘               │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                      CAPA DE ANÁLISIS TEMPORAL                          │
│  ┌─────────────────────┐  ┌─────────────────────┐                      │
│  │ consolidation       │  │ effect_latency      │                      │
│  │ _detector.py        │  │ _analyzer.py        │                      │
│  │ (48-72h, Ebbinghaus)│  │ (config → efecto)   │                      │
│  └─────────────────────┘  └─────────────────────┘                      │
│  ┌─────────────────────┐  ┌─────────────────────┐                      │
│  │ temporal            │  │ temporal_config     │                      │
│  │ _dynamics.py        │  │ _advisor.py         │                      │
│  │ (micro/meso/macro)  │  │ (recomendaciones)   │                      │
│  └─────────────────────┘  └─────────────────────┘                      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                      CAPA DE SOPORTE DOCENTE                            │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐   │
│  │ teacher           │  │ teacher           │  │ teacher_agency    │   │
│  │ _calibration.py   │  │ _notification.py  │  │ _longitudinal.py  │   │
│  │ (mental model)    │  │ (alertas)         │  │ (WP5 data)        │   │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                      CAPA DE ADAPTACIÓN                                 │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐   │
│  │ udl_adapter.py    │  │ metacognitive     │  │ learning_design   │   │
│  │ (UDL CAST)        │  │ _nudges.py        │  │ _generator.py     │   │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                      CAPA DE INVESTIGACIÓN                              │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐   │
│  │ o1_feedback       │  │ pilot_design.py   │  │ paper_drafting    │   │
│  │ _engine.py        │  │ (WP5 automatiz)   │  │ _engine.py        │   │
│  │ (O3→O1 loop)      │  └───────────────────┘  │ (LaTeX gen)       │   │
│  └───────────────────┘                         └───────────────────┘   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                      CAPA DE INFRAESTRUCTURA                            │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐           │
│  │ api.py    │  │database.py│  │ auth.py   │  │ gdpr      │           │
│  │ (FastAPI) │  │(SQLAlchemy│  │ (JWT)     │  │_anonymizer│           │
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘           │
│  ┌───────────┐  ┌───────────────────────────────────────────┐          │
│  │ lti       │  │ rag_pipeline.py + llm_client.py           │          │
│  │_integr.py │  │ (ChromaDB + OpenAI/Anthropic)             │          │
│  └───────────┘  └───────────────────────────────────────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📋 ESTADO DETALLADO POR MÓDULO

### 🟢 CORE (95% completado)

| Módulo | Líneas | Clases | Tests | Estado | Pendiente |
|--------|--------|--------|-------|--------|-----------|
| `middleware.py` | 443 | 3 | ✅ | 100% | — |
| `rag_pipeline.py` | 377 | 2 | — | 95% | Re-ranking cross-encoder |
| `llm_client.py` | 221 | 3 | — | 100% | — |
| `cognitive_engine.py` | 359 | 3 | — | 90% | Calibración español |
| `app.py` | 636 | 0 | — | 95% | Pulir CSS mobile |
| `cognitive_analyzer.py` | 81 | 2 | — | 100% | — |

**Lo que funciona:** Flujo end-to-end estudiante→middleware→RAG→LLM→respuesta.  
**Lo que falta:** Evaluación RAGAS del retrieval con materiales reales.

---

### 🟢 ANALYTICS (90% completado)

| Módulo | Líneas | Tests | Estado | Innovación |
|--------|--------|-------|--------|------------|
| `cognitive_profiler.py` | 613 | — | 90% | Bloom + ICAP combinados |
| `nd_patterns.py` | 501 | ✅ | 95% | 6 patrones neurodivergentes |
| `epistemic_autonomy.py` | 474 | — | 85% | 4 fases de dependencia |
| `epistemic_ecology.py` | 546 | — | 85% | Análisis colectivo |
| `trust_dynamics.py` | 268 | — | 90% | Lee & See (2004) |

**Lo que funciona:** Detección de niveles Bloom, patrones ND, fases de autonomía.  
**Lo que falta:** Validación con datos reales del piloto WP5.

---

### 🟢 TEMPORAL (85% completado)

| Módulo | Líneas | Tests | Estado | Contribución teórica |
|--------|--------|-------|--------|----------------------|
| `consolidation_detector.py` | 1211 | ✅ | 90% | Ventana 48-72h Ebbinghaus |
| `effect_latency_analyzer.py` | 630 | ✅ | 85% | Delay config→efecto |
| `temporal_dynamics.py` | 587 | — | 80% | Micro/meso/macro |
| `temporal_config_advisor.py` | 618 | — | 75% | Recomendaciones |

**Innovación:** Primer sistema que modela consolidación inter-sesión en chatbots educativos.  
**Lo que falta:** Umbrales calibrados con datos longitudinales.

---

### 🟢 DETECTION (80% completado)

| Módulo | Líneas | Tests | Estado | Problema que ataca |
|--------|--------|-------|--------|-------------------|
| `cognitive_gap_detector.py` | 1594 | ✅ | 85% | Dunning-Kruger |
| `epistemic_silence_detector.py` | 848 | — | 75% | Ausencias informativas |
| `hhh_alignment_detector.py` | 737 | — | 70% | Brecha declaración/ejecución |
| `ach_diagnostic.py` | 517 | — | 80% | ACH de Heuer |

**Lo que funciona:** Detección de gaps metacognitivos, silencios, desalineamiento.  
**Lo que falta:** Integración con metacognitive_nudges para intervención.

---

### 🟡 TEACHER (70% completado)

| Módulo | Líneas | Tests | Estado | Pendiente |
|--------|--------|-------|--------|-----------|
| `teacher_agency_longitudinal.py` | 1124 | — | 75% | UI para visualización |
| `teacher_calibration.py` | 894 | — | 70% | Feedback loop |
| `teacher_notification_engine.py` | 406 | — | 65% | Canales (email, Slack) |

**Lo que funciona:** Modelo teórico de agencia docente, calibración mental model.  
**Lo que falta:** Conexión con canales de notificación reales.

---

### 🟡 RESEARCH (75% completado)

| Módulo | Líneas | Estado | Función |
|--------|--------|--------|---------|
| `o1_feedback_engine.py` | 1819 | 80% | Retroalimentación O3→O1 |
| `pilot_design.py` | 1275 | 75% | Automatización WP5 |
| `paper_drafting_engine.py` | 801 | 70% | Generación LaTeX |
| `researcher_view.py` | 621 | 75% | Dashboard investigador |

**Lo que funciona:** Detección de anomalías, propuestas de revisión de escenarios.  
**Lo que falta:** Templates de papers específicos para venues (LAK, L@S, CHI).

---

### 🟡 ADAPTATION (65% completado)

| Módulo | Líneas | Estado | Framework base |
|--------|--------|--------|----------------|
| `learning_design_generator.py` | 958 | 70% | Generates learning designs |
| `udl_adapter.py` | 776 | 65% | UDL CAST |
| `metacognitive_nudges.py` | 728 | 60% | Intervenciones |

**Lo que funciona:** Estructura de adaptaciones UDL, tipos de nudges.  
**Lo que falta:** Conexión bidireccional con middleware para aplicar adaptaciones.

---

### 🟠 INFRASTRUCTURE (50% completado) — CRÍTICO

| Módulo | Líneas | Estado | Problema |
|--------|--------|--------|----------|
| `api.py` | 850 | 60% | Endpoints definidos, no probados |
| `database.py` | 329 | 55% | SQLAlchemy models, no migrations |
| `lti_integration.py` | 587 | 40% | LTI 1.3 parcial |
| `auth.py` | 138 | 50% | JWT básico |
| `gdpr_anonymizer.py` | 353 | 70% | K-anonymity implementado |
| `api__2_.py` | 576 | 45% | Versión alternativa |

**BLOQUEANTE:** La infraestructura necesita trabajo antes del piloto.  
**Lo que falta:**
1. Migrations Alembic
2. Tests de API con pytest
3. LTI 1.3 completo para Moodle
4. OAuth2 flow

---

### 🟡 VIZ (60% completado)

| Módulo | Líneas | Estado | Dependencias |
|--------|--------|--------|--------------|
| `config_impact_panel.py` | 833 | 65% | Plotly |
| `autonomy_viz.py` | 771 | 60% | Plotly |
| `ecosystem_dashboard.py` | 502 | 55% | Streamlit + Plotly |

**Lo que funciona:** Gráficos básicos.  
**Lo que falta:** Responsividad, exportación, interactividad avanzada.

---

### 🟡 META (70% completado)

| Módulo | Líneas | Estado | Función |
|--------|--------|--------|---------|
| `system_event_logger.py` | 784 | 75% | Audit trail |
| `system_reflexivity.py` | 722 | 70% | Meta-análisis |
| `cross_node_signal.py` | 641 | 65% | Multi-instancia |
| `config_genome.py` | 540 | 70% | Fingerprints |

---

## 🎯 PRIORIDADES PARA CURSOR

### P0: CRÍTICO (antes del piloto)

```
# 1. Arreglar imports rotos
mv cognitive_analyzer__1_.py cognitive_analyzer.py
mv analytics_bridge__1_.py analytics_bridge.py
mv api__2_.py  # eliminar o consolidar con api.py

# 2. Database migrations
cd infrastructure && alembic init migrations
alembic revision --autogenerate -m "Initial schema"

# 3. Tests de integración
pytest test_integration.py test_middleware.py -v
```

### P1: ALTA (semana 1)

1. **LTI 1.3 completo** — `lti_integration.py`
   - Implementar Deep Linking
   - Probar con Moodle sandbox

2. **API tests** — `api.py`
   - Añadir fixtures pytest
   - Probar todos los endpoints

3. **RAG evaluation** — `rag_pipeline.py`
   - Integrar RAGAS
   - Métricas: faithfulness, relevance, context_recall

### P2: MEDIA (semana 2-3)

1. **Conexión adaptation→middleware**
   - `metacognitive_nudges.py` → `middleware.py`
   - Aplicar nudges en tiempo real

2. **Teacher notifications**
   - `teacher_notification_engine.py` → email/Slack
   - Webhooks configurables

3. **Dashboard responsive**
   - `ecosystem_dashboard.py`
   - Mobile-friendly

### P3: BAJA (post-piloto)

1. Paper templates para LAK, L@S, CHI
2. Cross-node signals para multi-instancia
3. Exportación de visualizaciones

---

## 📁 ESTRUCTURA DE DIRECTORIOS RECOMENDADA

```
genie-learn/
├── README.md
├── pyproject.toml          # Poetry/pip config
├── alembic.ini             # Database migrations
│
├── core/                   # Módulos esenciales
│   ├── __init__.py
│   ├── middleware.py
│   ├── rag_pipeline.py
│   ├── llm_client.py
│   └── cognitive_engine.py
│
├── analytics/              # Análisis cognitivo
│   ├── __init__.py
│   ├── cognitive_profiler.py
│   ├── nd_patterns.py
│   ├── epistemic_autonomy.py
│   ├── epistemic_ecology.py
│   └── trust_dynamics.py
│
├── temporal/               # Análisis temporal
│   ├── __init__.py
│   ├── consolidation_detector.py
│   ├── effect_latency_analyzer.py
│   ├── temporal_dynamics.py
│   └── temporal_config_advisor.py
│
├── detection/              # Detectores especializados
│   ├── __init__.py
│   ├── cognitive_gap_detector.py
│   ├── epistemic_silence_detector.py
│   ├── hhh_alignment_detector.py
│   └── ach_diagnostic.py
│
├── teacher/                # Soporte docente
│   ├── __init__.py
│   ├── teacher_calibration.py
│   ├── teacher_notification_engine.py
│   └── teacher_agency_longitudinal.py
│
├── adaptation/             # Adaptación UDL
│   ├── __init__.py
│   ├── udl_adapter.py
│   ├── metacognitive_nudges.py
│   └── learning_design_generator.py
│
├── research/               # Herramientas investigador
│   ├── __init__.py
│   ├── o1_feedback_engine.py
│   ├── pilot_design.py
│   ├── paper_drafting_engine.py
│   └── researcher_view.py
│
├── infrastructure/         # Backend
│   ├── __init__.py
│   ├── api.py
│   ├── database.py
│   ├── auth.py
│   ├── lti_integration.py
│   ├── gdpr_anonymizer.py
│   └── migrations/
│
├── viz/                    # Visualizaciones
│   ├── __init__.py
│   ├── autonomy_viz.py
│   ├── ecosystem_dashboard.py
│   └── config_impact_panel.py
│
├── meta/                   # Meta-sistema
│   ├── __init__.py
│   ├── system_reflexivity.py
│   ├── system_event_logger.py
│   ├── config_genome.py
│   └── cross_node_signal.py
│
├── ui/                     # Interfaces
│   ├── app.py              # Streamlit main
│   └── static/
│
├── tests/
│   ├── test_middleware.py
│   ├── test_nd_patterns.py
│   ├── test_consolidation_detector.py
│   ├── test_cognitive_gap_detector.py
│   ├── test_effect_latency_analyzer.py
│   └── test_config_interaction_analyzer.py
│
└── docs/
    ├── ARCHITECTURE.md     # Este documento
    ├── API.md
    └── DEPLOYMENT.md
```

---

## 🔧 .cursorrules RECOMENDADO

```yaml
# GENIE Learn — Cursor Rules
# ==========================

language: python
version: "3.11"

# Estilo de código
style:
  docstrings: google
  line_length: 100
  quotes: double
  
# Imports organizados
imports:
  group_order:
    - stdlib
    - third_party
    - local
  local_prefix: "genie_learn"

# Convenciones del proyecto
conventions:
  - Todos los módulos tienen docstring con PROBLEMA QUE ATACA
  - Dataclasses para DTOs
  - Type hints obligatorios
  - Tests con pytest (prefix test_)
  
# Dependencias del ecosistema
ecosystem_modules:
  core:
    - middleware.py      # 5 reglas pedagógicas
    - rag_pipeline.py    # ChromaDB + embeddings
    - llm_client.py      # OpenAI/Anthropic wrapper
    - cognitive_engine.py # Bloom + ICAP
    
  analytics:
    - cognitive_profiler.py
    - nd_patterns.py     # ADHD, ASD patterns
    - epistemic_autonomy.py
    - trust_dynamics.py  # Lee & See 2004
    
  temporal:
    - consolidation_detector.py  # Ebbinghaus 48-72h
    - effect_latency_analyzer.py
    
# Frameworks teóricos a respetar
theoretical_frameworks:
  - "Taxonomía de Bloom Revisada (Anderson & Krathwohl, 2001)"
  - "Framework ICAP (Chi & Wylie, 2014)"
  - "Trust in Automation (Lee & See, 2004)"
  - "Universal Design for Learning (CAST)"
  - "Spacing Effect (Cepeda et al., 2006)"
  - "Dunning-Kruger Effect (Kruger & Dunning, 1999)"

# Patrones a evitar
avoid:
  - print() en producción (usar logging)
  - Hardcoded API keys
  - Mutable default arguments
  - Imports circulares
  
# Tests requeridos para merge
required_tests:
  - test_middleware.py
  - test_nd_patterns.py
  - test_integration.py
```

---

## 📈 MÉTRICAS DE COMPLETITUD

```
CORE:           ████████████████████░░░ 95%
ANALYTICS:      ██████████████████░░░░░ 90%
TEMPORAL:       █████████████████░░░░░░ 85%
DETECTION:      ████████████████░░░░░░░ 80%
RESEARCH:       ███████████████░░░░░░░░ 75%
TEACHER:        ██████████████░░░░░░░░░ 70%
META:           ██████████████░░░░░░░░░ 70%
ADAPTATION:     █████████████░░░░░░░░░░ 65%
VIZ:            ████████████░░░░░░░░░░░ 60%
INFRASTRUCTURE: ██████████░░░░░░░░░░░░░ 50%  ← BLOQUEANTE
─────────────────────────────────────────────
GLOBAL:         ██████████████░░░░░░░░░ 72%
```

---

## 🚀 COMANDO PARA EMPEZAR

```bash
# En Cursor, abrir el proyecto y ejecutar:
cd genie-learn
python -c "
from middleware import PedagogicalMiddleware, PedagogicalConfig
from rag_pipeline import get_rag_pipeline, SAMPLE_COURSE_CONTENT
from llm_client import get_llm_client

# Test rápido
config = PedagogicalConfig(scaffolding_mode='socratic')
mw = PedagogicalMiddleware(config)
rag = get_rag_pipeline(use_openai=False)
rag.ingest_text(SAMPLE_COURSE_CONTENT, 'demo.pdf')
llm = get_llm_client()

result = mw.pre_process('test', '¿Qué es un bucle for?')
print(f'Scaffolding: {result[\"scaffolding_level\"]}')
print(f'Topics: {result[\"detected_topics\"]}')
print('✅ Sistema operativo')
"
```

---

*Documento generado para entrevista CP25/152 GSIC/EMIC-UVa · Febrero 2026*
