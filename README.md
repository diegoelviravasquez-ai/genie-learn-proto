# GENIE Learn — Ecosistema Pedagógico Inteligente

> *"No es un chatbot que responde. Es un ecosistema que OBSERVA, INTERPRETA y ADAPTA."*

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Research-green.svg)]()

Prototipo de middleware pedagógico para el proyecto GENIE Learn (CP25/152) — GSIC/EMIC, Universidad de Valladolid.

---

## 📊 Métricas del Ecosistema

| Métrica | Valor |
|---------|-------|
| Módulos Python | 59 |
| Archivos totales | 70+ |
| Líneas Python | ~35,000 |
| Líneas React | ~96,000 |
| Capas arquitectónicas | 8 |
| Modos de scaffolding | 8 |
| Vistas de usuario | 7 |

---

## 🚀 Instalación y Ejecución

```bash
git clone https://github.com/diegoelviravasquez-ai/genie-learn-proto.git
cd genie-learn-proto
pip install -r requirements.txt
pip install anthropic
streamlit run app.py
```

**Funciona en modo demo sin API keys** — respuestas simuladas pedagógicamente diferenciadas.

Para usar LLM real, añade tu API key en "Docente — Configuración" o en `.env`:
```
ANTHROPIC_API_KEY=tu_clave_aquí
```

---

## 🎯 7 Vistas de Usuario

| Vista | Rol | Función |
|-------|-----|---------|
| **Estudiante** | Alumno | Chat con el tutor IA |
| **Práctica Guiada** | Alumno | Mini-retos gamificados, logros, curiosidades epistémicas |
| **Docente — Configuración** | Profesor | 8 modos scaffolding, límites, RAG, API keys |
| **Docente — Analytics** | Profesor | Métricas, predictor de abandono, alertas |
| **Mapa Epistémico** | Ambos | Grafo de conceptos, dominio por tema, huecos |
| **Demo en Vivo** | Demo | Estudiante simulado con frustración y recuperación |
| **Investigador** | Investigador | Bloom, ICAP, ACH, ND patterns, meta-evaluación |

---

## 🧠 8 Modos de Scaffolding

| Modo | Estrategia | Cuándo usar |
|------|------------|-------------|
| `socratic` | Preguntas orientadoras | Fomentar pensamiento crítico |
| `hints` | Pistas progresivas | Desbloquear sin dar respuesta |
| `examples` | Ejemplos similares | Aprendizaje por analogía |
| `analogies` | Analogías del mundo real | Conectar con conocimiento previo |
| `direct` | Respuesta directa | Cuando el estudiante está bloqueado |
| `challenge` | Problema más simple | Reducir carga cognitiva |
| `rubber_duck` | Explicar paso a paso | Metacognición guiada |
| `progressive` | Escala automática | socrático → pistas → ejemplos → explicación |

---

## 🏗️ Arquitectura de 8 Capas

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPA 7: INVESTIGADOR                      │
│  paper_drafting_engine · pilot_design · o1_feedback_engine  │
├─────────────────────────────────────────────────────────────┤
│                    CAPA 6: ECOLÓGICA                         │
│     cross_node_signal · epistemic_ecology · gdpr_anonymizer │
├─────────────────────────────────────────────────────────────┤
│                    CAPA 5: TEMPORAL                          │
│  temporal_dynamics · consolidation_detector · effect_latency│
├─────────────────────────────────────────────────────────────┤
│                    CAPA 4: DOCENTE                           │
│  teacher_agency · config_genome · teacher_calibration       │
├─────────────────────────────────────────────────────────────┤
│                    CAPA 3: CALIDAD/ÉTICO                     │
│    hhh_alignment_detector · llm_judge · rag_quality_sensor  │
├─────────────────────────────────────────────────────────────┤
│                    CAPA 2: COGNITIVO                         │
│  cognitive_profiler · ach_diagnostic · epistemic_silence    │
│  nd_patterns · cognitive_gap_detector · metacognitive_nudges│
├─────────────────────────────────────────────────────────────┤
│                    CAPA 1: MIDDLEWARE                        │
│         middleware.py (8 modos) · llm_client · rag_pipeline │
├─────────────────────────────────────────────────────────────┤
│                    CAPA 0: NÚCLEO                            │
│            app.py · database · auth · lti_integration       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Inventario de Módulos (59 Python)

### Núcleo (8)
- `app.py` — Streamlit principal, 7 vistas
- `middleware.py` — 8 modos scaffolding, máquina de estados
- `llm_client.py` — Anthropic/OpenAI/Mock con fallback
- `rag_pipeline.py` — ChromaDB + TF-IDF fallback
- `database.py` — SQLite con seed de demo
- `auth.py` — MockLDAP (profesor_01, estudiante_01-04)
- `lti_integration.py` — MockLTI (curso FP-101)
- `integration.py` — Orquestador de módulos

### Cognitivo (12)
- `cognitive_profiler.py` — Perfil cognitivo del estudiante
- `cognitive_engine.py` — Motor de análisis cognitivo
- `cognitive_gap_detector.py` — Sondas epistémicas (Firlej & Hellens)
- `cognitive_pragmatics.py` — Pragmática conversacional
- `ach_diagnostic.py` — Análisis de Hipótesis Competitivas (Heuer/CIA)
- `epistemic_silence_detector.py` — Detecta lo que NO se pregunta
- `epistemic_autonomy.py` — Autonomía epistémica del estudiante
- `epistemic_ecology.py` — Ecología del conocimiento
- `nd_patterns.py` — Patrones neurodivergentes (TDAH, AACC, 2e)
- `consolidation_detector.py` — Spacing effect (Bjork, 1994)
- `metacognitive_nudges.py` — Intervenciones calibradas
- `cognitive_analyzer__1_.py` — Análisis adicional

### Docente (8)
- `teacher_agency_longitudinal.py` — Agencia docente (Priestley & Biesta)
- `teacher_calibration.py` — Calibración de configuraciones
- `teacher_notification_engine.py` — Alertas al docente
- `config_genome.py` — Fingerprint pedagógico
- `config_impact_panel.py` — Panel de impacto
- `config_interaction_analyzer.py` — Análisis de interacciones
- `temporal_config_advisor.py` — Recomendaciones temporales
- `learning_design_generator.py` — Generador de diseños

### Temporal/Dinámico (4)
- `temporal_dynamics.py` — Dinámicas temporales
- `trust_dynamics.py` — Modelo de confianza (Lee & See, 2004)
- `effect_latency_analyzer.py` — Latencia de efectos
- `cross_node_signal.py` — Señales entre universidades

### Calidad/Ético (5)
- `hhh_alignment_detector.py` — Auditoría HHH (Askell et al., 2021)
- `llm_judge.py` — LLM-as-Judge con rúbricas
- `rag_quality_sensor.py` — Calidad del retrieval
- `gdpr_anonymizer.py` — Privacidad by design
- `system_reflexivity.py` — Reflexividad del sistema

### Investigador (5)
- `researcher_view.py` — Vista completa de investigación
- `paper_drafting_engine.py` — Generación de secciones académicas
- `pilot_design.py` — Diseño de pilotos
- `o1_feedback_engine.py` — Bucle DSRM O3→O1 (Popper)
- `analytics_bridge__1_.py` — Puente de analytics

### Visualización (4)
- `autonomy_viz.py` — Visualización de autonomía
- `ecosystem_dashboard.py` — Dashboard del ecosistema
- `interaction_semiotics.py` — Semiótica de interacción
- `udl_adapter.py` — Adaptador UDL

### APIs (2)
- `api.py` — API REST principal
- `api__2_.py` — API alternativa

### Tests (8)
- `test_cognitive.py`
- `test_middleware.py`
- `test_nd_patterns.py`
- `test_cognitive_gap_detector.py`
- `test_consolidation_detector.py`
- `test_config_interaction_analyzer.py`
- `test_effect_latency_analyzer.py`
- `test_integration__1_.py`

---

## 📄 Documentación

| Archivo | Contenido |
|---------|-----------|
| `DESIGN_DECISIONS.md` | 20 ADRs documentados |
| `THEORETICAL_FOUNDATIONS.md` | Fundamentación teórica |
| `ECOSYSTEM_README.md` | Descripción del ecosistema |
| `docs/` | Mapas visuales interactivos |

---

## ⚛️ Frontend React (96K líneas)

- `genie_demo.jsx` — Demo interactiva (33K)
- `genie_learn_frontend.jsx` — Frontend completo (63K)

Preparado para migración a Next.js en Fase A (ver ADR-001 en DESIGN_DECISIONS.md).

---

## 🔬 Innovaciones Diferenciales

| Innovación | Módulo | ¿Existe en literatura? |
|------------|--------|------------------------|
| Middleware pedagógico ejecutable | `middleware.py` | Conceptual sí, implementación no |
| ACH para diagnóstico educativo | `ach_diagnostic.py` | No — transferencia de inteligencia |
| Detección de silencios epistémicos | `epistemic_silence_detector.py` | No — todos miden presencia |
| HHH alignment implementado | `hhh_alignment_detector.py` | Declarado en papers, no implementado |
| Bucle O3→O1 instrumentado | `o1_feedback_engine.py` | Double-loop teórico, no computacional |
| Patrones ND como adaptación | `nd_patterns.py` | Etiquetado sí, adaptación no |
| Cross-node signals anónimos | `cross_node_signal.py` | No en LA educativo |
| Gamificación con sondas epistémicas | Vista Práctica Guiada | No — gamificación sin diagnóstico |

---

## 📚 Referencias Teóricas Clave

- **Scaffolding:** Wood, Bruner & Ross (1976); Chi & Wylie (2014) ICAP
- **Metacognición:** Flavell (1979); Dunning-Kruger (1999)
- **Spacing Effect:** Bjork (1994) desirable difficulties
- **ACH:** Heuer (1999) Psychology of Intelligence Analysis
- **Neurodivergencia:** Barkley (2015); Silverman (2013); Reis et al. (2014)
- **Agencia Docente:** Priestley, Biesta & Robinson (2015)
- **Confianza:** Lee & See (2004) trust in automation
- **HHH:** Askell et al. (2021) Anthropic alignment
- **DSRM:** Peffers et al. (2007) Design Science Research
- **Double-loop:** Argyris & Schön (1978)

---

## 👤 Autor

**Diego Elvira Vásquez**
- 1º Premio HACK4EDU 2024 (UBUN.IA)
- Creador IA Trust Nexus, Eevee ESF, Sistema 27 Capitales
- Perfil 2e (AACC + TDAH)

---

## 📋 Proyecto

**GENIE Learn CP25/152**
- GSIC/EMIC — Universidad de Valladolid
- PIs: Bote-Lorenzo, Asensio-Pérez
- Best Paper CSEDU 2025
- Workshop LAK 2026 Bergen

---

*Febrero 2026*
