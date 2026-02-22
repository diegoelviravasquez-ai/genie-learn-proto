# Decisiones de Diseño — Registro de Arquitectura

> Este documento registra las decisiones técnicas y de diseño del prototipo GENIE Learn,
> incluyendo alternativas consideradas y razones de descarte. No es documentación
> de usuario — es el log de pensamiento de ingeniería que un revisor técnico
> necesita para entender POR QUÉ el sistema es como es.
>
> **Ecosistema:** 39 módulos · 35K líneas Python · 8 capas funcionales

---

## Índice de Decisiones

| ADR | Tema | Capa |
|-----|------|------|
| 001 | Streamlit sobre Next.js | Núcleo |
| 002 | Scaffolding con máquina de estados | Núcleo |
| 003 | TF-IDF como baseline de evaluación | RAG |
| 004 | ACH para diagnóstico sin datos etiquetados | Cognitivo |
| 005 | Neurodivergencia como adaptación, no etiqueta | Cognitivo |
| 006 | Alucinaciones pedagógicas controladas | Middleware |
| 007 | Mock LLM pedagógicamente diferenciado | Núcleo |
| 008 | Silencio epistémico — medir ausencias | Cognitivo |
| 009 | HHH Alignment como auditoría real | Ético |
| 010 | Bucle O3→O1 — retroalimentación de supuestos | Meta |
| 011 | Config Genome — fingerprint pedagógico | Docente |
| 012 | Teacher Agency Longitudinal | Docente |
| 013 | Consolidation Detector — spacing effect | Temporal |
| 014 | Cross-Node Signals — inteligencia colectiva | Ecológico |
| 015 | Paper Drafting Engine — auto-documentación | Investigador |
| 016 | Metacognitive Nudges — intervenciones calibradas | Estudiante |
| 017 | LLM-as-Judge para evaluación | Calidad |
| 018 | GDPR Anonymizer — privacidad by design | Ético |
| 019 | Cognitive Gap Detector — unknown unknowns | Cognitivo |
| 020 | Trust Dynamics — modelo de confianza | Relacional |

---

## ADR-001: Streamlit sobre Next.js para el prototipo

**Contexto:** El stack final del proyecto GENIE Learn usa React. Mi stack personal
incluye Next.js/React/TypeScript. La tentación era construir en Next.js para demostrar
dominio del stack de producción.

**Decisión:** Streamlit.

**Razones:**
- El prototipo es un instrumento de investigación, no un producto. La velocidad de iteración
  importa más que la estética del frontend.
- Los evaluadores (Bote-Lorenzo, Asensio-Pérez) son ingenieros de telecomunicaciones, no
  diseñadores de UI. Van a evaluar la lógica del middleware y la coherencia del pipeline RAG.
- Streamlit permite la vista multi-rol (estudiante/docente/investigador) con un radio button.
- **Deuda técnica aceptada:** Streamlit no escala a producción con 200 estudiantes concurrentes.
  Migración planificada en Fase A.

**Alternativas descartadas:**
- Next.js + FastAPI: correcto para producción, excesivo para prototipo de investigación.
- Gradio: demasiado limitado para las 4 vistas que necesito.

**Nota:** El repo incluye 96K líneas de React (`genie_demo.jsx`, `genie_learn_frontend.jsx`)
como demostración de capacidad técnica y preparación para migración.

---

## ADR-002: Scaffolding socrático con máquina de estados

**Contexto:** La forma "fácil" de implementar scaffolding es meter todo en el system prompt
y dejar que el LLM gestione la progresión. Esto es lo que hace el 90% de los wrappers educativos.

**Decisión:** Máquina de estados explícita en el middleware con 4 niveles discretos.

**Razones:**
- El LLM no tiene memoria fiable entre turnos. Si le dices "sé más directo porque es la
  tercera vez que pregunta", necesitas que el LLM cuente — y los LLMs cuentan mal.
- Con estados explícitos, el INVESTIGADOR puede medir la progresión: "el 60% de los
  estudiantes llegaron al nivel 3 en bucles, pero solo el 20% en recursión." Dato publicable.
- Separación de POLÍTICA (decisión pedagógica del docente) del MECANISMO (implementación).
  Principio de Saltzer et al. (1984).

**Limitación conocida:** La transición entre niveles es por conteo de intentos. En producción,
debería ser por análisis semántico de la respuesta del estudiante.

---

## ADR-003: TF-IDF como baseline de evaluación

**Contexto:** El pipeline RAG usa OpenAI embeddings + ChromaDB cuando hay API key, y cae
a TF-IDF cuando no la hay.

**Decisión:** El fallback no es solo "para que funcione sin API key" — es un **baseline de
evaluación** deliberado.

**Razones:**
- Necesito saber si el retrieval semántico MEJORA las respuestas respecto al retrieval por
  keywords. Sin baseline, no hay medida.
- Tres condiciones experimentales: RAG-semántico vs. RAG-keywords vs. no-RAG.
- TF-IDF es sorprendentemente bueno para documentación técnica donde los términos son específicos.

**Métrica pendiente:** Implementar RAGAS (faithfulness + answer relevancy + context recall).

---

## ADR-004: ACH para diagnóstico sin datos etiquetados

**Contexto:** La forma estándar de clasificar "tipos de dificultad" de un estudiante sería
entrenar un clasificador supervisado sobre datos etiquetados.

**Decisión:** Análisis de Hipótesis Competitivas (Heuer, 1999) como motor de diagnóstico.

**Razones:**
- No tengo datos etiquetados. Sin ground truth, no hay supervisado.
- ACH opera con razonamiento estructurado sobre evidencia cualitativa.
- **Beneficio a largo plazo:** Los diagnósticos ACH validados por docentes se convierten en
  etiquetas de entrenamiento para un clasificador supervisado en Fase C.

**Origen:** Máster en Análisis de Inteligencia (LISA-UDIMA). Transferencia genuina entre dominios.

**Módulo:** `ach_diagnostic.py` (517 líneas)

---

## ADR-005: Neurodivergencia como adaptación, no etiquetado

**Contexto:** El módulo `nd_patterns.py` detecta patrones de interacción asociados a TDAH,
AACC y perfiles 2e. Territorio éticamente sensible.

**Decisión:** El sistema detecta PATRONES FUNCIONALES y sugiere ADAPTACIONES. Nunca genera
etiquetas diagnósticas.

**Razones:**
- Diagnosticar es acto clínico. Un chatbot no diagnostica.
- El docente ve "interacción episódica con ráfagas de hiperfoco", no "posible TDAH".
- Fundamentación: Value-Sensitive Design (Friedman et al., 2017).

**Conexión personal:** Perfil 2e (AACC + TDAH diagnosticado). UBUN.IA (1er premio HACK4EDU 2024)
abordaba exactamente esto.

**Módulo:** `nd_patterns.py` (500 líneas)

---

## ADR-006: Alucinaciones pedagógicas controladas

**Contexto:** El paper LAK 2026 introduce "forced hallucinations" — inyectar errores
deliberados para fomentar lectura crítica. Solo el 50% de profesores la consideró útil.

**Decisión:** Implementación mínima con flag explícito visible al estudiante.

**Razones:**
- Pedagógicamente interesante (Bjork, 1994: "desirable difficulties") pero éticamente compleja.
- Si el estudiante no sabe que la respuesta puede ser incorrecta, el chatbot MIENTE.
- Compromiso conservador: advertencia visible reduce efecto pedagógico pero elimina problema ético.

**Para el piloto:** Diseñar experimento con grupo control vs. experimental.

---

## ADR-007: Mock LLM pedagógicamente diferenciado

**Contexto:** El prototipo funciona sin API key usando MockLLMClient.

**Decisión:** Las respuestas mock son pedagógicamente diferenciadas, no lorem ipsum.

**Razones:**
- Si alguien ejecuta sin API key, necesita VER la diferencia entre scaffolding socrático
  y respuesta directa.
- Las respuestas mock son ESPECIFICACIÓN: gold standard contra el que evalúo si el system
  prompt produce el comportamiento deseado con LLM real.

---

## ADR-008: Silencio Epistémico — medir ausencias

**Contexto:** Todos los sistemas de Learning Analytics registran lo que los estudiantes HACEN.
Nadie mide lo que el estudiante NO preguntó cuando estadísticamente debería haberlo hecho.

**Decisión:** Implementar detector de silencios epistémicos basado en distribuciones de
referencia colectivas.

**Fundamento teórico:**
- Metacognition (Flavell, 1979; Dunning-Kruger, 1999): "no saber que no se sabe"
- Principio SIGINT de "ausencia significativa" — cuando una fuente que normalmente genera
  tráfico deja de hacerlo, esa ausencia es señal
- Sherlock Holmes, "Silver Blaze": "El curioso incidente del perro en la noche fue que
  el perro no hizo nada."

**Innovación:** Ningún sistema de LA publicado mide AUSENCIA de eventos. Todos responden a EVENTOS.

**Módulo:** `epistemic_silence_detector.py` (848 líneas)

---

## ADR-009: HHH Alignment como auditoría real

**Contexto:** El paper CSEDU 2025 cita el framework HHH (Helpful, Honest, Harmless) de
Askell et al. (2021) como principio rector. Pero NINGÚN módulo evalúa si las respuestas
generadas CUMPLEN el marco.

**Decisión:** Implementar auditor HHH que evalúa cada respuesta DESPUÉS de generarla y
ANTES de entregarla al estudiante.

**Razones:**
- Value-Sensitive Design exige que los valores se IMPLEMENTEN, no solo se declaren.
- La definición de "helpful" en educación es no-trivial: a veces lo MÁS útil es NO responder
  directamente (Bjork, 1994: desirable difficulties).
- El módulo puede bloquear respuestas que no superen umbrales mínimos.

**Módulo:** `hhh_alignment_detector.py` (737 líneas)

---

## ADR-010: Bucle O3→O1 — retroalimentación de supuestos

**Contexto:** Los cuatro objetivos del proyecto (O1-O4) operan en flujo unidireccional.
Nadie retroalimenta O1 (escenarios HL) con evidencia empírica de los pilotos.

**Decisión:** Implementar motor de retroalimentación que cierra el ciclo DSRM.

**Fundamento teórico:**
- Design Science Research Methodology (Peffers et al., 2007)
- Double-loop learning (Argyris & Schön, 1978): cuestionar los supuestos que gobiernan
  las acciones, no solo ajustar las acciones
- Popper (1963): Un supuesto sin condiciones de falsación no es ciencia, es dogma

**Implementación:** Cada O1Assumption lleva un `evidence_threshold` explícito: el n mínimo
para considerar válida la prueba.

**Módulo:** `o1_feedback_engine.py` (1819 líneas — el más grande del sistema)

---

## ADR-011: Config Genome — fingerprint pedagógico

**Contexto:** El docente configura múltiples parámetros (scaffolding mode, límites, RAG, etc.).
Las combinaciones posibles son enormes. ¿Cómo caracterizar el "estilo pedagógico" emergente?

**Decisión:** Implementar un "genoma de configuración" que genera un fingerprint único
para cada combinación de parámetros.

**Razones:**
- Permite agrupar docentes por estilo pedagógico sin preguntar explícitamente
- Facilita análisis de correlación: "¿qué configuraciones producen mejores resultados?"
- El fingerprint es hasheable: permite lookup eficiente en base de datos

**Estilos detectados:**
- `scaffolded_explorer`: alto RAG, socrático, límites flexibles
- `strict_guardian`: límites estrictos, sin soluciones directas
- `permissive_guide`: todo permitido, respuestas directas
- `challenge_based`: alucinaciones activas, límites estrictos
- `mixed`: combinaciones no categorizables

**Módulo:** `config_genome.py` (580 líneas)

---

## ADR-012: Teacher Agency Longitudinal

**Contexto:** El proyecto investiga cómo el docente desarrolla agencia al co-diseñar con
tecnologías inteligentes. Pero la agencia no es un estado estático — evoluciona.

**Decisión:** Implementar tracker longitudinal de agencia docente basado en Priestley & Biesta.

**Framework:** Priestley, Biesta & Robinson (2015) — agencia como interacción de:
- Dimensión iterativa (pasado): experiencias previas con tecnología
- Dimensión proyectiva (futuro): intenciones y expectativas
- Dimensión práctica-evaluativa (presente): juicios en contexto

**Métricas:** IFS Score (Iteration-Forward-Situation), calculado sobre patrones de
cambio de configuración a lo largo del tiempo.

**Módulo:** `teacher_agency_longitudinal.py` (1250 líneas)

---

## ADR-013: Consolidation Detector — spacing effect

**Contexto:** El spacing effect (Bjork, 1994) indica que el aprendizaje se consolida mejor
con práctica distribuida. ¿Cómo detectar si un estudiante está consolidando o no?

**Decisión:** Implementar detector de consolidación basado en ventanas temporales de 48-72h.

**Señales de consolidación:**
- Preguntas sobre el mismo topic en sesiones separadas por 24-72h
- Progresión de nivel Bloom entre sesiones (no dentro de una sesión)
- Reducción de copy-paste score en preguntas subsiguientes
- Aumento de metacognición explícita

**Señales de NO consolidación:**
- Cramming: muchas preguntas del mismo topic en una sesión
- Reset: mismas preguntas Bloom-1 después de días sin actividad
- Abandono: silencio epistémico prolongado post-dificultad

**Módulo:** `consolidation_detector.py` (1210 líneas)

---

## ADR-014: Cross-Node Signals — inteligencia colectiva

**Contexto:** GENIE Learn se desplegará en múltiples universidades (UC3M, UVa, UPF).
Cada nodo genera datos independientes. ¿Cómo aprovechar la inteligencia colectiva?

**Decisión:** Implementar sistema de señales anónimas entre nodos.

**Principio:** Si el nodo UC3M detecta que el 80% de estudiantes tienen dificultad con
recursión en la semana 3, los nodos UVa y UPF reciben una alerta ANTES de llegar a esa semana.

**Privacidad:** Las señales son agregadas y anónimas. No se comparten datos individuales.
Solo patrones estadísticos: "topic X, semana Y, dificultad Z%".

**Beneficio:** Inteligencia colectiva sin comprometer privacidad. El todo es mayor que
la suma de las partes.

**Módulo:** `cross_node_signal.py` (680 líneas)

---

## ADR-015: Paper Drafting Engine — auto-documentación

**Contexto:** El proyecto generará papers académicos. Los datos están en el sistema.
¿Por qué escribir manualmente las secciones de resultados?

**Decisión:** Implementar motor que extrae datos del sistema y genera secciones académicas.

**Capacidades:**
- Genera tablas de estadísticos descriptivos en formato LaTeX
- Produce gráficos listos para publicación
- Redacta borradores de secciones de resultados con citas de los datos
- Exporta a formatos de conferencia (ACM, IEEE, Springer)

**Limitación:** El motor genera borradores. El investigador revisa, edita y aprueba.
No es generación automática de papers — es asistencia estructurada.

**Módulo:** `paper_drafting_engine.py` (920 líneas)

---

## ADR-016: Metacognitive Nudges — intervenciones calibradas

**Contexto:** El sistema detecta patrones (gaps, silencios, dificultades). ¿Cómo intervenir
sin ser intrusivo?

**Decisión:** Implementar sistema de nudges metacognitivos calibrados por perfil.

**Tipos de nudge:**
- **Reflexivo:** "¿Qué parte de tu pregunta te cuesta más formular?"
- **Estratégico:** "Antes de preguntar, ¿has intentado descomponer el problema?"
- **Afectivo:** "Es normal sentir frustración. Tómate un momento."
- **Metacognitivo:** "¿Qué diferencia hay entre esta pregunta y la anterior?"

**Calibración:** La frecuencia e intensidad de nudges se ajusta al perfil del estudiante.
Un estudiante con alta metacognición recibe menos nudges. Uno con patrón de frustración
recibe más nudges afectivos.

**Módulo:** `metacognitive_nudges.py` (830 líneas)

---

## ADR-017: LLM-as-Judge para evaluación

**Contexto:** Evaluar la calidad de las respuestas del chatbot requiere juicio humano.
Pero el juicio humano no escala. ¿Cómo automatizar sin perder validez?

**Decisión:** Implementar LLM-as-Judge con rúbricas explícitas y calibración contra gold standard.

**Proceso:**
1. Definir rúbrica con criterios y ejemplos (gold standard humano)
2. Calibrar el LLM-judge contra el gold standard
3. Medir inter-rater reliability (LLM vs. humano)
4. Usar LLM-judge solo donde la concordancia supera umbral

**Limitación:** El LLM-judge NO reemplaza evaluación humana para decisiones críticas.
Es filtro de primer nivel para volumen alto.

**Módulo:** `llm_judge.py` (310 líneas)

---

## ADR-018: GDPR Anonymizer — privacidad by design

**Contexto:** Los logs de interacción contienen datos personales. GDPR exige privacidad
by design, no privacidad como parche posterior.

**Decisión:** Implementar anonimizador en el pipeline de logging, no en post-proceso.

**Técnicas:**
- Pseudonimización de IDs de estudiante (hash + salt rotativo)
- Supresión de PII en texto libre (regex + NER)
- K-anonimidad para exports de investigación
- Differential privacy para estadísticos agregados

**Principio:** Los datos crudos NUNCA salen del sistema. Solo salen datos procesados
que cumplen k-anonimidad mínima.

**Módulo:** `gdpr_anonymizer.py` (350 líneas)

---

## ADR-019: Cognitive Gap Detector — unknown unknowns

**Contexto:** El estudiante pregunta sobre lo que sabe que no sabe. ¿Cómo detectar
lo que no sabe que no sabe?

**Decisión:** Implementar sondas epistémicas basadas en Firlej (2024) para descubrir
gaps cognitivos no manifestados.

**Método:** Inyectar preguntas de sondeo que revelan gaps:
- "¿Qué diferencia hay entre X e Y?" (si el estudiante no distingue, hay gap)
- "¿En qué caso usarías X en vez de Y?" (transferencia)
- "¿Qué pasaría si...?" (razonamiento contrafactual)

**Ética:** Las sondas NO son evaluación. Son diagnóstico para ajustar scaffolding.
El estudiante no recibe "nota" por las sondas.

**Módulo:** `cognitive_gap_detector.py` (1593 líneas — el más complejo conceptualmente)

---

## ADR-020: Trust Dynamics — modelo de confianza

**Contexto:** La confianza del estudiante en el chatbot afecta cómo usa la herramienta.
Confianza excesiva → dependencia. Confianza insuficiente → abandono.

**Decisión:** Implementar modelo de dinámicas de confianza basado en Lee & See (2004).

**Factores:**
- **Performance:** ¿El chatbot responde correctamente?
- **Process:** ¿El estudiante entiende CÓMO el chatbot genera respuestas?
- **Purpose:** ¿El estudiante percibe que el chatbot quiere ayudarle?

**Intervenciones:**
- Si confianza excesiva: aumentar scaffolding socrático, mostrar incertidumbre
- Si confianza insuficiente: mostrar fuentes RAG, explicar razonamiento

**Módulo:** `trust_dynamics.py` (267 líneas)

---

## Decisiones pendientes (para los 18 meses)

| ID | Decisión | Depende de | Fase |
|----|----------|------------|------|
| ADR-P1 | ¿ChromaDB vs. pgvector vs. Pinecone para producción? | Volumen de datos del piloto | A |
| ADR-P2 | ¿Chunk size óptimo para materiales de programación? | Evaluación RAGAS | A |
| ADR-P3 | ¿Cómo integrar con el LTI existente sin romper el flujo? | Acceso al código actual | A |
| ADR-P4 | ¿Mixed methods secuencial o concurrente para análisis del piloto? | Diseño experimental con PIs | B |
| ADR-P5 | ¿Re-ranking con cross-encoder o basta con coseno? | Métricas de retrieval del piloto | C |
| ADR-P6 | ¿Federación de modelos entre nodos o modelo centralizado? | Requisitos de privacidad | C |
| ADR-P7 | ¿Fine-tuning del LLM base o solo prompt engineering? | Volumen de datos + presupuesto | C |

---

## Resumen de Innovaciones Diferenciales

| Innovación | Módulo | ¿Existe en literatura? |
|------------|--------|------------------------|
| Middleware pedagógico como capa ejecutable | `middleware.py` | Conceptual sí, implementación no |
| ACH para diagnóstico educativo | `ach_diagnostic.py` | No — transferencia de inteligencia |
| Detección de silencios epistémicos | `epistemic_silence_detector.py` | No — todos miden presencia, no ausencia |
| HHH alignment implementado, no declarado | `hhh_alignment_detector.py` | Declarado en papers, no implementado |
| Bucle O3→O1 instrumentado | `o1_feedback_engine.py` | Double-loop teórico, no computacional |
| Patrones ND como adaptación | `nd_patterns.py` | Etiquetado sí, adaptación no |
| Cross-node signals anónimos | `cross_node_signal.py` | No en LA educativo |

---

*Documento generado: Febrero 2026*
*Autor: Diego Elvira Vásquez*
*Proyecto: GENIE Learn CP25/152 — GSIC/EMIC-UVa*
