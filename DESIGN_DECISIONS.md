# Decisiones de Diseño — Registro de Arquitectura

> Este documento registra las decisiones técnicas y de diseño del prototipo,
> incluyendo alternativas consideradas y razones de descarte. No es documentación
> de usuario — es el log de pensamiento de ingeniería que un revisor técnico
> necesita para entender POR QUÉ el sistema es como es.

---

## ADR-001: Streamlit sobre Next.js para el prototipo

**Contexto:** El stack final del proyecto GENIE Learn usa React (interfaz Gradio en el TFG,
pero el paper LAK 2026 ya menciona interfaces web independientes). Mi stack personal
incluye Next.js/React/TypeScript. La tentación era construir en Next.js para demostrar
dominio del stack de producción.

**Decisión:** Streamlit.

**Razones:**
- El prototipo es un instrumento de investigación, no un producto. La velocidad de iteración
  importa más que la estética del frontend. Con Streamlit, cambiar una configuración pedagógica
  y ver el efecto es cuestión de 3 líneas, no de un componente React con estado.
- Los evaluadores (Bote-Lorenzo, Asensio-Pérez) son ingenieros de telecomunicaciones, no
  diseñadores de UI. Van a evaluar la lógica del middleware y la coherencia del pipeline RAG,
  no si uso Tailwind correctamente.
- Streamlit permite la vista multi-rol (estudiante/docente/investigador) con un radio button.
  En Next.js esto requiere routing, auth mock, y layout management.
- **Deuda técnica aceptada:** Streamlit no escala a producción con 200 estudiantes concurrentes.
  Eso está documentado como migración planificada en la Fase A del plan de 18 meses.

**Alternativas descartadas:**
- Next.js + FastAPI: correcto para producción, excesivo para prototipo de investigación.
- Gradio (como el TFG): demasiado limitado para las 4 vistas que necesito.
- Jupyter + Voilà: bueno para análisis, malo para la demo interactiva del chat.

---

## ADR-002: Scaffolding socrático con máquina de estados, no con prompt engineering puro

**Contexto:** La forma "fácil" de implementar scaffolding es meter todo en el system prompt
("si el estudiante pregunta por primera vez, sé socrático; si insiste, da pistas...") y dejar
que el LLM gestione la progresión. Esto es lo que hace el 90% de los wrappers educativos.

**Decisión:** Máquina de estados explícita en el middleware con 4 niveles discretos.

**Razones:**
- El LLM no tiene memoria fiable entre turnos sin RAG conversacional. Si le dices "sé más
  directo porque es la tercera vez que pregunta", necesitas que el LLM cuente — y los LLMs
  cuentan mal.
- Con estados explícitos, el INVESTIGADOR puede medir la progresión: "el 60% de los
  estudiantes llegaron al nivel 3 en el tema de bucles, pero solo el 20% en recursión."
  Eso es un dato publicable. "El LLM fue más o menos socrático" no lo es.
- El docente configura el scaffolding mode, pero el middleware decide el nivel. Esto separa
  la POLÍTICA (decisión pedagógica del docente) del MECANISMO (implementación técnica).
  Principio de diseño de Saltzer et al. (1984) — separación de política y mecanismo.
- **Limitación conocida:** La transición entre niveles es por conteo de intentos (cada 2
  preguntas sobre el mismo tema se escala un nivel). En producción, debería ser por análisis
  semántico de la respuesta del estudiante. Si el estudiante muestra comprensión en nivel 1,
  puede saltar a nivel 3. Pero eso requiere un evaluador de comprensión que no tengo tiempo
  de construir ahora.

**Lo que probé y no funcionó:**
- Intenté usar el Bloom level del prompt como señal de escalado (si el estudiante sube de
  nivel cognitivo, desbloquear más ayuda). El problema: un estudiante puede hacer una pregunta
  Bloom-4 por copy-paste sin entender nada. El nivel cognitivo del prompt no indica comprensión,
  indica formulación. Son cosas distintas.

---

## ADR-003: TF-IDF como fallback de retrieval, no solo como degradación elegante

**Contexto:** El pipeline RAG usa OpenAI embeddings + ChromaDB cuando hay API key, y cae
a TF-IDF (Jaccard similarity sobre tokens) cuando no la hay.

**Decisión:** El fallback no es solo "para que funcione sin API key" — es un **baseline de
evaluación** deliberado.

**Razones:**
- Cuando tenga datos reales del piloto, necesito saber si el retrieval semántico MEJORA
  las respuestas respecto al retrieval por keywords. Si no tengo baseline, no tengo medida.
- En el paper LAK 2026, la evaluación compara RAG vs. no-RAG. Yo añado un tercer nivel:
  RAG-semántico vs. RAG-keywords vs. no-RAG. Tres condiciones experimentales.
- El TF-IDF es sorprendentemente bueno para documentación técnica de programación donde
  los términos son específicos ("bucle for", "array", "recursión"). La ventaja de embeddings
  aparece con preguntas semánticamente complejas ("¿cómo repito algo muchas veces?" → debe
  recuperar "bucles" sin que la palabra aparezca).
- **Métrica pendiente:** Implementar RAGAS (faithfulness + answer relevancy + context recall)
  para comparar ambos retrieval modes. Está en el plan de la Fase A.

---

## ADR-004: Por qué ACH y no un clasificador ML para diagnóstico

**Contexto:** La forma estándar de clasificar "tipos de dificultad" de un estudiante sería
entrenar un clasificador supervisado sobre datos etiquetados.

**Decisión:** Análisis de Hipótesis Competitivas (Heuer, 1999) como motor de diagnóstico.

**Razones:**
- No tengo datos etiquetados. No hay un dataset de "este estudiante tiene un prerrequisito
  ausente" vs "este tiene ansiedad ante la programación." Sin ground truth, no hay supervisado.
- ACH opera con razonamiento estructurado sobre evidencia cualitativa. Genera hipótesis,
  busca evidencia discriminante en los logs, y asigna consistencia/inconsistencia.
- **El beneficio a largo plazo:** Cuando tenga datos del piloto 1 (semestre Sep 2026-Feb 2027),
  los diagnósticos ACH validados por los docentes se convierten en etiquetas de entrenamiento
  para un clasificador supervisado en la Fase C. ACH es el método de bootstrapping para
  generar ground truth.
- Esto viene directamente de mi formación: Máster en Análisis de Inteligencia (LISA-UDIMA),
  donde ACH era una de las herramientas centrales del programa. No es algo que se me haya
  ocurrido leyendo papers de education — es transferencia genuina entre dominios.

**Alternativa para el futuro:**
- Cuando haya ~300 interacciones etiquetadas, entrenar un Random Forest o XGBoost ligero
  usando las features que ACH ya extrae (bloom progression, topic switching, copypaste score,
  latency patterns). El ACH genera las features Y las labels.

---

## ADR-005: Detección de neurodivergencia como adaptación, no como etiquetado

**Contexto:** El módulo nd_patterns.py detecta patrones de interacción asociados a TDAH,
AACC y perfiles 2e. Esto es un territorio éticamente sensible.

**Decisión:** El sistema detecta PATRONES FUNCIONALES y sugiere ADAPTACIONES. Nunca genera
etiquetas diagnósticas. Los patrones se reportan al docente con lenguaje funcional.

**Razones:**
- Diagnosticar es acto clínico. Un chatbot no diagnostica. Pero un chatbot SÍ puede detectar
  que un estudiante muestra "interacción episódica con ráfagas de hiperfoco" y sugerir al
  docente que ese estudiante quizá se beneficiaría de sesiones más cortas con más frecuencia.
- La fundamentación ética viene de Value-Sensitive Design (Friedman et al., 2017): los
  estudiantes neurodivergentes son stakeholders indirectos cuyas necesidades raramente informan
  el diseño de sistemas educativos.
- **Conexión personal:** Soy perfil 2e (AACC + TDAH diagnosticado). UBUN.IA, el proyecto
  con el que ganamos HACK4EDU 2024, abordaba exactamente esto — equidad educativa para
  perfiles neurodivergentes. No es un módulo añadido porque queda bien; es el módulo que
  construiría aunque nadie me lo pidiera.

**Riesgo mitigado:** El docente puede desactivar esta funcionalidad. Si la activa, ve
descripciones funcionales ("patrón de interacción episódica"), no etiquetas ("posible TDAH").

---

## ADR-006: Alucinaciones pedagógicas controladas — implementación conservadora

**Contexto:** El paper LAK 2026 introduce la idea de "forced hallucinations" — inyectar
errores deliberados en las respuestas para fomentar lectura crítica. Es la configuración
más controvertida: solo el 50% de los profesores evaluadores la consideró útil.

**Decisión:** Implementación mínima con flag explícito visible al estudiante.

**Razones:**
- La idea es pedagógicamente interesante (conecta con "desirable difficulties" de Bjork, 1994)
  pero éticamente compleja. Si el estudiante no sabe que la respuesta puede ser incorrecta,
  el chatbot está MINTIENDO, no enseñando.
- Mi implementación añade una advertencia visible cuando se inyecta una alucinación. Esto
  reduce el efecto pedagógico (el estudiante sabe que debe desconfiar) pero elimina el problema
  ético. Es un compromiso conservador deliberado.
- **Para el piloto real:** Diseñar un experimento con grupo control (sin alucinaciones) vs.
  grupo experimental (con alucinaciones + advertencia) vs. grupo experimental 2 (con
  alucinaciones sin advertencia, si el comité de ética lo aprueba). El paper que salga de
  esto es potencialmente interesante independientemente del resultado.

**Lo que NO hice:**
- No implementé alucinaciones semánticas sofisticadas (errores lógicos en código, conceptos
  casi-correctos). Eso requiere un segundo LLM que genere errores plausibles, lo cual
  duplica el coste de API y la complejidad. Para la Fase C del plan.

---

## ADR-007: Sobre el mock de LLM y por qué las respuestas simuladas importan

**Contexto:** El prototipo funciona sin API key usando un MockLLMClient que genera respuestas
predeterminadas según el nivel de scaffolding detectado.

**Decisión:** Las respuestas mock son pedagógicamente diferenciadas, no lorem ipsum.

**Razones:**
- Si alguien ejecuta el prototipo sin API key (que es lo más probable en una demo rápida),
  necesita VER la diferencia entre scaffolding socrático y respuesta directa. Si el mock
  devuelve "Lorem ipsum" en ambos casos, la demo no demuestra nada.
- Las respuestas mock están escritas manualmente para reflejar exactamente qué haría un tutor
  en cada nivel: nivel 0 pregunta, nivel 1 orienta, nivel 2 ejemplifica, nivel 3 explica.
- Esto también sirve como ESPECIFICACIÓN: cuando conecte un LLM real, las respuestas mock
  son el gold standard contra el que evalúo si el system prompt produce el comportamiento
  deseado.

---

## Decisiones pendientes (para los 18 meses)

| ID | Decisión | Depende de | Fase |
|----|----------|------------|------|
| ADR-P1 | ¿ChromaDB vs. pgvector vs. Pinecone para producción? | Volumen de datos del piloto | A |
| ADR-P2 | ¿Chunk size óptimo para materiales de programación? | Evaluación RAGAS | A |
| ADR-P3 | ¿Cómo integrar con el LTI existente sin romper el flujo? | Acceso al código actual | A |
| ADR-P4 | ¿Mixed methods secuencial o concurrente para análisis del piloto? | Diseño experimental con PIs | B |
| ADR-P5 | ¿Re-ranking con cross-encoder o basta con coseno? | Métricas de retrieval del piloto | C |
