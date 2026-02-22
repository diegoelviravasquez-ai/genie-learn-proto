"""
FUNDAMENTACIÓN DE DECISIONES DE DISEÑO
═══════════════════════════════════════════════════════════════════════
Tres módulos diferenciales para el prototipo GENIE Learn CP25/152

Documento destinado a: panel evaluador (Bote-Lorenzo, Asensio-Pérez)
Contexto: entrevista para contrato CP25/152, nodo UVa GENIE Learn
═══════════════════════════════════════════════════════════════════════

1. POR QUÉ ESTOS TRES MÓDULOS Y NO OTROS
─────────────────────────────────────────

Los tres módulos atacan las TRES LIMITACIONES EXPLÍCITAS del paper
LAK 2026 (Ortega-Arranz et al.), no inventos teóricos desconectados
de lo que el proyecto necesita:

┌──────────────────────────────────────────────────────────────────────┐
│ LIMITACIÓN LAK 2026              │ MÓDULO QUE LA ATACA              │
├──────────────────────────────────┼───────────────────────────────────┤
│ "Low-level analytics (raw data)" │ COGNITIVE ENGAGEMENT PROFILER    │
│ "Higher-level analytics that     │ → Perfiles de engagement con     │
│  derive engagement profiles"     │   taxonomía de Bloom + afecto    │
├──────────────────────────────────┼───────────────────────────────────┤
│ Topali SLR: "Gap in integration  │ EPISTEMIC AUTONOMY TRACKER       │
│  of learning theories in HCLA/   │ → Vygotsky (ZPD) + Bandura      │
│  HCAI design"                    │   (self-efficacy) + 4E cognition │
│  (solo 4/47 estudios integran    │   operacionalizados en código    │
│  teoría del aprendizaje)         │                                  │
├──────────────────────────────────┼───────────────────────────────────┤
│ Evaluadores piden "copy-paste    │ INTERACTION SEMIOTICS ENGINE     │
│  prevention" + "higher-level     │ → Speech Act Theory + Grice +    │
│  analytics"                      │   gaming detection (CI)          │
└──────────────────────────────────┴───────────────────────────────────┘


2. POR QUÉ CADA DECISIÓN TÉCNICA ES DIFERENCIAL
─────────────────────────────────────────────────

MÓDULO 1: COGNITIVE ENGAGEMENT PROFILER
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Decisión genérica (lo que haría cualquiera):
  Contabilizar keywords por tema y mostrar histogramas.

Decisión diferencial (lo que hago yo):
  Operacionalizar la taxonomía de Bloom como EVALUACIÓN COMPETITIVA
  (ACH de Heuer, 1999). El prompt se evalúa contra TODAS las
  categorías simultáneamente y se asigna a la que maximiza evidencia
  acumulada, no al primer regex que hace match.

  ¿Por qué? Porque un analista de inteligencia sabe que la primera
  hipótesis que encaja NO es necesariamente la correcta. El sesgo
  de anclaje (Kahneman) opera igual en clasificadores: si pones
  "recordar" primero en la lista, sobrerrepresentas ese nivel.
  La evaluación competitiva mitiga esto.

  Además: los marcadores lingüísticos NO son keywords sino PATRONES
  DISCURSIVOS. "Qué es X" (recordar) es sintácticamente distinto de
  "por qué X funciona diferente a Y" (analizar). La diferencia es
  pragmática (Austin, 1962), no léxica. Un sociólogo del conocimiento
  (Mannheim) sabe que las estructuras cognitivas se manifiestan en
  patrones de discurso, no en vocabulario aislado.

  Trayectorias temporales como pendiente de regresión: Bourdieu (1986)
  demostró que el capital cultural se acumula o erosiona en el tiempo.
  Un snapshot no captura esto. Una regresión lineal sobre la ventana
  temporal sí. Y es computacionalmente trivial.


MÓDULO 2: EPISTEMIC AUTONOMY TRACKER
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Decisión genérica (lo que haría cualquiera):
  Contar cuántas preguntas hace cada estudiante y asumir que más
  preguntas = más engagement.

Decisión diferencial (lo que hago yo):
  Operacionalizar la AGENCIA como constructo tridimensional
  (Emirbayer & Mische, 1998 — exactamente la referencia que usa
  Alonso-Prieto en su tesis dentro del GSIC):

  - Dimensión iteracional (habitus) → dependency_ratio
  - Dimensión projective (imaginación) → self_efficacy_proxy
  - Dimensión practical-evaluative → productive_struggle

  Esto NO es teoría por decorar. Es la fundamentación exacta que
  el grupo GSIC/EMIC usa en la tesis de Alonso-Prieto. Alinearme
  con su marco teórico demuestra que he leído sus papers y que
  puedo trabajar dentro de su ecosistema conceptual.

  Scaffolding fading: el middleware actual escala el scaffolding
  HACIA ARRIBA (más ayuda tras cada intento). Pero Wood, Bruner
  & Ross (1976) — la referencia fundacional del scaffolding —
  definen el scaffolding como una estructura que SE RETIRA cuando
  el aprendiz demuestra competencia. Mi módulo implementa el
  fading, no solo la escalación.

  Productive struggle (Kapur, 2008): corrección crucial. Si el
  estudiante está luchando productivamente (muestra intentos propios
  + razonamiento, sin frustración extrema), NO escalamos la ayuda.
  La dificultad deseable (Bjork, 1994) es pedagógicamente productiva.
  Un ingeniero intuye esto. Un educador lo sabe por la literatura.
  Yo lo implemento como condicional en el código.

  4E cognition (Newen, De Bruin & Gallagher, 2018) aplicada:
  - Extended: el chatbot como extensión cognitiva que debe
    desaparecer gradualmente (fading = salud epistémica)
  - Enacted: la autonomía se mide por ACCIÓN (intentos propios),
    no por declaración ("creo que entiendo")
  Esto conecta directamente con el Tractatus de Intelligentia
  Extensa: la inteligencia extendida implica que las herramientas
  cognitivas deben integrarse y luego internalizarse.


MÓDULO 3: INTERACTION SEMIOTICS ENGINE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Decisión genérica (lo que haría cualquiera):
  Binario copy-paste: sí/no, basado en longitud + ausencia de "?".

Decisión diferencial (lo que hago yo):
  Clasificación de ACTOS DE HABLA (Searle, 1969) que identifica
  la INTENCIÓN comunicativa del estudiante. "Dame el código"
  (directivo → valor pedagógico 10%) es cualitativamente distinto
  de "mi código falla en la línea 5, creo que es por el scope"
  (asertivo+directivo → valor pedagógico 75%).

  Máximas de Grice (1975) como framework de calidad interaccional.
  Un prompt que viola la máxima de cantidad (demasiado corto,
  sin contexto) será mal procesado por el LLM, lo que genera
  respuestas genéricas que frustran al estudiante. Detectar esto
  ANTES de enviar al LLM permite intervenir.

  Gaming detection inspirada en contrainteligencia cognitiva.
  Los estudiantes no solo copian-pegan: desarrollan ESTRATEGIAS:
  - Fragmentación: dividir el ejercicio en sub-preguntas
  - Paráfrasis evasiva: reformular para evadir detección
  - Explotación del scaffolding: repetir "no entiendo" para
    escalar a explicación directa
  - Sondeo de guardrails: probar formulaciones hasta encontrar
    una que pase

  Un ingeniero ve copy-paste. Un analista de inteligencia ve
  patrones de comportamiento estratégico adaptativos. La
  diferencia operativa es que mi detección es SECUENCIAL (analiza
  la historia de interacciones), no puntual.


3. CÓMO SE INTEGRA SIN ROMPER LO EXISTENTE
───────────────────────────────────────────

Patrón OBSERVER: los tres módulos observan cada interacción en
paralelo al flujo del middleware. No modifican pre_process() ni
post_process(). Se invocan DESPUÉS de log_interaction().

Esto es deliberado:
- El chatbot de GENIE Learn ya está en producción (TFG + evaluación
  con 16 profesores). No se toca.
- Los módulos ENRIQUECEN el dashboard, no lo sustituyen.
- Si un módulo falla, el chatbot sigue funcionando.
- El docente ve las nuevas métricas como CAPAS ADICIONALES sobre
  los analytics existentes.

Arquitectura:
  middleware.pre_process()
  → LLM
  → middleware.post_process()
  → middleware.log_interaction()
  → enhanced_analytics.analyze_interaction()  ← AQUÍ
  → dashboard extendido


4. QUÉ DEMUESTRA ESTO PARA EL PUESTO CP25/152
──────────────────────────────────────────────

  COMPETENCIA REQUERIDA          │ EVIDENCIA EN ESTOS MÓDULOS
  ───────────────────────────────┼──────────────────────────────────
  Desarrollo de prototipos GenAI │ Código Python funcional, testeable,
                                 │ integrable con el sistema existente.
  ───────────────────────────────┼──────────────────────────────────
  Análisis cuantitativo de datos │ Regresión lineal para trayectorias,
  educativos                     │ scoring compuesto con pesos justifi-
                                 │ cados, tipología derivada de datos.
  ───────────────────────────────┼──────────────────────────────────
  Fundamentación HCAI            │ Bloom, Vygotsky, Bandura, Emirbayer,
                                 │ Wood/Bruner/Ross, Bjork, 4E cognition
                                 │ — exactamente el marco teórico que el
                                 │ grupo GSIC/EMIC usa.
  ───────────────────────────────┼──────────────────────────────────
  Learning Analytics             │ Analytics de alto nivel (perfiles,
                                 │ no datos brutos), atacando la
                                 │ limitación explícita de LAK 2026.
  ───────────────────────────────┼──────────────────────────────────
  Valor diferencial del perfil   │ Speech Act Theory, Grice, análisis de
                                 │ inteligencia (ACH), Bourdieu, Mannheim,
                                 │ detección de gaming por CI — ningún
                                 │ candidato de ingeniería pura trae esto.


5. LO QUE HARÍA EN LOS 18 MESES DEL CONTRATO
──────────────────────────────────────────────

Meses 1-4:   Integrar estos módulos en el sistema real (Canvas LTI),
             con ChromaDB persistente y logging a PostgreSQL.

Meses 4-8:   Validar los perfiles de engagement con datos reales de
             pilotos. Iterar los pesos y umbrales con evidencia empírica.
             Evaluación con RAGAS (faithfulness, relevance) del RAG.

Meses 6-14:  Diseño experimental para pilotos en cursos reales UVa.
             Análisis estadístico (descriptivo + inferencial) de los
             datos de interacción. Correlación entre perfiles de
             engagement y resultados académicos.

Meses 8-18:  Paper sobre analytics de alto nivel en GenAI educativo
             (target: CSEDU 2026, EC-TEL 2026 o LAK 2027).
             Paper sobre resultados del piloto (target: journal —
             Computers & Education, BIT, IEEE TLT).


═══════════════════════════════════════════════════════════════════════
Diego Elvira Vásquez · Febrero 2026 · Prototipo para CP25/152
═══════════════════════════════════════════════════════════════════════
"""
