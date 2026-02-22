# FUNDAMENTACIÓN TEÓRICA
## Arquitectura epistémica del prototipo GENIE Learn

*Diego Elvira Vásquez · Febrero 2026*

---

## Declaración de posición

Este prototipo no es un chatbot con funcionalidades pedagógicas añadidas. Es un **instrumento de investigación** cuya arquitectura de software materializa decisiones teóricas explícitas. Cada módulo responde a un compromiso epistémico que puede ser discutido, refutado o sustituido — exactamente como debe funcionar un sistema en el marco DSRM (Peffers et al., 2007).

La estructura que sigue no es documentación técnica sino **genealogía intelectual**: de dónde viene cada decisión de diseño y por qué se tomó esa y no otra.

---

## 1. El middleware como materialización de la agencia docente

**Compromiso teórico:** La agencia del docente sobre la IA educativa no puede ser declarativa (el docente "opina" sobre el sistema) sino operativa (el docente *configura* el sistema y sus decisiones se ejecutan). Esta distinción separa el Value-Sensitive Design (Friedman et al., 2017) del mero user testing.

**Implementación:** El `middleware.py` no es una capa de filtrado — es un **motor de reglas pedagógicas** donde cada regla corresponde a una decisión docente. El system prompt del LLM se *construye dinámicamente* a partir de las configuraciones del profesor. Esto significa que dos profesores usando el mismo chatbot con los mismos materiales producen comportamientos pedagógicos completamente distintos.

**Referencia clave del proyecto:** Ortega-Arranz et al. (LAK 2026) identifican como limitación que sus "GenAI Analytics are low-level analytics (i.e., raw data)". Los módulos `cognitive_analyzer.py` y `researcher_view.py` responden directamente a esta limitación declarada elevando el nivel de análisis.

**Tensión no resuelta:** ¿Hasta qué punto la configuración docente debe poder contradecir lo que los datos de interacción sugieren? Si los datos muestran que el scaffolding socrático frustra a un estudiante, ¿el sistema lo desactiva automáticamente o espera a que el docente decida? Esta tensión entre *automation* y *agency* es exactamente lo que investiga el proyecto GENIE Learn (Delgado-Kloos et al., CSEDU 2025, Objetivo O3).

---

## 2. Taxonomía de Bloom como proxy de profundidad cognitiva

**Compromiso teórico:** Los prompts de los estudiantes no son solo preguntas — son *acciones epistémicas* (Clark & Chalmers, 1998). La pregunta que un estudiante formula al chatbot revela qué operación cognitiva está intentando realizar. Clasificar estas operaciones con la Taxonomía de Bloom Revisada (Anderson & Krathwohl, 2001) permite trazar trayectorias de aprendizaje observable.

**Por qué Bloom y no otra taxonomía:**
- **SOLO (Biggs & Collis, 1982)** es más precisa pero requiere analizar la *respuesta* del estudiante, no su *pregunta*. En un chatbot, el input del estudiante es una pregunta, no una producción evaluable.
- **Webb's Depth of Knowledge** distingue tipos de conocimiento pero no operaciones cognitivas — complementa a Bloom, no lo reemplaza.
- **ICAP (Chi & Wylie, 2014)** correlaciona tipo de actividad con aprendizaje pero opera a nivel de actividad global, no de interacción individual.

La solución implementada en `cognitive_analyzer.py` usa Bloom como clasificador primario y mapea a ICAP como indicador secundario.

**Limitación explícita:** La clasificación por keywords y patrones regex tiene un techo de precisión (~70-80% estimado). En un sistema de producción, esto requiere embeddings semánticos (e.g., sentence-transformers fine-tuned en español académico) o clasificación con LLM-as-judge. El prototipo usa heurísticas porque la prioridad es demostrar la arquitectura analítica, no la precisión del NLP.

---

## 3. ACH como metodología diagnóstica

**Compromiso teórico:** Cuando un estudiante "no entiende", el docente opera con una **hipótesis implícita** sobre la causa (generalmente la más disponible cognitivamente: "no ha estudiado"). El Analysis of Competing Hypotheses (Heuer, 1999) fuerza la consideración simultánea de múltiples causas y busca evidencia *discriminante* — evidencia que descarta hipótesis, no que confirma la favorita.

**Adaptación al contexto educativo:** Las 6 hipótesis diagnósticas del `ach_diagnostic.py` no son arbitrarias — mapean a la taxonomía de errores de VanLehn (2006):
- *Bugs* (errores en el modelo mental) → H_PREREQ, H_TRANSFER
- *Slips* (errores de ejecución) → H_SYNTAX
- *Misconceptions* → H_METACOG
- Factores no-cognitivos → H_AFFECTIVE, H_MOTIVATION

**Decisión de diseño contra-intuitiva:** El scoring ACH penaliza la evidencia inconsistente 1.5× más que premia la consistente. Esto es el principio Heuer aplicado: la hipótesis ganadora no es la que tiene más evidencia a favor sino la que tiene *menos en contra*. En un contexto educativo, esto contrarresta la tendencia del docente a confirmar su impresión inicial sobre un estudiante.

**Ruta a validación:** Los diagnósticos ACH generados durante los pilotos se pueden validar con entrevistas semi-estructuradas a estudiantes y triangular con datos de evaluación. Esto convierte el módulo ACH en un *generador de ground truth labels* para futuros clasificadores ML — resolviendo el problema de cold start de datos etiquetados que tiene cualquier piloto nuevo.

---

## 4. Confianza como variable dinámica

**Compromiso teórico:** La confianza del estudiante en la IA no es un rasgo estable sino un *proceso dinámico* que evoluciona con la interacción (Lee & See, 2004). El framework tripartito de Lee & See (performance trust, process trust, purpose trust) se adapta al contexto educativo:
- Performance trust: ¿el chatbot responde correctamente?
- Process trust: ¿entiendo cómo llega a esa respuesta?
- Purpose trust: ¿creo que el chatbot existe para ayudarme a aprender (no para vigilarme)?

**La paradoja de la confianza en IA educativa:** Un estudiante que confía *demasiado* en el chatbot no aprende (automation bias, Parasuraman & Riley 1997). Un estudiante que confía *demasiado poco* no lo usa (algorithm aversion). El sweet spot es **confianza calibrada**: suficiente para usarlo como herramienta, insuficiente para delegar el pensamiento.

**Operacionalización conductual:** El `trust_dynamics.py` no mide confianza con cuestionarios (eso es post-hoc y declarativo) sino con **señales conductuales en tiempo real**: latencia entre respuesta y siguiente prompt, presencia de marcadores de verificación ("¿estás seguro?"), aceptación acrítica ("ok gracias, siguiente"), frustración con el scaffolding. Cada señal tiene un vector de trust_direction que alimenta el score de calibración.

---

## 5. Patrones neurodivergentes como variable de diseño

**Compromiso teórico:** Un sistema educativo que ignora la neurodivergencia no es "neutral" — está diseñado implícitamente para el cerebro neurotípico y penaliza sistemáticamente a quien opera distinto. Esto no es una afirmación política sino un dato de diseño: si tu sistema asume distribución uniforme de interacción temporal y un estudiante con TDAH opera en ráfagas de hiperfoco, tu límite diario de prompts lo penaliza.

**Principio de diseño:** El `nd_patterns.py` detecta patrones para *adaptar el scaffolding*, no para diagnosticar. Las etiquetas son funcionales ("interacción episódica", "saltos cognitivos no-lineales", "rendimiento temáticamente asimétrico") no clínicas ("TDAH", "gifted", "2e").

**Patrones implementados y su fundamentación:**

| Patrón | Señal | Referencia | Adaptación |
|--------|-------|------------|------------|
| Episódico | CV temporal >1.2 | Barkley (2015) | Limites por ventana, no diarios |
| Topic switching | Cambio >55% | Brown (2017) | Anclas temáticas opcionales |
| Saltos cognitivos | ≥3 niveles Bloom | Silverman (2013) | Scaffolding reducido a 2 niveles |
| Frustración selectiva | Varianza inter-topic >2.0 | Reis et al. (2014) | Scaffolding diferenciado por tema |
| Twice-exceptional | Combinación de ≥3 anteriores | Reis et al. (2014) | Scaffolding completo diferenciado |
| Re-asking | Regresión tras Bloom ≥3 | Barkley (2015) | Recordatorios de contexto previo |

**Conexión con UBUN.IA:** Este módulo es la evolución conceptual del sistema ganador del HACK4EDU 2024 — donde el foco era equidad educativa para neurodivergencia. La diferencia: UBUN.IA era un diseño conceptual; este módulo es una implementación operativa con detección de patrones sobre datos reales.

---

## 6. Perfiles de engagement como analytics de alto nivel

**Compromiso teórico:** Los 5 perfiles del `EngagementProfiler` operacionalizan computacionalmente la distinción entre *deep approach* y *surface approach* de Biggs (1987), extendida con las categorías ICAP de Chi & Wylie (2014).

El perfil NO es una etiqueta fija — es un snapshot que puede evolucionar. La comparación longitudinal de perfiles (¿el estudiante pasó de surface a deep durante el semestre?) es una medida directa de la efectividad del scaffolding pedagógico.

**Respuesta directa a la limitación del LAK 2026:** El paper de Ortega-Arranz et al. reconoce explícitamente: "it would be interesting to explore higher-level analytics that combine multiple indicators to derive participants' profiles or engagement levels with AI." Los perfiles de engagement son exactamente eso.

---

## 7. Arquitectura como argumento

La decisión de separar el sistema en 8 módulos no es una elección técnica sino epistemológica. Cada módulo encapsula una **teoría sobre un aspecto del aprendizaje mediado por IA**:

| Módulo | Teoría encapsulada | Se puede reemplazar por |
|--------|-------------------|------------------------|
| `middleware.py` | Agencia docente (VSD) | Cualquier motor de reglas |
| `rag_pipeline.py` | Contextualización curricular (Lewis et al., 2020) | Fine-tuning, prompt caching |
| `cognitive_analyzer.py` | Operaciones cognitivas (Bloom/ICAP) | Clasificador ML, LLM-as-judge |
| `trust_dynamics.py` | Confianza calibrada (Lee & See, 2004) | Cuestionarios post-hoc (peor) |
| `ach_diagnostic.py` | Diagnóstico por hipótesis competitivas (Heuer) | Árboles de decisión, redes bayesianas |
| `nd_patterns.py` | Diseño neurodiverso (Reis/Barkley) | Clasificador de series temporales |
| `researcher_view.py` | Visualización para publicación | Notebook Jupyter, R Shiny |
| `llm_client.py` | Agnóstica — abstracción técnica | Cualquier LLM |

Esta modularidad no es accidental — es lo que permite que el sistema funcione como **instrumento de investigación** donde cada componente teórico se puede evaluar independientemente.

---

## Referencias

- Anderson, L. W. & Krathwohl, D. R. (2001). *A Taxonomy for Learning, Teaching, and Assessing*. Longman.
- Barkley, R. A. (2015). *Attention-Deficit Hyperactivity Disorder: A Handbook for Diagnosis and Treatment*. Guilford.
- Biggs, J. & Collis, K. (1982). *Evaluating the Quality of Learning: The SOLO Taxonomy*. Academic Press.
- Brown, T. E. (2017). *Outside the Box: Rethinking ADD/ADHD in Children and Adults*. American Psychiatric Association.
- Chi, M. T. H. & Wylie, R. (2014). The ICAP Framework. *Educational Psychologist*, 49(4), 219-243.
- Clark, A. & Chalmers, D. (1998). The Extended Mind. *Analysis*, 58(1), 7-19.
- Delgado-Kloos, C. et al. (2025). The GENIE Learn Project. *CSEDU 2025*. [Best Paper]
- Friedman, B., Hendry, D. G. & Borning, A. (2017). A Survey of Value Sensitive Design Methods. *Foundations and Trends in HCI*, 11(2), 63-125.
- Heuer, R. (1999). *Psychology of Intelligence Analysis*. CIA Center for the Study of Intelligence.
- Lee, J. D. & See, K. A. (2004). Trust in Automation. *Human Factors*, 46(1), 50-80.
- Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- Ortega-Arranz, A. et al. (2026). GenAI Analytics and Pedagogical Configurations. *LAK 2026*.
- Parasuraman, R. & Riley, V. (1997). Humans and Automation. *Human Factors*, 39(2), 230-253.
- Peffers, K. et al. (2007). A Design Science Research Methodology. *JMIS*, 24(3), 45-77.
- Reis, S. M. et al. (2014). Twice-Exceptional Students. *Gifted Child Quarterly*, 58(3), 169-183.
- Silverman, L. K. (2013). *Giftedness 101*. Springer.
- Topali, P. et al. (2024). Human-Centered Learning Analytics and Human-Centered AI in Education: A Systematic Literature Review. *BIT*.
- VanLehn, K. (2006). The Behavior of Tutoring Systems. *International Journal of AI in Education*, 16, 227-265.
