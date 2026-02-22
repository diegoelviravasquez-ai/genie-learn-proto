# DÍA 1 — Metacognición para el Estudiante + Modo Sandbox

## Qué cambia

Estas 4 ficheros (1 nuevo + 3 modificados) transforman GENIE Learn de
"herramienta que mide al estudiante" a "herramienta que ayuda al estudiante
a entender su propio aprendizaje".

## Archivos

### NUEVO: `metacognitive_nudges.py` (727 líneas)
Motor de nudges metacognitivos. No muestra métricas — genera intervenciones
comunicativas que provocan reflexión.

6 tipos de nudge:
- **Progresión** 📈 — cuando sube de nivel Bloom
- **Esfuerzo productivo** 💪 — cuando demuestra productive struggle (Kapur 2008)
- **Repaso espaciado** 🔄 — cuando vuelve a un topic tras 24-168h (Bjork 1994)
- **Desacople** 🦋 — cuando el scaffolding fading indica independencia
- **Reflexión** 🪞 — periódico, invita a mirar hacia atrás
- **Sandbox** 🏖️ — bienvenida al modo práctica libre

3 tonos configurables por el docente: cálido / neutro / académico.
Frecuencia configurable (mínimo N interacciones entre nudges).
Incluye `generate_demo_nudge_sequence()` para demo con 15 interacciones.

### MODIFICADO: `middleware.py` (+30 líneas)
- Campo `sandbox_mode: bool` en `PedagogicalConfig`
- `log_interaction()` devuelve `None` en sandbox (no registra contenido)
- Contador `_sandbox_interaction_count` para QoS
- `sandbox_interactions` en `get_analytics_summary()`

### MODIFICADO: `app.py` (+120 líneas netas)
Vista estudiante:
- Banner sandbox visible cuando modo activo
- Nudges se muestran como `<div class="nudge-box">` bajo cada respuesta
- Nudges persisten en `chat_history` (se replayan al scroll)
- Métrica "Modo" muestra "🏖️ Sandbox" cuando activo
- Metadata oculta en sandbox (no muestra Bloom/trust al estudiante)

Vista docente:
- Card **🏖️ Modo Sandbox** con toggle + explicación
- Card **🪞 Nudges Metacognitivos** con:
  - Toggle activar/desactivar
  - Slider frecuencia (cada N interacciones)
  - Selector de tono (cálido/neutro/académico)
  - Multiselect de tipos de nudge activos
  - Toggle mostrar referencia teórica al estudiante
- Config summary JSON actualizado con sandbox + nudge state
- Botón "Aplicar" recrea `nudge_generator` con nueva config

Sidebar:
- Badge "SANDBOX" cuando modo activo
- Contador de nudges entregados

### MODIFICADO: `researcher_view.py` (+81 líneas)
Sección 5: "🪞 Nudges Metacognitivos — Analytics PARA el estudiante"
- Métricas: interacciones simuladas, nudges generados, ratio
- Timeline visual de la secuencia demo (15 interacciones)
- Distribución por tipo de nudge
- Insight box sobre tipología de perfiles metacognitivos

## Instalación

Copiar los 4 archivos al directorio raíz del proyecto, reemplazando
los existentes:

```bash
cp metacognitive_nudges.py  /path/to/genie-learn-proto/
cp middleware.py             /path/to/genie-learn-proto/
cp app.py                   /path/to/genie-learn-proto/
cp researcher_view.py       /path/to/genie-learn-proto/
```

No se requieren dependencias nuevas. Todo usa stdlib + lo que ya estaba
en `requirements.txt`.

## Test rápido

```bash
python metacognitive_nudges.py
```

Muestra la secuencia demo de 15 interacciones con 5 nudges generados.

## Frases para la entrevista

**Metacognición:**
> "Los learning analytics existentes informan al docente sobre el estudiante.
> Nuestros nudges metacognitivos informan al estudiante sobre sí mismo —
> convierten los analytics de vigilancia pasiva en herramienta de
> autorregulación del aprendizaje."

**Sandbox:**
> "Implementamos el derecho del estudiante a equivocarse sin ser medido.
> Un sistema que mide todo el tiempo produce estudiantes que performan
> para el sistema en lugar de aprender genuinamente. Es la traducción
> técnica del principio de harmlessness del framework HHH."

**Conexión Villa-Torrano:**
> "Villa-Torrano estudia regulación socialmente compartida del aprendizaje.
> Los nudges metacognitivos son la versión individual de esa misma
> capacidad — y la arquitectura permite escalar a nudges grupales
> cuando el piloto WP5 incluya trabajo colaborativo."
