# GENIE Learn — Ecosystem Extension
## Seis módulos de inteligencia educativa distribuida

Diego Elvira Vásquez · CP25/152 GSIC/EMIC-UVa · Feb 2026

---

### Arquitectura del ecosistema

```
FLUJO DE DATOS:
  app.py ──► ecosystem_dashboard.py ──► [todos los módulos]
                                         │
                 ┌───────────────────────┼────────────────────────┐
                 │                       │                        │
    system_event_logger.py   config_genome.py     rag_quality_sensor.py
         (sensor universal)  (fingerprints)        (calidad RAG)
                 │                       │                        │
    teacher_calibration.py  cross_node_signal.py  temporal_config_advisor.py
    (fidelidad docente)     (UC3M↔UVa↔UPF)        (calendario académico)
```

### Los seis módulos

| Módulo | Agudeza | Paper habilitado |
|--------|---------|-----------------|
| `system_event_logger.py` | Flujo de eventos unificado con las 4 columnas diferenciales | Base de todos los papers |
| `config_genome.py` | Fingerprinting pedagógico + análisis atribucional | WP2: Design Analytics |
| `rag_quality_sensor.py` | Calidad RAG implícita por rephrase sequences | WP3: HCAI |
| `teacher_calibration.py` | **La inversión del paradigma**: el docente como sujeto de análisis | WP3: Teacher Agency |
| `cross_node_signal.py` | Inteligencia colectiva UC3M↔UVa↔UPF | WP4: Infrastructure |
| `temporal_config_advisor.py` | Configuración contextual al calendario académico | WP2: Actionable LA |
| `ecosystem_dashboard.py` | Hub de integración de los seis módulos | Todos |

### Las 4 columnas diferenciales (SystemEvent)

```python
config_snapshot         # qué config estaba activa en este momento exacto
student_bloom_estimate  # nivel cognitivo en este instante
session_pressure_index  # presión académica contextual [0-1]
node_cohort_state       # estado del cohorte del nodo
```

Con estas 4 columnas añadidas al log existente, el sistema puede responder:
- ¿Qué configuración produce qué cambio en Bloom? (O2/O3)
- ¿El modo socrático funciona igual bajo presión pre-examen? (WP3)
- ¿Las intervenciones docentes están calibradas al estado real del estudiante? (teacher_calibration)

### Inserción en el middleware existente (no invasiva)

```python
# En PedagogicalMiddleware.pre_process(), añadir al final:
event = create_student_prompt_event(
    student_id=student_id,
    prompt=raw_prompt,
    topics=result["detected_topics"],
    copy_paste_score=result["copy_paste_score"],
    config_snapshot=asdict(self.config),    # ← la columna diferencial clave
    bloom_estimate=estimated_bloom,
    pressure_index=temporal_advisor.compute_pressure_profile().pressure_index,
)
self.event_logger.log_event(event)
# Una línea por punto de inserción. No modifica la lógica existente.
```

### Demo de cada módulo

```bash
python system_event_logger.py   # flujo de eventos + silencio epistémico
python config_genome.py         # fingerprints Prof. A vs Prof. B
python rag_quality_sensor.py    # detección de rephrase sequences
python teacher_calibration.py   # IFS: intervención bien/mal calibrada
python cross_node_signal.py     # UC3M emite, UVa recibe con 11 días de anticipación
python temporal_config_advisor.py # sugerencias + rechazo con razón
python ecosystem_dashboard.py   # integración completa de los 6 módulos
```

### Principio arquitectónico

El chatbot es el punto de entrada. El ecosistema es el producto real.

Cada evento en el sistema es un sensor. Tratados como incidentes aislados (implementación actual): datos de analytics estándar. Tratados como flujo correlacionado con las 4 columnas diferenciales: el dataset más rico en comportamiento humano-IA educativa de España durante los próximos 3 años del proyecto.

Los papers de WP2, WP3 y WP4 que GENIE Learn escribirá en 2027 están en la tabla `system_events`. Solo falta el análisis.
