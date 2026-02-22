"""
SYSTEM EVENT LOGGER — El Sistema Nervioso del Ecosistema GENIE Learn
═══════════════════════════════════════════════════════════════════════
Módulo fundacional que convierte GENIE Learn de chatbot pedagógico
en ecosistema de inteligencia educativa distribuida.

PRINCIPIO ARQUITECTÓNICO:
Cada acción en el sistema — prompt de estudiante, cambio de configuración
docente, recuperación RAG, gap de sesión — es un sensor. Tratados como
eventos aislados (implementación actual): incidentes. Tratados como flujo
correlacionado: el dataset más rico en comportamiento humano-IA educativa
de España en los próximos 3 años.

POSICIÓN EN EL STACK:
    middleware.py (pre_process) → SystemEventLogger.log_event()
    middleware.py (post_process) → SystemEventLogger.log_event()
    config_genome.py → SystemEventLogger.log_event() [ConfigChange]
    rag_quality_sensor.py → SystemEventLogger.log_event() [RAGDegradation]

INSERCIÓN EN MIDDLEWARE EXISTENTE (no invasiva):
    En PedagogicalMiddleware.pre_process(), añadir al final:
        self.event_logger.log_event(SystemEvent(
            event_type="student_prompt",
            actor="student",
            student_id=student_id,
            config_snapshot=asdict(self.config),
            ...
        ))
    Una línea por punto de inserción. No modifica la lógica existente.

Los papers de WP2, WP3 y WP4 que GENIE Learn escribirá en 2027
están en la tabla que genera este módulo. Solo falta el análisis.

Autor: Diego Elvira Vásquez · Ecosistema GENIE Learn · Feb 2026
Fundamentación: Tabuenca et al. (2021) Sense-Analyze-React model;
               Dimitriadis (2021) Actionable Learning Analytics.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Literal, Optional, Dict, List, Any
from pathlib import Path
import sqlite3
import threading


# ──────────────────────────────────────────────────────────────
# TIPOS DE EVENTO — Taxonomía completa del ecosistema
# ──────────────────────────────────────────────────────────────

EventType = Literal[
    # Eventos del estudiante
    "student_prompt",           # prompt enviado al chatbot
    "student_rephrase",         # reformulación de prompt previo (señal RAG degradada)
    "student_session_start",    # inicio de sesión
    "student_session_end",      # fin de sesión
    "epistemic_silence",        # ausencia de pregunta donde se esperaba (Kapur 2008)

    # Eventos del docente
    "config_change",            # cambio en configuraciones pedagógicas
    "teacher_intervention",     # intervención directa (mensaje, ajuste manual)
    "teacher_dashboard_view",   # el docente abrió el dashboard

    # Eventos del sistema
    "rag_retrieval",            # recuperación de chunks RAG
    "rag_degradation",          # señal de calidad RAG baja por rephrase
    "llm_response",             # respuesta generada por el LLM
    "hallucination_injected",   # alucinación pedagógica activada
    "prompt_blocked",           # prompt bloqueado por límite o regla

    # Eventos inter-nodo
    "cross_node_signal",        # señal recibida de otro nodo universitario
]

NodeID = Literal["uva", "uc3m", "upf", "upm"]  # nodos del proyecto GENIE Learn
Actor = Literal["student", "teacher", "system", "external_node"]
PressureLevel = Literal["low", "medium", "high", "critical"]


# ──────────────────────────────────────────────────────────────
# DATACLASS PRINCIPAL — El sensor universal
# ──────────────────────────────────────────────────────────────

@dataclass
class SystemEvent:
    """
    Unidad atómica de información del ecosistema.
    
    Cada campo es un sensor. La correlación entre campos
    es la inteligencia que los papers del GSIC necesitan.
    
    COLUMNAS CRÍTICAS (las que no existen hoy en ningún sistema):
    - config_snapshot: qué configuración estaba activa en este momento exacto
    - student_bloom_estimate: nivel cognitivo del estudiante en este momento
    - session_pressure_index: presión académica contextual (pre-examen, entrega...)
    - node_id: qué nodo universitario generó este evento
    
    Con estas cuatro columnas adicionales sobre el log existente,
    se pueden responder las preguntas de O2 y O3 del contrato.
    """
    # ─── Identificadores ───
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # ─── Clasificación del evento ───
    event_type: EventType = "student_prompt"
    actor: Actor = "student"
    node_id: NodeID = "uva"

    # ─── Participantes ───
    student_id: Optional[str] = None
    teacher_id: Optional[str] = None
    course_id: Optional[str] = None

    # ─── LAS CUATRO COLUMNAS DIFERENCIALES ───────────────────
    # (no existen en ningún sistema LA publicado con este nivel de detalle)

    config_snapshot: Dict = field(default_factory=dict)
    # Snapshot completo de PedagogicalConfig en el momento del evento.
    # Permite: "¿qué configuración estaba activa cuando esto ocurrió?"
    # Paper habilitado: análisis atribucional config → comportamiento (WP2)

    student_bloom_estimate: int = 0
    # Nivel Bloom estimado del estudiante EN ESTE MOMENTO.
    # Permite: "¿la intervención docente era apropiada para el nivel cognitivo real?"
    # Paper habilitado: fidelidad de interpretación docente (WP3)

    session_pressure_index: float = 0.0
    # [0.0 - 1.0] Presión académica contextual calculada por temporal_config_advisor.
    # 0.0 = semana normal; 1.0 = día antes de examen final.
    # Permite: "¿el comportamiento del estudiante cambia bajo presión?"
    # Paper habilitado: efectos contextuales sobre autonomía epistémica (WP3)

    node_cohort_state: Dict = field(default_factory=dict)
    # Estado del cohorte del nodo en este momento: distribución Bloom media,
    # topics activos, ratio de gaming, autonomía media del curso.
    # Permite: análisis inter-nodo y propagación de señales (WP4)
    # ──────────────────────────────────────────────────────────

    # ─── Payload específico por tipo de evento ───
    payload: Dict = field(default_factory=dict)
    # Contenido variable según event_type:
    # "student_prompt": {prompt, topics, copy_paste_score, response_time_ms}
    # "config_change": {param_changed, value_before, value_after, analytics_trigger}
    # "rag_retrieval": {query, chunks_retrieved, relevance_scores, threshold}
    # "epistemic_silence": {topic, expected_questions, silence_duration_min}
    # "cross_node_signal": {source_node, signal_type, topic, pattern}

    # ─── Outcome (llenado retroactivamente) ───
    outcome: Dict = field(default_factory=dict)
    # Efecto medible del evento, registrado cuando se conoce:
    # Para "config_change": {bloom_delta_7d, autonomy_delta_7d, gaming_rate_delta}
    # Para "student_prompt": {next_bloom_level, response_quality_tier}
    # Permite: análisis causal config → efecto (el análisis atribucional completo)

    # ─── Metadatos ───
    session_id: Optional[str] = None
    sequence_in_session: int = 0


# ──────────────────────────────────────────────────────────────
# MOTOR DE LOGGING — Persistencia + Consulta
# ──────────────────────────────────────────────────────────────

class SystemEventLogger:
    """
    Motor central del ecosistema GENIE Learn.

    Responsabilidades:
    1. Persistir cada evento en SQLite (local, sin dependencias cloud)
    2. Exponer queries analíticas para los módulos del ecosistema
    3. Propagar señales inter-nodo cuando se detectan patrones
    4. Mantener el estado del cohorte actualizado en tiempo real

    El patrón de integración es OBSERVER:
    cada módulo (middleware, RAG, config_genome...) llama a log_event()
    sin conocer la implementación del logger.
    """

    def __init__(self, db_path: str = "genie_events.db", node_id: NodeID = "uva"):
        self.node_id = node_id
        self.db_path = db_path
        self._lock = threading.Lock()
        self._session_sequences: Dict[str, int] = {}
        self._cohort_cache: Dict = {}  # actualizado cada N eventos
        self._cache_dirty_count = 0
        self._cache_refresh_interval = 10  # recalcular cohorte cada 10 eventos

        self._init_db()

    def _init_db(self):
        """Crea el esquema SQLite si no existe."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_events (
                    event_id     TEXT PRIMARY KEY,
                    timestamp    TEXT NOT NULL,
                    event_type   TEXT NOT NULL,
                    actor        TEXT NOT NULL,
                    node_id      TEXT NOT NULL,
                    student_id   TEXT,
                    teacher_id   TEXT,
                    course_id    TEXT,
                    session_id   TEXT,
                    sequence_in_session INTEGER DEFAULT 0,

                    -- Las cuatro columnas diferenciales (JSON serializado)
                    config_snapshot       TEXT DEFAULT '{}',
                    student_bloom_estimate INTEGER DEFAULT 0,
                    session_pressure_index REAL DEFAULT 0.0,
                    node_cohort_state     TEXT DEFAULT '{}',

                    -- Payload y outcome
                    payload  TEXT DEFAULT '{}',
                    outcome  TEXT DEFAULT '{}'
                )
            """)
            # Índices para las consultas analíticas más frecuentes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_student ON system_events(student_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON system_events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON system_events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON system_events(session_id)")
            conn.commit()

    def log_event(self, event: SystemEvent) -> str:
        """
        Registra un evento en el ecosistema.

        Flujo:
        1. Asigna node_id y sequence_in_session
        2. Actualiza el cache del cohorte si necesario
        3. Enriquece el evento con node_cohort_state actual
        4. Persiste en SQLite
        5. Retorna event_id para referencias futuras

        Thread-safe: usa lock interno.
        """
        with self._lock:
            # Asignar node_id del logger
            event.node_id = self.node_id

            # Secuencia dentro de la sesión
            if event.session_id:
                seq = self._session_sequences.get(event.session_id, 0)
                event.sequence_in_session = seq
                self._session_sequences[event.session_id] = seq + 1

            # Refrescar cache del cohorte periódicamente
            self._cache_dirty_count += 1
            if self._cache_dirty_count >= self._cache_refresh_interval:
                self._refresh_cohort_cache()
                self._cache_dirty_count = 0

            # Enriquecer con estado del cohorte
            if self._cohort_cache:
                event.node_cohort_state = self._cohort_cache

            # Persistir
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO system_events VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?,
                        ?, ?
                    )
                """, (
                    event.event_id,
                    event.timestamp,
                    event.event_type,
                    event.actor,
                    event.node_id,
                    event.student_id,
                    event.teacher_id,
                    event.course_id,
                    event.session_id,
                    event.sequence_in_session,
                    json.dumps(event.config_snapshot),
                    event.student_bloom_estimate,
                    event.session_pressure_index,
                    json.dumps(event.node_cohort_state),
                    json.dumps(event.payload),
                    json.dumps(event.outcome),
                ))
                conn.commit()

            return event.event_id

    def update_outcome(self, event_id: str, outcome: Dict):
        """
        Actualiza el outcome de un evento pasado.

        Se usa cuando el efecto de una intervención se conoce
        después del momento del evento (e.g., cambio en Bloom
        a los 7 días de un cambio de configuración).

        Este método es lo que permite el análisis atribucional:
        config_change → outcome_7d.
        """
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE system_events SET outcome = ? WHERE event_id = ?",
                    (json.dumps(outcome), event_id)
                )
                conn.commit()

    # ──────────────────────────────────────────────
    # QUERIES ANALÍTICAS
    # ──────────────────────────────────────────────

    def get_student_timeline(self, student_id: str) -> List[Dict]:
        """
        Línea de tiempo completa de un estudiante.

        El corazón del análisis de trayectoria epistémica:
        cada evento en secuencia, con config activa y nivel Bloom.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM system_events
                WHERE student_id = ?
                ORDER BY timestamp ASC
            """, (student_id,)).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def get_config_change_events(self, teacher_id: Optional[str] = None) -> List[Dict]:
        """
        Todos los eventos de cambio de configuración.

        Base para el análisis del Genoma de Configuración:
        qué docentes cambian qué parámetros, cuándo, y con qué efecto.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if teacher_id:
                rows = conn.execute("""
                    SELECT * FROM system_events
                    WHERE event_type = 'config_change' AND teacher_id = ?
                    ORDER BY timestamp ASC
                """, (teacher_id,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM system_events
                    WHERE event_type = 'config_change'
                    ORDER BY timestamp ASC
                """).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def get_rephrase_sequences(self, window_minutes: int = 5) -> List[Dict]:
        """
        Detecta secuencias de rephrase dentro de ventanas temporales.

        Un estudiante que envía dos prompts similares en < N minutos
        es señal de que la primera respuesta RAG falló.

        Esta query es el sensor de calidad RAG en tiempo real.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM system_events
                WHERE event_type IN ('student_prompt', 'student_rephrase')
                ORDER BY student_id, timestamp ASC
            """).fetchall()

        events = [self._row_to_dict(row) for row in rows]

        # Agrupar por estudiante y detectar pares cercanos temporalmente
        rephrase_sequences = []
        from itertools import groupby

        for student_id, group in groupby(events, key=lambda e: e.get("student_id")):
            student_events = list(group)
            for i in range(len(student_events) - 1):
                t1 = datetime.fromisoformat(student_events[i]["timestamp"])
                t2 = datetime.fromisoformat(student_events[i + 1]["timestamp"])
                delta_minutes = (t2 - t1).total_seconds() / 60

                if delta_minutes <= window_minutes:
                    rephrase_sequences.append({
                        "student_id": student_id,
                        "prompt_1": student_events[i].get("payload", {}).get("prompt", ""),
                        "prompt_2": student_events[i + 1].get("payload", {}).get("prompt", ""),
                        "delta_minutes": round(delta_minutes, 2),
                        "topics": student_events[i].get("payload", {}).get("topics", []),
                        "config_active": student_events[i].get("config_snapshot", {}),
                        "timestamp": student_events[i]["timestamp"],
                    })

        return rephrase_sequences

    def get_epistemic_silences(self) -> List[Dict]:
        """
        Retorna todos los silencios epistémicos registrados.

        El detector de silencio epistémico es el sensor más original:
        mide lo que el estudiante NO hizo cuando debería haberlo hecho.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM system_events
                WHERE event_type = 'epistemic_silence'
                ORDER BY timestamp DESC
            """).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def get_cross_node_signals(self, target_node: Optional[NodeID] = None) -> List[Dict]:
        """
        Señales inter-nodo disponibles para este nodo.

        La capa de inteligencia colectiva entre UC3M, UVa y UPF.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM system_events
                WHERE event_type = 'cross_node_signal'
                ORDER BY timestamp DESC
                LIMIT 50
            """).fetchall()

        return [self._row_to_dict(row) for row in rows]

    def get_cohort_bloom_over_time(self, course_id: Optional[str] = None) -> List[Dict]:
        """
        Evolución del nivel Bloom medio del cohorte en el tiempo.

        La curva que convierte el Eje Z (temporal-evolutivo) en argumento visual:
        ¿el curso sube colectivamente en nivel cognitivo a lo largo del semestre?
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = """
                SELECT
                    date(timestamp) as day,
                    AVG(student_bloom_estimate) as bloom_mean,
                    COUNT(*) as event_count,
                    COUNT(DISTINCT student_id) as active_students
                FROM system_events
                WHERE event_type = 'student_prompt'
            """
            params = []
            if course_id:
                query += " AND course_id = ?"
                params.append(course_id)

            query += " GROUP BY date(timestamp) ORDER BY day ASC"
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def export_for_analysis(self, output_path: str = "genie_events_export.json"):
        """
        Exporta todos los eventos como JSON para análisis externo.

        BERTopic, K-Means, Spearman — el pipeline analítico completo
        puede correr sobre este export sin acceso a la BD de producción.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM system_events ORDER BY timestamp ASC").fetchall()

        events = [self._row_to_dict(row) for row in rows]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(events, f, ensure_ascii=False, indent=2)

        return len(events)

    def get_stats(self) -> Dict:
        """Estadísticas generales del ecosistema."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM system_events").fetchone()[0]
            students = conn.execute(
                "SELECT COUNT(DISTINCT student_id) FROM system_events WHERE student_id IS NOT NULL"
            ).fetchone()[0]
            event_types = conn.execute(
                "SELECT event_type, COUNT(*) as n FROM system_events GROUP BY event_type ORDER BY n DESC"
            ).fetchall()
            rephrase_count = conn.execute(
                "SELECT COUNT(*) FROM system_events WHERE event_type = 'student_rephrase'"
            ).fetchone()[0]
            silences = conn.execute(
                "SELECT COUNT(*) FROM system_events WHERE event_type = 'epistemic_silence'"
            ).fetchone()[0]

        return {
            "total_events": total,
            "unique_students": students,
            "rephrase_events": rephrase_count,
            "epistemic_silences": silences,
            "event_type_distribution": dict(event_types),
        }

    # ──────────────────────────────────────────────
    # HELPERS INTERNOS
    # ──────────────────────────────────────────────

    def _refresh_cohort_cache(self):
        """Recalcula el estado del cohorte para enriquecer futuros eventos."""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute("""
                SELECT
                    AVG(student_bloom_estimate) as bloom_mean,
                    AVG(session_pressure_index) as pressure_mean,
                    COUNT(DISTINCT student_id) as active_students,
                    COUNT(*) as total_prompts_today
                FROM system_events
                WHERE event_type = 'student_prompt'
                  AND date(timestamp) = date('now')
            """).fetchone()

        if result and result[0] is not None:
            self._cohort_cache = {
                "bloom_mean": round(result[0], 2),
                "pressure_mean": round(result[1] or 0.0, 2),
                "active_students": result[2],
                "prompts_today": result[3],
            }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict:
        """Convierte una fila SQLite a dict, deserializando JSON fields."""
        d = dict(row)
        for json_field in ["config_snapshot", "node_cohort_state", "payload", "outcome"]:
            if json_field in d and d[json_field]:
                try:
                    d[json_field] = json.loads(d[json_field])
                except (json.JSONDecodeError, TypeError):
                    d[json_field] = {}
        return d


# ──────────────────────────────────────────────
# FACTORY — Para uso en el middleware
# ──────────────────────────────────────────────

def create_student_prompt_event(
    student_id: str,
    prompt: str,
    topics: List[str],
    copy_paste_score: float,
    config_snapshot: Dict,
    bloom_estimate: int,
    pressure_index: float,
    session_id: Optional[str] = None,
    course_id: Optional[str] = None,
) -> SystemEvent:
    """Factory para eventos de prompt estudiantil."""
    return SystemEvent(
        event_type="student_prompt",
        actor="student",
        student_id=student_id,
        course_id=course_id,
        session_id=session_id,
        config_snapshot=config_snapshot,
        student_bloom_estimate=bloom_estimate,
        session_pressure_index=pressure_index,
        payload={
            "prompt": prompt,
            "topics": topics,
            "copy_paste_score": copy_paste_score,
        }
    )


def create_config_change_event(
    teacher_id: str,
    param_changed: str,
    value_before: Any,
    value_after: Any,
    analytics_trigger: Optional[str],
    config_snapshot_after: Dict,
    course_id: Optional[str] = None,
) -> SystemEvent:
    """
    Factory para eventos de cambio de configuración docente.

    analytics_trigger: qué dato del dashboard llevó al docente a cambiar esto.
    Si es None, el docente cambió sin consultar el dashboard
    (dato relevante para la investigación de teacher agency).
    """
    return SystemEvent(
        event_type="config_change",
        actor="teacher",
        teacher_id=teacher_id,
        course_id=course_id,
        config_snapshot=config_snapshot_after,
        payload={
            "param_changed": param_changed,
            "value_before": value_before,
            "value_after": value_after,
            "analytics_trigger": analytics_trigger,
        }
    )


def create_epistemic_silence_event(
    student_id: str,
    topic: str,
    expected_question_density: float,
    observed_question_density: float,
    silence_duration_minutes: float,
    config_snapshot: Dict,
    bloom_estimate: int,
    session_id: Optional[str] = None,
) -> SystemEvent:
    """
    Factory para silencios epistémicos — la señal más valiosa y menos visible.

    El perro que no ladró (Conan Doyle vía SIGINT):
    la ausencia de pregunta cuando estadísticamente debería haber una
    es señal de metacognición deficiente o de concepto ciego.
    """
    return SystemEvent(
        event_type="epistemic_silence",
        actor="system",
        student_id=student_id,
        session_id=session_id,
        config_snapshot=config_snapshot,
        student_bloom_estimate=bloom_estimate,
        payload={
            "topic": topic,
            "expected_question_density": round(expected_question_density, 3),
            "observed_question_density": round(observed_question_density, 3),
            "density_deficit": round(expected_question_density - observed_question_density, 3),
            "silence_duration_minutes": round(silence_duration_minutes, 1),
            "anomaly_score": round(
                (expected_question_density - observed_question_density) /
                max(expected_question_density, 0.001),
                3
            ),
        }
    )


# ──────────────────────────────────────────────
# DEMO AUTOEJECTABLE
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import time as time_module

    # Limpiar BD de demo
    if os.path.exists("demo_events.db"):
        os.remove("demo_events.db")

    logger = SystemEventLogger(db_path="demo_events.db", node_id="uva")

    print("═" * 60)
    print("SYSTEM EVENT LOGGER — Demo del ecosistema GENIE Learn")
    print("═" * 60)

    # Simular una sesión de curso (semana 8, pre-entrega de práctica)
    session_id = "session_demo_001"
    config = {
        "scaffolding_mode": "socratic",
        "max_daily_prompts": 10,
        "block_direct_solutions": True,
        "forced_hallucination_pct": 0.0,
        "use_rag": True,
    }
    pressure = 0.65  # presión alta: semana de entrega

    # Estudiante A: patrón sano
    for i, (prompt, topics, bloom, cp) in enumerate([
        ("¿cómo funciona la recursión?", ["recursión"], 2, 0.05),
        ("¿por qué la recursión necesita un caso base?", ["recursión"], 3, 0.08),
        ("creo que entiendo: sin caso base hay stack overflow ¿correcto?", ["recursión"], 4, 0.02),
    ]):
        evt = create_student_prompt_event(
            student_id="est_A",
            prompt=prompt,
            topics=topics,
            copy_paste_score=cp,
            config_snapshot=config,
            bloom_estimate=bloom,
            pressure_index=pressure,
            session_id=session_id,
        )
        logger.log_event(evt)
        time_module.sleep(0.01)

    # Estudiante B: rephrase sequence (señal RAG degradada)
    for prompt, topics, bloom in [
        ("¿qué es la recursión?", ["recursión"], 1),
        ("explícame la recursión paso a paso", ["recursión"], 1),  # rephrase 2min después
    ]:
        evt = create_student_prompt_event(
            student_id="est_B",
            prompt=prompt,
            topics=topics,
            copy_paste_score=0.1,
            config_snapshot=config,
            bloom_estimate=bloom,
            pressure_index=pressure,
            session_id=session_id,
        )
        logger.log_event(evt)
        time_module.sleep(0.01)

    # Cambio de configuración del docente
    config_after = {**config, "scaffolding_mode": "direct"}
    evt_config = create_config_change_event(
        teacher_id="prof_01",
        param_changed="scaffolding_mode",
        value_before="socratic",
        value_after="direct",
        analytics_trigger="bloom_mean_below_2",
        config_snapshot_after=config_after,
    )
    evt_id_config = logger.log_event(evt_config)
    time_module.sleep(0.01)

    # Silencio epistémico: est_C pasó por recursión sin preguntar nada
    evt_silence = create_epistemic_silence_event(
        student_id="est_C",
        topic="recursión",
        expected_question_density=0.72,  # el 72% de estudiantes similares pregunta sobre recursión
        observed_question_density=0.0,   # est_C: ninguna pregunta
        silence_duration_minutes=45.0,
        config_snapshot=config,
        bloom_estimate=1,
        session_id=session_id,
    )
    logger.log_event(evt_silence)
    time_module.sleep(0.01)

    # Simular outcome del cambio de configuración (7 días después)
    logger.update_outcome(evt_id_config, {
        "bloom_delta_7d": +0.8,
        "gaming_rate_delta": -0.12,
        "autonomy_delta_7d": -0.15,  # ← dato negativo: el modo directo redujo autonomía
        "n_students_affected": 12,
    })

    # ─── Mostrar resultados ───
    stats = logger.get_stats()
    print(f"\n📊 ESTADÍSTICAS DEL ECOSISTEMA")
    print(f"   Total eventos: {stats['total_events']}")
    print(f"   Estudiantes únicos: {stats['unique_students']}")
    print(f"   Silencios epistémicos: {stats['epistemic_silences']}")
    print(f"   Distribución: {stats['event_type_distribution']}")

    timeline_A = logger.get_student_timeline("est_A")
    print(f"\n📈 TRAYECTORIA EPISTÉMICA — Est. A ({len(timeline_A)} eventos)")
    for evt in timeline_A:
        print(f"   [{evt['timestamp'][11:19]}] "
              f"Bloom={evt['student_bloom_estimate']} | "
              f"Topic={evt.get('payload', {}).get('topics', [])} | "
              f"Presión={evt['session_pressure_index']}")

    silences = logger.get_epistemic_silences()
    print(f"\n🔇 SILENCIOS EPISTÉMICOS DETECTADOS: {len(silences)}")
    for s in silences:
        p = s.get("payload", {})
        print(f"   Estudiante: {s['student_id']} | Topic: {p.get('topic')} | "
              f"Anomalía: {p.get('anomaly_score', 0):.0%} | "
              f"Duración: {p.get('silence_duration_minutes')} min")

    config_events = logger.get_config_change_events()
    print(f"\n⚙️  CAMBIOS DE CONFIGURACIÓN: {len(config_events)}")
    for c in config_events:
        p = c.get("payload", {})
        o = c.get("outcome", {})
        print(f"   [{c['timestamp'][11:19]}] {p.get('param_changed')}: "
              f"{p.get('value_before')} → {p.get('value_after')}")
        if o:
            print(f"   └─ Outcome 7d: Bloom Δ={o.get('bloom_delta_7d',0):+.1f}, "
                  f"Autonomía Δ={o.get('autonomy_delta_7d',0):+.2f}")

    print(f"\n✅ BD persistida en: demo_events.db")
    print(f"   Export: logger.export_for_analysis('export.json')")
    print("═" * 60)
