"""
TEMPORAL CONFIG ADVISOR — Configuración Consciente del Calendario Académico
═══════════════════════════════════════════════════════════════════════════════
El calendario académico tiene topología predecible: semanas de entrega,
períodos pre-examen, semanas de inicio y cierre. La configuración óptima
del chatbot pedagógico NO es constante — varía con el contexto temporal.

HIPÓTESIS PEDAGÓGICA:
Un hallucination_rate del 15% es valioso en la semana 6 (exploración activa)
y contraproducente en la semana anterior al examen parcial (el estudiante
necesita seguridad, no verificación). Un límite de 10 prompts diarios
es apropiado en semanas normales y opresivo en la semana de entrega de una
práctica que requiere consultas intensivas.

EL SISTEMA NO IMPONE — SUGIERE:
El docente aprueba o rechaza cada sugerencia. Cada rechazo con razón
es el mejor dato cualitativo del estudio. Esto es HCAI: el sistema
amplifica la agencia del docente, no la sustituye.

POSICIÓN EN EL ECOSISTEMA:
    calendar_config.json → (configurable por el docente)
    temporal_config_advisor.py → genera sugerencias contextuales
    cross_node_signal.py → señales inter-nodo informan las sugerencias
    system_event_logger.py → registra aprobaciones/rechazos
    app.py sidebar → muestra el panel "Sugerencias de Configuración"

Autor: Diego Elvira Vásquez · Ecosistema GENIE Learn · Feb 2026
Fundamentación: Alonso-Prieto et al. (2025) Teacher Agency;
               Pishtari et al. (2025) Reflection App LAK25.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Literal, Tuple, Any
from enum import Enum


# ──────────────────────────────────────────────────────────────
# TIPOS DE MOMENTO ACADÉMICO
# ──────────────────────────────────────────────────────────────

class AcademicMomentType(Enum):
    COURSE_START = "course_start"              # primeras 2 semanas
    EXPLORATION = "exploration"                # semanas normales de trabajo
    PRE_DELIVERY = "pre_delivery"              # 3-5 días antes de entrega práctica
    DELIVERY_DAY = "delivery_day"              # día de entrega
    POST_DELIVERY = "post_delivery"            # 2-3 días después de entrega
    PRE_MIDTERM = "pre_midterm"                # semana antes de parcial
    MIDTERM_WEEK = "midterm_week"              # semana de examen parcial
    PRE_FINAL = "pre_final"                    # semana antes de examen final
    FINAL_WEEK = "final_week"                  # semana de examen final
    HOLIDAY = "holiday"                        # festivos / vacaciones
    REVIEW_SESSION = "review_session"          # sesión de repaso planificada
    UNKNOWN = "unknown"


@dataclass
class AcademicEvent:
    """Un evento del calendario académico del curso."""
    event_id: str
    event_type: AcademicMomentType
    event_name: str
    start_date: str
    end_date: str
    topics_covered: List[str] = field(default_factory=list)
    pressure_weight: float = 0.5  # [0-1] peso de presión para este evento


@dataclass
class ConfigSuggestion:
    """
    Sugerencia de configuración para el docente.
    
    El docente ve esto en el sidebar de Streamlit y puede
    aprobar, rechazar, o modificar antes de aplicar.
    """
    suggestion_id: str
    generated_at: str
    valid_until: str

    # Contexto que motivó la sugerencia
    academic_moment: AcademicMomentType
    moment_description: str
    pressure_index: float

    # La sugerencia en sí
    params_to_change: Dict[str, Any]   # {param: new_value}
    params_to_keep: Dict[str, Any]     # {param: current_value} — no tocar

    # Justificación pedagógica (legible por el docente)
    pedagogical_rationale: str
    evidence_base: str                 # referencia a paper o dato del sistema

    # Si viene de una señal inter-nodo
    cross_node_signal_id: Optional[str] = None
    source_node: Optional[str] = None

    # Respuesta del docente (rellenable)
    teacher_decision: Optional[Literal["approved", "rejected", "modified"]] = None
    teacher_reason: Optional[str] = None    # clave para investigación
    decided_at: Optional[str] = None


@dataclass
class TemporalPressureProfile:
    """
    Perfil de presión académica de un punto en el tiempo.
    
    El índice de presión es la columna session_pressure_index
    que se añade a todos los SystemEvent.
    """
    date: str
    pressure_index: float               # [0.0 - 1.0]
    moment_type: AcademicMomentType
    active_events: List[str]            # nombres de eventos activos
    days_to_next_deadline: Optional[int]
    days_to_next_exam: Optional[int]


# ──────────────────────────────────────────────────────────────
# MOTOR DE SUGERENCIAS TEMPORALES
# ──────────────────────────────────────────────────────────────

class TemporalConfigAdvisor:
    """
    Genera sugerencias de configuración adaptadas al momento académico.
    
    No requiere ML. La topología del calendario académico es predecible
    y las reglas pedagógicas son suficientemente robustas para guiar
    sugerencias concretas.
    
    USO:
    - Se inicializa con el calendario del curso (fichero JSON simple)
    - Se consulta en cada inicio de sesión del docente
    - Genera sugerencias que el docente puede aprobar/rechazar
    - Los rechazos se registran como datos de investigación
    """

    # Plantillas de configuración por tipo de momento académico
    # Condensan las recomendaciones pedagógicas de la literatura
    CONFIG_TEMPLATES: Dict[AcademicMomentType, Dict] = {
        AcademicMomentType.COURSE_START: {
            "scaffolding_mode": "direct",        # primeras semanas: acceso fluido
            "max_daily_prompts": 25,             # límite alto para exploración
            "block_direct_solutions": False,     # no bloquear al inicio
            "forced_hallucination_pct": 0.0,     # sin alucinaciones al inicio
            "rationale": (
                "Primeras semanas del curso: el estudiante necesita aclimatarse al sistema. "
                "Alta accesibilidad + respuestas directas construyen la relación de confianza "
                "con el chatbot antes de introducir restricciones pedagógicas."
            ),
        },
        AcademicMomentType.EXPLORATION: {
            "scaffolding_mode": "socratic",
            "max_daily_prompts": 15,
            "block_direct_solutions": True,
            "forced_hallucination_pct": 0.10,    # alucinaciones moderadas para lectura crítica
            "rationale": (
                "Semana de trabajo normal: activar el modo socrático maximiza el valor "
                "pedagógico de cada interacción. Las alucinaciones pedagógicas (10%) "
                "fomentan la lectura crítica sin abrumar al estudiante."
            ),
        },
        AcademicMomentType.PRE_DELIVERY: {
            "scaffolding_mode": "hints",         # pistas, no soluciones
            "max_daily_prompts": 20,             # más prompts en período de entrega
            "block_direct_solutions": True,      # mantener bloqueo
            "forced_hallucination_pct": 0.0,     # QUITAR alucinaciones cerca de entrega
            "rationale": (
                "Pre-entrega: el estudiante necesita apoyo intensivo pero sin código directo. "
                "Se eliminan las alucinaciones para evitar confusión cuando el contexto "
                "es de alta presión. Se aumenta el límite de prompts."
            ),
        },
        AcademicMomentType.DELIVERY_DAY: {
            "scaffolding_mode": "hints",
            "max_daily_prompts": 30,             # máximo el día de entrega
            "block_direct_solutions": True,
            "forced_hallucination_pct": 0.0,
            "rationale": (
                "Día de entrega: máxima disponibilidad de soporte con límites claros. "
                "El límite alto de prompts reconoce la intensidad del trabajo, "
                "el bloqueo de soluciones directas mantiene la integridad académica."
            ),
        },
        AcademicMomentType.PRE_MIDTERM: {
            "scaffolding_mode": "direct",        # más apoyo directo pre-examen
            "max_daily_prompts": 20,
            "block_direct_solutions": False,     # permitir repaso de soluciones
            "forced_hallucination_pct": 0.0,     # NUNCA alucinaciones pre-examen
            "rationale": (
                "Pre-parcial: el estudiante necesita consolidar conocimiento, no explorar. "
                "Modo directo + sin alucinaciones + acceso a soluciones de referencia. "
                "Esta es la semana donde las alucinaciones pedagógicas son contraproducentes: "
                "la confianza epistémica es prioritaria sobre la lectura crítica."
            ),
        },
        AcademicMomentType.MIDTERM_WEEK: {
            "scaffolding_mode": "direct",
            "max_daily_prompts": 10,             # reducido: deben estudiar, no chatear
            "block_direct_solutions": False,
            "forced_hallucination_pct": 0.0,
            "rationale": (
                "Semana de parcial: disponibilidad reducida (el chatbot no debe sustituir "
                "al estudio) pero acceso directo para consultas de repaso. "
                "Sin alucinaciones en toda la semana."
            ),
        },
        AcademicMomentType.POST_DELIVERY: {
            "scaffolding_mode": "socratic",      # volver al modo reflexivo
            "max_daily_prompts": 12,
            "block_direct_solutions": True,
            "forced_hallucination_pct": 0.05,    # reintroducir suavemente
            "rationale": (
                "Post-entrega: momento de reflexión sobre el trabajo realizado. "
                "El modo socrático ayuda a consolidar el aprendizaje de la práctica "
                "mientras se reintroducen gradualmente las restricciones pedagógicas."
            ),
        },
    }

    def __init__(self, calendar: Optional[List[AcademicEvent]] = None):
        self.calendar: List[AcademicEvent] = calendar or []
        self.pending_suggestions: List[ConfigSuggestion] = []
        self.suggestion_history: List[ConfigSuggestion] = []

    @classmethod
    def from_json(cls, calendar_json_path: str) -> "TemporalConfigAdvisor":
        """Carga el calendario desde un fichero JSON configurado por el docente."""
        try:
            with open(calendar_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            events = []
            for ev in data.get("events", []):
                events.append(AcademicEvent(
                    event_id=ev["id"],
                    event_type=AcademicMomentType(ev["type"]),
                    event_name=ev["name"],
                    start_date=ev["start"],
                    end_date=ev["end"],
                    topics_covered=ev.get("topics", []),
                    pressure_weight=ev.get("pressure", 0.5),
                ))
            return cls(calendar=events)
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return cls(calendar=[])

    def compute_pressure_profile(self, target_date: Optional[date] = None) -> TemporalPressureProfile:
        """
        Calcula el índice de presión académica para una fecha dada.
        
        Este índice es la columna session_pressure_index que se añade
        a todos los SystemEvent, permitiendo correlacionar comportamiento
        estudiantil con contexto de presión académica.
        """
        if target_date is None:
            target_date = date.today()

        date_str = target_date.isoformat()
        pressure = 0.0
        moment_type = AcademicMomentType.UNKNOWN
        active_events = []
        days_to_deadline = None
        days_to_exam = None

        for event in self.calendar:
            start = date.fromisoformat(event.start_date)
            end = date.fromisoformat(event.end_date)

            if start <= target_date <= end:
                # Evento activo — contribuye a la presión
                active_events.append(event.event_name)
                pressure = max(pressure, self._moment_pressure(event.event_type))
                moment_type = event.event_type

            # Calcular distancias a deadlines y exámenes
            if event.event_type in (AcademicMomentType.DELIVERY_DAY, AcademicMomentType.PRE_DELIVERY):
                days_delta = (start - target_date).days
                if 0 <= days_delta <= 14:
                    if days_to_deadline is None or days_delta < days_to_deadline:
                        days_to_deadline = days_delta

            if event.event_type in (AcademicMomentType.MIDTERM_WEEK, AcademicMomentType.FINAL_WEEK,
                                     AcademicMomentType.PRE_MIDTERM, AcademicMomentType.PRE_FINAL):
                days_delta = (start - target_date).days
                if 0 <= days_delta <= 21:
                    if days_to_exam is None or days_delta < days_to_exam:
                        days_to_exam = days_delta

        # Añadir presión por proximidad (aunque el evento no esté activo aún)
        if days_to_deadline is not None and days_to_deadline <= 5:
            pressure = max(pressure, 0.7 + (5 - days_to_deadline) * 0.06)
        if days_to_exam is not None and days_to_exam <= 7:
            pressure = max(pressure, 0.6 + (7 - days_to_exam) * 0.04)

        if moment_type == AcademicMomentType.UNKNOWN:
            moment_type = AcademicMomentType.EXPLORATION
            pressure = max(pressure, 0.3)

        return TemporalPressureProfile(
            date=date_str,
            pressure_index=round(min(pressure, 1.0), 3),
            moment_type=moment_type,
            active_events=active_events,
            days_to_next_deadline=days_to_deadline,
            days_to_next_exam=days_to_exam,
        )

    def generate_suggestion(
        self,
        current_config: Dict,
        target_date: Optional[date] = None,
        cross_node_signals: Optional[List[Dict]] = None,
    ) -> Optional[ConfigSuggestion]:
        """
        Genera una sugerencia de configuración para el momento académico actual.
        
        Retorna None si la configuración actual ya es óptima para el momento.
        El docente ve esto en el sidebar de Streamlit.
        """
        pressure_profile = self.compute_pressure_profile(target_date)
        moment = pressure_profile.moment_type

        template = self.CONFIG_TEMPLATES.get(moment)
        if not template:
            return None

        # Calcular qué cambios son necesarios
        params_to_change = {}
        params_to_keep = {}

        template_config = {k: v for k, v in template.items() if k != "rationale"}

        for param, suggested_value in template_config.items():
            current_value = current_config.get(param)
            if current_value != suggested_value:
                params_to_change[param] = suggested_value
            else:
                params_to_keep[param] = current_value

        # Si no hay cambios necesarios, no generar sugerencia
        if not params_to_change:
            return None

        # Enriquecer con señales inter-nodo si las hay
        cross_signal_id = None
        source_node = None
        extra_rationale = ""
        if cross_node_signals:
            for sig in cross_node_signals:
                extra_rationale = (
                    f" Adicionalmente, {sig.get('source_node', 'otro nodo').upper()} "
                    f"detectó dificultades en '{sig.get('topic')}' con "
                    f"{sig.get('n_students_affected', 0)} estudiantes afectados."
                )
                cross_signal_id = sig.get("signal_id")
                source_node = sig.get("source_node")
                break  # usar solo la señal más relevante

        now = datetime.now()
        suggestion = ConfigSuggestion(
            suggestion_id=f"sug_{moment.value}_{now.date().isoformat()}",
            generated_at=now.isoformat(),
            valid_until=(now + timedelta(days=1)).isoformat(),
            academic_moment=moment,
            moment_description=pressure_profile.active_events[0] if pressure_profile.active_events
                              else moment.value.replace("_", " ").title(),
            pressure_index=pressure_profile.pressure_index,
            params_to_change=params_to_change,
            params_to_keep=params_to_keep,
            pedagogical_rationale=template["rationale"] + extra_rationale,
            evidence_base="GENIE Learn Temporal Config Advisor v1.0",
            cross_node_signal_id=cross_signal_id,
            source_node=source_node,
        )
        self.pending_suggestions.append(suggestion)
        return suggestion

    def record_teacher_decision(
        self,
        suggestion_id: str,
        decision: Literal["approved", "rejected", "modified"],
        reason: Optional[str] = None,
        applied_config: Optional[Dict] = None,
    ) -> bool:
        """
        Registra la decisión del docente sobre una sugerencia.
        
        El dato de investigación más valioso del módulo:
        ¿por qué el docente rechaza una sugerencia bien fundamentada?
        Esa razón es el acceso al modelo mental pedagógico del docente.
        """
        for sug in self.pending_suggestions:
            if sug.suggestion_id == suggestion_id:
                sug.teacher_decision = decision
                sug.teacher_reason = reason
                sug.decided_at = datetime.now().isoformat()
                self.suggestion_history.append(sug)
                self.pending_suggestions.remove(sug)
                return True
        return False

    def get_decision_analytics(self) -> Dict:
        """
        Analítica de decisiones docentes sobre sugerencias.
        
        El paper de WP2 sobre teacher agency puede construirse
        sobre esta analítica: qué sugerencias acepta/rechaza cada docente,
        con qué razonamientos, y en qué momentos del curso.
        """
        if not self.suggestion_history:
            return {"message": "Sin historial de decisiones aún."}

        total = len(self.suggestion_history)
        approved = sum(1 for s in self.suggestion_history if s.teacher_decision == "approved")
        rejected = sum(1 for s in self.suggestion_history if s.teacher_decision == "rejected")
        modified = sum(1 for s in self.suggestion_history if s.teacher_decision == "modified")

        # Patrones de rechazo: qué momento académico genera más rechazos
        rejection_by_moment: Dict[str, int] = {}
        rejection_reasons = []
        for s in self.suggestion_history:
            if s.teacher_decision == "rejected":
                key = s.academic_moment.value
                rejection_by_moment[key] = rejection_by_moment.get(key, 0) + 1
                if s.teacher_reason:
                    rejection_reasons.append(s.teacher_reason)

        return {
            "total_suggestions": total,
            "approval_rate": round(approved / total, 2),
            "rejection_rate": round(rejected / total, 2),
            "modification_rate": round(modified / total, 2),
            "rejection_by_moment": rejection_by_moment,
            "rejection_reasons": rejection_reasons,  # datos cualitativos de oro
        }

    def generate_calendar_json_template(self) -> str:
        """
        Genera un template JSON del calendario para que el docente lo configure.
        
        Se guarda como calendar_config.json y el docente lo edita
        con las fechas reales de su asignatura.
        """
        today = date.today()
        template = {
            "course_id": "MI_ASIGNATURA_2026",
            "node_id": "uva",
            "events": [
                {
                    "id": "start_2026",
                    "type": "course_start",
                    "name": "Inicio del semestre",
                    "start": today.isoformat(),
                    "end": (today + timedelta(weeks=2)).isoformat(),
                    "topics": ["variables", "tipos de dato"],
                    "pressure": 0.2,
                },
                {
                    "id": "pr1_2026",
                    "type": "pre_delivery",
                    "name": "Pre-entrega Práctica 1",
                    "start": (today + timedelta(weeks=5)).isoformat(),
                    "end": (today + timedelta(weeks=5, days=3)).isoformat(),
                    "topics": ["bucles", "funciones"],
                    "pressure": 0.7,
                },
                {
                    "id": "delivery1_2026",
                    "type": "delivery_day",
                    "name": "Entrega Práctica 1",
                    "start": (today + timedelta(weeks=5, days=4)).isoformat(),
                    "end": (today + timedelta(weeks=5, days=4)).isoformat(),
                    "topics": ["bucles", "funciones"],
                    "pressure": 0.9,
                },
                {
                    "id": "midterm_2026",
                    "type": "midterm_week",
                    "name": "Examen Parcial",
                    "start": (today + timedelta(weeks=9)).isoformat(),
                    "end": (today + timedelta(weeks=9, days=4)).isoformat(),
                    "topics": ["variables", "bucles", "funciones", "arrays"],
                    "pressure": 1.0,
                },
            ],
        }
        return json.dumps(template, ensure_ascii=False, indent=2)

    @staticmethod
    def _moment_pressure(moment: AcademicMomentType) -> float:
        """Presión base por tipo de momento académico."""
        pressure_map = {
            AcademicMomentType.COURSE_START: 0.15,
            AcademicMomentType.EXPLORATION: 0.30,
            AcademicMomentType.PRE_DELIVERY: 0.70,
            AcademicMomentType.DELIVERY_DAY: 0.90,
            AcademicMomentType.POST_DELIVERY: 0.35,
            AcademicMomentType.PRE_MIDTERM: 0.75,
            AcademicMomentType.MIDTERM_WEEK: 0.85,
            AcademicMomentType.PRE_FINAL: 0.80,
            AcademicMomentType.FINAL_WEEK: 0.95,
            AcademicMomentType.HOLIDAY: 0.05,
            AcademicMomentType.REVIEW_SESSION: 0.45,
            AcademicMomentType.UNKNOWN: 0.30,
        }
        return pressure_map.get(moment, 0.30)


# ──────────────────────────────────────────────
# DEMO AUTOEJECTABLE
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import date, timedelta

    advisor = TemporalConfigAdvisor()

    # Simular un calendario académico
    today = date.today()
    events = [
        AcademicEvent(
            event_id="ev1",
            event_type=AcademicMomentType.EXPLORATION,
            event_name="Semana normal – Tema: Recursión",
            start_date=(today - timedelta(days=1)).isoformat(),
            end_date=(today + timedelta(days=4)).isoformat(),
            topics_covered=["recursión"],
            pressure_weight=0.3,
        ),
        AcademicEvent(
            event_id="ev2",
            event_type=AcademicMomentType.PRE_DELIVERY,
            event_name="Pre-entrega Práctica 2",
            start_date=(today + timedelta(days=8)).isoformat(),
            end_date=(today + timedelta(days=10)).isoformat(),
            topics_covered=["recursión", "arrays"],
            pressure_weight=0.7,
        ),
        AcademicEvent(
            event_id="ev3",
            event_type=AcademicMomentType.PRE_MIDTERM,
            event_name="Semana previa al parcial",
            start_date=(today + timedelta(weeks=4)).isoformat(),
            end_date=(today + timedelta(weeks=4, days=4)).isoformat(),
            topics_covered=["todos"],
            pressure_weight=0.8,
        ),
    ]
    advisor.calendar = events

    print("═" * 65)
    print("TEMPORAL CONFIG ADVISOR — Demo de configuración contextual")
    print("═" * 65)

    # Config actual del docente (subóptima para el momento)
    current_config = {
        "scaffolding_mode": "direct",          # ← debería ser socrático en exploración
        "max_daily_prompts": 20,
        "block_direct_solutions": False,       # ← debería estar activado
        "forced_hallucination_pct": 0.0,       # ← podría aumentar en exploración
        "use_rag": True,
    }

    # Analizar el día de hoy
    pressure_today = advisor.compute_pressure_profile(today)
    print(f"\n📅 HOY ({today.isoformat()})")
    print(f"   Tipo de momento: {pressure_today.moment_type.value}")
    print(f"   Índice de presión: {pressure_today.pressure_index:.2f}")
    print(f"   Eventos activos: {pressure_today.active_events}")
    print(f"   Días a próxima entrega: {pressure_today.days_to_next_deadline}")

    # Generar sugerencia
    suggestion = advisor.generate_suggestion(current_config, today)
    if suggestion:
        print(f"\n💡 SUGERENCIA GENERADA")
        print(f"   Momento: {suggestion.moment_description}")
        print(f"   Presión: {suggestion.pressure_index:.0%}")
        print(f"\n   PARÁMETROS A CAMBIAR:")
        for param, value in suggestion.params_to_change.items():
            old = current_config.get(param, "?")
            print(f"   • {param}: {old} → {value}")
        print(f"\n   JUSTIFICACIÓN PEDAGÓGICA:")
        print(f"   {suggestion.pedagogical_rationale[:300]}...")
    else:
        print("\n✓ Configuración actual óptima para el momento académico.")

    # Simular rechazo con razón (el dato de investigación más valioso)
    if suggestion:
        advisor.record_teacher_decision(
            suggestion_id=suggestion.suggestion_id,
            decision="rejected",
            reason="Mis estudiantes son avanzados y el modo socrático les frustra en esta fase.",
        )
        print(f"\n   El docente RECHAZÓ la sugerencia.")
        print(f"   Razón registrada: '{suggestion.teacher_reason}'")
        print(f"   ← Esto es dato cualitativo de investigación sobre teacher agency.")

    # Analizar el período pre-parcial (futuro)
    pre_midterm_date = today + timedelta(weeks=4)
    pressure_exam = advisor.compute_pressure_profile(pre_midterm_date)
    suggestion_exam = advisor.generate_suggestion(current_config, pre_midterm_date)

    print(f"\n📅 PRE-PARCIAL ({pre_midterm_date.isoformat()})")
    print(f"   Índice de presión: {pressure_exam.pressure_index:.2f}")
    if suggestion_exam:
        print(f"   Cambios sugeridos: {suggestion_exam.params_to_change}")
        print(f"   NOTA: hallucination_pct → 0.0 siempre en períodos de examen.")

    # Template de calendario
    template = advisor.generate_calendar_json_template()
    print(f"\n📋 TEMPLATE DE CALENDARIO GENERADO (guardable como calendar_config.json)")
    print(f"   {len(json.loads(template)['events'])} eventos de plantilla.")

    print("\n═" * 65)
