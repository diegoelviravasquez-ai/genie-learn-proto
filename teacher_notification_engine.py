"""
TEACHER NOTIFICATION ENGINE — Cerrar el Bucle Analytics→Docente
═══════════════════════════════════════════════════════════════════════
Brecha central de O2: "los docentes configuran una vez y las dejan estáticas."

Este módulo cierra el bucle: cuando los analytics detectan un patrón
que requiere intervención, genera una NOTIFICACIÓN ACCIONABLE al docente.
El docente acepta, rechaza, o modifica. Cada decisión se registra como
evento — el dato más valioso para la investigación sobre teacher agency.

TIPOS DE NOTIFICACIÓN:
    config_suggestion    → Sugiere cambiar una configuración pedagógica
    cohort_alert         → Alerta sobre patrón preocupante del cohorte
    student_flag         → Señala un estudiante individual que necesita atención
    ld_recommendation    → Recomienda una secuencia de Learning Design
    rag_quality_alert    → Alerta de degradación del corpus RAG
    milestone            → Celebra un logro del cohorte

PRINCIPIO:
    El sistema SUGIERE, el docente DECIDE. Cada rechazo con razón es
    el mejor dato cualitativo del estudio. Esto es HCAI: amplificación
    de agencia, no sustitución (Shneiderman, 2022).

Autor: Diego Elvira Vásquez · CP25/152 GSIC/EMIC · Feb 2026
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum


class NotificationPriority(str, Enum):
    CRITICAL = "critical"    # Requiere acción inmediata
    HIGH = "high"            # Importante, acción recomendada hoy
    MEDIUM = "medium"        # Informativa, acción cuando convenga
    LOW = "low"              # Contexto, no requiere acción


class NotificationType(str, Enum):
    CONFIG_SUGGESTION = "config_suggestion"
    COHORT_ALERT = "cohort_alert"
    STUDENT_FLAG = "student_flag"
    LD_RECOMMENDATION = "ld_recommendation"
    RAG_QUALITY_ALERT = "rag_quality_alert"
    MILESTONE = "milestone"


class DecisionType(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    MODIFIED = "modified"
    DEFERRED = "deferred"


@dataclass
class TeacherNotification:
    """Una notificación accionable para el docente."""
    notification_id: str = ""
    timestamp: str = ""
    teacher_id: str = ""
    course_id: str = "default"

    # Contenido
    notification_type: str = ""
    priority: str = "medium"
    title: str = ""
    message: str = ""
    data_justification: str = ""     # Por qué el sistema sugiere esto
    theoretical_basis: str = ""       # Marco teórico que sustenta
    suggested_action: Dict = field(default_factory=dict)

    # Contexto del cohorte al momento de generar la notificación
    context_snapshot: Dict = field(default_factory=dict)

    # Decisión docente
    decision: str = "pending"
    decision_timestamp: str = ""
    decision_rationale: str = ""     # ORO para Teacher Agency
    modified_action: Dict = field(default_factory=dict)

    # Meta
    was_read: bool = False
    read_timestamp: str = ""


class TeacherNotificationEngine:
    """
    Motor de notificaciones que cierra el bucle analytics→docente.
    
    Consume: analytics de todos los módulos del ecosistema
    Produce: notificaciones accionables priorizadas
    Registra: decisiones docentes como eventos de investigación
    """

    def __init__(self):
        self.notifications: List[TeacherNotification] = []
        self.rules: List[dict] = self._init_rules()

    def _init_rules(self) -> List[dict]:
        """Reglas de generación de notificaciones basadas en umbrales."""
        return [
            # Bloom bajo sostenido
            {
                "id": "bloom_low_sustained",
                "type": NotificationType.CONFIG_SUGGESTION,
                "priority": NotificationPriority.HIGH,
                "condition": lambda ctx: ctx.get("avg_bloom", 3) < 2.5 and ctx.get("interactions", 0) > 15,
                "title": "Nivel cognitivo bajo sostenido",
                "message_template": (
                    "El {pct_low}% del cohorte opera en Bloom 1-2 tras {n} interacciones. "
                    "El scaffolding actual no está elevando la distribución cognitiva."
                ),
                "suggested_action": {"scaffolding_mode": "hints", "rationale": "Hints concretos pueden ser más efectivos que socrático puro para Bloom < 2.5"},
                "theoretical_basis": "Vygotsky (1978): ZPD + Kalyuga (2003): Expertise Reversal — scaffolding inadecuado al nivel actual",
            },
            # Gaming alto
            {
                "id": "gaming_alert",
                "type": NotificationType.COHORT_ALERT,
                "priority": NotificationPriority.CRITICAL,
                "condition": lambda ctx: ctx.get("gaming_rate", 0) > 0.25,
                "title": "Tasa de gaming elevada",
                "message_template": (
                    "{gaming_pct}% de interacciones muestran señales de gaming. "
                    "Los estudiantes podrían estar usando el chatbot para obtener soluciones sin aprender."
                ),
                "suggested_action": {"block_direct_solutions": True, "max_daily_prompts": 8},
                "theoretical_basis": "Baker et al. (2008): gaming the system correlaciona con peor aprendizaje",
            },
            # RAG degradado
            {
                "id": "rag_degraded",
                "type": NotificationType.RAG_QUALITY_ALERT,
                "priority": NotificationPriority.HIGH,
                "condition": lambda ctx: ctx.get("rag_quality", 1.0) < 0.5,
                "title": "Calidad RAG degradada",
                "message_template": (
                    "La calidad de las respuestas basadas en el material del curso ha bajado a {rag_q}%. "
                    "Considere actualizar el corpus RAG o revisar los documentos cargados."
                ),
                "suggested_action": {"action": "review_rag_corpus"},
                "theoretical_basis": "Es et al. (2023): RAGAS — faithfulness < 0.5 indica respuestas no ancladas al material",
            },
            # Dependencia creciente
            {
                "id": "dependency_increasing",
                "type": NotificationType.CONFIG_SUGGESTION,
                "priority": NotificationPriority.MEDIUM,
                "condition": lambda ctx: ctx.get("dependency_ratio", 0) > 0.7 and ctx.get("week", 1) > 3,
                "title": "Dependencia cognitiva creciente",
                "message_template": (
                    "{dep_pct}% de estudiantes en fase 'dependiente' en la semana {week}. "
                    "El scaffolding no está produciendo fading gradual."
                ),
                "suggested_action": {"max_daily_prompts": 10, "scaffolding_mode": "hints"},
                "theoretical_basis": "Wood, Bruner & Ross (1976): scaffolding debe reducirse gradualmente. Bandura (1997): dependencia erosiona self-efficacy.",
            },
            # Hito positivo
            {
                "id": "bloom_improvement",
                "type": NotificationType.MILESTONE,
                "priority": NotificationPriority.LOW,
                "condition": lambda ctx: ctx.get("bloom_trend", 0) > 0.3 and ctx.get("interactions", 0) > 20,
                "title": "¡Mejora cognitiva del cohorte!",
                "message_template": (
                    "La trayectoria Bloom del cohorte muestra tendencia positiva (+{trend}). "
                    "Las configuraciones actuales parecen estar funcionando."
                ),
                "suggested_action": {"action": "maintain_current_config"},
                "theoretical_basis": "Anderson & Krathwohl (2001): progresión Bloom indica aprendizaje profundo",
            },
            # Copy-paste masivo
            {
                "id": "copypaste_spike",
                "type": NotificationType.COHORT_ALERT,
                "priority": NotificationPriority.HIGH,
                "condition": lambda ctx: ctx.get("high_copypaste_pct", 0) > 0.4,
                "title": "Oleada de copy-paste detectada",
                "message_template": (
                    "{cp_pct}% de interacciones recientes tienen alta detección de copy-paste. "
                    "Posible entrega de práctica en curso — los estudiantes copian enunciados."
                ),
                "suggested_action": {"block_direct_solutions": True, "system_addon": "Si detectas un enunciado de ejercicio, pide al estudiante que formule su duda específica."},
                "theoretical_basis": "Evaluadores LAK 2026: 'prevent students from copy-pasting exercises'",
            },
        ]

    def evaluate_context(self, context: dict, teacher_id: str = "default") -> List[TeacherNotification]:
        """
        Evalúa el contexto actual contra las reglas y genera notificaciones.
        
        Args:
            context: dict con métricas actuales del cohorte
            teacher_id: ID del docente destinatario
        """
        new_notifications = []

        for rule in self.rules:
            try:
                if rule["condition"](context):
                    # Evitar duplicados: no generar si ya hay una pendiente del mismo tipo
                    existing = [n for n in self.notifications
                               if n.notification_type == rule["type"]
                               and n.decision == "pending"
                               and n.title == rule["title"]]
                    if existing:
                        continue

                    notification = TeacherNotification(
                        notification_id=str(uuid.uuid4())[:8],
                        timestamp=datetime.now().isoformat(),
                        teacher_id=teacher_id,
                        notification_type=rule["type"],
                        priority=rule["priority"],
                        title=rule["title"],
                        message=rule["message_template"].format(
                            pct_low=round(context.get("pct_bloom_low", 0) * 100),
                            n=context.get("interactions", 0),
                            gaming_pct=round(context.get("gaming_rate", 0) * 100),
                            rag_q=round(context.get("rag_quality", 0) * 100),
                            dep_pct=round(context.get("dependency_ratio", 0) * 100),
                            week=context.get("week", 1),
                            trend=round(context.get("bloom_trend", 0), 2),
                            cp_pct=round(context.get("high_copypaste_pct", 0) * 100),
                        ),
                        data_justification=f"Basado en {context.get('interactions', 0)} interacciones",
                        theoretical_basis=rule["theoretical_basis"],
                        suggested_action=rule["suggested_action"],
                        context_snapshot=context,
                    )
                    self.notifications.append(notification)
                    new_notifications.append(notification)
            except Exception:
                continue

        return new_notifications

    def record_decision(
        self,
        notification_id: str,
        decision: str,
        rationale: str = "",
        modified_action: dict = None,
    ) -> Optional[TeacherNotification]:
        """
        Registra la decisión del docente sobre una notificación.
        
        ESTE DATO ES ORO:
        - accepted + rationale = el docente entiende y acepta la sugerencia
        - rejected + rationale = el docente tiene razones pedagógicas propias
        - modified + modified_action = el docente adapta la sugerencia
        
        Para Teacher Agency (Alonso-Prieto LASI 2025), los rechazos con
        rationale son más valiosos que las aceptaciones.
        """
        for n in self.notifications:
            if n.notification_id == notification_id:
                n.decision = decision
                n.decision_timestamp = datetime.now().isoformat()
                n.decision_rationale = rationale
                if modified_action:
                    n.modified_action = modified_action
                return n
        return None

    def get_pending(self, teacher_id: str = None) -> List[dict]:
        """Notificaciones pendientes para un docente."""
        pending = [n for n in self.notifications if n.decision == "pending"]
        if teacher_id:
            pending = [n for n in pending if n.teacher_id == teacher_id]

        # Ordenar por prioridad
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        pending.sort(key=lambda n: priority_order.get(n.priority, 99))

        return [
            {
                "id": n.notification_id,
                "type": n.notification_type,
                "priority": n.priority,
                "title": n.title,
                "message": n.message,
                "suggested_action": n.suggested_action,
                "theoretical_basis": n.theoretical_basis,
                "timestamp": n.timestamp,
            }
            for n in pending
        ]

    def get_decision_history(self, teacher_id: str = None) -> List[dict]:
        """Historial de decisiones — dataset para Teacher Agency."""
        decided = [n for n in self.notifications if n.decision != "pending"]
        if teacher_id:
            decided = [n for n in decided if n.teacher_id == teacher_id]

        return [
            {
                "notification_id": n.notification_id,
                "type": n.notification_type,
                "title": n.title,
                "decision": n.decision,
                "rationale": n.decision_rationale,
                "modified_action": n.modified_action,
                "original_suggestion": n.suggested_action,
                "notification_timestamp": n.timestamp,
                "decision_timestamp": n.decision_timestamp,
                "context_at_decision": n.context_snapshot,
            }
            for n in decided
        ]

    def get_agency_metrics(self, teacher_id: str = None) -> dict:
        """Métricas de agencia docente derivadas de las decisiones."""
        decisions = [n for n in self.notifications if n.decision != "pending"]
        if teacher_id:
            decisions = [n for n in decisions if n.teacher_id == teacher_id]

        if not decisions:
            return {"total_decisions": 0}

        n = len(decisions)
        accepted = sum(1 for d in decisions if d.decision == "accepted")
        rejected = sum(1 for d in decisions if d.decision == "rejected")
        modified = sum(1 for d in decisions if d.decision == "modified")
        with_rationale = sum(1 for d in decisions if d.decision_rationale)

        return {
            "total_decisions": n,
            "accepted": accepted,
            "rejected": rejected,
            "modified": modified,
            "deferred": n - accepted - rejected - modified,
            "acceptance_rate": round(accepted / n, 3),
            "modification_rate": round(modified / n, 3),
            "rejection_rate": round(rejected / n, 3),
            "rationale_rate": round(with_rationale / n, 3),
            "agency_index": round((rejected + modified) / n, 3),
            # agency_index alto = docente ejerce agencia activamente
            "interpretation": (
                "HIGH_AGENCY" if (rejected + modified) / n > 0.4 else
                "MODERATE_AGENCY" if (rejected + modified) / n > 0.2 else
                "LOW_AGENCY"
            ),
        }


# ═══════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = TeacherNotificationEngine()

    # Simular contexto problemático
    context = {
        "interactions": 45,
        "avg_bloom": 2.1,
        "pct_bloom_low": 0.72,
        "gaming_rate": 0.30,
        "high_copypaste_pct": 0.15,
        "rag_quality": 0.75,
        "dependency_ratio": 0.65,
        "bloom_trend": -0.1,
        "week": 5,
    }

    # Evaluar
    new = engine.evaluate_context(context, teacher_id="prof_01")
    print(f"Notificaciones generadas: {len(new)}")
    for n in new:
        print(f"  [{n.priority}] {n.title}")

    # Mostrar pendientes
    pending = engine.get_pending("prof_01")
    print(f"\nPendientes para prof_01: {len(pending)}")

    # Docente toma decisiones
    if pending:
        # Acepta la primera
        engine.record_decision(
            pending[0]["id"], "accepted",
            rationale="Tiene sentido, los estudiantes no están progresando")

        # Rechaza la segunda con razón
        if len(pending) > 1:
            engine.record_decision(
                pending[1]["id"], "rejected",
                rationale="Prefiero mantener la configuración actual porque estamos cerca del examen")

    # Métricas de agencia
    metrics = engine.get_agency_metrics("prof_01")
    print(f"\nAgency metrics:")
    print(f"  Agency index: {metrics.get('agency_index', 0)}")
    print(f"  Interpretation: {metrics.get('interpretation', '')}")

    # Historial
    history = engine.get_decision_history("prof_01")
    print(f"\nHistorial de decisiones: {len(history)}")
    for h in history:
        print(f"  {h['decision']}: {h['rationale'][:60]}")

    print("\n✓ Teacher Notification Engine operativo")
