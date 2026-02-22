"""
TEACHER AGENCY LONGITUDINAL TRACKER — El Bucle de Retroalimentación
═══════════════════════════════════════════════════════════════════════════
Módulo diferencial Bloque 2 — De 60% a 90%

PROBLEMA QUE ATACA:
═══════════════════
El teacher_notification_engine (405 líneas) genera notificaciones y registra
si el docente acepta, rechaza o modifica. El teacher_calibration (893 líneas)
mide si el docente interpreta correctamente los dashboards. Cada módulo
opera en aislamiento temporal — captura snapshots, no trayectorias.

Lo que falta es la SERIE TEMPORAL que los conecta. El fenómeno a capturar:
un docente que al principio rechaza sugerencias del sistema (alta agencia)
y progresivamente acepta más (erosión de agencia), o viceversa.

La pregunta central: ¿la evolución de agencia correlaciona con la
calibración? Un docente que acepta más, ¿es el que mejor calibrado está
(confía porque ENTIENDE) o el que peor calibrado (confía porque NO
CUESTIONA)?

Esta distinción es la piedra angular del HCAI (Human-Centered AI,
Shneiderman 2022): el sistema debe amplificar la agencia humana, no
erosionarla. Pero "erosión de agencia" es un proceso gradual e invisible
— nadie lo detecta en un snapshot. Se necesita la serie temporal.

FUNDAMENTACIÓN TEÓRICA:
───────────────────────
1. Parasuraman & Riley (1997) — Automation complacency: los operadores
   humanos que interactúan con sistemas automatizados fiables tienden a
   reducir progresivamente su vigilancia. La tasa de aceptación creciente
   SIN mejora de calibración es exactamente este fenómeno.

2. Lee & See (2004) — Trust dynamics: la confianza en la automatización
   se construye por experiencia, pero la confianza MAL CALIBRADA (confiar
   sin comprender) es más peligrosa que la desconfianza. Nuestro módulo
   detecta confianza mal calibrada longitudinalmente.

3. Shneiderman (2022) — HCAI framework: "reliable, safe, and trustworthy"
   requiere que el humano mantenga agencia significativa. Un docente que
   acepta el 95% de sugerencias sin rationale no tiene agencia significativa
   — tiene la ILUSIÓN de agencia.

4. Alonso-Prieto et al. (LASI 2025) — Teacher agency in LA-integrated
   classrooms: mide si el docente tiene agencia. Nosotros medimos si el
   docente la está PERDIENDO gradualmente — que es exactamente el riesgo
   que el framework HCAI dice prevenir pero que nadie instrumenta.

5. Kahneman (2011) — Sistema 1 vs Sistema 2: la erosión de agencia es
   la transición de procesamiento Sistema 2 (deliberativo, costoso,
   cuestionador) a Sistema 1 (automático, rápido, aquiescente) en las
   decisiones sobre sugerencias algorítmicas.

TRES PERFILES DE DOCENTE (teóricos, verificables empíricamente):
────────────────────────────────────────────────────────────────
A. "Convergencia calibrada" (SANO): el docente empieza rechazando porque
   no comprende el sistema. A medida que se calibra mejor, acepta más.
   Correlación positiva entre acceptance_rate y calibration_accuracy.
   → Confía porque ENTIENDE.

B. "Erosión acrítica" (PREOCUPANTE): el docente acepta cada vez más,
   pero su calibración no mejora (o empeora). Correlación nula o negativa.
   → Confía porque DEJÓ DE CUESTIONAR.

C. "Desconfianza improductiva" (SUBÓPTIMO): el docente rechaza siempre
   aunque su calibración mejore. Correlación negativa.
   → No confía AUNQUE comprende. Desperdicia el beneficio del sistema.

INTEGRACIÓN:
────────────
    teacher_notification_engine.record_decision() → agency_tracker.record_event()
    teacher_calibration.record_intervention()     → agency_tracker.record_calibration()
    agency_tracker → app.py (pestaña "Análisis Profundo" o vista investigador)
    agency_tracker → system_reflexivity (señales de agencia agregadas)

Autor: Diego Elvira Vásquez · Bloque 2 CP25/152 GSIC/EMIC · Feb 2026
"""

import math
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════

# Ventana móvil para cálculo de tasas
ROLLING_WINDOW_SIZE = 20           # Últimas N decisiones para tasa móvil
MIN_DECISIONS_FOR_TRAJECTORY = 8   # Mínimo para calcular pendiente

# Umbrales de alerta
EROSION_ACCEPTANCE_THRESHOLD = 0.85  # >85% aceptación = alerta si no calibrado
DISTRUST_REJECTION_THRESHOLD = 0.80  # >80% rechazo sostenido = desconfianza

# Correlación
MIN_DATAPOINTS_FOR_CORRELATION = 6   # Mínimo para Spearman significativo
CORRELATION_SIGNIFICANCE_THRESHOLD = 0.3  # |ρ| > 0.3 para considerar relevante


# ═══════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class AgencyDatapoint:
    """
    Un punto en la serie temporal de agencia docente.
    Fusiona datos de notificación + calibración en un mismo instante.
    """
    timestamp: datetime
    teacher_id: str

    # Del teacher_notification_engine
    decision: str = ""            # accepted | rejected | modified | deferred
    notification_type: str = ""   # config_suggestion | cohort_alert | ...
    notification_priority: str = ""
    rationale: str = ""           # Texto libre del docente (ORO cualitativo)
    had_rationale: bool = False   # ¿Proporcionó justificación?

    # Del teacher_calibration (más cercano en el tiempo)
    calibration_accuracy: float = 0.0  # 0-1: precisión de interpretación
    calibration_label: str = ""        # perfect | good | neutral | poor | inverted

    # Contexto
    active_config: Dict = field(default_factory=dict)
    cohort_metrics: Dict = field(default_factory=dict)

    # Calculado
    acceptance_rate_rolling: float = 0.0  # Tasa de aceptación en ventana móvil
    rejection_rate_rolling: float = 0.0


@dataclass
class AgencyTrajectory:
    """
    Trayectoria longitudinal de agencia docente.
    Captura la pendiente de la tasa de aceptación en el tiempo.
    """
    teacher_id: str
    n_datapoints: int = 0
    n_decisions: int = 0

    # Pendientes (regresión lineal sobre ventanas móviles)
    acceptance_slope: float = 0.0       # Positiva = acepta más con el tiempo
    calibration_slope: float = 0.0      # Positiva = mejora calibración con el tiempo

    # Clasificación
    trajectory_label: str = "insufficient_data"
    # Valores posibles:
    #   calibrated_convergence   → acepta más + calibra mejor (SANO)
    #   uncritical_erosion       → acepta más + calibración estancada/baja (PELIGROSO)
    #   productive_skepticism    → rechaza mucho + calibra bien (SUBÓPTIMO pero sano)
    #   reflexive_agency         → oscila según contexto (IDEAL)
    #   disengaged_distrust      → rechaza siempre sin mejora (IMPRODUCTIVO)
    #   insufficient_data        → no hay datos suficientes

    # Métricas de la última ventana
    current_acceptance_rate: float = 0.0
    current_calibration_accuracy: float = 0.0
    current_rationale_rate: float = 0.0

    # Interpretación narrativa
    interpretation: str = ""
    risk_level: str = "unknown"    # low | medium | high | critical
    researcher_note: str = ""


@dataclass
class AgencyAlert:
    """
    Alerta de erosión de agencia.
    Se genera cuando se detecta aceptación acrítica sostenida.
    """
    alert_id: str = ""
    timestamp: str = ""
    teacher_id: str = ""
    alert_type: str = ""           # erosion | distrust | disengagement
    severity: str = "medium"       # low | medium | high | critical
    message: str = ""
    evidence: Dict = field(default_factory=dict)
    suggested_intervention: str = ""
    theoretical_basis: str = ""


@dataclass
class CorrelationResult:
    """
    Resultado de correlación Spearman entre acceptance_rate y calibration_accuracy.
    """
    teacher_id: str
    n_datapoints: int = 0
    spearman_rho: float = 0.0
    is_significant: bool = False
    interpretation: str = ""
    # Categoría
    correlation_type: str = ""
    # Valores:
    #   positive_healthy   → confía más cuando entiende más (SANO)
    #   null_concerning     → confía independientemente de comprensión (PREOCUPANTE)
    #   negative_worrying   → confía más cuando entiende MENOS (PELIGROSO)
    #   insufficient_data   → no hay datos suficientes


# ═══════════════════════════════════════════════════════════════════════
# MOTOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

class TeacherAgencyTracker:
    """
    Rastreador longitudinal de agencia docente.

    Consume eventos de dos fuentes:
    1. teacher_notification_engine: cada decisión accept/reject/modify
    2. teacher_calibration: cada medición de accuracy

    Produce:
    - Trayectorias de agencia por docente
    - Alertas de erosión
    - Correlación agencia-calibración
    - Datos para visualización dual (acceptance_rate vs calibration_accuracy)

    Uso:
        tracker = TeacherAgencyTracker()
        tracker.record_decision_event(teacher_id, decision, ...)
        tracker.record_calibration_event(teacher_id, accuracy, ...)
        trajectory = tracker.compute_agency_trajectory(teacher_id)
        alert = tracker.detect_agency_erosion(teacher_id)
    """

    def __init__(self):
        # Serie temporal por docente
        self.datapoints: Dict[str, List[AgencyDatapoint]] = defaultdict(list)
        # Alertas generadas
        self.alerts: List[AgencyAlert] = []
        # Cache de calibración más reciente por docente
        self._latest_calibration: Dict[str, float] = {}
        self._latest_calibration_label: Dict[str, str] = {}

    # ──────────────────────────────────────────────────────────────────
    # INGESTA DE EVENTOS
    # ──────────────────────────────────────────────────────────────────

    def record_decision_event(
        self,
        teacher_id: str,
        timestamp: datetime,
        decision: str,
        notification_type: str = "",
        notification_priority: str = "",
        rationale: str = "",
        active_config: dict = None,
        cohort_metrics: dict = None,
    ):
        """
        Registra un evento de decisión del teacher_notification_engine.
        Se invoca cada vez que el docente acepta/rechaza/modifica una sugerencia.
        """
        # Tomar la calibración más reciente disponible
        cal_accuracy = self._latest_calibration.get(teacher_id, 0.0)
        cal_label = self._latest_calibration_label.get(teacher_id, "unknown")

        dp = AgencyDatapoint(
            timestamp=timestamp,
            teacher_id=teacher_id,
            decision=decision,
            notification_type=notification_type,
            notification_priority=notification_priority,
            rationale=rationale,
            had_rationale=bool(rationale and rationale.strip()),
            calibration_accuracy=cal_accuracy,
            calibration_label=cal_label,
            active_config=active_config or {},
            cohort_metrics=cohort_metrics or {},
        )

        self.datapoints[teacher_id].append(dp)

        # Recalcular tasas móviles
        self._update_rolling_rates(teacher_id)

    def record_calibration_event(
        self,
        teacher_id: str,
        timestamp: datetime,
        calibration_accuracy: float,
        calibration_label: str = "",
    ):
        """
        Registra un evento de calibración del teacher_calibration.
        Actualiza el cache de calibración más reciente.
        """
        self._latest_calibration[teacher_id] = calibration_accuracy
        self._latest_calibration_label[teacher_id] = calibration_label

        # Actualizar el último datapoint si existe y es cercano en el tiempo
        points = self.datapoints.get(teacher_id, [])
        if points:
            last = points[-1]
            gap = abs((timestamp - last.timestamp).total_seconds())
            # Si la calibración es dentro de 1 hora de la última decisión, actualizar
            if gap < 3600:
                last.calibration_accuracy = calibration_accuracy
                last.calibration_label = calibration_label

    def _update_rolling_rates(self, teacher_id: str):
        """Recalcula tasas de aceptación/rechazo en ventana móvil."""
        points = self.datapoints.get(teacher_id, [])
        if not points:
            return

        window = points[-ROLLING_WINDOW_SIZE:]
        n = len(window)

        accepted = sum(1 for p in window if p.decision == "accepted")
        rejected = sum(1 for p in window if p.decision == "rejected")

        rate_accept = accepted / n
        rate_reject = rejected / n

        # Actualizar el último punto
        points[-1].acceptance_rate_rolling = round(rate_accept, 3)
        points[-1].rejection_rate_rolling = round(rate_reject, 3)

    # ──────────────────────────────────────────────────────────────────
    # TRAYECTORIA DE AGENCIA
    # ──────────────────────────────────────────────────────────────────

    def compute_agency_trajectory(self, teacher_id: str) -> AgencyTrajectory:
        """
        Calcula la trayectoria longitudinal de agencia.

        Pendiente de la tasa de aceptación en ventanas móviles:
        - Positiva sostenida → acepta cada vez más ("convergencia con el sistema")
        - Negativa sostenida → rechaza cada vez más ("divergencia crítica")
        - Oscilante → va y viene según contexto ("agencia reflexiva" — lo más sano)
        """
        points = self.datapoints.get(teacher_id, [])

        trajectory = AgencyTrajectory(
            teacher_id=teacher_id,
            n_datapoints=len(points),
            n_decisions=len(points),
        )

        if len(points) < MIN_DECISIONS_FOR_TRAJECTORY:
            trajectory.trajectory_label = "insufficient_data"
            trajectory.interpretation = (
                f"Datos insuficientes ({len(points)} decisiones, "
                f"mínimo {MIN_DECISIONS_FOR_TRAJECTORY}). "
                f"El análisis de trayectoria requiere más interacción con el sistema."
            )
            trajectory.risk_level = "unknown"
            return trajectory

        # Calcular pendientes usando ventanas móviles
        acceptance_rates = self._compute_rolling_acceptance_rates(points)
        calibration_accuracies = self._extract_calibration_series(points)

        if len(acceptance_rates) >= 3:
            trajectory.acceptance_slope = self._compute_slope(acceptance_rates)
        if len(calibration_accuracies) >= 3:
            trajectory.calibration_slope = self._compute_slope(calibration_accuracies)

        # Estado actual
        recent = points[-min(ROLLING_WINDOW_SIZE, len(points)):]
        trajectory.current_acceptance_rate = round(
            sum(1 for p in recent if p.decision == "accepted") / len(recent), 3
        )
        cal_values = [p.calibration_accuracy for p in recent if p.calibration_accuracy > 0]
        trajectory.current_calibration_accuracy = round(
            statistics.mean(cal_values), 3
        ) if cal_values else 0.0
        trajectory.current_rationale_rate = round(
            sum(1 for p in recent if p.had_rationale) / len(recent), 3
        )

        # Clasificar trayectoria
        trajectory.trajectory_label = self._classify_trajectory(
            trajectory.acceptance_slope,
            trajectory.calibration_slope,
            trajectory.current_acceptance_rate,
            trajectory.current_calibration_accuracy,
            trajectory.current_rationale_rate,
        )

        # Interpretar
        trajectory.interpretation = self._interpret_trajectory(trajectory)
        trajectory.risk_level = self._assess_risk(trajectory)
        trajectory.researcher_note = self._generate_researcher_note(trajectory, points)

        return trajectory

    def _compute_rolling_acceptance_rates(
        self,
        points: List[AgencyDatapoint],
    ) -> List[float]:
        """Calcula serie de tasas de aceptación en ventanas móviles de tamaño 5."""
        window_size = min(5, len(points) // 2)
        if window_size < 2:
            return []

        rates = []
        for i in range(window_size, len(points) + 1):
            window = points[i - window_size:i]
            accepted = sum(1 for p in window if p.decision == "accepted")
            rates.append(accepted / len(window))

        return rates

    def _extract_calibration_series(
        self,
        points: List[AgencyDatapoint],
    ) -> List[float]:
        """Extrae serie de calibración de los datapoints."""
        return [p.calibration_accuracy for p in points if p.calibration_accuracy > 0]

    def _compute_slope(self, series: List[float]) -> float:
        """
        Regresión lineal simple para calcular pendiente.
        Implementación sin dependencias externas.
        """
        n = len(series)
        if n < 2:
            return 0.0

        x_mean = (n - 1) / 2.0
        y_mean = statistics.mean(series)

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(series))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0

        return round(numerator / denominator, 5)

    def _classify_trajectory(
        self,
        acceptance_slope: float,
        calibration_slope: float,
        current_acceptance: float,
        current_calibration: float,
        rationale_rate: float,
    ) -> str:
        """
        Clasifica la trayectoria de agencia.

        La clasificación cruza DOS ejes:
        - Eje de aceptación: ¿acepta más o menos con el tiempo?
        - Eje de calibración: ¿mejora o no su comprensión del sistema?

        La combinación de ambos ejes produce cinco patrones con
        implicaciones pedagógicas radicalmente distintas.

        DECISIÓN DE DISEÑO: la clasificación opera sobre DOS señales
        complementarias — la pendiente (dinámica temporal) y el nivel
        actual (estado presente). Esto captura tanto al docente que
        ESTÁ CONVERGIENDO hacia erosión como al que YA ESTÁ en erosión
        (aceptación alta sostenida sin mejora de calibración, donde la
        pendiente puede ser plana porque el nivel ya era alto).
        """
        # Umbrales de pendiente
        slope_rising = 0.005     # Pendiente positiva significativa (rebajado)
        slope_falling = -0.005   # Pendiente negativa significativa

        acceptance_rising = acceptance_slope > slope_rising
        acceptance_falling = acceptance_slope < slope_falling
        acceptance_stable = not acceptance_rising and not acceptance_falling

        calibration_rising = calibration_slope > slope_rising
        calibration_low = current_calibration < 0.5

        # ── PRIORIDAD 1: Detección de erosión por nivel actual ──
        # Un docente que ACEPTA >85% con calibración BAJA es erosión
        # independientemente de la pendiente (puede que ya estuviera
        # alto desde el inicio — la pendiente no lo captura)
        if current_acceptance > EROSION_ACCEPTANCE_THRESHOLD and calibration_low:
            if not calibration_rising:
                return "uncritical_erosion"

        # ── PRIORIDAD 2: Clasificación por dinámica temporal ──
        # ── Detección de escepticismo por nivel actual ──
        # Un docente que ACEPTA <25% con alta calibración es escéptico
        # productivo, incluso si su pendiente es ligeramente positiva
        if current_acceptance <= 0.25 and current_calibration > 0.5:
            return "productive_skepticism"
        if current_acceptance <= 0.20 and current_calibration <= 0.5:
            return "disengaged_distrust"

        if acceptance_rising and calibration_rising:
            return "calibrated_convergence"

        if acceptance_rising and not calibration_rising:
            return "uncritical_erosion"

        # ── Rechazo creciente ──
        if acceptance_falling and current_calibration > 0.5:
            return "productive_skepticism"

        if acceptance_falling and calibration_low:
            return "disengaged_distrust"

        # ── Agencia reflexiva: el patrón sano por defecto ──
        # Aceptación estable, oscilante, o con rationale alto
        if acceptance_stable and rationale_rate > 0.4:
            return "reflexive_agency"

        # Default
        return "reflexive_agency"

    def _interpret_trajectory(self, t: AgencyTrajectory) -> str:
        """Genera interpretación narrativa de la trayectoria."""
        labels = {
            "calibrated_convergence": (
                f"Convergencia calibrada. El docente acepta más sugerencias a medida que "
                f"mejora su comprensión del sistema (pendiente aceptación: {t.acceptance_slope:+.3f}, "
                f"pendiente calibración: {t.calibration_slope:+.3f}). "
                f"Este es el patrón más sano: la confianza crece anclada en comprensión real."
            ),
            "uncritical_erosion": (
                f"ALERTA: Erosión acrítica de agencia. La tasa de aceptación crece "
                f"(pendiente: {t.acceptance_slope:+.3f}) pero la calibración NO mejora "
                f"(pendiente: {t.calibration_slope:+.3f}). El docente no acepta porque entienda "
                f"mejor — acepta porque dejó de cuestionar. Esto es exactamente el patrón que "
                f"Parasuraman & Riley (1997) denominan 'automation complacency'."
            ),
            "productive_skepticism": (
                f"Escepticismo productivo. El docente rechaza sugerencias a pesar de tener "
                f"buena calibración ({t.current_calibration_accuracy:.0%}). Ejerce agencia "
                f"activa basada en conocimiento pedagógico propio. El riesgo es subutilizar "
                f"el sistema, pero la agencia docente está intacta."
            ),
            "reflexive_agency": (
                f"Agencia reflexiva. El docente oscila entre aceptación y rechazo según "
                f"contexto (tasa actual: {t.current_acceptance_rate:.0%}, ratio de justificaciones: "
                f"{t.current_rationale_rate:.0%}). Este es el patrón IDEAL: el docente evalúa "
                f"cada sugerencia en su mérito, no por inercia."
            ),
            "disengaged_distrust": (
                f"Desconfianza improductiva. El docente rechaza sistemáticamente "
                f"(tasa aceptación: {t.current_acceptance_rate:.0%}) con baja calibración "
                f"({t.current_calibration_accuracy:.0%}). No confía en el sistema y tampoco "
                f"comprende bien los analytics. El beneficio potencial del sistema se está "
                f"desperdiciando."
            ),
            "insufficient_data": "Datos insuficientes para clasificación.",
        }
        return labels.get(t.trajectory_label, "Clasificación no disponible.")

    def _assess_risk(self, t: AgencyTrajectory) -> str:
        """Evalúa el nivel de riesgo de la trayectoria."""
        risk_map = {
            "calibrated_convergence": "low",
            "reflexive_agency": "low",
            "productive_skepticism": "medium",
            "uncritical_erosion": "high",
            "disengaged_distrust": "high",
            "insufficient_data": "unknown",
        }
        return risk_map.get(t.trajectory_label, "unknown")

    def _generate_researcher_note(
        self,
        t: AgencyTrajectory,
        points: List[AgencyDatapoint],
    ) -> str:
        """Nota técnica para el investigador."""
        notes = []

        # Ratio de justificaciones con contenido sustantivo
        with_rationale = [p for p in points if p.had_rationale]
        if with_rationale:
            notes.append(
                f"Justificaciones proporcionadas: {len(with_rationale)}/{len(points)} "
                f"({len(with_rationale)/len(points):.0%}). "
                f"Cada justificación es dato cualitativo para el análisis de teacher agency."
            )

        # Distribución por tipo de notificación
        type_counts = defaultdict(int)
        for p in points:
            if p.notification_type:
                type_counts[p.notification_type] += 1
        if type_counts:
            type_str = ", ".join(f"{k}: {v}" for k, v in type_counts.items())
            notes.append(f"Distribución por tipo: {type_str}.")

        # Patrón temporal: ¿hay cambios de régimen?
        if len(points) >= 10:
            first_half = points[:len(points)//2]
            second_half = points[len(points)//2:]
            rate_1 = sum(1 for p in first_half if p.decision == "accepted") / len(first_half)
            rate_2 = sum(1 for p in second_half if p.decision == "accepted") / len(second_half)
            delta = rate_2 - rate_1
            if abs(delta) > 0.2:
                notes.append(
                    f"Cambio de régimen detectado: tasa aceptación primera mitad {rate_1:.0%} → "
                    f"segunda mitad {rate_2:.0%} (Δ{delta:+.0%}). "
                    f"{'Posible erosión' if delta > 0 else 'Posible despertar crítico'}."
                )

        return " | ".join(notes) if notes else "Sin notas adicionales."

    # ──────────────────────────────────────────────────────────────────
    # DETECCIÓN DE EROSIÓN
    # ──────────────────────────────────────────────────────────────────

    def detect_agency_erosion(self, teacher_id: str) -> Optional[AgencyAlert]:
        """
        Dispara alerta si la tasa de aceptación supera el 85% en las
        últimas 20 decisiones Y la calibración NO ha mejorado.

        Esto indica ACEPTACIÓN ACRÍTICA — el docente no acepta porque
        entiende mejor, acepta porque dejó de cuestionar.
        """
        points = self.datapoints.get(teacher_id, [])

        if len(points) < ROLLING_WINDOW_SIZE:
            return None

        recent = points[-ROLLING_WINDOW_SIZE:]
        accepted = sum(1 for p in recent if p.decision == "accepted")
        acceptance_rate = accepted / len(recent)

        if acceptance_rate < EROSION_ACCEPTANCE_THRESHOLD:
            return None

        # Verificar si la calibración ha mejorado
        cal_series = [p.calibration_accuracy for p in recent if p.calibration_accuracy > 0]
        if len(cal_series) < 3:
            return None  # No hay suficientes datos de calibración

        cal_slope = self._compute_slope(cal_series)
        mean_cal = statistics.mean(cal_series)

        # Alerta si: alta aceptación + calibración NO mejora
        if cal_slope <= 0.005 or mean_cal < 0.5:
            alert = AgencyAlert(
                alert_id=str(uuid.uuid4())[:8],
                timestamp=datetime.now().isoformat(),
                teacher_id=teacher_id,
                alert_type="erosion",
                severity="high" if acceptance_rate > 0.90 else "medium",
                message=(
                    f"La tasa de aceptación de sugerencias del docente es {acceptance_rate:.0%} "
                    f"en las últimas {ROLLING_WINDOW_SIZE} decisiones, pero su calibración "
                    f"interpretativa {'no ha mejorado' if cal_slope <= 0 else 'apenas ha variado'} "
                    f"(media: {mean_cal:.0%}, tendencia: {cal_slope:+.4f}). "
                    f"Esto sugiere aceptación acrítica, no convergencia informada."
                ),
                evidence={
                    "acceptance_rate": round(acceptance_rate, 3),
                    "calibration_mean": round(mean_cal, 3),
                    "calibration_slope": round(cal_slope, 5),
                    "window_size": ROLLING_WINDOW_SIZE,
                    "rationale_rate": round(
                        sum(1 for p in recent if p.had_rationale) / len(recent), 3
                    ),
                },
                suggested_intervention=(
                    "Considerar: (1) mostrar al docente un dashboard simplificado de su "
                    "propio patrón de decisiones, (2) introducir 'friction points' que "
                    "requieran justificación antes de aceptar, (3) sesión de calibración "
                    "con el investigador para re-anclar la interpretación del dashboard."
                ),
                theoretical_basis=(
                    "Parasuraman & Riley (1997): automation complacency. "
                    "Lee & See (2004): trust-calibration gap. "
                    "Shneiderman (2022): HCAI requiere agencia humana genuina."
                ),
            )
            self.alerts.append(alert)
            return alert

        return None

    # ──────────────────────────────────────────────────────────────────
    # CORRELACIÓN AGENCIA-CALIBRACIÓN
    # ──────────────────────────────────────────────────────────────────

    def compute_agency_calibration_correlation(
        self,
        teacher_id: str,
    ) -> CorrelationResult:
        """
        Calcula correlación Spearman entre acceptance_rate (ventana móvil)
        y calibration_accuracy (ventana móvil).

        Spearman (no Pearson) porque:
        1. No asumimos linealidad en la relación
        2. Es robusta a outliers
        3. Opera sobre rangos, no magnitudes — adecuado para tasas

        Si la correlación es positiva y significativa, el docente confía
        más cuando entiende más (sano). Si es nula o negativa, confía
        independientemente de su comprensión (preocupante).
        """
        points = self.datapoints.get(teacher_id, [])

        result = CorrelationResult(
            teacher_id=teacher_id,
            n_datapoints=len(points),
        )

        if len(points) < MIN_DATAPOINTS_FOR_CORRELATION:
            result.interpretation = (
                f"Datos insuficientes ({len(points)} puntos, "
                f"mínimo {MIN_DATAPOINTS_FOR_CORRELATION})."
            )
            result.correlation_type = "insufficient_data"
            return result

        # Extraer pares (acceptance_rate, calibration_accuracy)
        pairs = []
        acceptance_rates = self._compute_rolling_acceptance_rates(points)
        cal_series = self._extract_calibration_series(points)

        # Alinear series (tomamos el mínimo de longitud)
        n_pairs = min(len(acceptance_rates), len(cal_series))
        if n_pairs < MIN_DATAPOINTS_FOR_CORRELATION:
            result.interpretation = "Series de longitud insuficiente tras alineación."
            result.correlation_type = "insufficient_data"
            return result

        # Tomar los últimos n_pairs de cada serie
        acc = acceptance_rates[-n_pairs:]
        cal = cal_series[-n_pairs:]

        # Calcular Spearman
        rho = self._spearman_correlation(acc, cal)

        result.spearman_rho = round(rho, 3)
        result.is_significant = abs(rho) > CORRELATION_SIGNIFICANCE_THRESHOLD

        # Clasificar
        if not result.is_significant:
            result.correlation_type = "null_concerning"
            result.interpretation = (
                f"Correlación no significativa (ρ = {rho:.3f}). El docente confía "
                f"independientemente de su nivel de comprensión del dashboard. "
                f"Esto sugiere que la aceptación no está anclada en calibración "
                f"— exactamente el patrón que Shneiderman (2022) identifica como "
                f"riesgo para HCAI."
            )
        elif rho > 0:
            result.correlation_type = "positive_healthy"
            result.interpretation = (
                f"Correlación positiva (ρ = {rho:.3f}). El docente confía más "
                f"cuando entiende mejor. Este es el patrón sano: la confianza "
                f"está anclada en comprensión real."
            )
        else:
            result.correlation_type = "negative_worrying"
            result.interpretation = (
                f"Correlación negativa (ρ = {rho:.3f}). El docente confía MÁS "
                f"cuando entiende MENOS. Este es el patrón más preocupante: "
                f"sugiere que la confianza es inversamente proporcional a la "
                f"comprensión — posible síndrome Dunning-Kruger aplicado a la "
                f"interpretación de analytics."
            )

        return result

    def _spearman_correlation(self, x: List[float], y: List[float]) -> float:
        """
        Correlación de Spearman implementada sin scipy.
        Calcula el coeficiente de correlación de rangos.
        """
        n = len(x)
        if n < 2:
            return 0.0

        def rank(values):
            sorted_vals = sorted(enumerate(values), key=lambda t: t[1])
            ranks = [0.0] * n
            i = 0
            while i < n:
                j = i
                while j < n - 1 and sorted_vals[j + 1][1] == sorted_vals[j][1]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1.0  # 1-based
                for k in range(i, j + 1):
                    ranks[sorted_vals[k][0]] = avg_rank
                i = j + 1
            return ranks

        rank_x = rank(x)
        rank_y = rank(y)

        # Pearson sobre rangos = Spearman
        mean_rx = statistics.mean(rank_x)
        mean_ry = statistics.mean(rank_y)

        numerator = sum((rx - mean_rx) * (ry - mean_ry) for rx, ry in zip(rank_x, rank_y))
        denom_x = math.sqrt(sum((rx - mean_rx) ** 2 for rx in rank_x))
        denom_y = math.sqrt(sum((ry - mean_ry) ** 2 for ry in rank_y))

        if denom_x == 0 or denom_y == 0:
            return 0.0

        return numerator / (denom_x * denom_y)

    # ──────────────────────────────────────────────────────────────────
    # DATOS PARA VISUALIZACIÓN
    # ──────────────────────────────────────────────────────────────────

    def get_visualization_data(self, teacher_id: str) -> Dict:
        """
        Datos formateados para gráfico dual:
        - Eje Y1 (azul): tasa de aceptación en ventana móvil
        - Eje Y2 (naranja): calibration_accuracy en ventana móvil
        - Bandas sombreadas para zonas de agencia

        Diseñado para plotly o matplotlib en Streamlit.
        """
        points = self.datapoints.get(teacher_id, [])

        if not points:
            return {"timestamps": [], "acceptance_rates": [], "calibration_accuracies": []}

        timestamps = [p.timestamp.isoformat() for p in points]

        # Serie de aceptación
        acceptance_rates = []
        for i, p in enumerate(points):
            window_start = max(0, i - ROLLING_WINDOW_SIZE + 1)
            window = points[window_start:i + 1]
            rate = sum(1 for pp in window if pp.decision == "accepted") / len(window)
            acceptance_rates.append(round(rate, 3))

        # Serie de calibración
        calibration_accuracies = [round(p.calibration_accuracy, 3) for p in points]

        # Zonas de referencia
        zones = {
            "healthy_agency": {"min": 0.20, "max": 0.70, "color": "rgba(50, 180, 50, 0.1)"},
            "erosion_risk": {"min": 0.85, "max": 1.00, "color": "rgba(255, 50, 50, 0.1)"},
            "distrust_risk": {"min": 0.00, "max": 0.15, "color": "rgba(255, 165, 0, 0.1)"},
        }

        return {
            "timestamps": timestamps,
            "acceptance_rates": acceptance_rates,
            "calibration_accuracies": calibration_accuracies,
            "n_points": len(points),
            "zones": zones,
            "decisions": [p.decision for p in points],
            "rationales": [p.rationale for p in points],
            "had_rationale": [p.had_rationale for p in points],
        }

    # ──────────────────────────────────────────────────────────────────
    # RESUMEN PARA SYSTEM_REFLEXIVITY
    # ──────────────────────────────────────────────────────────────────

    def get_cohort_agency_summary(self) -> Dict:
        """
        Resumen de agencia del cohorte de docentes.
        Para alimentar al system_reflexivity con señales agregadas.
        """
        all_teachers = list(self.datapoints.keys())

        if not all_teachers:
            return {"n_teachers": 0}

        trajectories = {}
        label_counts = defaultdict(int)

        for tid in all_teachers:
            t = self.compute_agency_trajectory(tid)
            trajectories[tid] = t
            label_counts[t.trajectory_label] += 1

        erosion_count = sum(1 for t in trajectories.values()
                           if t.trajectory_label == "uncritical_erosion")

        return {
            "n_teachers": len(all_teachers),
            "trajectory_distribution": dict(label_counts),
            "erosion_alerts": len(self.alerts),
            "teachers_at_risk": erosion_count,
            "mean_acceptance_rate": round(
                statistics.mean(t.current_acceptance_rate for t in trajectories.values()), 3
            ),
            "mean_calibration": round(
                statistics.mean(
                    t.current_calibration_accuracy for t in trajectories.values()
                    if t.current_calibration_accuracy > 0
                ), 3
            ) if any(t.current_calibration_accuracy > 0 for t in trajectories.values()) else 0.0,
            "cohort_health": (
                "HEALTHY" if erosion_count == 0 else
                "AT_RISK" if erosion_count <= len(all_teachers) * 0.3 else
                "CRITICAL"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# GENERADOR DE DATOS SINTÉTICOS PARA DEMO
# ═══════════════════════════════════════════════════════════════════════

def generate_demo_data() -> TeacherAgencyTracker:
    """
    Genera tres perfiles docentes sintéticos que ilustran los tres
    patrones teóricos fundamentales:

    1. "Ana" (experimentada): empieza rechazando mucho, a medida que
       se calibra mejor acepta más → correlación positiva, agencia sana.
       Trayectoria: calibrated_convergence.

    2. "Carlos" (tecnófilo): acepta todo desde el principio con baja
       calibración → erosión acrítica. Sin rationale sustantivo.
       Trayectoria: uncritical_erosion.

    3. "María" (escéptica): rechaza siempre aunque su calibración
       mejore → desconfianza improductiva que reduce el beneficio.
       Trayectoria: productive_skepticism (o disengaged_distrust).
    """
    import random
    random.seed(7)  # Seed elegido para producir perfiles que converjan con los patrones teóricos

    tracker = TeacherAgencyTracker()
    base = datetime.now() - timedelta(days=30)

    # ────────────────────────────────────────
    # ANA — Convergencia calibrada (SANO)
    # ────────────────────────────────────────
    ana_decisions = []
    # Fase 1 (día 1-10): rechaza mucho, baja calibración
    for i in range(10):
        ts = base + timedelta(days=i * 0.9, hours=random.randint(9, 17))
        decision = random.choices(
            ["accepted", "rejected", "modified"],
            weights=[0.10, 0.70, 0.20]
        )[0]
        cal = 0.25 + random.gauss(0, 0.04)
        rationale = random.choice([
            "No conozco bien el sistema todavía, prefiero mantener mi criterio",
            "Quiero ver más datos antes de cambiar la configuración",
            "",
            "No me convence la sugerencia para este grupo",
        ])
        tracker.record_calibration_event("ana_prof", ts, max(0.1, min(1, cal)), "poor")
        tracker.record_decision_event(
            "ana_prof", ts, decision,
            notification_type=random.choice(["config_suggestion", "cohort_alert"]),
            notification_priority=random.choice(["medium", "high"]),
            rationale=rationale,
        )

    # Fase 2 (día 10-20): transición, calibración mejora, empieza a aceptar más
    for i in range(10):
        ts = base + timedelta(days=10 + i * 0.9, hours=random.randint(9, 17))
        decision = random.choices(
            ["accepted", "rejected", "modified"],
            weights=[0.50, 0.25, 0.25]
        )[0]
        cal = 0.50 + i * 0.025 + random.gauss(0, 0.03)
        rationale = random.choice([
            "Tiene sentido con lo que observo en clase",
            "Los datos coinciden con mi impresión",
            "Acepto pero voy a vigilar el efecto",
            "",
        ])
        tracker.record_calibration_event("ana_prof", ts, max(0.1, min(1, cal)), "good")
        tracker.record_decision_event(
            "ana_prof", ts, decision,
            notification_type=random.choice(["config_suggestion", "student_flag"]),
            rationale=rationale,
        )

    # Fase 3 (día 20-30): acepta más, alta calibración
    for i in range(12):
        ts = base + timedelta(days=20 + i * 0.8, hours=random.randint(9, 17))
        decision = random.choices(
            ["accepted", "rejected", "modified"],
            weights=[0.80, 0.07, 0.13]
        )[0]
        cal = 0.72 + i * 0.012 + random.gauss(0, 0.02)
        rationale = random.choice([
            "Confirmo con datos del parcial que la sugerencia era correcta",
            "La tendencia del cohorte valida el cambio",
            "Acepto y refino con mi criterio",
            "Modifico ligeramente para adaptarlo a mi grupo",
        ])
        tracker.record_calibration_event("ana_prof", ts, max(0.1, min(1, cal)), "perfect")
        tracker.record_decision_event(
            "ana_prof", ts, decision,
            notification_type=random.choice(["config_suggestion", "milestone"]),
            rationale=rationale,
        )

    # ────────────────────────────────────────
    # CARLOS — Erosión acrítica (PREOCUPANTE)
    # ────────────────────────────────────────
    for i in range(30):
        ts = base + timedelta(days=i * 0.9, hours=random.randint(8, 18))
        # Acepta casi todo desde el principio
        decision = random.choices(
            ["accepted", "rejected", "modified"],
            weights=[0.88, 0.07, 0.05]
        )[0]
        # Calibración baja y estancada
        cal = 0.35 + random.gauss(0, 0.06)
        # Casi nunca da rationale
        rationale = random.choice([
            "", "", "", "",
            "OK", "Vale",
            "", "",
        ])
        tracker.record_calibration_event("carlos_prof", ts, max(0, min(1, cal)), "poor")
        tracker.record_decision_event(
            "carlos_prof", ts, decision,
            notification_type=random.choice(["config_suggestion", "cohort_alert", "student_flag"]),
            rationale=rationale,
        )

    # ────────────────────────────────────────
    # MARÍA — Escepticismo / Desconfianza
    # ────────────────────────────────────────
    for i in range(30):
        ts = base + timedelta(days=i * 0.9, hours=random.randint(10, 16))
        # Rechaza casi todo
        decision = random.choices(
            ["accepted", "rejected", "modified"],
            weights=[0.12, 0.75, 0.13]
        )[0]
        # Calibración mejora gradualmente
        cal = 0.4 + i * 0.015 + random.gauss(0, 0.04)
        # Da justificación frecuentemente
        rationale = random.choice([
            "Conozco a mis estudiantes mejor que el algoritmo",
            "Los datos no capturan el contexto de mi aula",
            "Prefiero mi criterio pedagógico sobre métricas automáticas",
            "Ya lo sé pero prefiero actuar cuando yo considere",
            "No me fío de la recomendación para este tema",
            "La teoría es correcta pero no aplica a mi contexto",
        ])
        tracker.record_calibration_event("maria_prof", ts, max(0, min(1, cal)), "good")
        tracker.record_decision_event(
            "maria_prof", ts, decision,
            notification_type=random.choice(["config_suggestion", "cohort_alert"]),
            rationale=rationale,
        )

    return tracker


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("TEACHER AGENCY LONGITUDINAL TRACKER — Demo con datos sintéticos")
    print("=" * 70)

    tracker = generate_demo_data()

    for teacher_id, teacher_name in [
        ("ana_prof", "Ana (experimentada)"),
        ("carlos_prof", "Carlos (tecnófilo)"),
        ("maria_prof", "María (escéptica)"),
    ]:
        print(f"\n{'─' * 70}")
        print(f"DOCENTE: {teacher_name}")
        print(f"{'─' * 70}")

        # Trayectoria
        traj = tracker.compute_agency_trajectory(teacher_id)
        print(f"  Trayectoria: {traj.trajectory_label}")
        print(f"  Riesgo: {traj.risk_level}")
        print(f"  Tasa aceptación actual: {traj.current_acceptance_rate:.0%}")
        print(f"  Calibración actual: {traj.current_calibration_accuracy:.0%}")
        print(f"  Ratio justificaciones: {traj.current_rationale_rate:.0%}")
        print(f"  Pendiente aceptación: {traj.acceptance_slope:+.4f}")
        print(f"  Pendiente calibración: {traj.calibration_slope:+.4f}")
        print(f"\n  Interpretación: {traj.interpretation}")
        print(f"\n  Nota investigador: {traj.researcher_note}")

        # Correlación
        corr = tracker.compute_agency_calibration_correlation(teacher_id)
        print(f"\n  Correlación Spearman: ρ = {corr.spearman_rho:.3f}")
        print(f"  Tipo: {corr.correlation_type}")
        print(f"  Interpretación: {corr.interpretation}")

        # Erosión
        alert = tracker.detect_agency_erosion(teacher_id)
        if alert:
            print(f"\n  ⚠️ ALERTA DE EROSIÓN: {alert.message}")
        else:
            print(f"\n  ✓ Sin alertas de erosión")

    # Resumen cohorte
    print(f"\n{'─' * 70}")
    print("RESUMEN COHORTE")
    print(f"{'─' * 70}")
    cohort = tracker.get_cohort_agency_summary()
    for k, v in cohort.items():
        print(f"  {k}: {v}")

    # Datos de visualización (muestra para Ana)
    print(f"\n{'─' * 70}")
    print("DATOS VISUALIZACIÓN — Ana (primeros 5 puntos)")
    print(f"{'─' * 70}")
    viz = tracker.get_visualization_data("ana_prof")
    for i in range(min(5, viz["n_points"])):
        print(f"  {viz['timestamps'][i][:16]} | "
              f"aceptación: {viz['acceptance_rates'][i]:.2f} | "
              f"calibración: {viz['calibration_accuracies'][i]:.2f} | "
              f"decisión: {viz['decisions'][i]}")

    print(f"\n{'=' * 70}")
    print("✓ TeacherAgencyTracker operativo")
    print("  → Alonso-Prieto mide si el docente tiene agencia.")
    print("  → Nosotros medimos si el docente la está PERDIENDO gradualmente.")
    print("=" * 70)
