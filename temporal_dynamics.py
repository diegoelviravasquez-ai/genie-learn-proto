"""
DINÁMICAS TEMPORALES DEL APRENDIZAJE — DIMENSIÓN 3
=====================================================
Eje temporal-evolutivo: el aprendizaje como proceso dinámico con fases,
velocidad, aceleración y transiciones — no como snapshot estático.

PROBLEMA QUE RESUELVE:
  Las dos dimensiones actuales de GENIE Learn (GenAI Analytics + Pedagogical
  Configurations) producen radiografías estáticas. El docente ve que un
  estudiante tiene nivel medio Bloom 2.3, pero no sabe si está ascendiendo
  desde 1.0 (progreso) o descendiendo desde 4.0 (regresión). El mismo
  número codifica realidades pedagógicas opuestas.

  Este módulo introduce el TIEMPO como variable constitutiva. No registra
  timestamps (eso ya lo hace el middleware). Modela TRAYECTORIAS.

FUNDAMENTACIÓN TEÓRICA:
  - Vygotsky (1978): ZPD como territorio dinámico, no punto fijo
  - van Geert (1998): Modelos de sistemas dinámicos del desarrollo cognitivo
  - Bjork (1994): Desirable difficulties — la dificultad temporal produce
    aprendizaje más robusto a largo plazo
  - Wood, Bruner & Ross (1976): El andamiaje se retira cuando el aprendiz
    demuestra competencia PROGRESIVA, no tras N intentos
  - Bandura (1997): Autoeficacia como predictor de rendimiento — se
    manifiesta en patrones conductuales observables

INTEGRACIÓN:
  Patrón OBSERVER. Se invoca después de middleware.log_interaction().
  No modifica pre_process() ni post_process(). Si falla, el chatbot
  sigue funcionando como antes.

  middleware.log_interaction()
  → temporal_dynamics.update(interaction_data)
  → dashboard extendido con trayectorias

Autor: Diego Elvira Vásquez · Dimensión 3 para GENIE Learn CP25/152
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta
from collections import defaultdict
import math


# ═══════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════

@dataclass
class TemporalSnapshot:
    """Un punto en la serie temporal del estudiante."""
    timestamp: datetime
    bloom_level: int
    scaffolding_level: int
    copy_paste_score: float
    is_metacognitive: bool
    topic: str
    prompt_length: int
    response_latency_seconds: float  # tiempo desde respuesta anterior
    session_id: str  # agrupa interacciones de la misma sesión


@dataclass
class LearningVelocity:
    """Velocidad de aprendizaje en una ventana temporal."""
    window_start: datetime
    window_end: datetime
    n_interactions: int
    slope: float              # pendiente Bloom/interacción (velocidad)
    acceleration: float       # cambio de pendiente (aceleración)
    direction: str            # "ascending", "plateau", "descending", "erratic"
    mean_level: float
    confidence: float         # R² de la regresión


@dataclass
class PhaseTransition:
    """Transición de fase detectada en la trayectoria cognitiva."""
    timestamp: datetime
    student_id: str
    from_regime: str          # "accumulation", "construction", "consolidation", "regression"
    to_regime: str
    trigger_indicator: str    # qué métrica disparó la detección
    bloom_before: float
    bloom_after: float
    significance: float       # 0-1, cuán abrupto es el cambio


@dataclass
class EpistemicResilience:
    """Capacidad de recuperación cognitiva tras dificultad."""
    student_id: str
    n_drops: int              # veces que Bloom bajó ≥2 niveles
    n_recoveries: int         # veces que recuperó nivel previo
    mean_recovery_time: float # interacciones promedio para recuperar
    resilience_score: float   # 0-1: ratio recuperación/caída


@dataclass
class SelfEfficacyProxy:
    """
    Proxy conductual de autoeficacia (Bandura, 1997).
    No mide autoeficacia directamente (eso requiere cuestionario)
    sino señales conductuales que la literatura correlaciona con ella.
    """
    student_id: str
    prompt_complexity_trend: float   # tendencia de complejidad sintáctica
    persistence_after_error: float   # ratio de continuación tras dificultad
    autonomy_index: float            # ratio reformulaciones propias / peticiones directas
    language_stance: str             # "helpless", "neutral", "agentic"
    composite_score: float           # 0-1 compuesto


# ═══════════════════════════════════════════════════════════════
# MOTOR DE ANÁLISIS TEMPORAL
# ═══════════════════════════════════════════════════════════════

class TemporalDynamicsAnalyzer:
    """
    Analiza la dimensión temporal del aprendizaje.

    La hipótesis central: el PATRÓN temporal de interacción contiene
    más información pedagógica que cualquier métrica puntual.

    Un estudiante con Bloom medio 3.0 que lleva trayectoria ascendente
    está en un estado radicalmente distinto al que tiene Bloom medio 3.0
    con trayectoria descendente. El primero está aprendiendo; el segundo
    está olvidando o desmotivándose. Tratarlos igual es mala pedagogía.
    """

    # Umbrales calibrados para detección de regímenes
    # Estos son valores iniciales — se iterarán con datos reales del piloto
    VELOCITY_THRESHOLD_ASCENDING = 0.15    # pendiente > 0.15 = ascenso significativo
    VELOCITY_THRESHOLD_DESCENDING = -0.15  # pendiente < -0.15 = descenso significativo
    ACCELERATION_THRESHOLD = 0.05          # cambio de pendiente para transición de fase
    DROP_THRESHOLD = 2                     # caída de ≥2 niveles Bloom = "dificultad"
    RECOVERY_WINDOW = 5                    # interacciones para considerar "recuperación"
    SESSION_GAP_MINUTES = 30               # pausa > 30min = nueva sesión

    # Marcadores lingüísticos de autoeficacia (Bandura, 1997 + Connolly et al., 2018)
    HELPLESS_MARKERS = [
        "no entiendo nada", "no sé qué hacer", "me rindo",
        "es imposible", "no puedo", "no me sale", "me pierdo",
        "estoy perdido", "no tengo ni idea", "no sirvo para esto",
    ]
    AGENTIC_MARKERS = [
        "creo que", "mi hipótesis es", "he probado",
        "se me ocurre", "podría ser porque", "mi idea es",
        "he intentado", "voy a probar", "a ver si",
        "entonces lo que pasa es", "ya entiendo que",
    ]

    def __init__(self):
        self.student_timelines: dict[str, list[TemporalSnapshot]] = defaultdict(list)
        self.velocity_cache: dict[str, list[LearningVelocity]] = defaultdict(list)
        self.phase_transitions: dict[str, list[PhaseTransition]] = defaultdict(list)

    # ─── INGESTA ───

    def update(
        self,
        student_id: str,
        bloom_level: int,
        scaffolding_level: int,
        copy_paste_score: float,
        is_metacognitive: bool,
        topic: str,
        prompt_text: str,
        timestamp: Optional[datetime] = None,
    ):
        """
        Registra una nueva interacción y actualiza análisis temporal.
        Llamar después de middleware.log_interaction().
        """
        ts = timestamp or datetime.now()

        # Calcular latencia respecto a interacción anterior
        timeline = self.student_timelines[student_id]
        if timeline:
            latency = (ts - timeline[-1].timestamp).total_seconds()
        else:
            latency = 0.0

        # Determinar sesión
        if timeline and latency < self.SESSION_GAP_MINUTES * 60:
            session_id = timeline[-1].session_id
        else:
            session_id = f"s_{ts.strftime('%Y%m%d_%H%M')}"

        snapshot = TemporalSnapshot(
            timestamp=ts,
            bloom_level=bloom_level,
            scaffolding_level=scaffolding_level,
            copy_paste_score=copy_paste_score,
            is_metacognitive=is_metacognitive,
            topic=topic,
            prompt_length=len(prompt_text.split()),
            response_latency_seconds=latency,
            session_id=session_id,
        )
        timeline.append(snapshot)

        # Recalcular velocidad si hay suficientes datos
        if len(timeline) >= 5:
            velocity = self._compute_velocity(timeline[-5:])
            self.velocity_cache[student_id].append(velocity)

            # Detectar transiciones de fase
            if len(self.velocity_cache[student_id]) >= 2:
                transition = self._detect_phase_transition(
                    student_id,
                    self.velocity_cache[student_id][-2],
                    self.velocity_cache[student_id][-1],
                    ts
                )
                if transition:
                    self.phase_transitions[student_id].append(transition)

    # ─── VELOCIDAD DE APRENDIZAJE ───

    def _compute_velocity(self, window: list[TemporalSnapshot]) -> LearningVelocity:
        """
        Regresión lineal simple sobre niveles Bloom en la ventana.

        La pendiente es la velocidad de aprendizaje: cuántos niveles
        Bloom gana (o pierde) el estudiante por interacción.

        Fundamentación: van Geert (1998) modela el desarrollo cognitivo
        como sistema dinámico con tasa de crecimiento variable. La
        pendiente local es la tasa instantánea.
        """
        n = len(window)
        levels = [s.bloom_level for s in window]

        # Regresión lineal: y = ax + b
        x_mean = (n - 1) / 2
        y_mean = sum(levels) / n

        numerator = sum((i - x_mean) * (levels[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        slope = numerator / denominator if denominator > 0 else 0

        # R² para confianza
        y_pred = [y_mean + slope * (i - x_mean) for i in range(n)]
        ss_res = sum((levels[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((levels[i] - y_mean) ** 2 for i in range(n))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Clasificación de dirección
        if slope > self.VELOCITY_THRESHOLD_ASCENDING:
            direction = "ascending"
        elif slope < self.VELOCITY_THRESHOLD_DESCENDING:
            direction = "descending"
        elif abs(slope) <= 0.05 and max(levels) - min(levels) <= 1:
            direction = "plateau"
        else:
            direction = "erratic"

        # Aceleración: diferencia con pendiente anterior
        acceleration = 0.0  # se calcula externamente comparando velocidades

        return LearningVelocity(
            window_start=window[0].timestamp,
            window_end=window[-1].timestamp,
            n_interactions=n,
            slope=round(slope, 4),
            acceleration=acceleration,
            direction=direction,
            mean_level=round(y_mean, 2),
            confidence=round(max(0, r_squared), 3),
        )

    # ─── TRANSICIONES DE FASE ───

    def _detect_phase_transition(
        self,
        student_id: str,
        prev_velocity: LearningVelocity,
        curr_velocity: LearningVelocity,
        timestamp: datetime,
    ) -> Optional[PhaseTransition]:
        """
        Detecta transiciones de fase en la trayectoria cognitiva.

        Una transición de fase es un cambio cualitativo en el régimen
        de aprendizaje — no solo un cambio de nivel, sino un cambio
        en la DINÁMICA del cambio de nivel.

        Inspiración: teoría de catástrofes (Thom, 1975) aplicada a
        psicología del desarrollo (van der Maas & Molenaar, 1992).
        Los saltos cualitativos en el aprendizaje exhiben las mismas
        propiedades que las transiciones de fase en sistemas físicos:
        bimodalidad, histéresis, divergencia.
        """
        slope_change = curr_velocity.slope - prev_velocity.slope

        if abs(slope_change) < self.ACCELERATION_THRESHOLD:
            return None

        # Mapear a regímenes cognitivos
        regime_map = {
            "ascending": "construction",
            "plateau": "consolidation",
            "descending": "regression",
            "erratic": "exploration",
        }
        from_regime = regime_map.get(prev_velocity.direction, "unknown")
        to_regime = regime_map.get(curr_velocity.direction, "unknown")

        if from_regime == to_regime:
            return None

        return PhaseTransition(
            timestamp=timestamp,
            student_id=student_id,
            from_regime=from_regime,
            to_regime=to_regime,
            trigger_indicator=f"slope_change={slope_change:+.3f}",
            bloom_before=prev_velocity.mean_level,
            bloom_after=curr_velocity.mean_level,
            significance=min(abs(slope_change) / 0.5, 1.0),
        )

    # ─── RESILIENCIA EPISTÉMICA ───

    def compute_resilience(self, student_id: str) -> EpistemicResilience:
        """
        Mide la capacidad del estudiante de recuperarse tras dificultades.

        Bjork (1994) demostró que las dificultades deseables producen
        aprendizaje más duradero, pero SOLO si el aprendiz puede
        recuperarse. La resiliencia epistémica distingue entre
        dificultad productiva y dificultad destructiva.

        Un estudiante con alta resiliencia puede soportar scaffolding
        socrático intenso. Uno con baja resiliencia necesita modo más
        directo temporalmente para reconstruir autoeficacia.
        """
        timeline = self.student_timelines.get(student_id, [])
        if len(timeline) < 5:
            return EpistemicResilience(
                student_id=student_id,
                n_drops=0, n_recoveries=0,
                mean_recovery_time=0, resilience_score=0.5,
            )

        levels = [s.bloom_level for s in timeline]
        drops = []
        recoveries = []

        for i in range(1, len(levels)):
            if levels[i] - levels[i-1] <= -self.DROP_THRESHOLD:
                # Caída detectada
                pre_drop_level = levels[i-1]
                drops.append(i)

                # Buscar recuperación dentro de la ventana
                for j in range(i+1, min(i + self.RECOVERY_WINDOW + 1, len(levels))):
                    if levels[j] >= pre_drop_level - 1:
                        recoveries.append(j - i)  # tiempo de recuperación
                        break

        n_drops = len(drops)
        n_recoveries = len(recoveries)
        mean_recovery = sum(recoveries) / len(recoveries) if recoveries else 0

        resilience = n_recoveries / n_drops if n_drops > 0 else 0.5

        return EpistemicResilience(
            student_id=student_id,
            n_drops=n_drops,
            n_recoveries=n_recoveries,
            mean_recovery_time=round(mean_recovery, 1),
            resilience_score=round(resilience, 3),
        )

    # ─── PROXY DE AUTOEFICACIA ───

    def estimate_self_efficacy(self, student_id: str, recent_prompts: list[str]) -> SelfEfficacyProxy:
        """
        Estima autoeficacia computacional a partir de señales conductuales.

        Bandura (1997) identificó 4 fuentes de autoeficacia:
        1. Experiencias de dominio (éxito previo)
        2. Experiencias vicarias (ver a otros)
        3. Persuasión verbal (feedback)
        4. Estados fisiológicos (ansiedad)

        En un chatbot, solo podemos observar proxies de la 1ª y la 4ª:
        - Complejidad creciente de prompts = indicador de dominio percibido
        - Lenguaje de desamparo = indicador de baja autoeficacia
        - Persistencia tras dificultad = indicador de expectativa de éxito
        """
        timeline = self.student_timelines.get(student_id, [])
        if len(timeline) < 3 or not recent_prompts:
            return SelfEfficacyProxy(
                student_id=student_id,
                prompt_complexity_trend=0,
                persistence_after_error=0.5,
                autonomy_index=0.5,
                language_stance="neutral",
                composite_score=0.5,
            )

        # 1. Tendencia de complejidad de prompts (longitud como proxy)
        lengths = [s.prompt_length for s in timeline[-10:]]
        if len(lengths) >= 3:
            complexity_trend = self._simple_slope(lengths)
        else:
            complexity_trend = 0

        # 2. Persistencia: ¿continúa en el mismo tema tras dificultad?
        drops_with_continuation = 0
        drops_total = 0
        for i in range(1, len(timeline)):
            if timeline[i].bloom_level < timeline[i-1].bloom_level - 1:
                drops_total += 1
                if i + 1 < len(timeline) and timeline[i+1].topic == timeline[i].topic:
                    drops_with_continuation += 1
        persistence = drops_with_continuation / drops_total if drops_total > 0 else 0.5

        # 3. Autonomía: reformulaciones propias vs peticiones directas
        agentic_count = 0
        helpless_count = 0
        for prompt in recent_prompts:
            prompt_lower = prompt.lower()
            if any(m in prompt_lower for m in self.AGENTIC_MARKERS):
                agentic_count += 1
            if any(m in prompt_lower for m in self.HELPLESS_MARKERS):
                helpless_count += 1
        total_marked = agentic_count + helpless_count
        autonomy = agentic_count / total_marked if total_marked > 0 else 0.5

        # 4. Stance lingüístico
        if helpless_count > agentic_count * 2:
            stance = "helpless"
        elif agentic_count > helpless_count * 2:
            stance = "agentic"
        else:
            stance = "neutral"

        # Compuesto
        composite = (
            min(max(complexity_trend + 0.5, 0), 1) * 0.2 +
            persistence * 0.3 +
            autonomy * 0.3 +
            {"helpless": 0.1, "neutral": 0.5, "agentic": 0.9}[stance] * 0.2
        )

        return SelfEfficacyProxy(
            student_id=student_id,
            prompt_complexity_trend=round(complexity_trend, 4),
            persistence_after_error=round(persistence, 3),
            autonomy_index=round(autonomy, 3),
            language_stance=stance,
            composite_score=round(composite, 3),
        )

    # ─── SCAFFOLDING ADAPTATIVO BASADO EN TRAYECTORIA ───

    def recommend_scaffolding(self, student_id: str) -> dict:
        """
        Recomienda nivel de scaffolding basado en trayectoria temporal.

        Wood, Bruner & Ross (1976): el andamiaje se retira cuando el
        aprendiz demuestra competencia PROGRESIVA. No después de N intentos.

        Reglas de decisión:
        - Trayectoria ascendente + resilencia alta → mantener o reducir scaffolding
        - Trayectoria descendente + resilencia baja → aumentar soporte
        - Plateau prolongado → introducir perturbación (desirable difficulty)
        - Transición construction→regression → alerta docente
        """
        velocities = self.velocity_cache.get(student_id, [])
        resilience = self.compute_resilience(student_id)

        if not velocities:
            return {
                "recommended_level": 1,  # pista por defecto
                "confidence": 0.0,
                "reasoning": "Datos insuficientes. Scaffolding estándar por defecto.",
                "alert_teacher": False,
            }

        current = velocities[-1]
        timeline = self.student_timelines.get(student_id, [])
        current_scaffolding = timeline[-1].scaffolding_level if timeline else 1

        if current.direction == "ascending" and resilience.resilience_score > 0.5:
            # Progresando bien → reducir andamiaje gradualmente
            recommended = max(0, current_scaffolding - 1)
            reasoning = (
                f"Trayectoria ascendente (pendiente {current.slope:+.3f}) con "
                f"resiliencia alta ({resilience.resilience_score:.0%}). "
                f"El estudiante demuestra competencia progresiva. "
                f"Recomendación: retirar andamiaje gradualmente (Wood et al., 1976)."
            )
            alert = False

        elif current.direction == "descending" and resilience.resilience_score < 0.3:
            # Regresión + baja resiliencia → aumentar soporte
            recommended = min(3, current_scaffolding + 1)
            reasoning = (
                f"Trayectoria descendente (pendiente {current.slope:+.3f}) con "
                f"resiliencia baja ({resilience.resilience_score:.0%}). "
                f"La dificultad actual excede la capacidad de recuperación. "
                f"Recomendación: aumentar soporte para reconstruir autoeficacia (Bandura, 1997)."
            )
            alert = True

        elif current.direction == "plateau":
            # Meseta → perturbación controlada
            recommended = current_scaffolding
            reasoning = (
                f"Meseta en nivel Bloom {current.mean_level:.1f}. "
                f"El estudiante opera consistentemente pero no progresa. "
                f"Recomendación: introducir pregunta de nivel superior como "
                f"perturbación controlada (desirable difficulty, Bjork 1994)."
            )
            alert = len(velocities) > 3 and all(v.direction == "plateau" for v in velocities[-3:])

        else:
            recommended = current_scaffolding
            reasoning = (
                f"Trayectoria errática (σ alta). Puede indicar exploración activa "
                f"o confusión. Mantener nivel actual y monitorizar."
            )
            alert = False

        return {
            "recommended_level": recommended,
            "current_level": current_scaffolding,
            "confidence": current.confidence,
            "reasoning": reasoning,
            "alert_teacher": alert,
            "velocity": current.slope,
            "resilience": resilience.resilience_score,
            "direction": current.direction,
        }

    # ─── UTILIDADES ───

    def _simple_slope(self, values: list) -> float:
        """Regresión lineal simple sobre lista de valores."""
        n = len(values)
        if n < 2:
            return 0
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den > 0 else 0

    def get_student_summary(self, student_id: str) -> dict:
        """Resumen completo de la dimensión temporal para un estudiante."""
        timeline = self.student_timelines.get(student_id, [])
        velocities = self.velocity_cache.get(student_id, [])
        transitions = self.phase_transitions.get(student_id, [])
        resilience = self.compute_resilience(student_id)
        scaffolding = self.recommend_scaffolding(student_id)

        return {
            "student_id": student_id,
            "n_interactions": len(timeline),
            "n_sessions": len(set(s.session_id for s in timeline)) if timeline else 0,
            "current_velocity": {
                "slope": velocities[-1].slope if velocities else None,
                "direction": velocities[-1].direction if velocities else None,
                "confidence": velocities[-1].confidence if velocities else None,
            },
            "phase_transitions": [{
                "timestamp": t.timestamp.isoformat(),
                "from": t.from_regime,
                "to": t.to_regime,
                "significance": t.significance,
            } for t in transitions[-5:]],
            "resilience": {
                "score": resilience.resilience_score,
                "drops": resilience.n_drops,
                "recoveries": resilience.n_recoveries,
                "mean_recovery_time": resilience.mean_recovery_time,
            },
            "scaffolding_recommendation": scaffolding,
        }
