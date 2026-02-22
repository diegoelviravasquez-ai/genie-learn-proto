"""
CONSOLIDATION DETECTOR — Temporalidad Meso: La Ventana de 48-72 Horas
═══════════════════════════════════════════════════════════════════════════
Módulo diferencial Bloque 4 — De 0% a 60%

PROBLEMA QUE ATACA:
═══════════════════
Los learning analytics actuales miden lo que ocurre DURANTE la sesión.
Ningún sistema publicado modela lo que ocurre ENTRE sesiones — la ventana
de consolidación mnésica donde la información se convierte en conocimiento
integrado o se disipa sin dejar rastro significativo.

El indicador de consolidación NO es que el estudiante recuerde (eso es
memoria declarativa pura). El indicador es que el estudiante REFORMULE
lo aprendido con mayor profundidad cuando retorna. Si el martes preguntó
"¿cómo hago un bucle for?" (Bloom 2) y el jueves pregunta "¿por qué un
for es mejor que un while aquí?" (Bloom 4) sobre el mismo tema, la
diferencia de nivel Bloom entre sesiones es EVIDENCIA OBSERVABLE de
consolidación — el conocimiento se ha integrado en esquemas más ricos.

FUNDAMENTACIÓN TEÓRICA:
───────────────────────
1. Ebbinghaus (1885) — Curva del olvido: la retención decae exponencialmente
   sin repaso. La ventana de 24-72h es crítica: el conocimiento que sobrevive
   este período tiene alta probabilidad de pasar a memoria a largo plazo.

2. Bjork (1994) — Desirable difficulties y spacing effect: la recuperación
   espaciada (no la mera reexposición) produce aprendizaje más robusto.
   Nuestro indicador de consolidación mide implícitamente si el spacing
   natural del estudiante está produciendo el efecto deseado.

3. Karpicke & Roediger (2008) — Retrieval practice: el acto de recuperar
   información de memoria es más efectivo que re-estudiarla. Cuando un
   estudiante vuelve a preguntar sobre un tema en un nivel Bloom superior,
   está ejecutando retrieval practice espontáneo — la señal más fuerte
   de aprendizaje genuino.

4. Cepeda et al. (2006) — Meta-análisis del spacing effect: la distribución
   óptima de repaso está en torno al 10-20% del intervalo de retención
   deseado. Para retención a 30 días, el repaso óptimo es a los 3-6 días.
   Nuestra ventana de 48-168h captura exactamente este rango.

5. Tulving (1972) — Memoria episódica vs semántica: la consolidación
   convierte recuerdos episódicos (qué hice en clase) en conocimiento
   semántico (cómo funcionan los bucles). El bloom_delta positivo entre
   sesiones es evidencia indirecta de esta transición.

INNOVACIÓN:
───────────
Ningún sistema de chatbot educativo publicado implementa detección de
consolidación inter-sesión. Los sistemas existentes (VillaTorrano 2025,
García-Zarza 2025) miden autorregulación DENTRO de la sesión. Este módulo
es el primero en modelar la ventana temporal donde la información cruda
se transmuta en conocimiento estructurado — o se evapora.

No tenemos datos longitudinales reales todavía, pero la lógica de detección
está implementada y lista para operar sobre datos reales del piloto WP5.
Los datos sintéticos demuestran que el detector funciona correctamente
sobre los tres patrones teóricamente previsibles: consolidación positiva,
regresión, y ausencia (candidato a intervención de repaso espaciado).

INTEGRACIÓN:
────────────
    middleware.interaction_logs → consolidation_detector.ingest()
    consolidation_detector → teacher_notification_engine (sugerencias spaced_repetition)
    consolidation_detector → system_reflexivity (señales de consolidación agregadas)

Autor: Diego Elvira Vásquez · Bloque 4 CP25/152 GSIC/EMIC · Feb 2026
"""

import math
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTES — Ventanas temporales derivadas de la literatura
# ═══════════════════════════════════════════════════════════════════════

# Cepeda et al. (2006): para retención a 30 días, repaso óptimo 3-6 días
# Ebbinghaus (1885): la curva del olvido muestra caída más abrupta en primeras 24h
# Ventana observable: 24h (mínimo para distinguir sesiones) a 168h (7 días, máximo
# antes de que la ausencia se explique por factores externos al aprendizaje)
MIN_GAP_HOURS = 24.0    # Mínimo para considerar inter-sesión (no intra-sesión)
MAX_GAP_HOURS = 168.0   # Máximo antes de que el gap sea ambiguo
OPTIMAL_GAP_MIN = 48.0  # Inicio de ventana óptima de consolidación
OPTIMAL_GAP_MAX = 72.0  # Fin de ventana óptima de consolidación

# Umbrales de señal
STRONG_CONSOLIDATION_BLOOM_DELTA = 1.5   # +1.5 niveles Bloom = consolidación fuerte
WEAK_CONSOLIDATION_BLOOM_DELTA = 0.5     # +0.5 niveles = consolidación débil
REGRESSION_BLOOM_DELTA = -0.5            # -0.5 niveles = regresión

# Ventana para sugerencias de repaso espaciado
SPACED_REPETITION_WINDOW_START = 48.0    # Horas post-primera interacción
SPACED_REPETITION_WINDOW_END = 96.0      # Horas: la ventana se cierra


# ═══════════════════════════════════════════════════════════════════════
# BLOOM WEIGHT MAP — Para cálculos numéricos
# ═══════════════════════════════════════════════════════════════════════

BLOOM_WEIGHT_MAP = {
    "remember": 1.0,    "recordar": 1.0,
    "understand": 2.0,  "comprender": 2.0,
    "apply": 3.0,       "aplicar": 3.0,
    "analyze": 4.0,     "analizar": 4.0,
    "evaluate": 5.0,    "evaluar": 5.0,
    "create": 6.0,      "crear": 6.0,
}


# ═══════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TopicEncounter:
    """
    Un encuentro del estudiante con un topic específico.
    Unidad atómica del análisis de consolidación.
    """
    timestamp: datetime
    student_id: str
    topic: str
    bloom_level: float          # Peso numérico 1.0-6.0
    bloom_label: str            # Etiqueta textual del nivel
    prompt_text: str            # Prompt original (para análisis de reformulación)
    session_id: str = ""        # Para distinguir intra vs inter sesión
    scaffolding_level: int = 0  # Nivel de scaffolding en ese momento


@dataclass
class ConsolidationWindow:
    """
    Ventana de consolidación: par de encuentros sobre el mismo topic
    separados por el intervalo temporal relevante (24h-168h).

    Es la estructura central del módulo. Cada instancia captura un
    EVENTO DE CONSOLIDACIÓN OBSERVABLE (o su ausencia).
    """
    student_id: str
    topic: str

    # Primer encuentro
    first_encounter: TopicEncounter = None
    first_bloom: float = 0.0
    first_prompt: str = ""

    # Retorno
    return_encounter: TopicEncounter = None
    return_bloom: float = 0.0
    return_prompt: str = ""

    # Métricas temporales
    gap_hours: float = 0.0

    # Métricas de consolidación
    bloom_delta: float = 0.0        # return_bloom - first_bloom
    reformulation_depth: float = 0.0  # 0-1: similitud semántica baja + Bloom alto
    # → el estudiante pregunta DISTINTO y MÁS PROFUNDO

    # Clasificación
    consolidation_signal: str = "absent"  # strong | weak | absent | regression

    # Metadatos para investigador
    in_optimal_window: bool = False       # ¿El retorno ocurrió en 48-72h?
    scaffolding_delta: int = 0            # Cambio en nivel de scaffolding


@dataclass
class TopicConsolidationProfile:
    """Perfil de consolidación de un topic para un estudiante."""
    topic: str
    n_encounters: int = 0
    n_windows: int = 0
    mean_bloom_delta: float = 0.0
    max_bloom_delta: float = 0.0
    consolidation_rate: float = 0.0    # % de ventanas con señal positiva
    regression_rate: float = 0.0       # % de ventanas con señal negativa
    last_encounter: Optional[datetime] = None
    hours_since_last: float = 0.0
    needs_spaced_repetition: bool = False
    spaced_repetition_urgency: str = "none"  # none | low | medium | high


@dataclass
class TopicSuggestion:
    """Sugerencia de repaso espaciado para el teacher_notification_engine."""
    student_id: str
    topic: str
    hours_since_encounter: float
    original_bloom_level: float
    urgency: str                        # low | medium | high
    message: str
    theoretical_basis: str
    suggested_prompt: str               # Prompt sugerido para re-engagement


@dataclass
class StudentConsolidationReport:
    """Informe agregado de consolidación para un estudiante."""
    student_id: str
    consolidation_index: float = 0.0     # Índice global 0-1
    n_topics_tracked: int = 0
    n_consolidation_windows: int = 0
    strong_consolidations: int = 0
    weak_consolidations: int = 0
    regressions: int = 0
    absences: int = 0                    # Topics sin retorno en ventana óptima
    topic_profiles: Dict[str, TopicConsolidationProfile] = field(default_factory=dict)
    spaced_repetition_suggestions: List[TopicSuggestion] = field(default_factory=list)
    interpretation: str = ""
    researcher_note: str = ""


# ═══════════════════════════════════════════════════════════════════════
# MOTOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

class MemoryConsolidationTracker:
    """
    Detecta patrones de consolidación mnésica entre sesiones.

    Opera sobre la serie temporal de interacciones del middleware,
    agrupando por topic y detectando pares de encuentros separados
    por la ventana temporal relevante para la consolidación (24-168h).

    Para cada par, calcula:
    - bloom_delta: diferencia de nivel cognitivo entre sesiones
    - reformulation_depth: grado de reformulación del prompt
    - consolidation_signal: clasificación del patrón observado

    Uso:
        tracker = MemoryConsolidationTracker()
        tracker.ingest_interaction(log)  # tras cada interacción
        report = tracker.get_student_report("est_01")
        suggestions = tracker.generate_spaced_repetition_suggestions("est_01")
    """

    def __init__(self):
        # Almacén de encuentros por estudiante y topic
        # {student_id: {topic: [TopicEncounter, ...]}}
        self.encounters: Dict[str, Dict[str, List[TopicEncounter]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Ventanas de consolidación detectadas
        self.consolidation_windows: List[ConsolidationWindow] = []
        # Cache de reportes
        self._report_cache: Dict[str, StudentConsolidationReport] = {}
        self._cache_dirty: Dict[str, bool] = defaultdict(lambda: True)

    # ──────────────────────────────────────────────────────────────────
    # INGESTA DE DATOS
    # ──────────────────────────────────────────────────────────────────

    def ingest_interaction(
        self,
        student_id: str,
        timestamp: datetime,
        topics: List[str],
        bloom_level: float,
        bloom_label: str,
        prompt_text: str,
        session_id: str = "",
        scaffolding_level: int = 0,
    ):
        """
        Ingesta una interacción y busca pares de consolidación.

        Se invoca tras cada middleware.log_interaction() con los datos
        ya procesados por el cognitive_profiler.
        """
        for topic in topics:
            encounter = TopicEncounter(
                timestamp=timestamp,
                student_id=student_id,
                topic=topic,
                bloom_level=bloom_level,
                bloom_label=bloom_label,
                prompt_text=prompt_text,
                session_id=session_id,
                scaffolding_level=scaffolding_level,
            )
            self.encounters[student_id][topic].append(encounter)

            # Buscar si este nuevo encuentro cierra alguna ventana
            self._check_for_consolidation_window(student_id, topic, encounter)

        self._cache_dirty[student_id] = True

    def ingest_from_interaction_log(self, log, bloom_level: float = 2.0, bloom_label: str = "comprender"):
        """
        Conveniencia: ingesta directamente desde un InteractionLog del middleware.
        Requiere que el Bloom se pase como argumento (viene del cognitive_profiler).
        """
        try:
            ts = datetime.fromisoformat(log.timestamp) if isinstance(log.timestamp, str) else log.timestamp
        except (ValueError, AttributeError):
            ts = datetime.now()

        self.ingest_interaction(
            student_id=log.student_id,
            timestamp=ts,
            topics=log.detected_topics if log.detected_topics else ["otro"],
            bloom_level=bloom_level,
            bloom_label=bloom_label,
            prompt_text=log.prompt_raw,
            scaffolding_level=log.scaffolding_level,
        )

    # ──────────────────────────────────────────────────────────────────
    # DETECCIÓN DE VENTANAS DE CONSOLIDACIÓN
    # ──────────────────────────────────────────────────────────────────

    def _check_for_consolidation_window(
        self,
        student_id: str,
        topic: str,
        new_encounter: TopicEncounter,
    ):
        """
        Verifica si el nuevo encuentro forma una ventana de consolidación
        con algún encuentro anterior sobre el mismo topic.

        Lógica: recorremos los encuentros anteriores del más reciente al
        más antiguo. Si encontramos uno en la ventana temporal [24h, 168h],
        computamos la ventana de consolidación.
        """
        encounters = self.encounters[student_id][topic]
        if len(encounters) < 2:
            return

        # Recorrer encuentros anteriores (excluyendo el nuevo)
        for prev in reversed(encounters[:-1]):
            gap = (new_encounter.timestamp - prev.timestamp).total_seconds() / 3600.0

            # Solo nos interesan gaps en la ventana de consolidación
            if gap < MIN_GAP_HOURS:
                continue  # Misma sesión o demasiado cercano
            if gap > MAX_GAP_HOURS:
                break  # Demasiado lejano, los anteriores serán aún más

            # Encontramos un par válido — calcular consolidación
            window = self._compute_consolidation_window(prev, new_encounter, gap)
            self.consolidation_windows.append(window)

            # Solo tomamos el encuentro más reciente dentro de la ventana
            # (el más relevante para consolidación)
            break

    def _compute_consolidation_window(
        self,
        first: TopicEncounter,
        returning: TopicEncounter,
        gap_hours: float,
    ) -> ConsolidationWindow:
        """
        Calcula las métricas de consolidación para un par de encuentros.

        El bloom_delta captura el cambio de profundidad cognitiva.
        El reformulation_depth captura si el estudiante reformuló la pregunta
        (no repitió literalmente la misma — lo cual sería mera repetición,
        no consolidación).
        """
        bloom_delta = returning.bloom_level - first.bloom_level
        reformulation_depth = self._compute_reformulation_depth(
            first.prompt_text, returning.prompt_text, bloom_delta
        )

        # Clasificar señal de consolidación
        signal = self._classify_consolidation_signal(bloom_delta, reformulation_depth)

        in_optimal = OPTIMAL_GAP_MIN <= gap_hours <= OPTIMAL_GAP_MAX

        return ConsolidationWindow(
            student_id=first.student_id,
            topic=first.topic,
            first_encounter=first,
            first_bloom=first.bloom_level,
            first_prompt=first.prompt_text,
            return_encounter=returning,
            return_bloom=returning.bloom_level,
            return_prompt=returning.prompt_text,
            gap_hours=gap_hours,
            bloom_delta=bloom_delta,
            reformulation_depth=reformulation_depth,
            consolidation_signal=signal,
            in_optimal_window=in_optimal,
            scaffolding_delta=returning.scaffolding_level - first.scaffolding_level,
        )

    def _compute_reformulation_depth(
        self,
        first_prompt: str,
        return_prompt: str,
        bloom_delta: float,
    ) -> float:
        """
        Estima la profundidad de reformulación entre dos prompts.

        Concepto: consolidación genuina = el estudiante pregunta DISTINTO
        (baja similitud léxica) y MÁS PROFUNDO (alto bloom_delta).

        Implementación simplificada: similitud de Jaccard invertida × bloom_delta
        normalizado. En producción, usaríamos embeddings semánticos para
        capturar reformulación que preserva el tema pero cambia el nivel.

        Un Jaccard alto con bloom_delta alto significa que el estudiante
        repitió las mismas palabras pero desde un nivel superior — posible
        pero menos interesante que reformulación genuina.
        """
        if not first_prompt or not return_prompt:
            return 0.0

        # Tokenización simplificada
        tokens_first = set(first_prompt.lower().split())
        tokens_return = set(return_prompt.lower().split())

        # Jaccard: 0 = completamente distintos, 1 = idénticos
        intersection = tokens_first & tokens_return
        union = tokens_first | tokens_return

        if not union:
            return 0.0

        jaccard = len(intersection) / len(union)

        # Invertimos Jaccard: baja similitud → alta reformulación
        dissimilarity = 1.0 - jaccard

        # Normalizar bloom_delta a [0, 1]
        bloom_factor = min(max(bloom_delta / 4.0, 0.0), 1.0)

        # Reformulación profunda = pregunta distinta + nivel cognitivo superior
        # Ponderamos: 60% dissimilarity + 40% bloom_factor
        # La disimilitud pesa más porque sin reformulación, un bloom_delta
        # alto podría ser artefacto del clasificador, no del estudiante
        depth = 0.6 * dissimilarity + 0.4 * bloom_factor

        return round(min(depth, 1.0), 3)

    def _classify_consolidation_signal(
        self,
        bloom_delta: float,
        reformulation_depth: float,
    ) -> str:
        """
        Clasifica la señal de consolidación.

        Cuatro categorías:
        - strong: Bloom sube significativamente + reformulación alta
        - weak: Bloom sube ligeramente o reformulación moderada
        - absent: No hay cambio significativo
        - regression: Bloom baja — el conocimiento se evaporó
        """
        if bloom_delta >= STRONG_CONSOLIDATION_BLOOM_DELTA and reformulation_depth >= 0.3:
            return "strong"
        elif bloom_delta >= WEAK_CONSOLIDATION_BLOOM_DELTA:
            return "weak"
        elif bloom_delta <= REGRESSION_BLOOM_DELTA:
            return "regression"
        else:
            return "absent"

    # ──────────────────────────────────────────────────────────────────
    # CÓMPUTO DEL ÍNDICE DE CONSOLIDACIÓN
    # ──────────────────────────────────────────────────────────────────

    def compute_consolidation_index(self, student_id: str) -> float:
        """
        Índice global de consolidación para un estudiante.

        Promedio ponderado de los bloom_delta positivos en ventanas de
        consolidación, normalizado a [0, 1].

        Un estudiante con índice alto demuestra que el aprendizaje se
        está integrando entre sesiones. Uno con índice bajo o negativo
        sugiere que el conocimiento se evapora entre sesiones.

        Ponderación: las ventanas en la franja óptima (48-72h) pesan
        el doble que las de fuera, siguiendo Cepeda et al. (2006).
        """
        windows = [w for w in self.consolidation_windows if w.student_id == student_id]

        if not windows:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for w in windows:
            # Peso: 2.0 si está en la ventana óptima, 1.0 si no
            weight = 2.0 if w.in_optimal_window else 1.0

            # Solo deltas positivos contribuyen al índice de consolidación
            # Los negativos se capturan en regression_rate
            delta_contribution = max(w.bloom_delta, 0.0)

            # Bonus por reformulación profunda
            # La reformulación indica que no es mera repetición
            reformulation_bonus = w.reformulation_depth * 0.5

            weighted_sum += (delta_contribution + reformulation_bonus) * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        # Normalizar a [0, 1] — bloom_delta máximo teórico es 5 (de Bloom 1 a 6)
        # + reformulation_bonus máximo 0.5 → máximo teórico 5.5
        raw_index = weighted_sum / total_weight
        normalized = min(raw_index / 5.5, 1.0)

        return round(normalized, 3)

    # ──────────────────────────────────────────────────────────────────
    # DETECCIÓN DE PATRONES POR TOPIC
    # ──────────────────────────────────────────────────────────────────

    def detect_consolidation_patterns(
        self,
        student_id: str,
    ) -> List[ConsolidationWindow]:
        """
        Devuelve todas las ventanas de consolidación de un estudiante,
        ordenadas cronológicamente.
        """
        windows = [w for w in self.consolidation_windows if w.student_id == student_id]
        windows.sort(key=lambda w: w.first_encounter.timestamp)
        return windows

    def get_topic_profile(
        self,
        student_id: str,
        topic: str,
        reference_time: Optional[datetime] = None,
    ) -> TopicConsolidationProfile:
        """
        Perfil de consolidación de un topic específico para un estudiante.
        """
        ref_time = reference_time or datetime.now()
        encounters = self.encounters.get(student_id, {}).get(topic, [])
        windows = [
            w for w in self.consolidation_windows
            if w.student_id == student_id and w.topic == topic
        ]

        profile = TopicConsolidationProfile(
            topic=topic,
            n_encounters=len(encounters),
            n_windows=len(windows),
        )

        if encounters:
            last = max(encounters, key=lambda e: e.timestamp)
            profile.last_encounter = last.timestamp
            profile.hours_since_last = (ref_time - last.timestamp).total_seconds() / 3600.0

        if windows:
            deltas = [w.bloom_delta for w in windows]
            profile.mean_bloom_delta = round(statistics.mean(deltas), 2)
            profile.max_bloom_delta = max(deltas)

            positive = [w for w in windows if w.consolidation_signal in ("strong", "weak")]
            negative = [w for w in windows if w.consolidation_signal == "regression"]

            profile.consolidation_rate = round(len(positive) / len(windows), 3)
            profile.regression_rate = round(len(negative) / len(windows), 3)

        # Detectar necesidad de repaso espaciado
        if profile.n_encounters == 1 and profile.hours_since_last:
            if SPACED_REPETITION_WINDOW_START <= profile.hours_since_last <= SPACED_REPETITION_WINDOW_END:
                profile.needs_spaced_repetition = True
                profile.spaced_repetition_urgency = "high"
            elif profile.hours_since_last < SPACED_REPETITION_WINDOW_START:
                profile.needs_spaced_repetition = False
                profile.spaced_repetition_urgency = "none"
            elif profile.hours_since_last > SPACED_REPETITION_WINDOW_END:
                profile.needs_spaced_repetition = True
                profile.spaced_repetition_urgency = "medium"  # Ventana cerrándose

        return profile

    # ──────────────────────────────────────────────────────────────────
    # SUGERENCIAS DE REPASO ESPACIADO
    # ──────────────────────────────────────────────────────────────────

    def generate_spaced_repetition_suggestions(
        self,
        student_id: str,
        reference_time: Optional[datetime] = None,
    ) -> List[TopicSuggestion]:
        """
        Genera sugerencias de repaso espaciado para el teacher_notification_engine.

        Para cada topic donde el estudiante NO ha vuelto en la ventana
        óptima post-primera interacción, genera una sugerencia accionable.

        Formato diseñado para integrarse directamente como notificación
        de tipo 'spaced_repetition' en el TeacherNotificationEngine.
        """
        ref_time = reference_time or datetime.now()
        suggestions = []

        student_topics = self.encounters.get(student_id, {})

        for topic, encounters in student_topics.items():
            if not encounters:
                continue

            profile = self.get_topic_profile(student_id, topic, ref_time)

            if not profile.needs_spaced_repetition:
                continue

            first = encounters[0]
            hours = profile.hours_since_last

            # Determinar urgencia según posición en la curva de Ebbinghaus
            if hours >= SPACED_REPETITION_WINDOW_END:
                urgency = "high"
                time_msg = f"hace {hours:.0f} horas"
                urgency_msg = "La ventana de consolidación está cerrándose"
            elif hours >= SPACED_REPETITION_WINDOW_START:
                urgency = "medium"
                time_msg = f"hace {hours:.0f} horas"
                urgency_msg = "Momento óptimo para repaso espaciado"
            else:
                urgency = "low"
                time_msg = f"hace {hours:.0f} horas"
                urgency_msg = "La ventana de consolidación aún no ha abierto"

            # Generar prompt sugerido que escale el Bloom
            suggested_prompt = self._generate_escalation_prompt(topic, first.bloom_level)

            suggestion = TopicSuggestion(
                student_id=student_id,
                topic=topic,
                hours_since_encounter=round(hours, 1),
                original_bloom_level=first.bloom_level,
                urgency=urgency,
                message=(
                    f"El estudiante vio '{topic}' {time_msg} (Bloom {first.bloom_level:.0f}) "
                    f"y no ha vuelto. {urgency_msg} — considerar un prompt de repaso."
                ),
                theoretical_basis=(
                    "Ebbinghaus (1885): la retención decae exponencialmente sin repaso. "
                    "Cepeda et al. (2006): el repaso óptimo ocurre al 10-20% del intervalo "
                    "de retención deseado. Karpicke & Roediger (2008): retrieval practice "
                    "es más efectivo que re-estudio pasivo."
                ),
                suggested_prompt=suggested_prompt,
            )
            suggestions.append(suggestion)

        # Ordenar por urgencia (high primero)
        urgency_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: urgency_order.get(s.urgency, 99))

        return suggestions

    def _generate_escalation_prompt(self, topic: str, current_bloom: float) -> str:
        """
        Genera un prompt sugerido que escale un nivel Bloom por encima
        del último encuentro del estudiante.

        Diseñado para que el docente pueda enviarlo como prompt de
        re-engagement, o usarlo como inspiración para diseñar su propio
        prompt de repaso.
        """
        escalation_templates = {
            1.0: "¿Puedes explicar con tus palabras cómo funciona {topic}?",
            2.0: "¿Podrías dar un ejemplo concreto donde usarías {topic}?",
            3.0: "¿Qué diferencias encuentras entre {topic} y [alternativa]? ¿Cuándo elegirías uno sobre otro?",
            4.0: "Diseña una solución para [problema] usando {topic}. Justifica tu elección.",
            5.0: "¿Qué problemas o limitaciones ves en cómo {topic} resuelve [situación]?",
            6.0: "Propón una variación o mejora de {topic} para un caso que no se cubra en clase.",
        }

        # Escalar un nivel (o quedarse en 6 si ya está en máximo)
        target_bloom = min(current_bloom + 1.0, 6.0)

        # Buscar la plantilla más cercana
        closest_key = min(escalation_templates.keys(), key=lambda k: abs(k - target_bloom))
        template = escalation_templates[closest_key]

        return template.format(topic=topic)

    # ──────────────────────────────────────────────────────────────────
    # REPORTE COMPLETO POR ESTUDIANTE
    # ──────────────────────────────────────────────────────────────────

    def get_student_report(
        self,
        student_id: str,
        reference_time: Optional[datetime] = None,
    ) -> StudentConsolidationReport:
        """
        Informe completo de consolidación para un estudiante.
        Agrega todas las ventanas, perfiles por topic, y sugerencias.
        """
        if not self._cache_dirty.get(student_id, True) and student_id in self._report_cache:
            return self._report_cache[student_id]

        ref_time = reference_time or datetime.now()
        windows = self.detect_consolidation_patterns(student_id)
        topics = self.encounters.get(student_id, {})

        # Perfiles por topic
        topic_profiles = {}
        for topic in topics:
            topic_profiles[topic] = self.get_topic_profile(student_id, topic, ref_time)

        # Contadores de señal
        strong = sum(1 for w in windows if w.consolidation_signal == "strong")
        weak = sum(1 for w in windows if w.consolidation_signal == "weak")
        regression = sum(1 for w in windows if w.consolidation_signal == "regression")
        absent_count = sum(1 for p in topic_profiles.values() if p.needs_spaced_repetition)

        # Índice global
        consolidation_index = self.compute_consolidation_index(student_id)

        # Sugerencias
        suggestions = self.generate_spaced_repetition_suggestions(student_id, ref_time)

        # Interpretación narrativa
        interpretation = self._generate_interpretation(
            consolidation_index, strong, weak, regression, absent_count, len(windows)
        )

        # Nota para investigador
        researcher_note = self._generate_researcher_note(
            windows, topic_profiles, consolidation_index
        )

        report = StudentConsolidationReport(
            student_id=student_id,
            consolidation_index=consolidation_index,
            n_topics_tracked=len(topics),
            n_consolidation_windows=len(windows),
            strong_consolidations=strong,
            weak_consolidations=weak,
            regressions=regression,
            absences=absent_count,
            topic_profiles=topic_profiles,
            spaced_repetition_suggestions=suggestions,
            interpretation=interpretation,
            researcher_note=researcher_note,
        )

        self._report_cache[student_id] = report
        self._cache_dirty[student_id] = False
        return report

    def _generate_interpretation(
        self,
        index: float,
        strong: int,
        weak: int,
        regression: int,
        absences: int,
        total_windows: int,
    ) -> str:
        """Genera interpretación narrativa del patrón de consolidación."""
        if total_windows == 0 and absences == 0:
            return "Datos insuficientes: el estudiante no ha visitado ningún topic más de una vez."

        if total_windows == 0 and absences > 0:
            return (
                f"El estudiante ha visitado {absences} topic(s) sin volver a ninguno. "
                f"No hay evidencia de consolidación. Posible candidato para intervención "
                f"de repaso espaciado."
            )

        parts = []

        if index >= 0.5:
            parts.append(
                f"Consolidación FUERTE (índice {index:.2f}). El aprendizaje se está "
                f"integrando entre sesiones."
            )
        elif index >= 0.2:
            parts.append(
                f"Consolidación MODERADA (índice {index:.2f}). Hay indicios de integración "
                f"pero con variabilidad entre topics."
            )
        else:
            parts.append(
                f"Consolidación DÉBIL (índice {index:.2f}). El conocimiento parece disiparse "
                f"entre sesiones."
            )

        if strong > 0:
            parts.append(f"{strong} ventana(s) con consolidación fuerte (Bloom +1.5).")
        if regression > 0:
            parts.append(
                f"ATENCIÓN: {regression} ventana(s) con regresión. El estudiante retorna "
                f"a un nivel inferior al de su primer encuentro."
            )
        if absences > 0:
            parts.append(
                f"{absences} topic(s) sin retorno en ventana óptima — candidatos a repaso espaciado."
            )

        return " ".join(parts)

    def _generate_researcher_note(
        self,
        windows: List[ConsolidationWindow],
        topic_profiles: Dict[str, TopicConsolidationProfile],
        index: float,
    ) -> str:
        """Nota técnica para el investigador (no visible al docente)."""
        notes = []

        # Ventanas en franja óptima vs fuera
        if windows:
            optimal = sum(1 for w in windows if w.in_optimal_window)
            notes.append(
                f"De {len(windows)} ventanas, {optimal} están en la franja óptima (48-72h). "
                f"Ratio: {optimal/len(windows):.0%}."
            )

        # Topics con regresión consistente
        regression_topics = [
            t for t, p in topic_profiles.items() if p.regression_rate > 0.5
        ]
        if regression_topics:
            notes.append(
                f"Topics con regresión >50%: {', '.join(regression_topics)}. "
                f"Investigar si el material de estos topics es inadecuado o si "
                f"el scaffolding está produciendo efecto inverso."
            )

        # Patrón de reformulación
        if windows:
            mean_reform = statistics.mean([w.reformulation_depth for w in windows])
            notes.append(
                f"Profundidad media de reformulación: {mean_reform:.2f}. "
                f"{'Alta reformulación indica integración genuina, no repetición mecánica.' if mean_reform > 0.4 else 'Baja reformulación — posible repetición mecánica sin integración.'}"
            )

        return " | ".join(notes) if notes else "Sin notas adicionales."

    # ──────────────────────────────────────────────────────────────────
    # AGREGADOS PARA COHORTE
    # ──────────────────────────────────────────────────────────────────

    def get_cohort_consolidation_summary(self) -> Dict:
        """
        Resumen de consolidación del cohorte completo.
        Para alimentar al system_reflexivity con señales agregadas.
        """
        all_students = list(self.encounters.keys())

        if not all_students:
            return {"n_students": 0, "mean_consolidation_index": 0.0}

        indices = []
        total_strong = 0
        total_weak = 0
        total_regression = 0
        total_windows = 0

        for sid in all_students:
            report = self.get_student_report(sid)
            indices.append(report.consolidation_index)
            total_strong += report.strong_consolidations
            total_weak += report.weak_consolidations
            total_regression += report.regressions
            total_windows += report.n_consolidation_windows

        return {
            "n_students": len(all_students),
            "mean_consolidation_index": round(statistics.mean(indices), 3) if indices else 0.0,
            "median_consolidation_index": round(statistics.median(indices), 3) if indices else 0.0,
            "std_consolidation_index": round(statistics.stdev(indices), 3) if len(indices) > 1 else 0.0,
            "total_windows": total_windows,
            "total_strong": total_strong,
            "total_weak": total_weak,
            "total_regression": total_regression,
            "strong_rate": round(total_strong / total_windows, 3) if total_windows else 0.0,
            "regression_rate": round(total_regression / total_windows, 3) if total_windows else 0.0,
            "cohort_interpretation": (
                "CONSOLIDATION_HEALTHY" if statistics.mean(indices) > 0.3 else
                "CONSOLIDATION_AT_RISK" if statistics.mean(indices) > 0.15 else
                "CONSOLIDATION_CRITICAL"
            ) if indices else "NO_DATA",
        }


# ═══════════════════════════════════════════════════════════════════════
# GENERADOR DE DATOS SINTÉTICOS PARA DEMO
# ═══════════════════════════════════════════════════════════════════════

def generate_demo_data() -> MemoryConsolidationTracker:
    """
    Genera un estudiante sintético con 20 interacciones distribuidas
    en 10 días. Tres patrones diferenciados:

    1. BUCLES: consolidación positiva
       Día 1: "¿qué es un bucle for?" (Bloom 2)
       Día 3: "¿por qué un for es mejor que un while aquí?" (Bloom 4)
       → bloom_delta = +2, señal STRONG

    2. FUNCIONES: regresión
       Día 2: "¿cómo paso parámetros por referencia?" (Bloom 3)
       Día 7: "¿qué es una función?" (Bloom 1)
       → bloom_delta = -2, señal REGRESSION

    3. RECURSIÓN: ausencia (candidato para intervención)
       Día 4: "¿qué es recursión?" (Bloom 2)
       Sin retorno → candidato para spaced repetition

    4. ARRAYS: consolidación débil
       Día 1: "¿cómo recorro un array?" (Bloom 2)
       Día 5: "¿puedo usar un bucle para recorrer un array?" (Bloom 2.5)
       → bloom_delta = +0.5, señal WEAK

    5. VARIABLES: consolidación fuerte con reformulación profunda
       Día 2: "¿qué tipos de variables hay?" (Bloom 1)
       Día 4: "¿por qué conviene usar constantes en lugar de variables globales?" (Bloom 5)
       → bloom_delta = +4, reformulación alta, señal STRONG
    """
    tracker = MemoryConsolidationTracker()

    # Día base: hace 10 días
    base = datetime.now() - timedelta(days=10)
    student = "est_demo_01"

    # ── BUCLES: consolidación positiva ──
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=0, hours=10),
        topics=["bucles"],
        bloom_level=2.0,
        bloom_label="comprender",
        prompt_text="¿Qué es un bucle for y cómo funciona?",
        session_id="s01",
        scaffolding_level=0,
    )
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=0, hours=10, minutes=15),
        topics=["bucles"],
        bloom_level=2.0,
        bloom_label="comprender",
        prompt_text="¿Y el bucle while? Dame un ejemplo",
        session_id="s01",
        scaffolding_level=1,
    )
    # Día 3: retorna con nivel superior
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=2, hours=14),
        topics=["bucles"],
        bloom_level=4.0,
        bloom_label="analizar",
        prompt_text="¿Por qué un for es más eficiente que un while cuando conozco el número de iteraciones?",
        session_id="s03",
        scaffolding_level=0,
    )

    # ── FUNCIONES: regresión ──
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=1, hours=11),
        topics=["funciones"],
        bloom_level=3.0,
        bloom_label="aplicar",
        prompt_text="¿Cómo paso un array como parámetro a una función por referencia?",
        session_id="s02",
        scaffolding_level=0,
    )
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=1, hours=11, minutes=20),
        topics=["funciones"],
        bloom_level=3.5,
        bloom_label="aplicar",
        prompt_text="Diseña una función que ordene un array usando parámetros de configuración",
        session_id="s02",
        scaffolding_level=1,
    )
    # Día 7: regresa con nivel inferior
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=6, hours=9),
        topics=["funciones"],
        bloom_level=1.0,
        bloom_label="recordar",
        prompt_text="¿Qué es una función?",
        session_id="s07",
        scaffolding_level=0,
    )

    # ── RECURSIÓN: ausencia (sin retorno) ──
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=3, hours=16),
        topics=["recursión"],
        bloom_level=2.0,
        bloom_label="comprender",
        prompt_text="¿Qué es la recursión? ¿Cómo se diferencia de un bucle?",
        session_id="s04",
        scaffolding_level=0,
    )

    # ── ARRAYS: consolidación débil ──
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=0, hours=11),
        topics=["arrays"],
        bloom_level=2.0,
        bloom_label="comprender",
        prompt_text="¿Cómo recorro un array con un for?",
        session_id="s01",
        scaffolding_level=0,
    )
    # Día 5: retorna con leve mejora
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=4, hours=10),
        topics=["arrays"],
        bloom_level=2.5,
        bloom_label="comprender",
        prompt_text="¿Puedo usar un bucle for-each para recorrer arrays en Java?",
        session_id="s05",
        scaffolding_level=0,
    )

    # ── VARIABLES: consolidación fuerte con reformulación profunda ──
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=1, hours=10),
        topics=["variables"],
        bloom_level=1.0,
        bloom_label="recordar",
        prompt_text="¿Qué tipos de variables hay en programación?",
        session_id="s02",
        scaffolding_level=0,
    )
    # Día 4: retorna con salto grande y reformulación profunda
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=3, hours=12),
        topics=["variables"],
        bloom_level=5.0,
        bloom_label="evaluar",
        prompt_text="¿Por qué conviene usar constantes en lugar de variables globales para la mantenibilidad del código?",
        session_id="s04",
        scaffolding_level=0,
    )

    # ── Interacciones adicionales para enriquecer el perfil ──
    # Día 5: bucles a nivel evaluación
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=4, hours=15),
        topics=["bucles"],
        bloom_level=5.0,
        bloom_label="evaluar",
        prompt_text="¿Cuáles son las ventajas e inconvenientes de usar recursión frente a iteración para problemas de tipo divide y vencerás?",
        session_id="s05",
        scaffolding_level=0,
    )

    # Día 6: entrada/salida nivel bajo (nuevo topic)
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=5, hours=9),
        topics=["entrada/salida"],
        bloom_level=1.0,
        bloom_label="recordar",
        prompt_text="¿Cómo se usa System.out.println en Java?",
        session_id="s06",
        scaffolding_level=0,
    )

    # Día 8: entrada/salida con mejora
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=7, hours=11),
        topics=["entrada/salida"],
        bloom_level=3.0,
        bloom_label="aplicar",
        prompt_text="Escribe un programa que lea datos de un fichero, los procese y escriba el resultado en otro",
        session_id="s08",
        scaffolding_level=0,
    )

    # Día 8: depuración (nuevo)
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=7, hours=14),
        topics=["depuración"],
        bloom_level=2.0,
        bloom_label="comprender",
        prompt_text="Mi programa compila pero no da el resultado correcto, ¿cómo puedo depurarlo?",
        session_id="s08",
        scaffolding_level=0,
    )

    # Día 9: variables a nivel crear (consolidación máxima)
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=8, hours=10),
        topics=["variables"],
        bloom_level=6.0,
        bloom_label="crear",
        prompt_text="Diseña un sistema de tipado personalizado para un DSL que valide tipos en tiempo de compilación",
        session_id="s09",
        scaffolding_level=0,
    )

    # Día 10: funciones intento de recuperación
    tracker.ingest_interaction(
        student_id=student,
        timestamp=base + timedelta(days=9, hours=11),
        topics=["funciones"],
        bloom_level=2.0,
        bloom_label="comprender",
        prompt_text="¿Para qué sirven los parámetros de una función?",
        session_id="s10",
        scaffolding_level=0,
    )

    return tracker


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("MEMORY CONSOLIDATION TRACKER — Demo con datos sintéticos")
    print("=" * 70)

    tracker = generate_demo_data()
    student_id = "est_demo_01"

    # Reporte completo
    report = tracker.get_student_report(student_id)

    print(f"\n{'─' * 70}")
    print(f"ESTUDIANTE: {student_id}")
    print(f"{'─' * 70}")
    print(f"  Índice de consolidación global: {report.consolidation_index:.3f}")
    print(f"  Topics rastreados: {report.n_topics_tracked}")
    print(f"  Ventanas de consolidación: {report.n_consolidation_windows}")
    print(f"    → Fuertes: {report.strong_consolidations}")
    print(f"    → Débiles: {report.weak_consolidations}")
    print(f"    → Regresiones: {report.regressions}")
    print(f"    → Topics sin retorno: {report.absences}")
    print(f"\n  Interpretación: {report.interpretation}")
    print(f"\n  Nota investigador: {report.researcher_note}")

    # Detalle por topic
    print(f"\n{'─' * 70}")
    print("PERFILES POR TOPIC")
    print(f"{'─' * 70}")
    for topic, profile in report.topic_profiles.items():
        status = "✓" if profile.consolidation_rate > 0.5 else "⚠" if profile.regression_rate > 0 else "○"
        print(f"  {status} {topic:20s} | encuentros: {profile.n_encounters:2d} | "
              f"ventanas: {profile.n_windows:2d} | Δbloom medio: {profile.mean_bloom_delta:+.1f} | "
              f"consolidación: {profile.consolidation_rate:.0%} | "
              f"{'REPASO NECESARIO' if profile.needs_spaced_repetition else 'OK'}")

    # Ventanas detalladas
    print(f"\n{'─' * 70}")
    print("VENTANAS DE CONSOLIDACIÓN DETALLADAS")
    print(f"{'─' * 70}")
    windows = tracker.detect_consolidation_patterns(student_id)
    for w in windows:
        opt = "⏰" if w.in_optimal_window else "  "
        print(f"  {opt} {w.topic:15s} | gap: {w.gap_hours:5.0f}h | "
              f"Bloom: {w.first_bloom:.0f}→{w.return_bloom:.0f} (Δ{w.bloom_delta:+.1f}) | "
              f"reformulación: {w.reformulation_depth:.2f} | señal: {w.consolidation_signal}")

    # Sugerencias de repaso
    print(f"\n{'─' * 70}")
    print("SUGERENCIAS DE REPASO ESPACIADO")
    print(f"{'─' * 70}")
    suggestions = report.spaced_repetition_suggestions
    if suggestions:
        for s in suggestions:
            print(f"  [{s.urgency:6s}] {s.topic}: {s.message}")
            print(f"          Prompt sugerido: {s.suggested_prompt}")
    else:
        print("  (Sin sugerencias pendientes)")

    # Resumen cohorte
    print(f"\n{'─' * 70}")
    print("RESUMEN COHORTE")
    print(f"{'─' * 70}")
    cohort = tracker.get_cohort_consolidation_summary()
    for k, v in cohort.items():
        print(f"  {k}: {v}")

    print(f"\n{'=' * 70}")
    print("✓ MemoryConsolidationTracker operativo")
    print("  → Los learning analytics actuales miden lo que pasa en la sesión.")
    print("  → Nosotros modelamos lo que pasa ENTRE sesiones.")
    print("=" * 70)
