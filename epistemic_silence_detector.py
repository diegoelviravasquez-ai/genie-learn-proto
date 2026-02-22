"""
EPISTEMIC SILENCE DETECTOR
═══════════════════════════════════════════════════════════════════════
Módulo diferencial #6 — El Detector de Silencio Epistémico

PROBLEMA QUE ATACA — EL PUNTO CIEGO UNIVERSAL:
════════════════════════════════════════════════
Todos los sistemas de learning analytics, incluyendo GENIE Learn, registran
lo que los estudiantes HACEN: prompts enviados, topics consultados, tiempo
de sesión, nivel Bloom de las preguntas.

Hay una categoría de dato completamente invisible en todos los sistemas
publicados hasta la fecha, incluyendo los 47 estudios del SLR de Topali
et al. (BIT, 2024):

    LO QUE EL ESTUDIANTE NO PREGUNTÓ CUANDO ESTADÍSTICAMENTE
    DEBERÍA HABERLO HECHO.

Un estudiante que encuentra un concepto difícil y pregunta → visible.
Un estudiante que encuentra el mismo concepto difícil, no reconoce que
no lo entiende, y avanza → INVISIBLE. Punto ciego. El dato más valioso.

FUNDAMENTO TEÓRICO:
────────────────────
1. Metacognition and Monitoring (Flavell, 1979; Dunning-Kruger, 1999)
   El "no saber que no se sabe" es la manifestación de metacognición
   deficiente: incapacidad de detectar los propios gaps de comprensión.
   La demanda de ayuda (help-seeking) requiere (a) detectar que hay un
   gap, (b) atribuirlo a comprensión insuficiente, (c) decidir que vale
   la pena pedir ayuda. El fracaso en (a) produce silencio donde
   debería haber pregunta — y ese silencio es datos.

2. Productive Failure y Desirable Difficulties (Kapur, 2008; Bjork, 1994)
   Kapur distingue entre "fracaso productivo" (el estudiante lucha pero
   aprende del proceso) y "fracaso silencioso" (el estudiante no sabe que
   está fracasando). Solo el segundo es sistémicamente dañino. Este módulo
   detecta el segundo tipo sin confundirlo con el primero.

3. Intercepts ausentes — principio SIGINT aplicado a educación
   En análisis de señales de inteligencia existe el concepto de
   "ausencia significativa": cuando una fuente que normalmente genera
   tráfico deja de hacerlo, esa ausencia es ella misma una señal.
   Buckingham Shum (2012) menciona el concepto de "traces" —
   huellas sin interpretación. Este módulo convierte la AUSENCIA de
   trazas en una traza con interpretación.

   La formulación más elegante: Sherlock Holmes, "Silver Blaze" (Conan
   Doyle, 1892). Inspector Gregory: "¿Hay algún otro punto al que llame
   mi atención?" Holmes: "Al curioso incidente del perro en la noche."
   Gregory: "El perro no hizo nada en la noche." Holmes: "Ese fue el
   curioso incidente." El GSIC mide lo que el perro ladra. Este módulo
   mide cuando el perro no ladra y debería haberlo hecho.

4. Aprendizaje maladaptativo y learned helplessness (Seligman, 1975)
   Tres patrones de silencio tienen interpretaciones distintas:
   - Silencio por competencia: el estudiante domina el topic (el perro
     no ladra porque no hay intruso real)
   - Silencio por desconexión: el estudiante ha dejado de intentarlo
     (learned helplessness — el perro ha aprendido que ladrar no sirve)
   - Silencio por metacognición deficiente: el estudiante no sabe que
     no sabe (el perro no detecta al intruso)
   Los tres producen el mismo dato observable (ausencia de pregunta) pero
   requieren intervenciones radicalmente distintas. El módulo los distingue
   usando el contexto de IEC, trayectoria de Bloom, y patrones de timing.

5. Estadística bayesiana aplicada a expectativas de pregunta
   No basta detectar ausencia: hay que modelar la EXPECTATIVA de presencia.
   Para eso necesitamos una distribución de referencia: cuántas preguntas
   genera este topic en estudiantes con este perfil. El módulo construye
   esa distribución a partir de datos históricos colectivos, permitiendo
   distinguir entre ausencia normal (este topic es fácil para este perfil)
   y ausencia anómala (este topic genera preguntas en el 80% de estudiantes
   similares, pero este no preguntó).

INNOVACIÓN METODOLÓGICA:
────────────────────────
Este módulo requiere cruzar datos históricos COLECTIVOS con el perfil
INDIVIDUAL — un paso de complejidad analítica que ningún sistema de LA
actual da porque están diseñados para responder a EVENTOS, no para
modelar la AUSENCIA de eventos.

Implementación: sin LLM, sin embeddings adicionales. Opera exclusivamente
sobre logs de interacción que ya existen en el sistema. Cero instrumentación
adicional al protocolo del piloto.

NO REQUIERE INSTRUMENTACIÓN ADICIONAL:
Usa los logs que ya existen:
- Lista de topics por sesión (ya calculada por middleware._detect_topics)
- Bloom por interacción (ya calculado por cognitive_analyzer)
- Timestamps (ya registrados por el sistema)
- Perfil epistémico (ya calculado por epistemic_autonomy)

Autor: Diego Elvira Vásquez · Prototipo CP25/152 · Feb 2026
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from datetime import datetime, timedelta
import math


# ═══════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TopicExposure:
    """
    Registro de la exposición de un estudiante a un topic.
    Incluye tanto lo que preguntó como lo que no preguntó.
    """
    student_id: str
    course_id: str
    topic: str
    timestamp: str

    # Lo que hizo el estudiante
    n_questions_on_topic: int = 0
    bloom_levels_on_topic: list = field(default_factory=list)
    first_contact_timestamp: str = ""

    # Contexto del curso en ese momento
    course_topic_density: float = 0.0      # cuántas preguntas genera este topic normalmente
    peer_n_questions_mean: float = 0.0     # media de preguntas de estudiantes similares

    # Señal de silencio
    silence_anomaly: float = 0.0           # 0 = normal, 1 = silencio crítico
    silence_type: str = "normal"           # normal | competence | disconnection | metacognitive_gap
    confidence_in_type: float = 0.0        # 0.0-1.0


@dataclass
class SilenceAlert:
    """
    Alerta de silencio epistémico para el docente.
    Generada cuando se detecta ausencia anómala persistente.
    """
    timestamp: str
    student_id: str
    course_id: str
    alert_level: str                        # low | medium | high | critical
    topics_affected: list                   # lista de topics con silencio anómalo
    n_consecutive_topics: int               # cuántos topics consecutivos

    # Interpretación
    silence_type: str                       # metacognitive_gap | disconnection | competence
    confidence: float                       # 0.0-1.0 confianza en la clasificación
    supporting_evidence: list               # qué datos apoyan esta clasificación

    # Mensaje para el docente (lenguaje pedagógico, no técnico)
    teacher_message: str = ""
    suggested_action: str = ""

    # Nota para el investigador
    researcher_note: str = ""

    # Para visualización en Streamlit
    probability_not_competence: float = 0.0


@dataclass
class CourseTopicProfile:
    """
    Perfil de un topic en un curso: cómo se comporta la población de estudiantes.
    Construido a partir de datos históricos para definir la "expectativa" de pregunta.
    """
    course_id: str
    topic: str
    n_students_exposed: int = 0
    n_students_who_asked: int = 0
    mean_questions_per_student: float = 0.0
    std_questions: float = 0.0
    mean_bloom_level: float = 0.0
    difficulty_score: float = 0.5           # 0 = fácil, 1 = muy difícil
    question_probability: float = 0.5       # probabilidad de que un estudiante pregunta sobre esto


# ═══════════════════════════════════════════════════════════════════════
# MOTOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

class EpistemicSilenceDetector:
    """
    Detecta silencio epistémico — la ausencia de pregunta cuando
    estadísticamente debería haberla habido.

    Flujo de uso:
    1. Registrar cada interacción del estudiante con un topic:
       detector.record_interaction(student_id, course_id, topic, bloom_level, ...)
    2. Registrar paso por un topic SIN pregunta:
       detector.record_topic_exposure_without_question(student_id, course_id, topic, ...)
    3. Consultar alertas activas:
       detector.get_active_alerts(course_id)
    4. Actualizar perfil histórico del curso (con datos de otros estudiantes):
       detector.update_course_topic_profile(course_id, topic, ...)
    """

    # Umbral de anomalía para generar alerta
    SILENCE_ANOMALY_THRESHOLD = 0.65       # > 65% de anomalía → alerta activa
    CONSECUTIVE_TOPICS_THRESHOLD = 3       # 3+ topics consecutivos → alerta alta
    CRITICAL_THRESHOLD = 5                 # 5+ → alerta crítica

    def __init__(self):
        self.topic_exposures: dict[str, list[TopicExposure]] = defaultdict(list)
        # student_id → list of TopicExposure

        self.course_profiles: dict[str, dict[str, CourseTopicProfile]] = defaultdict(dict)
        # course_id → topic → CourseTopicProfile

        self.active_alerts: dict[str, list[SilenceAlert]] = defaultdict(list)
        # course_id → list of SilenceAlert

        self.student_interaction_log: dict[str, list[dict]] = defaultdict(list)
        # student_id → list of {timestamp, topic, bloom, course_id}

    # ──────────────────────────────────────────────────────────────────
    # REGISTRO DE INTERACCIONES
    # ──────────────────────────────────────────────────────────────────

    def record_interaction(
        self,
        student_id: str,
        course_id: str,
        topic: str,
        bloom_level: str,
        bloom_weight: float,
        autonomy_score: float = 0.5,
        inter_prompt_latency: float = 60.0,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Registra UNA interacción (prompt enviado) sobre un topic.
        Actualiza el registro acumulado del estudiante en ese topic.
        """
        ts = timestamp or datetime.now().isoformat()

        self.student_interaction_log[student_id].append({
            "timestamp": ts,
            "topic": topic,
            "bloom_level": bloom_level,
            "bloom_weight": bloom_weight,
            "course_id": course_id,
            "autonomy_score": autonomy_score,
            "latency": inter_prompt_latency,
        })

        # Actualizar o crear registro de exposición al topic
        existing = next(
            (e for e in self.topic_exposures[student_id]
             if e.topic == topic and e.course_id == course_id),
            None
        )
        if existing:
            existing.n_questions_on_topic += 1
            existing.bloom_levels_on_topic.append(bloom_level)
        else:
            exposure = TopicExposure(
                student_id=student_id,
                course_id=course_id,
                topic=topic,
                timestamp=ts,
                n_questions_on_topic=1,
                bloom_levels_on_topic=[bloom_level],
                first_contact_timestamp=ts,
            )
            self.topic_exposures[student_id].append(exposure)

    def record_topic_exposure_without_question(
        self,
        student_id: str,
        course_id: str,
        topic: str,
        student_bloom_context: float,      # Bloom medio de las últimas 5 interacciones
        student_autonomy_score: float,
        student_iec: float = 0.5,          # Índice de extensión cognitiva
        timestamp: Optional[str] = None,
    ) -> Optional[SilenceAlert]:
        """
        Registra que el estudiante PASÓ POR UN TOPIC sin generar preguntas.

        Esto ocurre cuando:
        - El estudiante tuvo acceso al material del topic (p.ej., el docente
          asignó contenido sobre ese topic) pero no consultó al chatbot.
        - El chatbot detectó que el material menciona un topic pero el
          estudiante no hizo follow-up.

        Calcula la anomalía de silencio y genera alerta si supera umbral.
        """
        ts = timestamp or datetime.now().isoformat()

        # Comprobar si ya hay registro previo de preguntas sobre este topic
        existing_questions = next(
            (e for e in self.topic_exposures[student_id]
             if e.topic == topic and e.course_id == course_id),
            None
        )
        n_prior_questions = existing_questions.n_questions_on_topic if existing_questions else 0

        # Obtener perfil histórico del topic en el curso
        course_profile = self.course_profiles[course_id].get(topic)
        if course_profile:
            expected_density = course_profile.question_probability
            peer_mean = course_profile.mean_questions_per_student
            difficulty = course_profile.difficulty_score
        else:
            # Sin datos históricos: usar valores conservadores basados en Bloom
            # Topics difíciles (Bloom alto en el curso) tienen mayor expectativa de pregunta
            expected_density = 0.5 + (student_bloom_context / 12.0)  # base heurística
            peer_mean = 2.0
            difficulty = 0.5

        # Calcular anomalía de silencio
        anomaly = self._compute_silence_anomaly(
            n_actual_questions=n_prior_questions,
            expected_density=expected_density,
            peer_mean=peer_mean,
            student_bloom_context=student_bloom_context,
            student_autonomy_score=student_autonomy_score,
        )

        # Clasificar tipo de silencio
        silence_type, confidence = self._classify_silence_type(
            anomaly=anomaly,
            n_actual_questions=n_prior_questions,
            student_bloom_context=student_bloom_context,
            student_autonomy_score=student_autonomy_score,
            student_iec=student_iec,
        )

        # Registrar la exposición sin pregunta
        exposure = TopicExposure(
            student_id=student_id,
            course_id=course_id,
            topic=topic,
            timestamp=ts,
            n_questions_on_topic=n_prior_questions,
            course_topic_density=expected_density,
            peer_n_questions_mean=peer_mean,
            silence_anomaly=anomaly,
            silence_type=silence_type,
            confidence_in_type=confidence,
        )
        self.topic_exposures[student_id].append(exposure)

        # Generar alerta si supera umbral
        if anomaly >= self.SILENCE_ANOMALY_THRESHOLD and silence_type != "competence":
            return self._generate_alert(
                student_id, course_id, student_bloom_context, student_autonomy_score
            )
        return None

    # ──────────────────────────────────────────────────────────────────
    # CÁLCULO DE ANOMALÍA
    # ──────────────────────────────────────────────────────────────────

    def _compute_silence_anomaly(
        self,
        n_actual_questions: int,
        expected_density: float,
        peer_mean: float,
        student_bloom_context: float,
        student_autonomy_score: float,
    ) -> float:
        """
        Calcula cuán anómalo es el silencio observado.

        Lógica bayesiana simplificada:
        P(silencio anómalo) = P(silencio | no-dominio) × P(no-dominio | perfil)
                            / P(silencio)

        En términos operativos: un silencio es más anómalo cuando:
        1. La probabilidad histórica de que estudiantes similares pregunten es alta
        2. El estudiante tiene bajo historial de Bloom en topics similares
           (menos probable que sea dominio real)
        3. La autonomía del estudiante es baja (menos probable que sea autosuficiencia)

        Returns:
            anomaly score 0.0-1.0
        """
        if n_actual_questions > 0:
            # Ya preguntó algo → reducir anomalía proporcionalmente
            ratio = min(n_actual_questions / max(peer_mean, 1.0), 1.0)
            return max(0.0, expected_density * (1 - ratio))

        # Silencio total sobre el topic
        base_anomaly = expected_density

        # Ajuste por Bloom context: si el estudiante tiene Bloom alto en general,
        # es más probable que el silencio sea por competencia
        bloom_adjustment = (student_bloom_context - 3.0) / 6.0  # -0.5 a +0.5
        base_anomaly -= bloom_adjustment * 0.25

        # Ajuste por autonomía: alta autonomía → posible dominio → menos anomalía
        autonomy_adjustment = student_autonomy_score * 0.3
        base_anomaly -= autonomy_adjustment

        return max(0.0, min(1.0, base_anomaly))

    def _classify_silence_type(
        self,
        anomaly: float,
        n_actual_questions: int,
        student_bloom_context: float,
        student_autonomy_score: float,
        student_iec: float,
    ) -> tuple[str, float]:
        """
        Clasifica el tipo de silencio en tres categorías distintas
        con implicaciones pedagógicas radicalmente diferentes.

        Returns:
            (tipo, confianza)
        """
        if anomaly < self.SILENCE_ANOMALY_THRESHOLD:
            return "normal", 0.9

        # ── Silencio por competencia ──
        # Evidencias: Bloom alto, autonomía alta, IEC alto, algunas preguntas previas
        competence_score = 0.0
        if student_bloom_context >= 4.0:
            competence_score += 0.4
        if student_autonomy_score >= 0.65:
            competence_score += 0.3
        if student_iec >= 0.6:
            competence_score += 0.2
        if n_actual_questions > 0:
            competence_score += 0.1

        if competence_score >= 0.6:
            return "competence", competence_score

        # ── Silencio por desconexión (learned helplessness) ──
        # Evidencias: baja latencia en prompts previos (no reflexiona), IEC bajo,
        # patrón temporal errático
        disconnection_score = 0.0
        if student_autonomy_score < 0.2:
            disconnection_score += 0.35
        if student_iec < 0.2:
            disconnection_score += 0.3
        if student_bloom_context < 2.0:
            disconnection_score += 0.25

        # ── Silencio por gap metacognitivo ──
        # La categoría residual: ni competencia ni desconexión
        # El estudiante no sabe que no sabe
        meta_gap_score = 1.0 - max(competence_score, disconnection_score)

        scores = {
            "metacognitive_gap": meta_gap_score,
            "disconnection": disconnection_score,
            "competence": competence_score,
        }

        best_type = max(scores, key=scores.get)
        confidence = scores[best_type]

        return best_type, min(confidence, 0.95)

    # ──────────────────────────────────────────────────────────────────
    # GENERACIÓN DE ALERTAS
    # ──────────────────────────────────────────────────────────────────

    def _generate_alert(
        self,
        student_id: str,
        course_id: str,
        student_bloom_context: float,
        student_autonomy_score: float,
    ) -> Optional[SilenceAlert]:
        """
        Genera una alerta de silencio epistémico cuando el patrón es persistente.
        """
        # Obtener todos los silencios anómalos recientes del estudiante
        silent_exposures = [
            e for e in self.topic_exposures[student_id]
            if e.course_id == course_id and
               e.silence_anomaly >= self.SILENCE_ANOMALY_THRESHOLD and
               e.silence_type != "competence"
        ]

        if not silent_exposures:
            return None

        # Identificar topics afectados (sin duplicados, preservando orden)
        topics_seen = set()
        topics_affected = []
        for e in silent_exposures:
            if e.topic not in topics_seen:
                topics_seen.add(e.topic)
                topics_affected.append(e.topic)

        n_consecutive = len(topics_affected)

        # Nivel de alerta según número de topics afectados
        if n_consecutive >= self.CRITICAL_THRESHOLD:
            alert_level = "critical"
        elif n_consecutive >= self.CONSECUTIVE_TOPICS_THRESHOLD:
            alert_level = "high"
        elif n_consecutive >= 2:
            alert_level = "medium"
        else:
            alert_level = "low"

        # Tipo dominante de silencio
        type_counts = defaultdict(int)
        for e in silent_exposures:
            type_counts[e.silence_type] += 1
        dominant_type = max(type_counts, key=type_counts.get)
        dominant_confidence = sum(
            e.confidence_in_type for e in silent_exposures
            if e.silence_type == dominant_type
        ) / max(type_counts[dominant_type], 1)

        # Probabilidad de que NO sea competencia (la variable más importante para el docente)
        prob_not_competence = 1.0 - (
            sum(1 for e in silent_exposures if e.silence_type == "competence") /
            len(silent_exposures)
        )

        # Evidencias de apoyo
        evidence = [
            f"Topics sin preguntas: {', '.join(topics_affected[:5])}",
            f"Bloom medio del estudiante: {student_bloom_context:.1f}/6.0",
            f"Autonomía actual: {student_autonomy_score:.0%}",
            f"Tipo de silencio dominante: {dominant_type} "
            f"(confianza {dominant_confidence:.0%})",
        ]

        # Mensajes según tipo
        if dominant_type == "metacognitive_gap":
            teacher_message = (
                f"El estudiante {student_id} ha pasado por {n_consecutive} topic(s) "
                f"({', '.join(topics_affected[:3])}{', ...' if len(topics_affected) > 3 else ''}) "
                f"sin ninguna consulta. "
                f"Dado su perfil de Bloom ({student_bloom_context:.1f}/6) y autonomía actual "
                f"({student_autonomy_score:.0%}), la probabilidad de que esto sea dominio "
                f"real es del {(1.0 - prob_not_competence):.0%}. "
                f"Hipótesis alternativa: el estudiante no está detectando sus propios gaps."
            )
            suggested_action = (
                "Intervención recomendada: una pregunta diagnóstica directa sobre uno de "
                "los topics afectados. No activar scaffolding socrático todavía — primero "
                "verificar si hay comprensión real o la apariencia de ella."
            )
            researcher_note = (
                "Patrón consistente con metacognición deficiente (Flavell, 1979; "
                "Dunning-Kruger, 1999). El estudiante no genera demanda de ayuda "
                "porque no detecta el gap — el silencio es señal de ausencia de "
                "monitoreo metacognitivo, no de dominio. "
                "Clasificación: intercept ausente tipo II (silencio no informativo "
                "para el estudiante, altamente informativo para el analista)."
            )
        elif dominant_type == "disconnection":
            teacher_message = (
                f"El estudiante {student_id} muestra un patrón de baja interacción "
                f"consistente en {n_consecutive} topic(s) recientes. "
                f"El patrón es distinto al silencio por competencia: la autonomía "
                f"({student_autonomy_score:.0%}) y el Bloom contextual ({student_bloom_context:.1f}) "
                f"son insuficientes para explicar la ausencia de preguntas."
            )
            suggested_action = (
                "Posible desenganche del sistema. Considerar: (1) contacto directo "
                "con el estudiante fuera del chatbot, (2) revisar si hay dificultades "
                "externas al curso, (3) incrementar temporalmente el límite de prompts "
                "para reducir la fricción de uso."
            )
            researcher_note = (
                "Patrón consistente con learned helplessness (Seligman, 1975): "
                "el estudiante ha aprendido que el sistema no le es útil y ha dejado "
                "de intentarlo. IEC bajo y autonomía baja confirman ausencia de "
                "extensión cognitiva — el chatbot no está siendo usado como herramienta "
                "de pensamiento. Clasificación: silencio de evitación."
            )
        else:
            teacher_message = f"Patrón de silencio atípico detectado en {student_id}."
            suggested_action = "Revisión manual recomendada."
            researcher_note = "Tipo de silencio no concluyente."

        alert = SilenceAlert(
            timestamp=datetime.now().isoformat(),
            student_id=student_id,
            course_id=course_id,
            alert_level=alert_level,
            topics_affected=topics_affected,
            n_consecutive_topics=n_consecutive,
            silence_type=dominant_type,
            confidence=dominant_confidence,
            supporting_evidence=evidence,
            teacher_message=teacher_message,
            suggested_action=suggested_action,
            researcher_note=researcher_note,
            probability_not_competence=prob_not_competence,
        )

        self.active_alerts[course_id].append(alert)
        return alert

    # ──────────────────────────────────────────────────────────────────
    # ACTUALIZACIÓN DE PERFILES HISTÓRICOS
    # ──────────────────────────────────────────────────────────────────

    def update_course_topic_profile(
        self,
        course_id: str,
        topic: str,
        n_students_exposed: int,
        n_students_who_asked: int,
        mean_questions: float,
        std_questions: float,
        mean_bloom: float,
    ) -> CourseTopicProfile:
        """
        Actualiza el perfil histórico de un topic en un curso.
        Se invoca al final de cada unidad o sesión con datos agregados.
        """
        question_probability = (
            n_students_who_asked / n_students_exposed
            if n_students_exposed > 0 else 0.5
        )

        # Dificultad inferida: topics con Bloom alto y alta tasa de preguntas → más difíciles
        difficulty = min(1.0, (mean_bloom / 6.0) * question_probability * 1.5)

        profile = CourseTopicProfile(
            course_id=course_id,
            topic=topic,
            n_students_exposed=n_students_exposed,
            n_students_who_asked=n_students_who_asked,
            mean_questions_per_student=mean_questions,
            std_questions=std_questions,
            mean_bloom_level=mean_bloom,
            difficulty_score=difficulty,
            question_probability=question_probability,
        )

        self.course_profiles[course_id][topic] = profile
        return profile

    # ──────────────────────────────────────────────────────────────────
    # INTERFACES PÚBLICAS
    # ──────────────────────────────────────────────────────────────────

    def get_active_alerts(
        self,
        course_id: str,
        min_level: str = "medium",
    ) -> list[SilenceAlert]:
        """
        Devuelve alertas activas ordenadas por severidad.

        min_level: "low" | "medium" | "high" | "critical"
        """
        level_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        min_score = level_order.get(min_level, 1)

        alerts = [
            a for a in self.active_alerts[course_id]
            if level_order.get(a.alert_level, 0) >= min_score
        ]

        return sorted(
            alerts,
            key=lambda a: level_order.get(a.alert_level, 0),
            reverse=True,
        )

    def get_student_silence_report(self, student_id: str, course_id: str) -> dict:
        """
        Informe de silencio epistémico para un estudiante.
        Para el investigador — incluye clasificación técnica.
        """
        exposures = [
            e for e in self.topic_exposures[student_id]
            if e.course_id == course_id
        ]

        if not exposures:
            return {"student_id": student_id, "status": "no_data"}

        silent = [e for e in exposures if e.silence_anomaly >= self.SILENCE_ANOMALY_THRESHOLD]
        active = [e for e in exposures if e.n_questions_on_topic > 0]

        type_distribution = defaultdict(int)
        for e in silent:
            type_distribution[e.silence_type] += 1

        return {
            "student_id": student_id,
            "course_id": course_id,
            "n_topics_exposed": len(set(e.topic for e in exposures)),
            "n_topics_with_questions": len(set(e.topic for e in active)),
            "n_topics_with_silence_anomaly": len(silent),
            "silence_type_distribution": dict(type_distribution),
            "mean_anomaly_score": (
                sum(e.silence_anomaly for e in silent) / len(silent)
                if silent else 0.0
            ),
            "active_alerts": len([
                a for a in self.active_alerts[course_id]
                if a.student_id == student_id
            ]),
        }

    def get_course_silence_overview(self, course_id: str) -> dict:
        """
        Vista panorámica del silencio epistémico en el curso.
        Para el docente e investigador.
        """
        all_alerts = self.active_alerts[course_id]

        students_with_alerts = set(a.student_id for a in all_alerts)
        critical_students = set(
            a.student_id for a in all_alerts if a.alert_level == "critical"
        )

        topic_problem_map = defaultdict(int)
        for a in all_alerts:
            for t in a.topics_affected:
                topic_problem_map[t] += 1

        # Topics más problemáticos (generan más silencios anómalos)
        top_problematic = sorted(
            topic_problem_map.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return {
            "course_id": course_id,
            "n_students_with_alerts": len(students_with_alerts),
            "n_critical_students": len(critical_students),
            "critical_student_ids": list(critical_students),
            "total_alerts": len(all_alerts),
            "top_problematic_topics": top_problematic,
            "metacognitive_gap_cases": sum(
                1 for a in all_alerts if a.silence_type == "metacognitive_gap"
            ),
            "disconnection_cases": sum(
                1 for a in all_alerts if a.silence_type == "disconnection"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════
# DEMO / TEST
# ═══════════════════════════════════════════════════════════════════════

def _demo():
    """
    Demuestra el módulo simulando una secuencia de exposiciones a topics
    con distintos patrones de silencio.
    """
    detector = EpistemicSilenceDetector()
    course_id = "PROG101"

    # Poblar perfil histórico del curso
    for topic, (n_exp, n_asked, mean_q, mean_b) in {
        "bucles":      (60, 48, 3.2, 2.5),
        "recursión":   (60, 55, 5.1, 3.8),
        "punteros":    (60, 58, 6.2, 4.1),
        "funciones":   (60, 40, 2.8, 2.2),
        "variables":   (60, 25, 1.5, 1.5),
    }.items():
        detector.update_course_topic_profile(
            course_id, topic,
            n_students_exposed=n_exp,
            n_students_who_asked=n_asked,
            mean_questions=mean_q,
            std_questions=1.5,
            mean_bloom=mean_b,
        )

    print("=" * 65)
    print("DEMO: EPISTEMIC SILENCE DETECTOR")
    print("=" * 65)

    # ── Estudiante A: metacognitive gap ──
    print("\n[Estudiante A — gap metacognitivo esperado]")
    print("Historial: Bloom bajo (1.8), autonomía baja (0.22)")
    print("Escenario: pasa por recursión, punteros y funciones sin preguntar")

    for topic in ["recursión", "punteros", "funciones"]:
        alert = detector.record_topic_exposure_without_question(
            student_id="S_A",
            course_id=course_id,
            topic=topic,
            student_bloom_context=1.8,
            student_autonomy_score=0.22,
            student_iec=0.15,
        )
        if alert:
            print(f"\n  ⚠️  ALERTA [{alert.alert_level.upper()}] — {alert.silence_type}")
            print(f"  Topics afectados: {alert.topics_affected}")
            print(f"  P(no dominio): {alert.probability_not_competence:.0%}")
            print(f"  → {alert.teacher_message[:120]}...")
            print(f"  Acción: {alert.suggested_action[:90]}...")

    # ── Estudiante B: competencia real ──
    print("\n[Estudiante B — competencia real esperada]")
    print("Historial: Bloom alto (4.8), autonomía alta (0.82), IEC alto (0.75)")

    # Primero registrar algunas interacciones previas de calidad
    for i in range(5):
        detector.record_interaction(
            student_id="S_B", course_id=course_id, topic="bucles",
            bloom_level="analizar", bloom_weight=4.0,
            autonomy_score=0.82, inter_prompt_latency=180.0,
        )

    alert_b = detector.record_topic_exposure_without_question(
        student_id="S_B",
        course_id=course_id,
        topic="funciones",
        student_bloom_context=4.8,
        student_autonomy_score=0.82,
        student_iec=0.75,
    )
    if alert_b:
        print(f"  → Alerta: {alert_b.silence_type} ({alert_b.alert_level})")
    else:
        print("  → Sin alerta (silencio clasificado como competencia o normal)")

    # ── Panorama del curso ──
    print("\n" + "=" * 65)
    print("PANORAMA DEL CURSO:")
    overview = detector.get_course_silence_overview(course_id)
    print(f"  Estudiantes con alertas: {overview['n_students_with_alerts']}")
    print(f"  Casos críticos: {overview['n_critical_students']}")
    print(f"  Gap metacognitivo: {overview['metacognitive_gap_cases']} casos")
    print(f"  Desconexión: {overview['disconnection_cases']} casos")
    if overview['top_problematic_topics']:
        print(f"  Topics más problemáticos: "
              f"{', '.join(t for t, _ in overview['top_problematic_topics'][:3])}")

    # ── Informe individual ──
    print("\nINFORME ESTUDIANTE A:")
    report = detector.get_student_silence_report("S_A", course_id)
    print(f"  Topics expuestos: {report['n_topics_exposed']}")
    print(f"  Con preguntas: {report['n_topics_with_questions']}")
    print(f"  Con silencio anómalo: {report['n_topics_with_silence_anomaly']}")
    print(f"  Distribución de tipos: {report['silence_type_distribution']}")
    print(f"  Score medio de anomalía: {report['mean_anomaly_score']:.2f}")


if __name__ == "__main__":
    _demo()
