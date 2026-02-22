"""
ECOLOGÍA EPISTÉMICA — DIMENSIÓN 4
=====================================
Eje ecológico-relacional: el chatbot como artefacto mediador en un
sistema de actividad distribuido — no como herramienta aislada.

PROBLEMA QUE RESUELVE:
  El modelo actual trata la interacción estudiante-chatbot como sistema
  cerrado. Pero 60 estudiantes usando el mismo chatbot configurado por
  el mismo docente no son 60 sistemas independientes: son un ecosistema
  cognitivo acoplado. Las preguntas de un estudiante reflejan las
  dificultades del material, no solo sus limitaciones individuales.

  Si el 40% de los estudiantes pregunta variantes de la misma cuestión
  sobre recursividad, el problema no está en los estudiantes — está en
  el material o en la instrucción. Ningún analytics individual captura
  esto. Se necesita análisis de nivel ecológico.

FUNDAMENTACIÓN TEÓRICA:
  - Engeström (1987): Teoría de la Actividad expandida — la herramienta
    media la relación sujeto-objeto y está condicionada por reglas,
    comunidad, y división de trabajo.
  - Hutchins (1995): Cognición distribuida — la cognición en sistemas
    complejos no reside en individuos sino en el sistema sociotécnico.
  - Clark & Chalmers (1998): Mente extendida — el chatbot como extensión
    del sistema cognitivo del estudiante.
  - Emirbayer & Mische (1998): Agencia como capacidad de apropiarse y
    transformar las condiciones de acción — aplicada a la relación
    estudiante-chatbot-docente.
  - Latour (2005): Actor-Network Theory — el chatbot como actante no-humano
    con agencia en la red pedagógica.

INTEGRACIÓN:
  Patrón OBSERVER. Recibe datos de TODOS los estudiantes del mismo curso.
  Detecta patrones de nivel ecológico invisibles al análisis individual.

  middleware.log_interaction() × N estudiantes
  → epistemic_ecology.update_collective(all_interactions)
  → dashboard extendido con alertas de nivel curso

Autor: Diego Elvira Vásquez · Dimensión 4 para GENIE Learn CP25/152
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import math
import re


# ═══════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════

@dataclass
class CognitiveExtensionIndex:
    """
    Índice de Extensión Cognitiva (IEC).

    Mide el grado en que el estudiante usa el chatbot como extensión
    de su razonamiento (Clark & Chalmers, 1998) versus como oráculo
    de consulta puntual.

    IEC alto: cadenas de seguimiento, reformulaciones, verificaciones
    → el chatbot es parte del sistema cognitivo del estudiante.

    IEC bajo: preguntas atómicas sin relación entre sí
    → el chatbot es un buscador con interfaz conversacional.
    """
    student_id: str
    chain_ratio: float          # ratio de prompts que continúan el tema anterior
    reformulation_ratio: float  # ratio de reformulaciones sobre total
    verification_ratio: float   # ratio de preguntas de verificación
    mean_chain_length: float    # longitud media de cadenas de seguimiento
    composite_score: float      # 0-1 compuesto
    interpretation: str


@dataclass
class AgencyDistribution:
    """
    Distribución de agencia epistémica entre los tres actores
    del sistema: estudiante, chatbot, docente (a través de configs).

    Emirbayer & Mische (1998): la agencia no es propiedad de un actor
    sino una relación dinámica que se distribuye entre actores.
    """
    student_id: str
    student_agency: float   # 0-1: grado en que el estudiante dirige su aprendizaje
    chatbot_agency: float   # 0-1: grado en que el chatbot determina la respuesta
    teacher_agency: float   # 0-1: grado en que las configs docentes median la interacción
    balance_type: str        # "student_led", "chatbot_led", "teacher_mediated", "balanced"
    recommendation: str


@dataclass
class CollectiveDifficultySignal:
    """
    Señal de dificultad sistémica detectada a nivel de curso.

    Hutchins (1995): cuando múltiples agentes en un sistema distribuido
    convergen en el mismo problema, la causa es sistémica, no individual.
    """
    topic: str
    n_students_asking: int
    total_students: int
    convergence_ratio: float       # % de estudiantes que preguntan variantes
    mean_bloom_level: float        # nivel medio de las preguntas sobre este topic
    urgency: str                   # "low", "medium", "high", "critical"
    teacher_recommendation: str
    sample_prompts: list           # hasta 3 prompts representativos (anonimizados)


@dataclass
class EcologicalSnapshot:
    """Snapshot del estado ecológico del curso completo."""
    timestamp: datetime
    course_id: str
    n_active_students: int
    collective_difficulties: list  # CollectiveDifficultySignal
    agency_distribution_mean: dict # media de distribución de agencia
    extension_index_mean: float    # media de IEC del curso
    cohesion_score: float          # 0-1: qué tan similares son las interacciones
    diversity_score: float         # 0-1: diversidad temática del curso


# ═══════════════════════════════════════════════════════════════
# MOTOR DE ECOLOGÍA EPISTÉMICA
# ═══════════════════════════════════════════════════════════════

class EpistemicEcologyAnalyzer:
    """
    Analiza la dimensión ecológica de la interacción colectiva
    con el chatbot pedagógico.

    Metáfora operativa: el docente no solo ve a cada estudiante
    individualmente — ve el PAISAJE de aprendizaje del aula completa.
    Los analytics individuales son árboles; la ecología es el bosque.
    """

    # Marcadores de continuación temática
    CONTINUATION_MARKERS = [
        "entonces", "y si", "pero", "continuando",
        "sobre lo anterior", "respecto a eso", "en ese caso",
        "siguiendo con", "volviendo a", "sobre eso",
    ]

    # Marcadores de reformulación (señal de extensión cognitiva)
    REFORMULATION_MARKERS = [
        "o sea que", "es decir", "a ver si entiendo",
        "si he entendido bien", "reformulo", "en otras palabras",
        "lo que quieres decir es", "entonces lo que dices es",
    ]

    # Marcadores de verificación (señal de confianza calibrada)
    VERIFICATION_MARKERS = [
        "estás seguro", "es correcto", "puedo fiarme",
        "verificar", "contrastar", "seguro que",
    ]

    def __init__(self):
        self.course_interactions: dict[str, list[dict]] = defaultdict(list)
        # course_id -> lista de {student_id, topic, prompt, bloom, timestamp, ...}
        self.student_chains: dict[str, list[list[dict]]] = defaultdict(list)
        # student_id -> lista de cadenas de interacción

    # ─── ÍNDICE DE EXTENSIÓN COGNITIVA ───

    def compute_extension_index(
        self,
        student_id: str,
        interactions: list[dict],
    ) -> CognitiveExtensionIndex:
        """
        Calcula el Índice de Extensión Cognitiva (IEC).

        Clark & Chalmers (1998): si un artefacto es usado de forma
        regular y sistemática como soporte del razonamiento, se convierte
        en parte del sistema cognitivo del agente. El IEC mide cuánto
        el chatbot funciona como extensión cognitiva vs. herramienta puntual.
        """
        if len(interactions) < 3:
            return CognitiveExtensionIndex(
                student_id=student_id,
                chain_ratio=0, reformulation_ratio=0,
                verification_ratio=0, mean_chain_length=1,
                composite_score=0.5,
                interpretation="Datos insuficientes para estimar extensión cognitiva."
            )

        n = len(interactions)

        # Calcular ratio de continuaciones temáticas
        continuations = 0
        for i in range(1, n):
            prompt_lower = interactions[i].get("prompt", "").lower()
            if (interactions[i].get("topic") == interactions[i-1].get("topic") or
                    any(m in prompt_lower for m in self.CONTINUATION_MARKERS)):
                continuations += 1
        chain_ratio = continuations / (n - 1) if n > 1 else 0

        # Calcular ratio de reformulaciones
        reformulations = sum(
            1 for inter in interactions
            if any(m in inter.get("prompt", "").lower() for m in self.REFORMULATION_MARKERS)
        )
        reformulation_ratio = reformulations / n

        # Calcular ratio de verificaciones
        verifications = sum(
            1 for inter in interactions
            if any(m in inter.get("prompt", "").lower() for m in self.VERIFICATION_MARKERS)
        )
        verification_ratio = verifications / n

        # Longitud media de cadenas
        chains = self._extract_chains(interactions)
        mean_chain = sum(len(c) for c in chains) / len(chains) if chains else 1

        # Compuesto
        composite = (
            chain_ratio * 0.35 +
            reformulation_ratio * 0.25 +
            verification_ratio * 0.20 +
            min(mean_chain / 5, 1) * 0.20
        )

        # Interpretación
        if composite > 0.6:
            interpretation = (
                "Extensión cognitiva alta: el estudiante usa el chatbot como "
                "parte de su sistema de razonamiento. Las interacciones forman "
                "cadenas argumentativas, no consultas aisladas."
            )
        elif composite > 0.3:
            interpretation = (
                "Extensión cognitiva media: uso mixto entre consulta puntual "
                "y razonamiento extendido. El chatbot es herramienta útil "
                "pero no está integrado como extensión cognitiva."
            )
        else:
            interpretation = (
                "Extensión cognitiva baja: uso predominantemente como oráculo. "
                "Preguntas atómicas sin conexión entre sí. El chatbot no está "
                "siendo utilizado como herramienta de razonamiento."
            )

        return CognitiveExtensionIndex(
            student_id=student_id,
            chain_ratio=round(chain_ratio, 3),
            reformulation_ratio=round(reformulation_ratio, 3),
            verification_ratio=round(verification_ratio, 3),
            mean_chain_length=round(mean_chain, 1),
            composite_score=round(composite, 3),
            interpretation=interpretation,
        )

    def _extract_chains(self, interactions: list[dict]) -> list[list[dict]]:
        """Extrae cadenas de interacciones temáticamente conectadas."""
        if not interactions:
            return []
        chains = [[interactions[0]]]
        for i in range(1, len(interactions)):
            prompt_lower = interactions[i].get("prompt", "").lower()
            if (interactions[i].get("topic") == interactions[i-1].get("topic") or
                    any(m in prompt_lower for m in self.CONTINUATION_MARKERS)):
                chains[-1].append(interactions[i])
            else:
                chains.append([interactions[i]])
        return chains

    # ─── DISTRIBUCIÓN DE AGENCIA ───

    def compute_agency_distribution(
        self,
        student_id: str,
        interactions: list[dict],
        config: dict,
    ) -> AgencyDistribution:
        """
        Estima la distribución de agencia epistémica.

        Emirbayer & Mische (1998): la agencia es la capacidad de los
        actores de apropiarse y transformar las condiciones de su acción.
        En GENIE Learn, tres actores tienen agencia:
        - Estudiante: dirige preguntas, reformula, verifica
        - Chatbot: genera respuestas, determina contenido
        - Docente: configura scaffolding, limita, contextualiza

        La distribución ideal NO es máxima agencia del estudiante.
        Es agencia CALIBRADA según el nivel de competencia (Vygotsky).
        """
        if len(interactions) < 3:
            return AgencyDistribution(
                student_id=student_id,
                student_agency=0.33, chatbot_agency=0.33, teacher_agency=0.33,
                balance_type="balanced",
                recommendation="Datos insuficientes."
            )

        # Agencia del estudiante: formulaciones propias, metacognición, verificación
        student_signals = 0
        for inter in interactions:
            prompt = inter.get("prompt", "").lower()
            if inter.get("is_metacognitive", False):
                student_signals += 2
            if any(m in prompt for m in self.REFORMULATION_MARKERS):
                student_signals += 1
            if any(m in prompt for m in self.VERIFICATION_MARKERS):
                student_signals += 1
            if inter.get("bloom_level", 1) >= 4:
                student_signals += 1
        student_raw = student_signals / (len(interactions) * 3)

        # Agencia del docente: cuántas configs activas restringen/dirigen
        teacher_restrictions = 0
        if config.get("scaffolding_mode") == "socratic":
            teacher_restrictions += 2
        if config.get("block_direct_solutions", False):
            teacher_restrictions += 1
        if config.get("max_daily_prompts", 99) < 20:
            teacher_restrictions += 1
        if config.get("forced_hallucination_pct", 0) > 0:
            teacher_restrictions += 1
        if config.get("use_rag", False):
            teacher_restrictions += 1
        teacher_raw = teacher_restrictions / 6

        # Agencia del chatbot: lo que no es del estudiante ni del docente
        chatbot_raw = max(0, 1 - student_raw - teacher_raw)

        # Normalizar
        total = student_raw + chatbot_raw + teacher_raw
        if total == 0:
            total = 1
        student_norm = student_raw / total
        chatbot_norm = chatbot_raw / total
        teacher_norm = teacher_raw / total

        # Clasificar balance
        if student_norm > 0.5:
            balance = "student_led"
            recommendation = (
                "El estudiante dirige activamente su aprendizaje. "
                "Configuración apropiada para niveles avanzados."
            )
        elif chatbot_norm > 0.5:
            balance = "chatbot_led"
            recommendation = (
                "El chatbot domina la interacción. Considerar aumentar "
                "scaffolding para devolver agencia al estudiante."
            )
        elif teacher_norm > 0.5:
            balance = "teacher_mediated"
            recommendation = (
                "Las configuraciones docentes median fuertemente. "
                "Apropiado en fases iniciales; revisar si persiste."
            )
        else:
            balance = "balanced"
            recommendation = (
                "Distribución equilibrada de agencia. "
                "Monitorizar para detectar desbalances emergentes."
            )

        return AgencyDistribution(
            student_id=student_id,
            student_agency=round(student_norm, 3),
            chatbot_agency=round(chatbot_norm, 3),
            teacher_agency=round(teacher_norm, 3),
            balance_type=balance,
            recommendation=recommendation,
        )

    # ─── DIFICULTADES COLECTIVAS ───

    def detect_collective_difficulties(
        self,
        course_id: str,
        all_interactions: dict[str, list[dict]],
        # student_id -> interactions
    ) -> list[CollectiveDifficultySignal]:
        """
        Detecta dificultades sistémicas a nivel de curso.

        Hutchins (1995): cuando múltiples agentes en un sistema distribuido
        convergen en el mismo problema independientemente, la causa es
        sistémica, no individual.

        Si el 40% de los estudiantes pregunta variantes de recursividad,
        el docente debería reforzar ese tema en clase — no esperar a que
        cada estudiante lo resuelva individualmente con el chatbot.
        """
        total_students = len(all_interactions)
        if total_students < 3:
            return []

        # Agregar topics por estudiante (sin duplicar dentro del mismo estudiante)
        topic_students: dict[str, set] = defaultdict(set)
        topic_blooms: dict[str, list] = defaultdict(list)
        topic_prompts: dict[str, list] = defaultdict(list)

        for student_id, interactions in all_interactions.items():
            student_topics = set()
            for inter in interactions:
                for topic in inter.get("topics", [inter.get("topic", "otro")]):
                    if topic not in student_topics:
                        topic_students[topic].add(student_id)
                        student_topics.add(topic)
                    topic_blooms[topic].append(inter.get("bloom_level", 1))
                    if len(topic_prompts[topic]) < 10:
                        topic_prompts[topic].append(inter.get("prompt", ""))

        # Identificar topics con alta convergencia
        difficulties = []
        for topic, students in topic_students.items():
            n = len(students)
            ratio = n / total_students

            if ratio < 0.25:  # menos del 25% no es sistémico
                continue

            mean_bloom = sum(topic_blooms[topic]) / len(topic_blooms[topic])

            # Urgencia basada en convergencia + nivel cognitivo bajo
            if ratio > 0.6 and mean_bloom < 2.5:
                urgency = "critical"
                recommendation = (
                    f"URGENTE: {n}/{total_students} estudiantes ({ratio:.0%}) preguntan "
                    f"sobre '{topic}' con nivel cognitivo bajo (Bloom {mean_bloom:.1f}). "
                    f"Dificultad sistémica del material. Reforzar en clase."
                )
            elif ratio > 0.4:
                urgency = "high"
                recommendation = (
                    f"{n}/{total_students} estudiantes ({ratio:.0%}) preguntan sobre '{topic}'. "
                    f"Considerar refuerzo en próxima clase o material adicional."
                )
            elif ratio > 0.25:
                urgency = "medium"
                recommendation = (
                    f"{n}/{total_students} estudiantes ({ratio:.0%}) preguntan sobre '{topic}'. "
                    f"Monitorizar; posible necesidad de refuerzo."
                )
            else:
                urgency = "low"
                recommendation = f"Topic '{topic}' consultado por {n} estudiantes. Normal."

            # Anonimizar prompts de muestra
            samples = topic_prompts.get(topic, [])[:3]
            anonymized = [self._anonymize_prompt(p) for p in samples]

            difficulties.append(CollectiveDifficultySignal(
                topic=topic,
                n_students_asking=n,
                total_students=total_students,
                convergence_ratio=round(ratio, 3),
                mean_bloom_level=round(mean_bloom, 2),
                urgency=urgency,
                teacher_recommendation=recommendation,
                sample_prompts=anonymized,
            ))

        # Ordenar por urgencia
        urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        difficulties.sort(key=lambda d: urgency_order.get(d.urgency, 4))

        return difficulties

    def _anonymize_prompt(self, prompt: str) -> str:
        """Anonimiza un prompt eliminando información identificable."""
        # Eliminar nombres propios (simplificado)
        prompt = re.sub(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b', '[NOMBRE]', prompt)
        # Truncar a 100 chars
        if len(prompt) > 100:
            prompt = prompt[:97] + "..."
        return prompt

    # ─── SNAPSHOT ECOLÓGICO ───

    def compute_ecological_snapshot(
        self,
        course_id: str,
        all_interactions: dict[str, list[dict]],
        config: dict,
    ) -> EcologicalSnapshot:
        """
        Genera un snapshot completo del estado ecológico del curso.
        Combina todas las métricas de la dimensión 4.
        """
        difficulties = self.detect_collective_difficulties(course_id, all_interactions)

        # IEC medio del curso
        iec_scores = []
        agency_distributions = []
        for student_id, interactions in all_interactions.items():
            if len(interactions) >= 3:
                iec = self.compute_extension_index(student_id, interactions)
                iec_scores.append(iec.composite_score)
                agency = self.compute_agency_distribution(student_id, interactions, config)
                agency_distributions.append(agency)

        mean_iec = sum(iec_scores) / len(iec_scores) if iec_scores else 0

        # Distribución media de agencia
        if agency_distributions:
            mean_agency = {
                "student": round(sum(a.student_agency for a in agency_distributions) / len(agency_distributions), 3),
                "chatbot": round(sum(a.chatbot_agency for a in agency_distributions) / len(agency_distributions), 3),
                "teacher": round(sum(a.teacher_agency for a in agency_distributions) / len(agency_distributions), 3),
            }
        else:
            mean_agency = {"student": 0.33, "chatbot": 0.33, "teacher": 0.33}

        # Diversidad temática (Shannon entropy)
        all_topics = []
        for interactions in all_interactions.values():
            for inter in interactions:
                all_topics.extend(inter.get("topics", [inter.get("topic", "otro")]))
        topic_counts = Counter(all_topics)
        total_topics = sum(topic_counts.values())
        if total_topics > 0:
            entropy = -sum(
                (c / total_topics) * math.log2(c / total_topics)
                for c in topic_counts.values() if c > 0
            )
            max_entropy = math.log2(len(topic_counts)) if len(topic_counts) > 1 else 1
            diversity = entropy / max_entropy if max_entropy > 0 else 0
        else:
            diversity = 0

        # Cohesión: inverso de diversidad (alta cohesión = todos preguntan lo mismo)
        cohesion = 1 - diversity

        return EcologicalSnapshot(
            timestamp=datetime.now(),
            course_id=course_id,
            n_active_students=len(all_interactions),
            collective_difficulties=difficulties,
            agency_distribution_mean=mean_agency,
            extension_index_mean=round(mean_iec, 3),
            cohesion_score=round(cohesion, 3),
            diversity_score=round(diversity, 3),
        )
