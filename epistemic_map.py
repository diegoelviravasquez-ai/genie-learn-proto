"""
EPISTEMIC MAP — Mapa de Conceptos y Prerequisitos
═══════════════════════════════════════════════════════════════
PROBLEMA QUE ATACA:
Visualizar el grafo de conceptos del curso y el dominio del estudiante
para detectar huecos epistemológicos que bloquean el avance.

FUNDAMENTACIÓN TEÓRICA:
- Prerequisite learning (Gagne, 1985)
- Knowledge graphs for adaptive learning (Nesbit & Adesope, 2006)
- Epistemic network analysis (Shaffer et al., 2009)

POSICIÓN EN EL ECOSISTEMA:
cognitive_engine → epistemic_map (usa Bloom/topics)
epistemic_map → app.py (visualización)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

logger = logging.getLogger(__name__)

# Grafo de conceptos: nodos = temas, aristas = prerequisitos (de → a)
CONCEPT_GRAPH = {
    "variables": [],
    "condicionales": ["variables"],
    "bucles": ["variables", "condicionales"],
    "arrays": ["variables", "bucles"],
    "funciones": ["variables", "condicionales", "bucles"],
    "recursión": ["funciones", "condicionales"],
    "objetos": ["funciones", "arrays"],
}

TOPICS_ORDER = ["variables", "condicionales", "bucles", "arrays", "funciones", "recursión", "objetos"]


@dataclass
class ConceptNode:
    """Nodo del mapa epistémico."""
    topic: str
    mastery_pct: float  # 0-100
    question_count: int
    color: str  # red (0-30), yellow (30-70), green (70-100)
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class EpistemicGap:
    """Hueco epistémico: prerequisito no dominado."""
    topic: str
    missing_prereq: str
    missing_mastery: float
    recommendation: str


@dataclass
class EpistemicMapResult:
    """Resultado del análisis del mapa epistémico."""
    nodes: List[ConceptNode]
    edges: List[tuple]  # (from, to)
    gaps: List[EpistemicGap]
    insight: Optional[str] = None
    recommendation: Optional[str] = None


class EpistemicMap:
    """
    Mapa epistémico: grafo de conceptos con dominio del estudiante.
    
    Uso:
        map_ = EpistemicMap()
        result = map_.analyze(interactions, student_id="est_01")
    """

    def __init__(self):
        self.concept_graph = CONCEPT_GRAPH
        self.topics_order = TOPICS_ORDER

    def _mastery_color(self, pct: float) -> str:
        if pct <= 0.30:
            return "#e53935"  # rojo
        if pct <= 0.70:
            return "#ffc107"  # amarillo
        return "#43a047"  # verde

    def _compute_mastery(
        self,
        topic: str,
        interactions: List[Dict],
        student_id: Optional[str] = None,
    ) -> float:
        """
        Estima dominio 0-1 basado en Bloom de interacciones sobre el tema.
        Bloom medio alto → dominio alto.
        """
        topic_interactions = [
            i for i in interactions
            if topic in i.get("detected_topics", []) or topic in i.get("topics", [])
        ]
        if student_id:
            topic_interactions = [
                i for i in topic_interactions
                if i.get("student_id") == student_id
            ]
        if not topic_interactions:
            return 0.0
        bloom_sum = 0.0
        n = 0
        for i in topic_interactions:
            bl = i.get("bloom_level") or i.get("bloom")
            if bl is not None:
                bloom_sum += int(bl)
                n += 1
        if n == 0:
            return 0.0
        # Bloom 0-6, normalizado a 0-1 (4+ = dominio razonable)
        avg_bloom = bloom_sum / n
        return min(1.0, avg_bloom / 5.0)  # 5/5 ≈ 1.0

    def _count_questions(self, topic: str, interactions: List[Dict], student_id: Optional[str] = None) -> int:
        count = 0
        for i in interactions:
            topics = i.get("detected_topics", []) or i.get("topics", [])
            if topic in topics:
                if student_id and i.get("student_id") != student_id:
                    continue
                count += 1
        return count

    def analyze(
        self,
        interactions: List[Dict],
        student_id: Optional[str] = None,
    ) -> EpistemicMapResult:
        """
        Analiza interacciones y construye mapa con dominio y huecos.
        """
        nodes: List[ConceptNode] = []
        edges: List[tuple] = []
        gaps: List[EpistemicGap] = []
        mastery_by_topic: Dict[str, float] = {}

        for topic in self.topics_order:
            mastery = self._compute_mastery(topic, interactions, student_id)
            mastery_by_topic[topic] = mastery
            qcount = self._count_questions(topic, interactions, student_id)
            prereqs = self.concept_graph.get(topic, [])
            for p in prereqs:
                edges.append((p, topic))

            nodes.append(ConceptNode(
                topic=topic,
                mastery_pct=mastery * 100,
                question_count=qcount,
                color=self._mastery_color(mastery),
                prerequisites=prereqs,
            ))

        # Detectar huecos: tema con prerequisitos no dominados
        insight = None
        recommendation = None
        for topic in self.topics_order:
            prereqs = self.concept_graph.get(topic, [])
            for p in prereqs:
                p_mastery = mastery_by_topic.get(p, 0.0)
                t_mastery = mastery_by_topic.get(topic, 0.0)
                if p_mastery < 0.5 and t_mastery < 0.5:
                    gaps.append(EpistemicGap(
                        topic=topic,
                        missing_prereq=p,
                        missing_mastery=p_mastery * 100,
                        recommendation=f"Reforzar {p} antes de continuar con {topic}",
                    ))
                    if not recommendation:
                        recommendation = f"Reforzar {p} antes de continuar con {topic}"
                    if not insight:
                        insight = f"El estudiante evita {topic} porque no domina {p}"

        return EpistemicMapResult(
            nodes=nodes,
            edges=edges,
            gaps=gaps,
            insight=insight,
            recommendation=recommendation or "Sin recomendaciones específicas.",
        )

    def get_interactions_by_topic(
        self,
        topic: str,
        interactions: List[Dict],
        student_id: Optional[str] = None,
    ) -> List[Dict]:
        """Devuelve interacciones filtradas por tema."""
        result = []
        for i in interactions:
            topics = i.get("detected_topics", []) or i.get("topics", [])
            if topic in topics:
                if student_id and i.get("student_id") != student_id:
                    continue
                result.append(i)
        return result
