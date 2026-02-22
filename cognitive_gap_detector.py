"""
COGNITIVE GAP DETECTOR — Unknown Unknowns Elicitor
═══════════════════════════════════════════════════════════════════════
Módulo diferencial #7 — El Elicitador Cognitivo

PROBLEMA QUE ATACA — EL PUNTO CIEGO DE SEGUNDO ORDEN:
══════════════════════════════════════════════════════
El Bloque 6 (Epistemic Silence Detector) detecta CUANDO un estudiante
calla donde estadísticamente debería haber preguntado. Este módulo opera
un nivel más arriba: identifica QUÉ DEBERÍA ESTAR PREGUNTANDO y genera
sondas que hacen emerger esos gaps en la conversación natural.

La distinción es operacionalmente crítica:
- Silencio Detector: "Este estudiante no preguntó sobre X" (observación)
- Gap Detector: "X es prerrequisito de Y que sí está usando; la ausencia
  de X amenaza la integridad de su comprensión de Y" (diagnóstico)
- Probe Generator: "Pregúntale esto para verificar si X es un gap real
  o conocimiento adquirido por otra vía" (intervención)

FUNDAMENTO TEÓRICO — TRES TRADICIONES CONVERGENTES:
═══════════════════════════════════════════════════
1. Knowledge Elicitation (Firlej & Hellens, 1991)
   El Knowledge Elicitation Handbook describe técnicas para hacer manifiesto
   el conocimiento tácito del experto — "unclustering" de schemas automatizados.
   La inversión pedagógica: esas mismas técnicas detectan AUSENCIA de schemas
   en el aprendiz. Preguntas de contraste, límite, e hipotéticas se calibran
   para hacer emerger lo que no se formó, no lo que se automatizó.

   Técnicas adaptadas:
   - Contraste: "¿En qué se diferencia X de Y?" → detecta si el estudiante
     distingue conceptos que debería distinguir
   - Límite: "¿Cuándo deja de funcionar X?" → detecta comprensión profunda
     vs. memorización superficial
   - Hipotética: "¿Qué pasaría si X no existiera?" → detecta comprensión
     del rol funcional del concepto en el sistema

2. Key Intelligence Gaps — KIG Framework (CIA, DIA)
   En OSINT y análisis de señales: dados los objetivos de la misión (syllabus)
   y la información disponible (interacciones del estudiante), ¿qué información
   falta y cuál de esas ausencias es CRÍTICA para la misión?
   
   La formalización es directa:
   - Misión = objetivos de aprendizaje del curso
   - Información disponible = topics visitados + nivel Bloom alcanzado
   - Gap = topic del corpus no visitado
   - Gap CRÍTICO = gap que amenaza la integridad de topics ya visitados

3. Zone of Proximal Development (Vygotsky, 1978)
   Los gaps de adyacencia no son errores — son la ZPD del estudiante.
   El sistema identifica lo que está "al alcance" del estudiante dado su
   mapa actual, convirtiendo la detección de gaps en una brújula pedagógica.

NO REQUIERE INSTRUMENTACIÓN ADICIONAL:
Usa los logs que ya produce middleware.py (InteractionLog con detected_topics)
y los chunks que ya existen en rag_pipeline.py.

Autor: Diego Elvira Vásquez · Prototipo CP25/152 · Feb 2026
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from datetime import datetime, timedelta
import random
import math


# ═══════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class KnowledgeNode:
    """
    Nodo del grafo de conocimiento del corpus.
    Cada nodo es un topic/concepto extraído del RAG pipeline.
    
    Categoría ontológica: Entidad conceptual con relaciones de dependencia.
    La topología del grafo codifica la estructura epistémica del dominio:
    prerrequisitos (aristas dirigidas), co-ocurrencia (aristas no dirigidas),
    y dificultad estimada (propiedad del nodo).
    """
    topic: str
    chunk_indices: list = field(default_factory=list)
    # Qué chunks del RAG cubren este topic
    
    prerequisite_topics: list = field(default_factory=list)
    # Topics que hay que dominar ANTES de este
    
    dependent_topics: list = field(default_factory=list)
    # Topics que DEPENDEN de este (inverso de prerequisite)
    
    co_occurring_topics: list = field(default_factory=list)
    # Topics que aparecen en los mismos chunks (relación de adyacencia)
    
    difficulty_estimate: float = 0.5
    # 0.0 = trivial, 1.0 = máxima dificultad
    # Basada en nivel Bloom promedio de preguntas históricas sobre este topic
    
    bloom_histogram: dict = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0})
    # Distribución de niveles Bloom de todas las preguntas históricas sobre este topic


@dataclass
class StudentTopicState:
    """
    Estado de un estudiante respecto a UN topic específico.
    Subgrafo visitado: cada nodo visitado tiene este registro.
    """
    topic: str
    visited: bool = False
    visit_count: int = 0
    max_bloom_level: int = 0
    bloom_levels_seen: list = field(default_factory=list)
    last_visited: str = ""
    chunks_consumed: set = field(default_factory=set)
    # Indices de chunks RAG que el estudiante ha recibido para este topic
    
    first_visit: str = ""
    # Timestamp de primera interacción con este topic


@dataclass
class CognitiveGap:
    """
    Un gap detectado en el mapa cognitivo del estudiante.
    
    gap_type codifica la NATURALEZA del gap:
    - prerequisite_gap: construyendo sobre cimientos no verificados
    - adjacency_gap: zona proximal no explorada
    - metacognitive_gap: silencio donde debería haber habido preguntas
    
    severity codifica el RIESGO:
    - critical: amenaza directa a topics que el estudiante ya usa
    - moderate: oportunidad de aprendizaje perdida
    - low: gap periférico sin impacto inmediato
    """
    topic: str
    gap_type: str               # prerequisite_gap | adjacency_gap | metacognitive_gap
    severity: str               # critical | moderate | low
    
    dependent_topics_at_risk: list = field(default_factory=list)
    # Topics que el estudiante ya trabaja y que dependen de este gap
    
    evidence: list = field(default_factory=list)
    # Datos que sustentan la detección del gap
    
    suggested_probe: str = ""
    # Pregunta de sondeo para verificar si es gap real
    
    confidence: float = 0.0
    # 0.0-1.0 confianza en la detección
    
    detected_at: str = ""
    # Timestamp de detección


@dataclass
class EpistemicProbe:
    """
    Sonda epistémica generada por técnicas de Knowledge Elicitation (Firlej).
    
    Cada sonda tiene un tipo (contraste, límite, hipotética, directa) que
    determina qué mecanismo cognitivo activa en el estudiante.
    
    probe_type sigue la taxonomía de Firlej:
    - contrast: "¿En qué se diferencia X de Y?" → discriminación conceptual
    - boundary: "¿Cuándo deja de funcionar X?" → límites de aplicabilidad
    - hypothetical: "¿Qué pasaría si X no existiera?" → comprensión funcional
    - scaffolded: pregunta directa con andamiaje empático → gap metacognitivo
    """
    gap: CognitiveGap
    probe_type: str             # contrast | boundary | hypothetical | scaffolded
    probe_text: str             # La pregunta formulada en lenguaje natural
    target_bloom: int           # Nivel Bloom que la sonda intenta elicitar
    priority: float = 0.0       # 0.0-1.0, prioridad de inyección
    natural_context: str = ""   # Contexto conversacional donde encaja naturalmente
    
    # Metadatos para analytics
    firlej_technique: str = ""  # Nombre de la técnica Firlej aplicada
    expected_response_if_gap: str = ""      # Qué esperamos si HAY gap
    expected_response_if_competent: str = "" # Qué esperamos si NO hay gap


@dataclass
class KnowledgeMap:
    """
    Grafo de conocimiento completo del corpus.
    Topología: grafo dirigido (prerrequisitos) + no dirigido (co-ocurrencia).
    """
    nodes: dict = field(default_factory=dict)
    # topic → KnowledgeNode
    
    prerequisite_edges: list = field(default_factory=list)
    # [(from_topic, to_topic)] — from es prerrequisito de to
    
    cooccurrence_edges: list = field(default_factory=list)
    # [(topic_a, topic_b, weight)] — co-ocurren en chunks
    
    total_chunks: int = 0
    built_at: str = ""


@dataclass
class StudentKnowledgeMap:
    """
    Subgrafo del KnowledgeMap que el estudiante ha visitado.
    La DIFERENCIA entre KnowledgeMap y StudentKnowledgeMap = gaps.
    """
    student_id: str
    topic_states: dict = field(default_factory=dict)
    # topic → StudentTopicState
    
    total_interactions: int = 0
    last_updated: str = ""
    
    # Derived metrics
    coverage_ratio: float = 0.0
    # % de topics del corpus que el estudiante ha visitado
    
    avg_bloom_depth: float = 0.0
    # Bloom promedio ponderado por visitas


# ═══════════════════════════════════════════════════════════════════════
# PREREQUISITE GRAPH — Grafo estático del dominio de programación
# ═══════════════════════════════════════════════════════════════════════

# Definición estática de prerrequisitos para el dominio de programación
# introductoria. En producción, esto se extraería del syllabus o se
# inferiría de co-ocurrencia en chunks con un umbral de confianza.

PROGRAMMING_PREREQUISITES = {
    # topic → [topics que son prerrequisito]
    "variables": [],
    "entrada/salida": ["variables"],
    "bucles": ["variables"],
    "arrays": ["variables", "bucles"],
    "funciones": ["variables", "entrada/salida"],
    "recursión": ["funciones", "bucles"],
    "depuración": ["variables"],  # meta-skill, prerrequisito blando
    "conceptual": [],              # meta-topic, sin prerrequisitos
    "ejercicio": [],               # meta-topic
}

# Dificultad estimada (calibrada por experiencia docente)
TOPIC_DIFFICULTY = {
    "variables": 0.15,
    "entrada/salida": 0.25,
    "bucles": 0.40,
    "arrays": 0.50,
    "funciones": 0.55,
    "recursión": 0.85,
    "depuración": 0.45,
    "conceptual": 0.20,
    "ejercicio": 0.35,
}


# ═══════════════════════════════════════════════════════════════════════
# MOTOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

class CognitiveGapDetector:
    """
    Detecta unknown unknowns — lo que el estudiante no sabe que no sabe.
    
    Flujo de uso:
    1. build_knowledge_map(rag_documents) → construye grafo del corpus
    2. record_interaction(student_id, topics, bloom, chunks) → actualiza mapa estudiante
    3. detect_critical_gaps(student_id) → identifica gaps prioritarios
    4. generate_epistemic_probes(gap) → genera sondas Firlej
    5. get_probe_for_injection(student_id) → devuelve la sonda más prioritaria
       para inyectar en el system prompt del LLM
    
    Integración con middleware.py:
    En pre_process(), después de construir el system prompt, si hay gaps
    críticos y han pasado ≥3 interacciones desde la última sonda, inyectar
    la sonda como addendum al system prompt.
    """

    # Configuración de umbrales
    MIN_INTERACTIONS_BETWEEN_PROBES = 3
    # Mínimo de interacciones entre sondas para no ser intrusivo
    
    PREREQUISITE_GAP_CONFIDENCE = 0.85
    # Confianza base para gap de prerrequisito (alta porque la estructura es explícita)
    
    ADJACENCY_GAP_CONFIDENCE = 0.60
    # Confianza base para gap de adyacencia (media porque es heurística)
    
    METACOGNITIVE_GAP_CONFIDENCE = 0.70
    # Confianza base para gap metacognitivo (requiere corroboración del silence detector)

    def __init__(self):
        self.knowledge_map: Optional[KnowledgeMap] = None
        self.student_maps: dict[str, StudentKnowledgeMap] = {}
        # student_id → StudentKnowledgeMap
        
        self.detected_gaps: dict[str, list[CognitiveGap]] = defaultdict(list)
        # student_id → list of CognitiveGap
        
        self.probe_history: dict[str, list[dict]] = defaultdict(list)
        # student_id → [{timestamp, probe, gap_topic, interaction_number}]
        
        self.interaction_counters: dict[str, int] = defaultdict(int)
        # student_id → total interactions since last probe
        
        # Referencia cruzada con silence detector (si disponible)
        self.silence_detector = None

    def set_silence_detector(self, detector) -> None:
        """
        Inyecta referencia al EpistemicSilenceDetector para correlacionar
        silencios clasificados como metacognitive_gap con gaps estructurales.
        Acoplamiento débil: funciona sin él, mejora con él.
        """
        self.silence_detector = detector

    # ──────────────────────────────────────────────────────────────────
    # COMPONENTE 1 — Mapa del corpus como grafo de conocimiento
    # ──────────────────────────────────────────────────────────────────

    def build_knowledge_map(self, rag_documents: list[dict], topic_tags: list[str] = None) -> KnowledgeMap:
        """
        Construye el grafo de conocimiento a partir de los chunks del RAG pipeline.
        
        Cada chunk del RAG tiene: {id, text, source, chunk_index}.
        Los topic_tags vienen de PedagogicalConfig.topic_tags (middleware.py).
        
        Proceso:
        1. Para cada chunk, detectar qué topics cubre (keyword matching)
        2. Construir nodos con chunk_indices
        3. Aristas de prerrequisito: estáticas de PROGRAMMING_PREREQUISITES
        4. Aristas de co-ocurrencia: si dos topics aparecen en el mismo chunk
        
        Args:
            rag_documents: lista de chunks del RAG [{id, text, source, chunk_index}]
            topic_tags: lista de topics del dominio (default: los del middleware)
        
        Returns:
            KnowledgeMap completo
        """
        if topic_tags is None:
            topic_tags = list(PROGRAMMING_PREREQUISITES.keys())

        km = KnowledgeMap(
            total_chunks=len(rag_documents),
            built_at=datetime.now().isoformat(),
        )

        # Inicializar nodos para todos los topics del dominio
        for topic in topic_tags:
            km.nodes[topic] = KnowledgeNode(
                topic=topic,
                prerequisite_topics=PROGRAMMING_PREREQUISITES.get(topic, []),
                difficulty_estimate=TOPIC_DIFFICULTY.get(topic, 0.5),
            )

        # Construir inverso: dependent_topics
        for topic, prereqs in PROGRAMMING_PREREQUISITES.items():
            for prereq in prereqs:
                if prereq in km.nodes:
                    km.nodes[prereq].dependent_topics.append(topic)

        # Mapear chunks a topics (keyword detection — misma lógica que middleware)
        topic_keywords = {
            "variables": ["variable", "int ", "float", "string", "tipo de dato", "declarar",
                         "asignación", "valor", "constante"],
            "bucles": ["for", "while", "bucle", "iteración", "iterar", "repetir",
                      "recorrido", "ciclo"],
            "funciones": ["función", "funcion", "def ", "return", "parámetro", "argumento",
                         "llamada", "invocar", "método"],
            "arrays": ["array", "lista", "vector", "índice", "posición", "recorrer",
                      "colección", "elemento", "arreglo"],
            "recursión": ["recursión", "recursivo", "caso base", "llamada recursiva",
                         "pila de llamadas", "stack"],
            "entrada/salida": ["input", "output", "print", "scanner", "leer", "escribir",
                              "mostrar", "pantalla", "teclado"],
            "depuración": ["error", "bug", "fallo", "no funciona", "no compila", "excepción",
                          "depurar", "debug", "traza"],
            "conceptual": ["qué es", "qué significa", "diferencia entre", "para qué sirve",
                          "concepto", "teoría"],
            "ejercicio": ["ejercicio", "problema", "enunciado", "resolver", "implementar",
                         "programa", "código"],
        }

        # Para cada chunk, detectar topics presentes
        chunk_topic_map = defaultdict(set)  # chunk_index → set of topics
        
        for doc in rag_documents:
            text_lower = doc.get("text", "").lower()
            chunk_idx = doc.get("chunk_index", 0)
            
            for topic, keywords in topic_keywords.items():
                if topic in km.nodes and any(kw in text_lower for kw in keywords):
                    km.nodes[topic].chunk_indices.append(chunk_idx)
                    chunk_topic_map[chunk_idx].add(topic)

        # Construir aristas de co-ocurrencia a partir de chunks compartidos
        cooccurrence_counts = defaultdict(int)
        for chunk_idx, topics in chunk_topic_map.items():
            topics_list = sorted(topics)
            for i in range(len(topics_list)):
                for j in range(i + 1, len(topics_list)):
                    pair = (topics_list[i], topics_list[j])
                    cooccurrence_counts[pair] += 1
                    # Registrar como topics co-ocurrentes
                    if topics_list[j] not in km.nodes[topics_list[i]].co_occurring_topics:
                        km.nodes[topics_list[i]].co_occurring_topics.append(topics_list[j])
                    if topics_list[i] not in km.nodes[topics_list[j]].co_occurring_topics:
                        km.nodes[topics_list[j]].co_occurring_topics.append(topics_list[i])

        # Aristas de prerrequisito
        for topic, prereqs in PROGRAMMING_PREREQUISITES.items():
            for prereq in prereqs:
                km.prerequisite_edges.append((prereq, topic))

        # Aristas de co-ocurrencia con peso
        for (t1, t2), count in cooccurrence_counts.items():
            weight = count / max(len(rag_documents), 1)
            km.cooccurrence_edges.append((t1, t2, round(weight, 3)))

        self.knowledge_map = km
        return km

    # ──────────────────────────────────────────────────────────────────
    # COMPONENTE 2 — Mapa del estudiante como subgrafo visitado
    # ──────────────────────────────────────────────────────────────────

    def _ensure_student_map(self, student_id: str) -> StudentKnowledgeMap:
        """Crea o recupera el mapa del estudiante."""
        if student_id not in self.student_maps:
            self.student_maps[student_id] = StudentKnowledgeMap(
                student_id=student_id,
            )
            # Inicializar estados para todos los topics del knowledge map
            if self.knowledge_map:
                for topic in self.knowledge_map.nodes:
                    self.student_maps[student_id].topic_states[topic] = StudentTopicState(
                        topic=topic,
                    )
        return self.student_maps[student_id]

    def record_interaction(
        self,
        student_id: str,
        detected_topics: list[str],
        bloom_level: int = 2,
        chunk_indices: list[int] = None,
        timestamp: str = None,
    ) -> None:
        """
        Registra una interacción del estudiante para actualizar su mapa cognitivo.
        
        Llamado desde middleware.log_interaction() o directamente desde app.py.
        Cada interacción actualiza el subgrafo visitado del estudiante.
        
        Args:
            student_id: identificador del estudiante
            detected_topics: topics detectados en el prompt (de middleware._detect_topics)
            bloom_level: nivel Bloom de la pregunta (1-6)
            chunk_indices: indices de chunks RAG devueltos en esta interacción
            timestamp: momento de la interacción (ISO format)
        """
        ts = timestamp or datetime.now().isoformat()
        smap = self._ensure_student_map(student_id)
        
        for topic in detected_topics:
            if topic not in smap.topic_states:
                smap.topic_states[topic] = StudentTopicState(topic=topic)
            
            state = smap.topic_states[topic]
            state.visited = True
            state.visit_count += 1
            state.bloom_levels_seen.append(bloom_level)
            state.max_bloom_level = max(state.max_bloom_level, bloom_level)
            state.last_visited = ts
            
            if not state.first_visit:
                state.first_visit = ts
            
            if chunk_indices:
                state.chunks_consumed.update(chunk_indices)
        
        smap.total_interactions += 1
        smap.last_updated = ts
        
        # Actualizar métricas derivadas
        self._update_derived_metrics(student_id)
        
        # Incrementar contador de interacciones desde última sonda
        self.interaction_counters[student_id] += 1

    def _update_derived_metrics(self, student_id: str) -> None:
        """Recalcula coverage_ratio y avg_bloom_depth."""
        smap = self.student_maps[student_id]
        if not self.knowledge_map:
            return
        
        total_topics = len(self.knowledge_map.nodes)
        visited_topics = sum(
            1 for s in smap.topic_states.values() if s.visited
        )
        smap.coverage_ratio = visited_topics / max(total_topics, 1)
        
        # Bloom promedio ponderado por visitas
        total_bloom = 0
        total_visits = 0
        for state in smap.topic_states.values():
            if state.visited and state.bloom_levels_seen:
                total_bloom += sum(state.bloom_levels_seen)
                total_visits += len(state.bloom_levels_seen)
        
        smap.avg_bloom_depth = round(total_bloom / max(total_visits, 1), 2)

    def get_student_map(self, student_id: str) -> StudentKnowledgeMap:
        """Devuelve el mapa cognitivo del estudiante (para visualización)."""
        return self._ensure_student_map(student_id)

    # ──────────────────────────────────────────────────────────────────
    # COMPONENTE 3 — Detector de gaps críticos
    # ──────────────────────────────────────────────────────────────────

    def detect_critical_gaps(self, student_id: str) -> list[CognitiveGap]:
        """
        Compara KnowledgeMap completo con StudentKnowledgeMap.
        
        Un gap es un nodo del mapa completo que el estudiante no ha visitado
        (o ha visitado solo superficialmente).
        
        Un gap es CRÍTICO si cumple al menos una de tres condiciones:
        
        Condición 1 — Dependencia directa (prerequisite_gap):
          El estudiante trabaja con un topic que DEPENDE de un topic no visitado.
          Ejemplo: usa recursión sin haber abordado funciones.
          Severidad: critical. Confianza: alta (la estructura es explícita).
        
        Condición 2 — Adyacencia temática (adjacency_gap):
          El estudiante ha visitado topics vecinos pero no este.
          Ejemplo: trabaja variables y bucles pero no arrays (vecino de ambos).
          Severidad: moderate. Confianza: media (heurística de ZPD).
        
        Condición 3 — Corroboración del silence detector (metacognitive_gap):
          El EpistemicSilenceDetector ha clasificado un silencio como
          "metacognición deficiente" para este topic.
          Severidad: variable (hereda del silence detector).
        
        Returns:
            Lista de CognitiveGap ordenada por severidad y confianza.
        """
        if not self.knowledge_map:
            return []
        
        smap = self._ensure_student_map(student_id)
        gaps = []
        ts = datetime.now().isoformat()

        # ─── Condición 1: Prerequisite Gaps ───
        for topic, node in self.knowledge_map.nodes.items():
            student_state = smap.topic_states.get(topic, StudentTopicState(topic=topic))
            
            if not student_state.visited:
                continue  # Si no ha visitado este topic, no aplica C1
            
            # Verificar si los prerrequisitos están cubiertos
            for prereq in node.prerequisite_topics:
                prereq_state = smap.topic_states.get(prereq, StudentTopicState(topic=prereq))
                
                if not prereq_state.visited:
                    # Gap de prerrequisito CRÍTICO: usa topic sin su cimiento
                    gaps.append(CognitiveGap(
                        topic=prereq,
                        gap_type="prerequisite_gap",
                        severity="critical",
                        dependent_topics_at_risk=[topic],
                        evidence=[
                            f"Estudiante trabaja '{topic}' (visitado {student_state.visit_count}x, "
                            f"Bloom max {student_state.max_bloom_level}) pero nunca abordó "
                            f"su prerrequisito '{prereq}'",
                            f"Relación: '{prereq}' → '{topic}' en grafo de dependencias",
                        ],
                        confidence=self.PREREQUISITE_GAP_CONFIDENCE,
                        detected_at=ts,
                    ))
                elif prereq_state.max_bloom_level <= 1 and student_state.max_bloom_level >= 3:
                    # Gap parcial: prerrequisito visitado pero solo a nivel superficial
                    # mientras usa el dependiente a nivel profundo
                    gaps.append(CognitiveGap(
                        topic=prereq,
                        gap_type="prerequisite_gap",
                        severity="moderate",
                        dependent_topics_at_risk=[topic],
                        evidence=[
                            f"Prerrequisito '{prereq}' visitado solo a Bloom {prereq_state.max_bloom_level} "
                            f"(nivel superficial) pero dependiente '{topic}' se trabaja a "
                            f"Bloom {student_state.max_bloom_level} (nivel profundo)",
                            "Asimetría de profundidad sugiere comprensión frágil del cimiento",
                        ],
                        confidence=self.PREREQUISITE_GAP_CONFIDENCE * 0.7,
                        detected_at=ts,
                    ))

        # ─── Condición 2: Adjacency Gaps ───
        for topic, node in self.knowledge_map.nodes.items():
            student_state = smap.topic_states.get(topic, StudentTopicState(topic=topic))
            
            if student_state.visited:
                continue  # Solo buscamos topics NO visitados
            
            # Verificar si topics vecinos SÍ están visitados
            neighbors = set(node.co_occurring_topics + node.prerequisite_topics + node.dependent_topics)
            visited_neighbors = [
                n for n in neighbors
                if smap.topic_states.get(n, StudentTopicState(topic=n)).visited
            ]
            
            if len(visited_neighbors) >= 2:
                # Topic rodeado de visitados pero él mismo ausente → adjacency gap
                # Cuantos más vecinos visitados, más relevante es la ausencia
                confidence = min(
                    self.ADJACENCY_GAP_CONFIDENCE + (len(visited_neighbors) - 2) * 0.1,
                    0.90
                )
                gaps.append(CognitiveGap(
                    topic=topic,
                    gap_type="adjacency_gap",
                    severity="moderate",
                    dependent_topics_at_risk=visited_neighbors,
                    evidence=[
                        f"Topic '{topic}' no visitado pero {len(visited_neighbors)} topics "
                        f"vecinos sí: {visited_neighbors}",
                        f"Zona proximal de desarrollo: topic accesible dado el mapa actual",
                    ],
                    confidence=round(confidence, 2),
                    detected_at=ts,
                ))

        # ─── Condición 3: Metacognitive Gaps (cruzando con silence detector) ───
        if self.silence_detector:
            for topic in self.knowledge_map.nodes:
                student_state = smap.topic_states.get(topic, StudentTopicState(topic=topic))
                
                # Buscar silencios clasificados como metacognitive_gap
                student_exposures = self.silence_detector.topic_exposures.get(student_id, [])
                for exposure in student_exposures:
                    if (exposure.topic == topic and 
                        exposure.silence_type == "metacognitive_gap" and
                        exposure.confidence_in_type > 0.5):
                        
                        # Verificar que no sea duplicado de C1 o C2
                        already_detected = any(
                            g.topic == topic for g in gaps
                        )
                        if not already_detected:
                            gaps.append(CognitiveGap(
                                topic=topic,
                                gap_type="metacognitive_gap",
                                severity="moderate" if exposure.silence_anomaly > 0.7 else "low",
                                dependent_topics_at_risk=[],
                                evidence=[
                                    f"Silence Detector clasificó silencio en '{topic}' como "
                                    f"metacognición deficiente (anomalía: {exposure.silence_anomaly:.2f}, "
                                    f"confianza: {exposure.confidence_in_type:.2f})",
                                    f"Estudiante visitó {student_state.visit_count}x a Bloom max "
                                    f"{student_state.max_bloom_level} — insuficiente para dificultad "
                                    f"estimada {self.knowledge_map.nodes.get(topic, KnowledgeNode(topic=topic)).difficulty_estimate:.2f}",
                                ],
                                confidence=round(
                                    self.METACOGNITIVE_GAP_CONFIDENCE * exposure.confidence_in_type,
                                    2
                                ),
                                detected_at=ts,
                            ))
        
        # ─── También detectar gaps metacognitivos sin silence detector ───
        # Heurística: topic visitado pocas veces a Bloom bajo cuando la dificultad
        # estimada es alta → debería haber más preguntas
        for topic, node in self.knowledge_map.nodes.items():
            student_state = smap.topic_states.get(topic, StudentTopicState(topic=topic))
            
            if not student_state.visited:
                continue
            
            already_detected = any(g.topic == topic for g in gaps)
            if already_detected:
                continue
            
            # Criterio: visitado una vez a Bloom ≤ 2, dificultad ≥ 0.6
            if (student_state.visit_count <= 1 and
                student_state.max_bloom_level <= 2 and
                node.difficulty_estimate >= 0.6):
                
                gaps.append(CognitiveGap(
                    topic=topic,
                    gap_type="metacognitive_gap",
                    severity="low",
                    dependent_topics_at_risk=node.dependent_topics,
                    evidence=[
                        f"Topic '{topic}' (dificultad {node.difficulty_estimate:.2f}) visitado "
                        f"solo {student_state.visit_count}x a Bloom {student_state.max_bloom_level}",
                        "La ratio visitas/dificultad sugiere exploración insuficiente — "
                        "posible Dunning-Kruger o evitación",
                    ],
                    confidence=0.45,
                    detected_at=ts,
                ))

        # Consolidar: si un topic aparece en múltiples gaps, fusionar y escalar severidad
        gaps = self._consolidate_gaps(gaps)
        
        # Ordenar por severidad y confianza
        severity_order = {"critical": 0, "moderate": 1, "low": 2}
        gaps.sort(key=lambda g: (severity_order.get(g.severity, 3), -g.confidence))
        
        self.detected_gaps[student_id] = gaps
        return gaps

    def _consolidate_gaps(self, gaps: list[CognitiveGap]) -> list[CognitiveGap]:
        """
        Si un topic aparece en múltiples gaps (ej: es prerequisite_gap Y adjacency_gap),
        fusionar en un solo gap con la severidad más alta y evidencia combinada.
        """
        by_topic = defaultdict(list)
        for gap in gaps:
            by_topic[gap.topic].append(gap)
        
        consolidated = []
        for topic, topic_gaps in by_topic.items():
            if len(topic_gaps) == 1:
                consolidated.append(topic_gaps[0])
            else:
                # Fusionar: tomar la severidad más alta
                severity_order = {"critical": 0, "moderate": 1, "low": 2}
                topic_gaps.sort(key=lambda g: severity_order.get(g.severity, 3))
                primary = topic_gaps[0]
                
                # Combinar evidencia y dependencias
                all_evidence = []
                all_deps = set()
                all_types = set()
                max_confidence = 0
                
                for g in topic_gaps:
                    all_evidence.extend(g.evidence)
                    all_deps.update(g.dependent_topics_at_risk)
                    all_types.add(g.gap_type)
                    max_confidence = max(max_confidence, g.confidence)
                
                # Si aparece como multiple tipos, es más severo
                if len(all_types) >= 2 and primary.severity != "critical":
                    primary.severity = "critical" if "prerequisite_gap" in all_types else "moderate"
                
                primary.evidence = all_evidence
                primary.dependent_topics_at_risk = list(all_deps)
                primary.gap_type = "+".join(sorted(all_types))
                primary.confidence = min(max_confidence * 1.1, 1.0)  # Boost por convergencia
                consolidated.append(primary)
        
        return consolidated

    # ──────────────────────────────────────────────────────────────────
    # COMPONENTE 4 — Generador de sondas epistémicas (Firlej)
    # ──────────────────────────────────────────────────────────────────

    def generate_epistemic_probes(self, gap: CognitiveGap) -> list[EpistemicProbe]:
        """
        Genera 2-3 sondas epistémicas para verificar un gap cognitivo.
        
        Las técnicas de Firlej & Hellens (1991) para knowledge elicitation
        se invierten aquí: en vez de extraer conocimiento del experto que
        no sabe que sabe, se detecta la AUSENCIA de conocimiento en el
        estudiante que no sabe que no sabe.
        
        Tipos de sonda por tipo de gap:
        - prerequisite_gap → sonda de contraste + sonda hipotética
        - adjacency_gap → sonda scaffolded (no amenazante)
        - metacognitive_gap → sonda de límite + sonda scaffolded
        
        Cada sonda se formula en lenguaje natural, calibrada al nivel
        aparente del estudiante, para inyección orgánica en la conversación.
        """
        probes = []
        
        # ─── Biblioteca de templates de sondas por topic ───
        probe_templates = self._get_probe_templates()
        topic_probes = probe_templates.get(gap.topic, probe_templates.get("_default"))
        
        if gap.gap_type == "prerequisite_gap" or "prerequisite_gap" in gap.gap_type:
            probes.extend(self._generate_prerequisite_probes(gap, topic_probes))
        
        if gap.gap_type == "adjacency_gap" or "adjacency_gap" in gap.gap_type:
            probes.extend(self._generate_adjacency_probes(gap, topic_probes))
        
        if gap.gap_type == "metacognitive_gap" or "metacognitive_gap" in gap.gap_type:
            probes.extend(self._generate_metacognitive_probes(gap, topic_probes))
        
        # Si es tipo combinado y ya tenemos suficientes probes, recortar
        if len(probes) > 3:
            probes.sort(key=lambda p: -p.priority)
            probes = probes[:3]
        
        # Asegurar que al menos hay una sonda
        if not probes and topic_probes:
            probes.append(EpistemicProbe(
                gap=gap,
                probe_type="scaffolded",
                probe_text=topic_probes.get("scaffolded", 
                    f"Muchos estudiantes encuentran interesante el tema de {gap.topic}. "
                    f"¿Hay algo de ese tema que te gustaría explorar?"),
                target_bloom=2,
                priority=0.5,
                firlej_technique="open_probe",
                natural_context=f"Introducir el tema de {gap.topic} de forma no amenazante",
            ))
        
        return probes

    def _generate_prerequisite_probes(
        self, gap: CognitiveGap, templates: dict
    ) -> list[EpistemicProbe]:
        """Sondas para gaps de prerrequisito: contraste + hipotética."""
        probes = []
        deps = gap.dependent_topics_at_risk
        dep_str = deps[0] if deps else "lo que estás trabajando"
        
        # Sonda de contraste (Firlej: "contrast probe")
        contrast_text = templates.get("contrast_prereq", "").format(
            topic=gap.topic, dependent=dep_str
        ) if templates.get("contrast_prereq") else (
            f"Estás trabajando con {dep_str}, que se apoya en el concepto de "
            f"{gap.topic}. ¿Podrías explicar brevemente cómo se relacionan "
            f"{gap.topic} con {dep_str}?"
        )
        
        probes.append(EpistemicProbe(
            gap=gap,
            probe_type="contrast",
            probe_text=contrast_text,
            target_bloom=4,  # Analizar: establecer relaciones
            priority=0.9,
            firlej_technique="contrast_probe",
            natural_context=f"Cuando el estudiante mencione {dep_str}",
            expected_response_if_gap="Confusión, respuesta vaga, o admisión de no saber",
            expected_response_if_competent="Explicación clara de la relación causal",
        ))
        
        # Sonda hipotética (Firlej: "hypothetical probe")
        hypothetical_text = templates.get("hypothetical_prereq", "").format(
            topic=gap.topic, dependent=dep_str
        ) if templates.get("hypothetical_prereq") else (
            f"Imagina que no existiera el concepto de {gap.topic}. "
            f"¿Cómo afectaría eso a tu trabajo con {dep_str}?"
        )
        
        probes.append(EpistemicProbe(
            gap=gap,
            probe_type="hypothetical",
            probe_text=hypothetical_text,
            target_bloom=5,  # Evaluar: juzgar consecuencias
            priority=0.75,
            firlej_technique="hypothetical_probe",
            natural_context=f"Como pregunta de profundización sobre {dep_str}",
            expected_response_if_gap="Incapacidad de articular la dependencia",
            expected_response_if_competent="Identificación precisa del rol funcional",
        ))
        
        return probes

    def _generate_adjacency_probes(
        self, gap: CognitiveGap, templates: dict
    ) -> list[EpistemicProbe]:
        """Sondas para gaps de adyacencia: scaffolded (ZPD)."""
        probes = []
        neighbors = gap.dependent_topics_at_risk  # En adjacency, son los vecinos visitados
        neighbor_str = " y ".join(neighbors[:2]) if neighbors else "los temas que dominas"
        
        # Sonda scaffolded no amenazante
        scaffolded_text = templates.get("adjacency", "").format(
            topic=gap.topic, neighbors=neighbor_str
        ) if templates.get("adjacency") else (
            f"Has avanzado bien con {neighbor_str}. Un paso natural sería explorar "
            f"cómo se conectan con {gap.topic}. ¿Has pensado en ello?"
        )
        
        probes.append(EpistemicProbe(
            gap=gap,
            probe_type="scaffolded",
            probe_text=scaffolded_text,
            target_bloom=2,  # Comprender: establecer conexiones
            priority=0.6,
            firlej_technique="scaffolded_probe",
            natural_context=f"Transición natural desde {neighbor_str}",
            expected_response_if_gap="Curiosidad, reconocimiento de no haber explorado",
            expected_response_if_competent="Ya lo conozco, o explicación del concepto",
        ))
        
        return probes

    def _generate_metacognitive_probes(
        self, gap: CognitiveGap, templates: dict
    ) -> list[EpistemicProbe]:
        """Sondas para gaps metacognitivos: boundary + scaffolded."""
        probes = []
        
        # Sonda de límite (Firlej: "boundary probe")
        boundary_text = templates.get("boundary", "").format(
            topic=gap.topic
        ) if templates.get("boundary") else (
            f"El tema de {gap.topic} suele tener aspectos que parecen sencillos "
            f"hasta que uno se encuentra con los casos límite. ¿Hay alguna situación "
            f"donde {gap.topic} te resulte confuso o no tengas claro cómo funciona?"
        )
        
        probes.append(EpistemicProbe(
            gap=gap,
            probe_type="boundary",
            probe_text=boundary_text,
            target_bloom=5,  # Evaluar: identificar límites
            priority=0.7,
            firlej_technique="boundary_probe",
            natural_context=f"Cuando el estudiante parezca seguro sobre {gap.topic}",
            expected_response_if_gap="Admisión de incertidumbre o pregunta nueva",
            expected_response_if_competent="Identificación precisa de casos límite",
        ))
        
        # Sonda scaffolded empática
        scaffolded_text = templates.get("metacognitive", "").format(
            topic=gap.topic
        ) if templates.get("metacognitive") else (
            f"Muchos estudiantes encuentran complicado el concepto de {gap.topic}. "
            f"¿Te parece claro o hay algo que te gustaría repasar?"
        )
        
        probes.append(EpistemicProbe(
            gap=gap,
            probe_type="scaffolded",
            probe_text=scaffolded_text,
            target_bloom=2,
            priority=0.65,
            firlej_technique="empathic_scaffolded_probe",
            natural_context="Pregunta no amenazante que abre la puerta sin diagnosticar",
            expected_response_if_gap="Admisión de dificultad o pregunta exploratoria",
            expected_response_if_competent="Confirmación segura con ejemplos",
        ))
        
        return probes

    def _get_probe_templates(self) -> dict:
        """
        Templates de sondas específicas por topic del dominio de programación.
        
        Cada template usa {topic}, {dependent}, {neighbors} como placeholders.
        Las preguntas están calibradas para NO parecer un test y SÍ parecer
        continuación orgánica de la conversación.
        """
        return {
            "funciones": {
                "contrast_prereq": (
                    "Estás trabajando con {dependent} — que internamente usa "
                    "funciones. ¿Tienes claro cómo funciona el paso de "
                    "parámetros cuando una función llama a otra función?"
                ),
                "hypothetical_prereq": (
                    "Imagina que no pudieras usar funciones para resolver este "
                    "problema. ¿Cómo organizarías tu código? ¿Qué complicaciones "
                    "surgirían?"
                ),
                "boundary": (
                    "Las funciones son bastante intuitivas al principio, pero "
                    "¿qué pasa cuando una función necesita devolver más de un valor? "
                    "¿O cuando los parámetros son objetos complejos?"
                ),
                "metacognitive": (
                    "El concepto de función parece simple, pero los detalles de "
                    "alcance de variables (scope) y paso por referencia vs. valor "
                    "suelen generar confusión. ¿Te sientes cómodo con esos aspectos?"
                ),
                "adjacency": (
                    "Has avanzado bien con {neighbors}. Las funciones son el "
                    "siguiente paso natural para organizar código. ¿Has pensado "
                    "en cómo encapsularías en funciones lo que ya sabes hacer?"
                ),
            },
            "arrays": {
                "contrast_prereq": (
                    "Estás usando {dependent} que procesan múltiples datos. ¿Tienes "
                    "claro cómo se almacenan esos datos? ¿Sabrías explicar la "
                    "diferencia entre una variable simple y una colección de datos?"
                ),
                "hypothetical_prereq": (
                    "Si necesitaras manejar 100 elementos, ¿crearías 100 variables "
                    "individuales? ¿Qué alternativa conoces?"
                ),
                "boundary": (
                    "Los arrays son directos para datos del mismo tipo. "
                    "¿Qué pasa cuando necesitas mezclar tipos? ¿O cuando "
                    "no sabes de antemano cuántos elementos tendrás?"
                ),
                "metacognitive": (
                    "Muchos estudiantes creen que dominan los arrays hasta que "
                    "se encuentran con el desbordamiento de índice. ¿Has tenido "
                    "alguna experiencia con errores de ese tipo?"
                ),
                "adjacency": (
                    "Has dominado {neighbors}. Un paso natural sería explorar "
                    "cómo los bucles se combinan con colecciones de datos — "
                    "¿has pensado en cómo recorrerías una lista de 100 elementos?"
                ),
            },
            "recursión": {
                "contrast_prereq": (
                    "Has preguntado sobre recursión — una función que se llama a "
                    "sí misma. ¿Tienes claro cómo funciona la pila de llamadas? "
                    "¿Podrías trazar paso a paso qué pasa cuando una función "
                    "recursiva hace su segunda llamada?"
                ),
                "hypothetical_prereq": (
                    "Si no pudieras usar recursión, ¿cómo resolverías el mismo "
                    "problema? ¿Qué estructura de repetición usarías como alternativa?"
                ),
                "boundary": (
                    "La recursión funciona bien para problemas que se descomponen "
                    "naturalmente. Pero ¿cuándo NO conviene usar recursión? "
                    "¿Hay un límite práctico?"
                ),
                "metacognitive": (
                    "La recursión es estadísticamente el concepto más difícil de "
                    "este curso. Muchos estudiantes creen entenderla hasta que "
                    "intentan diseñar su propio caso recursivo. ¿Te animas a "
                    "intentarlo con un ejemplo nuevo?"
                ),
                "adjacency": (
                    "Has trabajado con {neighbors}. La recursión es una forma "
                    "diferente de pensar la repetición — ¿te gustaría explorar "
                    "cómo se compara con los bucles que ya conoces?"
                ),
            },
            "bucles": {
                "contrast_prereq": (
                    "Estás trabajando con bucles. ¿Tienes claro cómo se "
                    "actualiza una variable dentro de un bucle? ¿Sabrías "
                    "explicar qué es un acumulador?"
                ),
                "hypothetical_prereq": (
                    "Si no existieran los bucles, ¿cómo repetirías una "
                    "operación 10 veces? ¿Qué implicaciones tendría?"
                ),
                "boundary": (
                    "Los bucles for y while parecen intercambiables al principio. "
                    "¿Sabrías identificar un caso donde SOLO se pueda usar while "
                    "y no for?"
                ),
                "metacognitive": (
                    "El bucle while es donde muchos estudiantes se encuentran "
                    "con bucles infinitos. ¿Has pensado en qué condiciones "
                    "garantizan que un bucle termina?"
                ),
                "adjacency": (
                    "Dominas {neighbors}. Los bucles son esenciales para "
                    "procesar datos repetidamente — ¿has pensado en cómo "
                    "automatizar tareas que ahora haces manualmente?"
                ),
            },
            "depuración": {
                "contrast_prereq": (
                    "Has encontrado un error en tu código. ¿Tienes una "
                    "estrategia sistemática para encontrar dónde está el "
                    "problema, o vas cambiando cosas hasta que funciona?"
                ),
                "boundary": (
                    "Depurar errores de sintaxis es directo — el compilador "
                    "te dice dónde está. Pero ¿qué haces cuando el código "
                    "ejecuta sin errores pero da un resultado incorrecto?"
                ),
                "metacognitive": (
                    "La depuración es una habilidad que muchos estudiantes "
                    "subestiman. ¿Cuánto tiempo dedicas a leer mensajes de "
                    "error antes de buscar ayuda?"
                ),
                "adjacency": (
                    "Con lo que ya sabes de {neighbors}, es buen momento "
                    "para aprender a encontrar errores de forma sistemática. "
                    "¿Has usado alguna vez un depurador paso a paso?"
                ),
            },
            "variables": {
                "boundary": (
                    "Las variables parecen simples: guardas un valor y listo. "
                    "Pero ¿qué pasa cuando dos variables apuntan al mismo dato? "
                    "¿O cuando una variable cambia de tipo?"
                ),
                "metacognitive": (
                    "¿Tienes claro cuándo usar int vs float vs string? "
                    "Muchos errores sutiles vienen de tipos incorrectos."
                ),
            },
            "entrada/salida": {
                "boundary": (
                    "Leer un número del teclado parece trivial. ¿Qué pasa "
                    "cuando el usuario escribe texto donde esperabas un número?"
                ),
                "adjacency": (
                    "Sabes trabajar con {neighbors}. El siguiente paso es "
                    "interactuar con el usuario: ¿has pensado en cómo pedirle "
                    "datos y mostrar resultados?"
                ),
            },
            "_default": {
                "contrast_prereq": (
                    "Estás trabajando con {dependent}, que se apoya en "
                    "{topic}. ¿Podrías explicar brevemente cómo se "
                    "relacionan estos dos conceptos?"
                ),
                "hypothetical_prereq": (
                    "¿Qué crees que pasaría si no pudieras usar {topic}? "
                    "¿Cómo afectaría a tu trabajo con {dependent}?"
                ),
                "boundary": (
                    "¿Hay algún caso donde {topic} no funcione como "
                    "esperas? ¿Dónde están sus límites?"
                ),
                "metacognitive": (
                    "Muchos estudiantes encuentran complicado el concepto "
                    "de {topic}. ¿Te parece claro o hay algo que te "
                    "gustaría repasar?"
                ),
                "adjacency": (
                    "Has avanzado bien con {neighbors}. ¿Has explorado "
                    "cómo se conectan con {topic}?"
                ),
            },
        }

    # ──────────────────────────────────────────────────────────────────
    # COMPONENTE 5 — Integración con el flujo del chat
    # ──────────────────────────────────────────────────────────────────

    def get_probe_for_injection(self, student_id: str) -> Optional[str]:
        """
        Devuelve la sonda epistémica más prioritaria para inyectar en el
        system prompt del LLM, o None si no hay sonda pendiente.
        
        Condiciones para inyección:
        1. Hay gaps detectados para el estudiante
        2. Han pasado ≥ MIN_INTERACTIONS_BETWEEN_PROBES desde la última sonda
        3. La sonda no se ha usado ya para este gap
        
        Returns:
            String para añadir al system prompt, o None.
            Formato: "Si es pertinente en tu respuesta, introduce naturalmente
            esta pregunta: [probe_text]"
        """
        # Verificar que hay gaps
        gaps = self.detected_gaps.get(student_id, [])
        if not gaps:
            # Intentar detectar gaps si no se ha hecho
            gaps = self.detect_critical_gaps(student_id)
        
        if not gaps:
            return None
        
        # Verificar intervalo mínimo entre sondas
        interactions_since_probe = self.interaction_counters.get(student_id, 0)
        if interactions_since_probe < self.MIN_INTERACTIONS_BETWEEN_PROBES:
            return None
        
        # Seleccionar el gap más prioritario no sondeado recientemente
        recent_probed_topics = set()
        for p in self.probe_history.get(student_id, [])[-5:]:
            recent_probed_topics.add(p.get("gap_topic", ""))
        
        target_gap = None
        for gap in gaps:
            if gap.topic not in recent_probed_topics:
                target_gap = gap
                break
        
        if not target_gap:
            return None
        
        # Generar sondas y seleccionar la más prioritaria
        probes = self.generate_epistemic_probes(target_gap)
        if not probes:
            return None
        
        best_probe = max(probes, key=lambda p: p.priority)
        
        # Registrar en historial
        self.probe_history[student_id].append({
            "timestamp": datetime.now().isoformat(),
            "probe": best_probe.probe_text,
            "probe_type": best_probe.probe_type,
            "gap_topic": target_gap.topic,
            "gap_type": target_gap.gap_type,
            "gap_severity": target_gap.severity,
            "interaction_number": self.student_maps.get(student_id, StudentKnowledgeMap(student_id=student_id)).total_interactions,
        })
        
        # Reset contador de interacciones desde sonda
        self.interaction_counters[student_id] = 0
        
        # Formatear para inyección en system prompt
        injection = (
            f"\n\nSONDA EPISTÉMICA (prioridad: {target_gap.severity}): "
            f"Si es pertinente en tu respuesta, introduce naturalmente esta pregunta: "
            f"\"{best_probe.probe_text}\" "
            f"— No la presentes como un test. Debe fluir como curiosidad natural "
            f"dentro de tu respuesta al estudiante."
        )
        
        return injection

    def get_system_prompt_addon(self, student_id: str) -> str:
        """
        Alias de get_probe_for_injection, devuelve string vacío en vez de None
        para facilitar concatenación en middleware._build_system_prompt().
        """
        probe = self.get_probe_for_injection(student_id)
        return probe if probe else ""

    # ──────────────────────────────────────────────────────────────────
    # ANALYTICS Y VISUALIZACIÓN
    # ──────────────────────────────────────────────────────────────────

    def get_gap_summary(self, student_id: str) -> dict:
        """
        Resumen de gaps para el dashboard docente.
        
        Returns dict con:
        - total_gaps: int
        - critical_gaps: list[dict]
        - moderate_gaps: list[dict]
        - low_gaps: list[dict]
        - coverage_ratio: float
        - probe_history: list[dict]
        - knowledge_graph_stats: dict
        """
        gaps = self.detected_gaps.get(student_id, [])
        smap = self.student_maps.get(student_id)
        
        critical = [g for g in gaps if g.severity == "critical"]
        moderate = [g for g in gaps if g.severity == "moderate"]
        low = [g for g in gaps if g.severity == "low"]
        
        def gap_to_dict(g: CognitiveGap) -> dict:
            return {
                "topic": g.topic,
                "type": g.gap_type,
                "severity": g.severity,
                "confidence": g.confidence,
                "at_risk": g.dependent_topics_at_risk,
                "evidence": g.evidence,
                "probe": g.suggested_probe,
            }
        
        return {
            "total_gaps": len(gaps),
            "critical_gaps": [gap_to_dict(g) for g in critical],
            "moderate_gaps": [gap_to_dict(g) for g in moderate],
            "low_gaps": [gap_to_dict(g) for g in low],
            "coverage_ratio": smap.coverage_ratio if smap else 0.0,
            "avg_bloom_depth": smap.avg_bloom_depth if smap else 0.0,
            "total_interactions": smap.total_interactions if smap else 0,
            "probe_history": self.probe_history.get(student_id, []),
            "topics_visited": [
                t for t, s in (smap.topic_states if smap else {}).items() if s.visited
            ],
            "topics_not_visited": [
                t for t, s in (smap.topic_states if smap else {}).items() if not s.visited
            ],
            "knowledge_graph_stats": {
                "total_nodes": len(self.knowledge_map.nodes) if self.knowledge_map else 0,
                "total_prerequisite_edges": len(self.knowledge_map.prerequisite_edges) if self.knowledge_map else 0,
                "total_cooccurrence_edges": len(self.knowledge_map.cooccurrence_edges) if self.knowledge_map else 0,
            },
        }

    def get_knowledge_graph_data(self) -> dict:
        """
        Devuelve el grafo de conocimiento en formato apto para visualización
        (compatible con networkx/d3.js/Streamlit graph components).
        """
        if not self.knowledge_map:
            return {"nodes": [], "edges": []}
        
        nodes = []
        for topic, node in self.knowledge_map.nodes.items():
            nodes.append({
                "id": topic,
                "label": topic,
                "difficulty": node.difficulty_estimate,
                "n_chunks": len(node.chunk_indices),
                "n_prereqs": len(node.prerequisite_topics),
                "n_dependents": len(node.dependent_topics),
            })
        
        edges = []
        for src, dst in self.knowledge_map.prerequisite_edges:
            edges.append({
                "source": src,
                "target": dst,
                "type": "prerequisite",
                "weight": 1.0,
            })
        for t1, t2, w in self.knowledge_map.cooccurrence_edges:
            edges.append({
                "source": t1,
                "target": t2,
                "type": "cooccurrence",
                "weight": w,
            })
        
        return {"nodes": nodes, "edges": edges}

    def get_student_overlay_data(self, student_id: str) -> dict:
        """
        Datos para superponer el mapa del estudiante sobre el grafo de conocimiento.
        Para visualización: nodos visitados vs. no visitados, con gap highlighting.
        """
        smap = self.student_maps.get(student_id)
        gaps = self.detected_gaps.get(student_id, [])
        gap_topics = {g.topic: g for g in gaps}
        
        node_states = []
        if smap:
            for topic, state in smap.topic_states.items():
                gap = gap_topics.get(topic)
                node_states.append({
                    "id": topic,
                    "visited": state.visited,
                    "visit_count": state.visit_count,
                    "max_bloom": state.max_bloom_level,
                    "has_gap": gap is not None,
                    "gap_severity": gap.severity if gap else None,
                    "gap_type": gap.gap_type if gap else None,
                })
        
        return {
            "student_id": student_id,
            "node_states": node_states,
            "coverage_ratio": smap.coverage_ratio if smap else 0,
        }


# ═══════════════════════════════════════════════════════════════════════
# DATOS DEMO — Estudiante simulado para demostración
# ═══════════════════════════════════════════════════════════════════════

def generate_demo_data() -> tuple[CognitiveGapDetector, str]:
    """
    Genera un escenario de demostración completo.
    
    Estudiante simulado con 12 interacciones:
    - 4 sobre variables (Bloom 3-4): dominio sólido
    - 4 sobre bucles (Bloom 2-3): comprensión media
    - 3 sobre funciones (Bloom 2): superficie
    - 1 sobre recursión (Bloom 2): pregunta superficial, luego silencio
    
    Gaps esperados:
    - CRÍTICO: arrays (vecino de bucles y variables, nunca visitado;
      prerrequisito de muchos ejercicios prácticos)
    - MODERADO: depuración (prerrequisito práctico, nunca abordado)
    - METACOGNITIVO: recursión (visitado 1x a nivel bajo, silencio posterior
      donde estadísticamente debería haber preguntas dado que es el topic
      más difícil del corpus, difficulty=0.85)
    """
    detector = CognitiveGapDetector()
    student_id = "demo_student_001"
    
    # Construir knowledge map con documentos simulados del RAG
    # (En producción, estos son los chunks reales del rag_pipeline)
    fake_rag_docs = _generate_fake_rag_chunks()
    detector.build_knowledge_map(fake_rag_docs)
    
    # Simular 12 interacciones del estudiante
    base_time = datetime(2026, 2, 1, 9, 0, 0)
    
    # --- 4 interacciones sobre variables (Bloom 3-4, dominio sólido) ---
    for i in range(4):
        ts = (base_time + timedelta(hours=i*2)).isoformat()
        bloom = random.choice([3, 3, 4, 4])
        detector.record_interaction(
            student_id=student_id,
            detected_topics=["variables"],
            bloom_level=bloom,
            chunk_indices=[0, 1, 2],
            timestamp=ts,
        )
    
    # --- 4 interacciones sobre bucles (Bloom 2-3, comprensión media) ---
    for i in range(4):
        ts = (base_time + timedelta(hours=8 + i*2)).isoformat()
        bloom = random.choice([2, 2, 3, 3])
        detector.record_interaction(
            student_id=student_id,
            detected_topics=["bucles"],
            bloom_level=bloom,
            chunk_indices=[3, 4, 5],
            timestamp=ts,
        )
    
    # --- 3 interacciones sobre funciones (Bloom 2, superficie) ---
    for i in range(3):
        ts = (base_time + timedelta(hours=16 + i*3)).isoformat()
        detector.record_interaction(
            student_id=student_id,
            detected_topics=["funciones"],
            bloom_level=2,
            chunk_indices=[6, 7],
            timestamp=ts,
        )
    
    # --- 1 interacción sobre recursión (Bloom 2, superficial + silencio) ---
    ts = (base_time + timedelta(hours=25)).isoformat()
    detector.record_interaction(
        student_id=student_id,
        detected_topics=["recursión"],
        bloom_level=2,
        chunk_indices=[8],
        timestamp=ts,
    )
    
    # Forzar que el contador de interacciones desde última sonda permita inyección
    detector.interaction_counters[student_id] = 5
    
    return detector, student_id


def _generate_fake_rag_chunks() -> list[dict]:
    """
    Genera chunks RAG simulados para el dominio de programación.
    En producción, estos vienen de rag_pipeline.ingest_pdf().
    """
    chunks = [
        # Chunks sobre variables (0-2)
        {"id": "c0", "text": "Las variables son contenedores que almacenan valores en memoria. "
         "En Python, una variable se crea al asignarle un valor: x = 5. "
         "Los tipos de dato principales son int, float, string y bool.",
         "source": "tema1.pdf", "chunk_index": 0},
        {"id": "c1", "text": "La asignación de variables sigue la forma nombre = valor. "
         "Python usa tipado dinámico: el tipo se infiere del valor asignado. "
         "Las constantes se representan por convención en MAYÚSCULAS.",
         "source": "tema1.pdf", "chunk_index": 1},
        {"id": "c2", "text": "Las variables pueden cambiar de valor y tipo durante la ejecución. "
         "Para entrada de datos se usa input() que siempre devuelve string. "
         "Para mostrar resultados se usa print() con formato f-string.",
         "source": "tema1.pdf", "chunk_index": 2},
        # Chunks sobre bucles (3-5)
        {"id": "c3", "text": "Los bucles permiten repetir un bloque de código. "
         "El bucle for se usa cuando se conoce el número de iteraciones. "
         "El bucle while se usa cuando la condición de parada es dinámica.",
         "source": "tema2.pdf", "chunk_index": 3},
        {"id": "c4", "text": "La función range(n) genera una secuencia de 0 a n-1. "
         "Se puede usar con for para iterar un número fijo de veces. "
         "Las variables de control del bucle se actualizan en cada iteración.",
         "source": "tema2.pdf", "chunk_index": 4},
        {"id": "c5", "text": "Los bucles anidados permiten recorrer estructuras bidimensionales. "
         "Es importante controlar las variables de cada nivel de anidamiento. "
         "Los arrays y listas se recorren naturalmente con bucles for.",
         "source": "tema2.pdf", "chunk_index": 5},
        # Chunks sobre funciones (6-7)
        {"id": "c6", "text": "Una función es un bloque de código reutilizable definido con def. "
         "Las funciones reciben parámetros y devuelven valores con return. "
         "El paso de parámetros puede ser por posición o por nombre.",
         "source": "tema1.pdf", "chunk_index": 6},
        {"id": "c7", "text": "El alcance (scope) de las variables dentro de una función es local. "
         "Las variables globales se acceden con la palabra clave global. "
         "Las funciones pueden llamar a otras funciones, incluyendo a sí mismas.",
         "source": "tema1.pdf", "chunk_index": 7},
        # Chunks sobre recursión (8-9)
        {"id": "c8", "text": "La recursión es una técnica donde una función se llama a sí misma. "
         "Todo problema recursivo necesita un caso base que detenga la recursión. "
         "La pila de llamadas almacena el estado de cada invocación pendiente.",
         "source": "tema1.pdf", "chunk_index": 8},
        {"id": "c9", "text": "El factorial es un ejemplo clásico de recursión: n! = n * (n-1)! "
         "La recursión consume más memoria que un bucle equivalente por la pila. "
         "Fibonacci recursivo tiene complejidad exponencial sin memoización.",
         "source": "tema1.pdf", "chunk_index": 9},
        # Chunks sobre arrays (10-11)
        {"id": "c10", "text": "Las listas en Python son colecciones ordenadas y mutables. "
         "Se accede a elementos por índice: lista[0] es el primer elemento. "
         "Los bucles for son la forma natural de recorrer listas.",
         "source": "tema2.pdf", "chunk_index": 10},
        {"id": "c11", "text": "Las operaciones comunes con listas incluyen append, insert, remove. "
         "El slicing permite extraer sublistas: lista[1:3]. "
         "Las listas por comprensión combinan bucles y creación de listas.",
         "source": "tema2.pdf", "chunk_index": 11},
        # Chunks sobre depuración (12)
        {"id": "c12", "text": "Depurar es el proceso de encontrar y corregir errores en el código. "
         "Los errores de sintaxis se detectan antes de la ejecución. "
         "Los errores lógicos producen resultados incorrectos sin mensaje de error.",
         "source": "tema2.pdf", "chunk_index": 12},
    ]
    return chunks


def print_demo_report(detector: CognitiveGapDetector, student_id: str) -> str:
    """
    Genera un informe legible de la demo para presentación en entrevista.
    """
    lines = []
    lines.append("=" * 72)
    lines.append("COGNITIVE GAP DETECTOR — Informe de Demostración")
    lines.append("=" * 72)
    
    # Knowledge Map
    km = detector.knowledge_map
    lines.append(f"\n📊 KNOWLEDGE MAP: {len(km.nodes)} topics, "
                 f"{len(km.prerequisite_edges)} aristas prerequisito, "
                 f"{len(km.cooccurrence_edges)} aristas co-ocurrencia, "
                 f"{km.total_chunks} chunks RAG")
    
    lines.append("\n  Grafo de prerrequisitos:")
    for src, dst in km.prerequisite_edges:
        lines.append(f"    {src} → {dst}")
    
    # Student Map
    smap = detector.get_student_map(student_id)
    lines.append(f"\n👤 STUDENT MAP: {student_id}")
    lines.append(f"   Interacciones totales: {smap.total_interactions}")
    lines.append(f"   Cobertura: {smap.coverage_ratio:.0%}")
    lines.append(f"   Bloom promedio: {smap.avg_bloom_depth}")
    
    lines.append("\n  Topics visitados:")
    for topic, state in smap.topic_states.items():
        if state.visited:
            lines.append(f"    ✅ {topic}: {state.visit_count}x, "
                        f"Bloom max {state.max_bloom_level}, "
                        f"Bloom levels {state.bloom_levels_seen}")
    
    lines.append("\n  Topics NO visitados:")
    for topic, state in smap.topic_states.items():
        if not state.visited:
            lines.append(f"    ❌ {topic}")
    
    # Gaps
    gaps = detector.detect_critical_gaps(student_id)
    lines.append(f"\n🔍 GAPS DETECTADOS: {len(gaps)}")
    
    for gap in gaps:
        severity_icon = {"critical": "🔴", "moderate": "🟡", "low": "🟢"}.get(gap.severity, "⚪")
        lines.append(f"\n  {severity_icon} {gap.topic} [{gap.gap_type}] — {gap.severity.upper()}")
        lines.append(f"     Confianza: {gap.confidence:.0%}")
        if gap.dependent_topics_at_risk:
            lines.append(f"     Topics en riesgo: {gap.dependent_topics_at_risk}")
        for ev in gap.evidence:
            lines.append(f"     📋 {ev}")
    
    # Probes
    lines.append(f"\n🎯 SONDAS EPISTÉMICAS GENERADAS:")
    for gap in gaps:
        probes = detector.generate_epistemic_probes(gap)
        for probe in probes:
            lines.append(f"\n  [{probe.probe_type}] Para gap '{gap.topic}' ({gap.severity}):")
            lines.append(f"  Bloom objetivo: {probe.target_bloom} | Prioridad: {probe.priority}")
            lines.append(f"  Técnica Firlej: {probe.firlej_technique}")
            lines.append(f"  💬 \"{probe.probe_text}\"")
    
    # Probe injection
    injection = detector.get_probe_for_injection(student_id)
    lines.append(f"\n💉 INYECCIÓN EN SYSTEM PROMPT:")
    if injection:
        lines.append(f"  {injection}")
    else:
        lines.append("  [No hay sonda pendiente de inyección]")
    
    # Frase para entrevista
    lines.append("\n" + "─" * 72)
    lines.append("FRASE PARA ENTREVISTA:")
    lines.append("─" * 72)
    lines.append(
        '"Aplicamos técnicas de Knowledge Elicitation de la ingeniería del '
        'conocimiento — originalmente diseñadas para extraer conocimiento '
        'tácito de expertos — invertidas para detectar conocimiento ausente '
        'en estudiantes. Combinadas con gap analysis de metodología OSINT, '
        'el sistema identifica lo que el estudiante no sabe que no sabe y '
        'genera sondas epistémicas que hacen emerger esos gaps en la '
        'conversación natural."'
    )
    
    report = "\n".join(lines)
    return report


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    detector, student_id = generate_demo_data()
    report = print_demo_report(detector, student_id)
    print(report)
