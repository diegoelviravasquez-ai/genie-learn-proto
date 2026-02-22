"""
RAG QUALITY SENSOR — Sensor de Calidad RAG por Rephrase Sequences
═══════════════════════════════════════════════════════════════════════
Mide la calidad del sistema RAG desde la perspectiva del estudiante,
sin preguntárselo. El sensor es la secuencia de reformulación:
cuando un estudiante envía dos prompts semánticamente similares en
un intervalo corto, es una señal conductual de que la primera
respuesta no satisfizo su necesidad epistémica.

PRINCIPIO TÉCNICO:
La similitud semántica entre prompts consecutivos se mide con
TF-IDF + coseno (sin dependencias de GPU, ejecutable en cualquier
entorno académico). En producción, se substituiría por embeddings
text-embedding-3-small del pipeline RAG existente.

SEÑALES DETECTADAS:
1. Rephrase simple: mismo topic, reformulado (RAG no encontró chunk relevante)
2. Topic drift: el estudiante abandona el topic (frustración epistémica)
3. Precision failure: respuesta recuperada no era específica para el nivel del estudiante
4. Corpus gap: el topic no existe en el corpus RAG del curso

POSICIÓN EN EL ECOSISTEMA:
    rag_pipeline.py → [tras cada retrieval]
    rag_quality_sensor.py → detect_rephrase_sequence()
    system_event_logger.py → log_event(rag_degradation_event)
    researcher_view.py → panel "Calidad RAG en tiempo real"

INSERCIÓN EN RAG PIPELINE EXISTENTE (no invasiva):
    En rag_pipeline.py, tras cada retrieve():
        quality_report = rag_sensor.analyze_retrieval(
            student_id, query, chunks_retrieved, relevance_scores
        )
        if quality_report.degradation_detected:
            event_logger.log_event(quality_report.to_event())

Autor: Diego Elvira Vásquez · Ecosistema GENIE Learn · Feb 2026
Fundamentación: von Davier (2017) process data; Kapur (2008) productive failure.
"""

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ──────────────────────────────────────────────────────────────
# REPRESENTACIONES
# ──────────────────────────────────────────────────────────────

@dataclass
class RetrievalRecord:
    """Registro de una recuperación RAG individual."""
    student_id: str
    timestamp: str
    query: str
    chunks_retrieved: int
    relevance_scores: List[float]
    threshold_used: float
    above_threshold: int   # chunks que superaron el umbral
    topics_detected: List[str] = field(default_factory=list)


@dataclass
class RephrasePair:
    """
    Par de prompts consecutivos que constituyen una reformulación.
    
    La unidad de análisis del sensor de calidad RAG.
    """
    student_id: str
    timestamp_first: str
    timestamp_second: str
    delta_minutes: float
    
    prompt_first: str
    prompt_second: str
    semantic_similarity: float  # [0-1]
    
    # Diagnóstico
    degradation_type: str       # "rephrase" | "topic_drift" | "precision_failure" | "corpus_gap"
    severity: str               # "low" | "medium" | "high"
    
    # Contexto
    topics_first: List[str] = field(default_factory=list)
    topics_second: List[str] = field(default_factory=list)
    chunks_above_threshold_first: int = 0
    config_active: Dict = field(default_factory=dict)


@dataclass
class RAGQualityReport:
    """Informe de calidad RAG para el dashboard del investigador."""
    generated_at: str
    window_hours: float
    
    total_retrievals: int
    rephrase_pairs: List[RephrasePair]
    
    # Métricas agregadas
    rephrase_rate: float          # % de queries seguidas de rephrase
    mean_similarity_on_rephrase: float
    most_problematic_topics: List[str]
    corpus_gaps: List[str]        # topics con 0 chunks sobre umbral
    
    # Recomendaciones accionables
    recommendations: List[str]


# ──────────────────────────────────────────────────────────────
# MOTOR DE DETECCIÓN
# ──────────────────────────────────────────────────────────────

class RAGQualitySensor:
    """
    Sensor de calidad RAG basado en comportamiento observado del estudiante.
    
    No necesita ground truth de relevancia manual.
    No pregunta al estudiante si la respuesta fue útil.
    Detecta calidad degradada por las reformulaciones.
    
    TRUCO ANALÍTICO:
    El estudiante que reformula un prompt en menos de 3 minutos
    es más informativo que cualquier métrica NDCG calculada offline.
    Es evaluación implícita en tiempo real.
    """

    REPHRASE_WINDOW_MINUTES = 5       # ventana para considerar reformulación
    HIGH_SIMILARITY_THRESHOLD = 0.45  # por encima = mismo intent
    CORPUS_GAP_THRESHOLD = 0         # 0 chunks relevantes = gap en corpus

    def __init__(self):
        self._retrieval_history: List[RetrievalRecord] = []
        self._student_last_query: Dict[str, RetrievalRecord] = {}
        self._topic_corpus_stats: Dict[str, Dict] = defaultdict(
            lambda: {"total_queries": 0, "chunks_above_threshold": 0}
        )

    def record_retrieval(self, record: RetrievalRecord) -> Optional[RephrasePair]:
        """
        Registra una recuperación RAG y detecta si es una reformulación.
        
        Retorna RephrasePair si se detecta degradación, None si es primera query.
        Se llama DESPUÉS de cada retrieve() en rag_pipeline.py.
        """
        # Actualizar estadísticas por topic
        for topic in record.topics_detected:
            stats = self._topic_corpus_stats[topic]
            stats["total_queries"] += 1
            stats["chunks_above_threshold"] += record.above_threshold

        self._retrieval_history.append(record)

        # Verificar si hay una query previa del mismo estudiante
        prev = self._student_last_query.get(record.student_id)
        rephrase_pair = None

        if prev is not None:
            delta = self._delta_minutes(prev.timestamp, record.timestamp)
            
            if delta <= self.REPHRASE_WINDOW_MINUTES:
                similarity = self._compute_similarity(prev.query, record.query)
                
                if similarity >= self.HIGH_SIMILARITY_THRESHOLD:
                    # Es una reformulación — señal de calidad degradada
                    degradation_type = self._classify_degradation(
                        prev, record, similarity
                    )
                    severity = self._assess_severity(similarity, delta, prev.above_threshold)
                    
                    rephrase_pair = RephrasePair(
                        student_id=record.student_id,
                        timestamp_first=prev.timestamp,
                        timestamp_second=record.timestamp,
                        delta_minutes=round(delta, 2),
                        prompt_first=prev.query,
                        prompt_second=record.query,
                        semantic_similarity=round(similarity, 3),
                        degradation_type=degradation_type,
                        severity=severity,
                        topics_first=prev.topics_detected,
                        topics_second=record.topics_detected,
                        chunks_above_threshold_first=prev.above_threshold,
                    )

        # Actualizar last query
        self._student_last_query[record.student_id] = record
        return rephrase_pair

    def generate_quality_report(self, window_hours: float = 24.0) -> RAGQualityReport:
        """
        Genera informe completo de calidad RAG.
        
        El panel "Calidad RAG" del researcher_view: qué funciona,
        qué gaps existen, y qué añadir al corpus para mejorar.
        """
        cutoff = datetime.now() - timedelta(hours=window_hours)
        cutoff_str = cutoff.isoformat()

        recent = [r for r in self._retrieval_history if r.timestamp >= cutoff_str]

        # Detectar todos los rephrase pairs en la ventana
        all_rephrase_pairs = []
        student_recent: Dict[str, List[RetrievalRecord]] = defaultdict(list)
        for r in sorted(recent, key=lambda x: x.timestamp):
            student_recent[r.student_id].append(r)

        for student_id, records in student_recent.items():
            for i in range(len(records) - 1):
                prev = records[i]
                curr = records[i + 1]
                delta = self._delta_minutes(prev.timestamp, curr.timestamp)

                if delta <= self.REPHRASE_WINDOW_MINUTES:
                    sim = self._compute_similarity(prev.query, curr.query)
                    if sim >= self.HIGH_SIMILARITY_THRESHOLD:
                        deg_type = self._classify_degradation(prev, curr, sim)
                        severity = self._assess_severity(sim, delta, prev.above_threshold)
                        all_rephrase_pairs.append(RephrasePair(
                            student_id=student_id,
                            timestamp_first=prev.timestamp,
                            timestamp_second=curr.timestamp,
                            delta_minutes=round(delta, 2),
                            prompt_first=prev.query,
                            prompt_second=curr.query,
                            semantic_similarity=round(sim, 3),
                            degradation_type=deg_type,
                            severity=severity,
                            topics_first=prev.topics_detected,
                            topics_second=curr.topics_detected,
                            chunks_above_threshold_first=prev.above_threshold,
                        ))

        rephrase_rate = len(all_rephrase_pairs) / max(len(recent), 1)
        mean_similarity = (
            sum(p.semantic_similarity for p in all_rephrase_pairs) / len(all_rephrase_pairs)
            if all_rephrase_pairs else 0.0
        )

        # Topics más problemáticos
        topic_problems: Dict[str, int] = defaultdict(int)
        for p in all_rephrase_pairs:
            for t in p.topics_first:
                topic_problems[t] += 1
        most_problematic = sorted(topic_problems, key=topic_problems.get, reverse=True)[:5]

        # Corpus gaps: topics con 0 chunks sobre umbral consistentemente
        corpus_gaps = [
            topic
            for topic, stats in self._topic_corpus_stats.items()
            if stats["total_queries"] >= 3 and stats["chunks_above_threshold"] == 0
        ]

        recommendations = self._generate_recommendations(
            rephrase_rate, most_problematic, corpus_gaps, all_rephrase_pairs
        )

        return RAGQualityReport(
            generated_at=datetime.now().isoformat(),
            window_hours=window_hours,
            total_retrievals=len(recent),
            rephrase_pairs=all_rephrase_pairs,
            rephrase_rate=round(rephrase_rate, 3),
            mean_similarity_on_rephrase=round(mean_similarity, 3),
            most_problematic_topics=most_problematic,
            corpus_gaps=corpus_gaps,
            recommendations=recommendations,
        )

    def get_student_rag_quality(self, student_id: str) -> Dict:
        """
        Calidad RAG específica para un estudiante.
        
        Permite detectar si un estudiante específico tiene
        más dificultad con el corpus que sus compañeros —
        señal de que el corpus no cubre su nivel de entrada.
        """
        student_records = [r for r in self._retrieval_history if r.student_id == student_id]
        
        if not student_records:
            return {"student_id": student_id, "data": "no_records"}

        mean_relevance = sum(
            max(r.relevance_scores) if r.relevance_scores else 0.0
            for r in student_records
        ) / len(student_records)

        zero_threshold_rate = sum(
            1 for r in student_records if r.above_threshold == 0
        ) / len(student_records)

        return {
            "student_id": student_id,
            "total_queries": len(student_records),
            "mean_max_relevance": round(mean_relevance, 3),
            "zero_threshold_rate": round(zero_threshold_rate, 3),
            "corpus_coverage": round(1 - zero_threshold_rate, 3),
            "topics_queried": list({t for r in student_records for t in r.topics_detected}),
        }

    # ──────────────────────────────────────────────
    # HELPERS SEMÁNTICOS
    # ──────────────────────────────────────────────

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Similitud semántica aproximada con TF-IDF + coseno.
        
        Sin GPU, sin API calls, sin dependencias pesadas.
        Suficiente para detectar reformulaciones del mismo intent.
        En producción: sustituir por embeddings del pipeline RAG existente.
        """
        def tokenize(text: str) -> List[str]:
            text = text.lower()
            text = re.sub(r'[^\w\sáéíóúñü]', '', text)
            stopwords = {'el', 'la', 'los', 'las', 'un', 'una', 'de', 'del',
                        'en', 'con', 'y', 'o', 'que', 'qué', 'es', 'cómo',
                        'me', 'mi', 'a', 'al', 'por', 'para', 'no', 'lo',
                        'si', 'se', 'su', 'sus', 'he', 'ha', 'hay', 'the',
                        'how', 'what', 'is', 'are', 'can', 'i', 'my', 'you'}
            return [t for t in text.split() if t not in stopwords and len(t) > 2]

        tokens1 = tokenize(text1)
        tokens2 = tokenize(text2)

        if not tokens1 or not tokens2:
            return 0.0

        # Vocabulario combinado
        vocab = list(set(tokens1 + tokens2))
        
        # Vectores de frecuencia
        def freq_vector(tokens: List[str]) -> List[float]:
            counts = {}
            for t in tokens:
                counts[t] = counts.get(t, 0) + 1
            return [counts.get(w, 0) for w in vocab]

        v1 = freq_vector(tokens1)
        v2 = freq_vector(tokens2)

        # Coseno
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(b * b for b in v2))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot / (mag1 * mag2)

    def _classify_degradation(
        self,
        prev: RetrievalRecord,
        curr: RetrievalRecord,
        similarity: float,
    ) -> str:
        """Clasifica el tipo de degradación RAG."""
        # Gap de corpus: primer query tuvo 0 chunks relevantes
        if prev.above_threshold == 0:
            return "corpus_gap"

        # Cambio de topic: misma formulación pero topics distintos
        prev_topics = set(prev.topics_detected)
        curr_topics = set(curr.topics_detected)
        if prev_topics and curr_topics and not prev_topics.intersection(curr_topics):
            return "topic_drift"

        # Alta similitud pero respuesta insatisfactoria
        if similarity > 0.7:
            return "precision_failure"

        return "rephrase"

    def _assess_severity(self, similarity: float, delta_minutes: float, chunks_above: int) -> str:
        """Evalúa la severidad de la degradación."""
        severity_score = 0.0

        # Alta similitud = clara señal de degradación
        severity_score += similarity * 0.4

        # Respuesta muy rápida = el estudiante no leyó la respuesta o fue claramente inútil
        if delta_minutes < 1.0:
            severity_score += 0.4
        elif delta_minutes < 2.0:
            severity_score += 0.2

        # Sin chunks relevantes es la señal más severa
        if chunks_above == 0:
            severity_score += 0.3

        if severity_score >= 0.6:
            return "high"
        elif severity_score >= 0.3:
            return "medium"
        else:
            return "low"

    def _generate_recommendations(
        self,
        rephrase_rate: float,
        problematic_topics: List[str],
        corpus_gaps: List[str],
        pairs: List[RephrasePair],
    ) -> List[str]:
        """Genera recomendaciones accionables para el docente."""
        recs = []

        if rephrase_rate > 0.3:
            recs.append(
                f"Alta tasa de reformulación ({rephrase_rate:.0%}): "
                f"el corpus RAG no está cubriendo las necesidades del curso. "
                f"Considerar añadir más documentos o ajustar el umbral de relevancia."
            )

        for topic in corpus_gaps[:3]:
            recs.append(
                f"Gap de corpus en '{topic}': ninguna query sobre este topic "
                f"recupera chunks relevantes. Añadir material específico al corpus."
            )

        high_severity = [p for p in pairs if p.severity == "high"]
        if len(high_severity) > 5:
            recs.append(
                f"{len(high_severity)} casos de degradación severa detectados. "
                f"Topics más afectados: {problematic_topics[:2]}. "
                f"Prioridad alta para revisión del corpus."
            )

        if not recs:
            recs.append("Calidad RAG satisfactoria en esta ventana temporal.")

        return recs

    @staticmethod
    def _delta_minutes(t1_str: str, t2_str: str) -> float:
        """Diferencia en minutos entre dos timestamps ISO."""
        try:
            t1 = datetime.fromisoformat(t1_str)
            t2 = datetime.fromisoformat(t2_str)
            return abs((t2 - t1).total_seconds() / 60)
        except (ValueError, TypeError):
            return 999.0


# ──────────────────────────────────────────────
# DEMO AUTOEJECTABLE
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import time
    from datetime import datetime, timedelta

    sensor = RAGQualitySensor()

    print("═" * 60)
    print("RAG QUALITY SENSOR — Demo de detección de degradación")
    print("═" * 60)

    # Simular sesión con patrones de calidad variables
    now = datetime.now()

    retrievals = [
        # Est A: pregunta sobre recursión, obtiene chunk relevante — OK
        ("est_A", now - timedelta(minutes=30), "¿qué es la recursión?",
         3, [0.82, 0.71, 0.55], 0.4, 2, ["recursión"]),
        
        # Est B: pregunta sobre recursión, 0 chunks — corpus gap
        ("est_B", now - timedelta(minutes=25), "¿cómo funciona la recursión en Java?",
         1, [0.15], 0.4, 0, ["recursión"]),
        
        # Est B: reformula 2 minutos después — REPHRASE DETECTADO
        ("est_B", now - timedelta(minutes=23), "explícame recursión paso a paso Java",
         1, [0.22], 0.4, 0, ["recursión"]),
        
        # Est C: pregunta sobre punteros
        ("est_C", now - timedelta(minutes=15), "¿qué es un puntero?",
         2, [0.45, 0.38], 0.4, 1, ["variables"]),
        
        # Est C: reformula 1.5 min después — PRECISION FAILURE
        ("est_C", now - timedelta(minutes=13.5), "dime exactamente qué es un puntero en C",
         2, [0.48, 0.41], 0.4, 1, ["variables"]),
        
        # Est A: pregunta normal, diferente topic — OK
        ("est_A", now - timedelta(minutes=10), "¿cómo funciona el bucle for?",
         3, [0.91, 0.85, 0.72], 0.4, 3, ["bucles"]),
    ]

    detected_pairs = []
    for student_id, ts, query, n_chunks, scores, threshold, above, topics in retrievals:
        record = RetrievalRecord(
            student_id=student_id,
            timestamp=ts.isoformat(),
            query=query,
            chunks_retrieved=n_chunks,
            relevance_scores=scores,
            threshold_used=threshold,
            above_threshold=above,
            topics_detected=topics,
        )
        pair = sensor.record_retrieval(record)
        if pair:
            detected_pairs.append(pair)
            print(f"\n⚠️  REPHRASE DETECTADO — {student_id}")
            print(f"   Tipo: {pair.degradation_type} | Severidad: {pair.severity}")
            print(f"   Query 1: '{pair.prompt_first}'")
            print(f"   Query 2: '{pair.prompt_second}'")
            print(f"   Similitud: {pair.semantic_similarity:.2f} | Δt: {pair.delta_minutes} min")

    report = sensor.generate_quality_report(window_hours=1.0)
    print(f"\n📊 INFORME DE CALIDAD RAG")
    print(f"   Queries analizados: {report.total_retrievals}")
    print(f"   Tasa de rephrase: {report.rephrase_rate:.0%}")
    print(f"   Topics problemáticos: {report.most_problematic_topics}")
    print(f"   Corpus gaps: {report.corpus_gaps}")
    print(f"\n💡 RECOMENDACIONES:")
    for rec in report.recommendations:
        print(f"   • {rec}")

    print("\n═" * 60)
