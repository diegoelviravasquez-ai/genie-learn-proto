"""
CROSS NODE SIGNAL — Inteligencia Colectiva Inter-Institucional
═══════════════════════════════════════════════════════════════════════
Convierte GENIE Learn de tres sistemas independientes (UC3M, UVa, UPF)
en un ecosistema de inteligencia educativa distribuida.

HIPÓTESIS TÉCNICA:
Los tres nodos universitarios no están sincronizados en el temario.
UC3M (Madrid) suele ir 10-15 días por delante de UVa (Valladolid)
en asignaturas equivalentes de programación. Los patrones de dificultad
que UC3M experimenta hoy son señal predictiva para UVa.
Si UC3M muestra spike de gaming en recursión + Bloom medio bajo,
UVa puede preparar configuraciones anticipatorias antes de que
su propio cohorte llegue a ese concepto.

ESTO ES ÚNICO:
Ningún sistema de Learning Analytics en producción hace esto porque
ninguno tiene la arquitectura multi-nodo real que GENIE Learn ya tiene.
Los tres nodos están explícitamente en el paper CSEDU 2025.
La propagación de señales entre nodos solo requiere una API REST mínima
o incluso un fichero JSON compartido por SFTP seguro —
no requiere cloud, no requiere cambios de infraestructura.

PRIVACIDAD:
Las señales son AGREGADAS y ANÓNIMAS. Se propaga el patrón del cohorte,
nunca datos individuales de estudiantes. Esto es coherente con el
enfoque values-sensitive design del proyecto GENIE Learn.

POSICIÓN EN EL ECOSISTEMA:
    system_event_logger.py → log_event() [events locales]
    cross_node_signal.py   → emit_signal() [agrega + envía]
                           → receive_signal() [recibe + procesa]
    temporal_config_advisor.py → usa señales para sugerir configs anticipatorias

Autor: Diego Elvira Vásquez · Ecosistema GENIE Learn · Feb 2026
Fundamentación: Tabuenca et al. (2021) SLEs; Siemens (2005) Connectivism;
               Hernández-Leo et al. (2019) Community Analytics.
"""

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Literal
from collections import defaultdict
import hashlib


NodeID = Literal["uva", "uc3m", "upf", "upm"]
SignalType = Literal[
    "difficulty_spike",          # spike de dificultad en un topic
    "gaming_surge",              # incremento de gaming en un topic
    "bloom_plateau",             # estancamiento colectivo en nivel Bloom
    "autonomy_drop",             # caída de autonomía en el cohorte
    "topic_confusion_pattern",   # patrón específico de confusión conceptual
    "config_effectiveness",      # qué config funcionó bien en este nodo
]


# ──────────────────────────────────────────────────────────────
# REPRESENTACIONES
# ──────────────────────────────────────────────────────────────

@dataclass
class CrossNodeSignal:
    """
    Señal agregada y anónima de un nodo hacia los demás.
    
    NUNCA contiene datos individuales. Solo estadísticas del cohorte.
    La privacidad no es una restricción — es un principio de diseño.
    """
    # Identificación
    signal_id: str
    emitted_at: str
    source_node: NodeID
    target_nodes: List[NodeID]  # nodos destinatarios

    # Clasificación
    signal_type: SignalType
    topic: str                  # el concepto académico afectado
    urgency: Literal["informational", "advisory", "alert"]

    # Datos del cohorte (todos agregados y anónimos)
    n_students_affected: int
    bloom_mean: float
    gaming_rate: float
    rephrase_rate: float        # % de queries seguidas de reformulación
    autonomy_mean: float
    
    # La señal predictiva concreta
    predicted_days_until_arrival: Optional[int]  # días hasta que UVa llega a este topic
    
    # Recomendación configuracional específica
    recommended_config_changes: Dict  # qué cambiar en el nodo destino
    recommended_corpus_additions: List[str]  # qué añadir al RAG del nodo destino
    
    # Contexto adicional
    config_that_helped: Optional[Dict]  # config activa en el nodo emisor cuando mejoró
    evidence_strength: float  # [0-1] cuántos estudiantes respaldan la señal

    # Hash de verificación de integridad (no modifiable)
    integrity_hash: str = field(default="")

    def __post_init__(self):
        if not self.integrity_hash:
            payload = f"{self.signal_id}{self.source_node}{self.topic}{self.emitted_at}"
            self.integrity_hash = hashlib.md5(payload.encode()).hexdigest()[:8]


@dataclass
class NodeTemporalProfile:
    """
    Perfil temporal de un nodo: calendario académico y velocidad temática.
    
    Permite calcular el desfase entre nodos para señales predictivas.
    """
    node_id: NodeID
    course_start_date: str
    course_end_date: str
    topics_sequence: List[str]         # orden en que el curso cubre los topics
    topic_arrival_dates: Dict[str, str] # {topic: fecha esperada de llegada al topic}
    typical_days_per_topic: float       # velocidad media del curso


@dataclass
class ReceivedSignalAnalysis:
    """
    Análisis de una señal recibida: qué hacer con ella en el nodo destino.
    """
    signal: CrossNodeSignal
    received_at: str
    target_node: NodeID
    
    # ¿Es relevante para este nodo?
    is_relevant: bool
    relevance_reason: str
    
    # ¿Cuándo llegará el nodo a este topic?
    estimated_arrival_days: Optional[int]
    
    # Acciones recomendadas para el docente
    preemptive_config: Optional[Dict]
    corpus_additions_needed: List[str]
    urgency_for_target: str
    
    # Texto accionable para el dashboard docente
    teacher_alert: str


# ──────────────────────────────────────────────────────────────
# MOTOR DE SEÑALES INTER-NODO
# ──────────────────────────────────────────────────────────────

class CrossNodeSignalEngine:
    """
    Motor de inteligencia colectiva del ecosistema GENIE Learn.
    
    Corre UNA VEZ AL DÍA (cron job o Celery beat) en cada nodo.
    Agrega datos locales del día → emite señales hacia otros nodos →
    recibe señales de otros nodos → procesa y sirve al dashboard docente.
    
    El coste computacional es mínimo: es aritmética sobre conteos.
    No requiere modelos de ML ni GPU.
    """

    # Umbral mínimo para emitir señal (evitar ruido)
    MIN_STUDENTS_FOR_SIGNAL = 5       # al menos 5 estudiantes afectados
    DIFFICULTY_SPIKE_THRESHOLD = 0.35  # >35% de rephrase en un topic
    GAMING_SURGE_THRESHOLD = 0.4      # >40% de gaming en un topic
    BLOOM_PLATEAU_THRESHOLD = 0.5     # mismo nivel Bloom >50% del cohorte 5 días seguidos

    def __init__(self, node_id: NodeID, node_profile: Optional[NodeTemporalProfile] = None):
        self.node_id = node_id
        self.node_profile = node_profile
        self.emitted_signals: List[CrossNodeSignal] = []
        self.received_signals: List[CrossNodeSignal] = []
        self._topic_daily_stats: Dict[str, List[Dict]] = defaultdict(list)  # topic → stats diarios

    def record_daily_topic_stats(
        self,
        topic: str,
        n_students: int,
        bloom_mean: float,
        gaming_rate: float,
        rephrase_rate: float,
        autonomy_mean: float,
        active_config: Dict,
    ):
        """
        Registra estadísticas diarias por topic.
        
        Se llama al final de cada día con los agregados del día.
        Es el input del motor de señales.
        """
        self._topic_daily_stats[topic].append({
            "date": datetime.now().date().isoformat(),
            "n_students": n_students,
            "bloom_mean": bloom_mean,
            "gaming_rate": gaming_rate,
            "rephrase_rate": rephrase_rate,
            "autonomy_mean": autonomy_mean,
            "active_config": active_config,
        })

    def detect_and_emit_signals(
        self,
        target_nodes: List[NodeID],
    ) -> List[CrossNodeSignal]:
        """
        Detecta patrones en los datos locales y emite señales hacia otros nodos.
        
        Corre al final del día. Retorna las señales emitidas.
        """
        signals = []

        for topic, daily_stats in self._topic_daily_stats.items():
            recent = daily_stats[-3:] if len(daily_stats) >= 3 else daily_stats
            
            if not recent:
                continue

            n_students = recent[-1]["n_students"]
            
            if n_students < self.MIN_STUDENTS_FOR_SIGNAL:
                continue  # no emitir señal sin masa crítica

            current = recent[-1]

            # ─── Detectar spike de dificultad ───
            if current["rephrase_rate"] > self.DIFFICULTY_SPIKE_THRESHOLD:
                evidence = self._compute_evidence_strength(recent, "rephrase_rate", self.DIFFICULTY_SPIKE_THRESHOLD)
                signal = CrossNodeSignal(
                    signal_id=f"sig_{self.node_id}_{topic}_diff_{datetime.now().date().isoformat()}",
                    emitted_at=datetime.now().isoformat(),
                    source_node=self.node_id,
                    target_nodes=target_nodes,
                    signal_type="difficulty_spike",
                    topic=topic,
                    urgency="advisory" if current["rephrase_rate"] < 0.5 else "alert",
                    n_students_affected=n_students,
                    bloom_mean=current["bloom_mean"],
                    gaming_rate=current["gaming_rate"],
                    rephrase_rate=current["rephrase_rate"],
                    autonomy_mean=current["autonomy_mean"],
                    predicted_days_until_arrival=None,  # calculado por el receptor
                    recommended_config_changes=self._suggest_config_for_difficulty(current),
                    recommended_corpus_additions=self._suggest_corpus_additions(topic, current),
                    config_that_helped=None,  # se completará si el pattern se resuelve
                    evidence_strength=evidence,
                )
                signals.append(signal)

            # ─── Detectar surge de gaming ───
            if current["gaming_rate"] > self.GAMING_SURGE_THRESHOLD:
                evidence = self._compute_evidence_strength(recent, "gaming_rate", self.GAMING_SURGE_THRESHOLD)
                signal = CrossNodeSignal(
                    signal_id=f"sig_{self.node_id}_{topic}_gaming_{datetime.now().date().isoformat()}",
                    emitted_at=datetime.now().isoformat(),
                    source_node=self.node_id,
                    target_nodes=target_nodes,
                    signal_type="gaming_surge",
                    topic=topic,
                    urgency="alert",
                    n_students_affected=n_students,
                    bloom_mean=current["bloom_mean"],
                    gaming_rate=current["gaming_rate"],
                    rephrase_rate=current["rephrase_rate"],
                    autonomy_mean=current["autonomy_mean"],
                    predicted_days_until_arrival=None,
                    recommended_config_changes={
                        "block_direct_solutions": True,
                        "max_daily_prompts": "reduce_by_30pct",
                        "scaffolding_mode": "socratic",
                    },
                    recommended_corpus_additions=[
                        f"Ejercicios de comprensión conceptual sobre {topic} (no code-completion)",
                        f"Material de auto-evaluación sobre {topic}",
                    ],
                    config_that_helped=None,
                    evidence_strength=evidence,
                )
                signals.append(signal)

            # ─── Detectar plateau de Bloom ───
            if len(recent) >= 3:
                bloom_values = [s["bloom_mean"] for s in recent]
                bloom_variance = self._variance(bloom_values)
                if bloom_variance < 0.05 and bloom_values[-1] <= 2.5:
                    signal = CrossNodeSignal(
                        signal_id=f"sig_{self.node_id}_{topic}_plateau_{datetime.now().date().isoformat()}",
                        emitted_at=datetime.now().isoformat(),
                        source_node=self.node_id,
                        target_nodes=target_nodes,
                        signal_type="bloom_plateau",
                        topic=topic,
                        urgency="advisory",
                        n_students_affected=n_students,
                        bloom_mean=bloom_values[-1],
                        gaming_rate=current["gaming_rate"],
                        rephrase_rate=current["rephrase_rate"],
                        autonomy_mean=current["autonomy_mean"],
                        predicted_days_until_arrival=None,
                        recommended_config_changes={
                            "scaffolding_mode": "socratic",
                            "forced_hallucination_pct": 0.10,
                        },
                        recommended_corpus_additions=[
                            f"Preguntas de análisis sobre {topic} (Bloom 3-4)",
                            f"Casos de estudio comparativos sobre {topic}",
                        ],
                        config_that_helped=None,
                        evidence_strength=0.7,
                    )
                    signals.append(signal)

        self.emitted_signals.extend(signals)
        return signals

    def receive_signal(self, signal: CrossNodeSignal) -> ReceivedSignalAnalysis:
        """
        Procesa una señal recibida de otro nodo.
        
        Determina si es relevante para este nodo y con cuánta anticipación
        llega. Genera el alerta accionable para el dashboard docente.
        """
        self.received_signals.append(signal)

        # Verificar integridad
        if not self._verify_integrity(signal):
            return ReceivedSignalAnalysis(
                signal=signal,
                received_at=datetime.now().isoformat(),
                target_node=self.node_id,
                is_relevant=False,
                relevance_reason="Señal rechazada: hash de integridad inválido.",
                estimated_arrival_days=None,
                preemptive_config=None,
                corpus_additions_needed=[],
                urgency_for_target="none",
                teacher_alert="",
            )

        # Estimar cuándo llegará este topic a nuestro cohorte
        arrival_days = self._estimate_arrival_days(signal.topic)

        is_relevant = (
            arrival_days is not None and
            arrival_days > 0 and  # todavía no hemos llegado
            arrival_days <= 21    # dentro de los próximos 21 días
        )

        if not is_relevant:
            return ReceivedSignalAnalysis(
                signal=signal,
                received_at=datetime.now().isoformat(),
                target_node=self.node_id,
                is_relevant=False,
                relevance_reason=f"Topic '{signal.topic}' no relevante para este nodo en los próximos 21 días.",
                estimated_arrival_days=arrival_days,
                preemptive_config=None,
                corpus_additions_needed=[],
                urgency_for_target="none",
                teacher_alert="",
            )

        # Calcular urgencia local
        if arrival_days <= 5:
            urgency_local = "alert"
        elif arrival_days <= 10:
            urgency_local = "advisory"
        else:
            urgency_local = "informational"

        # Generar alerta para el docente
        teacher_alert = self._generate_teacher_alert(signal, arrival_days, urgency_local)

        return ReceivedSignalAnalysis(
            signal=signal,
            received_at=datetime.now().isoformat(),
            target_node=self.node_id,
            is_relevant=True,
            relevance_reason=f"Topic '{signal.topic}' llega en ~{arrival_days} días.",
            estimated_arrival_days=arrival_days,
            preemptive_config=signal.recommended_config_changes,
            corpus_additions_needed=signal.recommended_corpus_additions,
            urgency_for_target=urgency_local,
            teacher_alert=teacher_alert,
        )

    def get_pending_alerts(self) -> List[ReceivedSignalAnalysis]:
        """
        Retorna todas las alertas relevantes pendientes para el dashboard docente.
        """
        alerts = []
        for signal in self.received_signals:
            analysis = self.receive_signal(signal)
            if analysis.is_relevant:
                alerts.append(analysis)
        return alerts

    def export_signal(self, signal: CrossNodeSignal) -> str:
        """Serializa la señal para transmisión entre nodos (JSON)."""
        return json.dumps({
            "signal_id": signal.signal_id,
            "emitted_at": signal.emitted_at,
            "source_node": signal.source_node,
            "target_nodes": signal.target_nodes,
            "signal_type": signal.signal_type,
            "topic": signal.topic,
            "urgency": signal.urgency,
            "n_students_affected": signal.n_students_affected,
            "bloom_mean": signal.bloom_mean,
            "gaming_rate": signal.gaming_rate,
            "rephrase_rate": signal.rephrase_rate,
            "autonomy_mean": signal.autonomy_mean,
            "predicted_days_until_arrival": signal.predicted_days_until_arrival,
            "recommended_config_changes": signal.recommended_config_changes,
            "recommended_corpus_additions": signal.recommended_corpus_additions,
            "config_that_helped": signal.config_that_helped,
            "evidence_strength": signal.evidence_strength,
            "integrity_hash": signal.integrity_hash,
        }, ensure_ascii=False, indent=2)

    @staticmethod
    def import_signal(json_str: str) -> CrossNodeSignal:
        """Deserializa una señal recibida."""
        data = json.loads(json_str)
        sig = CrossNodeSignal(**{k: v for k, v in data.items()})
        return sig

    # ──────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────

    def _compute_evidence_strength(
        self, stats: List[Dict], metric: str, threshold: float
    ) -> float:
        """Qué fracción de los días recientes supera el umbral."""
        above = sum(1 for s in stats if s.get(metric, 0) > threshold)
        return above / len(stats)

    def _suggest_config_for_difficulty(self, stats: Dict) -> Dict:
        """Sugiere configuración anticipatoria basada en el patrón de dificultad."""
        config = {}
        if stats["bloom_mean"] < 2.5:
            config["scaffolding_mode"] = "socratic"
            config["max_response_length"] = 1500  # respuestas más detalladas
        if stats["rephrase_rate"] > 0.4:
            config["use_rag"] = True
            config["rag_top_k"] = 5  # recuperar más chunks
        if stats["autonomy_mean"] < 0.3:
            config["block_direct_solutions"] = False  # momentáneamente
        return config

    def _suggest_corpus_additions(self, topic: str, stats: Dict) -> List[str]:
        """Sugiere qué añadir al corpus RAG para este topic."""
        additions = []
        if stats["rephrase_rate"] > 0.35:
            additions.append(f"Explicaciones de {topic} en múltiples niveles de abstracción")
            additions.append(f"Ejemplos paso a paso de {topic} con errores comunes documentados")
        if stats["bloom_mean"] < 2.0:
            additions.append(f"Material introductorio de {topic} con analogías del mundo real")
        return additions

    def _estimate_arrival_days(self, topic: str) -> Optional[int]:
        """
        Estima cuántos días faltan para que este nodo llegue al topic.
        
        Si hay perfil temporal configurado, usa las fechas exactas.
        Si no, usa heurística basada en el estado local del topic.
        """
        if self.node_profile and topic in self.node_profile.topic_arrival_dates:
            arrival_date = datetime.fromisoformat(self.node_profile.topic_arrival_dates[topic])
            now = datetime.now()
            delta = (arrival_date - now).days
            return max(0, delta)

        # Heurística: si el topic no está en stats locales aún, estimar 14 días
        if topic not in self._topic_daily_stats:
            return 14

        # Si ya tenemos stats del topic, ya llegamos
        return 0

    def _verify_integrity(self, signal: CrossNodeSignal) -> bool:
        """Verifica que la señal no fue alterada en tránsito."""
        expected = hashlib.md5(
            f"{signal.signal_id}{signal.source_node}{signal.topic}{signal.emitted_at}".encode()
        ).hexdigest()[:8]
        return signal.integrity_hash == expected

    def _generate_teacher_alert(
        self, signal: CrossNodeSignal, arrival_days: int, urgency: str
    ) -> str:
        emoji_map = {"alert": "🔴", "advisory": "🟡", "informational": "🔵"}
        emoji = emoji_map.get(urgency, "⚪")

        signal_descriptions = {
            "difficulty_spike": "spike de dificultad",
            "gaming_surge": "incremento de gaming",
            "bloom_plateau": "estancamiento cognitivo colectivo",
            "autonomy_drop": "caída de autonomía",
            "topic_confusion_pattern": "patrón de confusión conceptual",
            "config_effectiveness": "configuración efectiva",
        }
        desc = signal_descriptions.get(signal.signal_type, signal.signal_type)

        alert = (
            f"{emoji} SEÑAL INTER-NODO ({signal.source_node.upper()} → {self.node_id.upper()}): "
            f"{desc.capitalize()} en '{signal.topic}'. "
            f"Llegará a tu curso en ~{arrival_days} días "
            f"({signal.n_students_affected} estudiantes afectados en {signal.source_node.upper()}, "
            f"Bloom medio: {signal.bloom_mean:.1f}). "
        )

        if signal.recommended_config_changes:
            changes = ", ".join(f"{k}={v}" for k, v in signal.recommended_config_changes.items())
            alert += f"Configuración anticipatoria sugerida: {changes}. "

        if signal.recommended_corpus_additions:
            alert += f"Añadir al corpus: {signal.recommended_corpus_additions[0]}."

        return alert

    @staticmethod
    def _variance(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)


# ──────────────────────────────────────────────
# DEMO AUTOEJECTABLE
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from datetime import date

    print("═" * 65)
    print("CROSS NODE SIGNAL — Demo de inteligencia colectiva inter-nodo")
    print("═" * 65)
    print()

    # ─── Nodo emisor: UC3M (Madrid) — va por delante en el temario ───
    profile_uc3m = NodeTemporalProfile(
        node_id="uc3m",
        course_start_date="2026-01-12",
        course_end_date="2026-06-15",
        topics_sequence=["variables", "bucles", "funciones", "arrays", "recursión", "punteros"],
        topic_arrival_dates={
            "recursión": "2026-03-01",
        },
        typical_days_per_topic=12,
    )
    uc3m_engine = CrossNodeSignalEngine(node_id="uc3m", node_profile=profile_uc3m)

    # Simular datos del día en UC3M sobre recursión
    uc3m_engine.record_daily_topic_stats(
        topic="recursión",
        n_students=28,
        bloom_mean=1.8,    # bajo: el cohorte no está entendiendo
        gaming_rate=0.22,
        rephrase_rate=0.48,  # alto: muchas reformulaciones → dificultad RAG + conceptual
        autonomy_mean=0.25,
        active_config={
            "scaffolding_mode": "direct",
            "max_daily_prompts": 20,
            "block_direct_solutions": False,
        },
    )

    # Día 2: sigue la dificultad
    uc3m_engine.record_daily_topic_stats(
        topic="recursión",
        n_students=30,
        bloom_mean=1.9,
        gaming_rate=0.35,
        rephrase_rate=0.51,
        autonomy_mean=0.22,
        active_config={"scaffolding_mode": "direct"},
    )

    # Día 3: persiste
    uc3m_engine.record_daily_topic_stats(
        topic="recursión",
        n_students=31,
        bloom_mean=2.0,
        gaming_rate=0.41,
        rephrase_rate=0.45,
        autonomy_mean=0.28,
        active_config={"scaffolding_mode": "socratic"},  # cambió a socrático
    )

    # Emitir señales
    signals = uc3m_engine.detect_and_emit_signals(target_nodes=["uva", "upf"])
    print(f"📡 UC3M emite {len(signals)} señal(es):")
    for s in signals:
        print(f"   [{s.signal_type}] Topic: '{s.topic}' | Urgencia: {s.urgency}")
        print(f"   Estudiantes afectados: {s.n_students_affected} | Bloom medio: {s.bloom_mean:.1f}")
        print(f"   Rephrase rate: {s.rephrase_rate:.0%} | Gaming: {s.gaming_rate:.0%}")
        print(f"   Config sugerida: {s.recommended_config_changes}")
        print(f"   Corpus sugerido: {s.recommended_corpus_additions[:1]}")
        print()

    # ─── Nodo receptor: UVa — llega a recursión en ~12 días ───
    profile_uva = NodeTemporalProfile(
        node_id="uva",
        course_start_date="2026-01-19",
        course_end_date="2026-06-22",
        topics_sequence=["variables", "bucles", "funciones", "arrays", "recursión", "punteros"],
        topic_arrival_dates={
            "recursión": (datetime.now() + timedelta(days=12)).strftime("%Y-%m-%d"),
        },
        typical_days_per_topic=14,
    )
    uva_engine = CrossNodeSignalEngine(node_id="uva", node_profile=profile_uva)

    print("─" * 65)
    print("📥 UVa recibe y procesa las señales de UC3M:")
    print()

    for signal in signals:
        # Serializar/deserializar (simula transmisión por red)
        signal_json = uc3m_engine.export_signal(signal)
        received_signal = CrossNodeSignalEngine.import_signal(signal_json)

        analysis = uva_engine.receive_signal(received_signal)
        print(f"   Señal: {analysis.signal.signal_type} sobre '{analysis.signal.topic}'")
        print(f"   ¿Relevante para UVa? {analysis.is_relevant}")
        if analysis.is_relevant:
            print(f"   Días hasta el topic: ~{analysis.estimated_arrival_days}")
            print(f"   Urgencia local: {analysis.urgency_for_target}")
            print(f"   ➤ ALERTA PARA EL DOCENTE:")
            print(f"   {analysis.teacher_alert}")
            if analysis.preemptive_config:
                print(f"   Config anticipatoria: {analysis.preemptive_config}")
        print()

    print("═" * 65)
