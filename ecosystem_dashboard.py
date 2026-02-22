"""
ECOSYSTEM DASHBOARD — Panel Unificado del Ecosistema GENIE Learn
═══════════════════════════════════════════════════════════════════════
Integra los seis módulos del ecosistema en una vista coherente.

ARQUITECTURA:
    system_event_logger    → flujo de eventos del ecosistema
    config_genome          → fingerprints y análisis atribucional
    rag_quality_sensor     → calidad RAG en tiempo real
    teacher_calibration    → fidelidad interpretativa docente
    cross_node_signal      → inteligencia colectiva inter-nodo
    temporal_config_advisor → configuración contextual al calendario

VISTAS DISPONIBLES:
    1. Vista del Investigador (researcher): métricas completas, datos crudos
    2. Vista del Docente (teacher): alertas accionables, sugerencias
    3. Vista de Nodo (node_admin): señales inter-nodo, estado del ecosistema

Este módulo es el punto de entrada único para el dashboard de Streamlit.
Reemplaza y extiende researcher_view.py con las seis dimensiones nuevas.

Autor: Diego Elvira Vásquez · Ecosistema GENIE Learn · Feb 2026
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json

from system_event_logger import SystemEventLogger, SystemEvent, create_student_prompt_event
from config_genome import ConfigGenomeAnalyzer
from rag_quality_sensor import RAGQualitySensor, RetrievalRecord
from teacher_calibration import TeacherCalibrationAnalyzer, StudentStateSnapshot
from cross_node_signal import CrossNodeSignalEngine, NodeTemporalProfile
from temporal_config_advisor import TemporalConfigAdvisor, AcademicEvent, AcademicMomentType


@dataclass
class EcosystemSnapshot:
    """
    Estado completo del ecosistema en un momento dado.
    
    La unidad de análisis para papers académicos:
    un snapshot captura todo lo que es relevante sobre el sistema
    en ese instante — desde el estado del cohorte hasta la calidad RAG.
    """
    generated_at: str
    node_id: str

    # Métricas del ecosistema
    total_events: int
    unique_students: int
    active_students_today: int
    
    # Calidad del sistema
    rag_quality_report: Dict
    rephrase_rate: float
    
    # Estado pedagógico
    cohort_bloom_mean: float
    cohort_autonomy_mean: float
    gaming_rate: float
    epistemic_silences_today: int
    
    # Estado configuracional
    current_config_fingerprint: Optional[str]
    pending_config_suggestions: int
    
    # Señales inter-nodo
    pending_cross_node_alerts: int
    
    # Calibración docente (si hay suficientes datos)
    teacher_fidelity_mean: Optional[float]
    
    # Presión académica actual
    pressure_index: float
    academic_moment: str


@dataclass
class TeacherDashboardView:
    """Vista simplificada para el docente en Streamlit."""
    # Lo urgente primero
    urgent_alerts: List[str]
    config_suggestions: List[Dict]
    
    # Estado del cohorte (legible)
    cohort_summary: str
    bloom_distribution: Dict[str, int]
    
    # Señales de atención
    students_at_risk: List[str]      # IDs de estudiantes con señales preocupantes
    gaming_detected: List[str]       # IDs con gaming activo
    epistemic_silences: List[Dict]   # silencios epistémicos detectados
    
    # Calidad RAG
    rag_status: str                  # "good" | "degraded" | "critical"
    corpus_gaps: List[str]
    
    # Cross-node
    cross_node_alerts: List[str]
    
    # Calibración (si el docente optó por activarla)
    calibration_summary: Optional[str]


class EcosystemDashboard:
    """
    Hub central del ecosistema GENIE Learn.
    
    Inicializa y coordina todos los módulos.
    Expone las vistas para Streamlit.
    Gestiona el flujo de datos entre módulos.
    """

    def __init__(
        self,
        node_id: str = "uva",
        db_path: str = "genie_events.db",
        calendar: Optional[List[AcademicEvent]] = None,
    ):
        self.node_id = node_id
        
        # ─── Inicializar los seis módulos del ecosistema ───
        self.event_logger = SystemEventLogger(db_path=db_path, node_id=node_id)
        self.config_genome = ConfigGenomeAnalyzer()
        self.rag_sensor = RAGQualitySensor()
        self.teacher_calibration = TeacherCalibrationAnalyzer()
        self.temporal_advisor = TemporalConfigAdvisor(calendar=calendar or [])
        
        # Cross-node (se inicializa con perfil temporal si existe)
        self.cross_node = CrossNodeSignalEngine(node_id=node_id)
        
        # Estado interno
        self._current_config: Dict = {}
        self._teacher_id: Optional[str] = None
        self._session_pressure: float = 0.3

    def configure(self, config: Dict, teacher_id: str):
        """Actualiza el contexto del dashboard con la config activa."""
        self._current_config = config
        self._teacher_id = teacher_id
        # Calcular presión académica del día
        pressure = self.temporal_advisor.compute_pressure_profile()
        self._session_pressure = pressure.pressure_index

    def process_student_interaction(
        self,
        student_id: str,
        prompt: str,
        topics: List[str],
        copy_paste_score: float,
        bloom_estimate: int,
        session_id: Optional[str] = None,
        rag_chunks: int = 0,
        rag_scores: Optional[List[float]] = None,
        rag_above_threshold: int = 0,
    ) -> Dict:
        """
        Punto de entrada de cada interacción estudiantil.
        
        Invoca todos los módulos relevantes y retorna análisis completo.
        Se llama desde app.py después de cada respuesta del LLM.
        """
        rag_scores = rag_scores or []

        # 1. Registrar en el event logger (el sensor universal)
        event = create_student_prompt_event(
            student_id=student_id,
            prompt=prompt,
            topics=topics,
            copy_paste_score=copy_paste_score,
            config_snapshot=self._current_config,
            bloom_estimate=bloom_estimate,
            pressure_index=self._session_pressure,
            session_id=session_id,
        )
        event_id = self.event_logger.log_event(event)

        # 2. Registrar en RAG quality sensor si hay datos de retrieval
        rag_pair = None
        if rag_chunks > 0 or rag_scores:
            record = RetrievalRecord(
                student_id=student_id,
                timestamp=datetime.now().isoformat(),
                query=prompt,
                chunks_retrieved=rag_chunks,
                relevance_scores=rag_scores,
                threshold_used=0.25,
                above_threshold=rag_above_threshold,
                topics_detected=topics,
            )
            rag_pair = self.rag_sensor.record_retrieval(record)

        return {
            "event_id": event_id,
            "pressure_index": self._session_pressure,
            "rag_degradation_detected": rag_pair is not None,
            "rag_degradation_type": rag_pair.degradation_type if rag_pair else None,
        }

    def get_teacher_dashboard(self) -> TeacherDashboardView:
        """
        Vista del docente: información accionable, sin ruido.
        """
        # Alertas urgentes
        urgent_alerts = []

        # Sugerencias de configuración temporales
        suggestion = self.temporal_advisor.generate_suggestion(self._current_config)
        config_suggestions = []
        if suggestion:
            config_suggestions.append({
                "id": suggestion.suggestion_id,
                "moment": suggestion.moment_description,
                "changes": suggestion.params_to_change,
                "rationale": suggestion.pedagogical_rationale[:200] + "...",
                "pressure": suggestion.pressure_index,
            })

        # Estado del cohorte desde el logger
        stats = self.event_logger.get_stats()
        bloom_timeline = self.event_logger.get_cohort_bloom_over_time()
        current_bloom = bloom_timeline[-1]["bloom_mean"] if bloom_timeline else 2.0

        cohort_summary = (
            f"{stats.get('unique_students', 0)} estudiantes activos. "
            f"Bloom medio del cohorte: {current_bloom:.1f}/6. "
            f"Silencios epistémicos: {stats.get('epistemic_silences', 0)}."
        )

        # Silencios epistémicos
        silences = self.event_logger.get_epistemic_silences()
        silence_dicts = [
            {
                "student_id": s["student_id"],
                "topic": s.get("payload", {}).get("topic"),
                "anomaly_score": s.get("payload", {}).get("anomaly_score", 0),
            }
            for s in silences[:5]  # top 5
        ]

        # Calidad RAG
        rag_report = self.rag_sensor.generate_quality_report()
        rag_status = (
            "critical" if rag_report.rephrase_rate > 0.5 else
            "degraded" if rag_report.rephrase_rate > 0.25 else
            "good"
        )
        if rag_status != "good":
            urgent_alerts.append(
                f"⚠️ Calidad RAG {rag_status.upper()}: "
                f"tasa de rephrase {rag_report.rephrase_rate:.0%}. "
                f"Revisar corpus."
            )

        # Señales inter-nodo
        cross_alerts = [
            analysis.teacher_alert
            for analysis in self.cross_node.get_pending_alerts()
            if analysis.teacher_alert
        ][:3]

        # Presión alta → alerta
        if self._session_pressure >= 0.75:
            urgent_alerts.append(
                f"🔴 PRESIÓN ACADÉMICA ALTA ({self._session_pressure:.0%}): "
                f"Considera adaptar configuración para el momento del curso."
            )

        return TeacherDashboardView(
            urgent_alerts=urgent_alerts,
            config_suggestions=config_suggestions,
            cohort_summary=cohort_summary,
            bloom_distribution={"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0},  # calculable desde DB
            students_at_risk=[s["student_id"] for s in silence_dicts if s["anomaly_score"] > 0.7],
            gaming_detected=[],
            epistemic_silences=silence_dicts,
            rag_status=rag_status,
            corpus_gaps=rag_report.corpus_gaps[:5],
            cross_node_alerts=cross_alerts,
            calibration_summary=None,
        )

    def get_ecosystem_snapshot(self) -> EcosystemSnapshot:
        """
        Snapshot completo del ecosistema para el investigador.
        """
        stats = self.event_logger.get_stats()
        rag_report = self.rag_sensor.generate_quality_report()
        pressure = self.temporal_advisor.compute_pressure_profile()
        bloom_timeline = self.event_logger.get_cohort_bloom_over_time()
        current_bloom = bloom_timeline[-1]["bloom_mean"] if bloom_timeline else 2.0

        return EcosystemSnapshot(
            generated_at=datetime.now().isoformat(),
            node_id=self.node_id,
            total_events=stats.get("total_events", 0),
            unique_students=stats.get("unique_students", 0),
            active_students_today=0,  # calculable con query adicional
            rag_quality_report={
                "rephrase_rate": rag_report.rephrase_rate,
                "corpus_gaps": rag_report.corpus_gaps,
                "problematic_topics": rag_report.most_problematic_topics,
            },
            rephrase_rate=rag_report.rephrase_rate,
            cohort_bloom_mean=round(current_bloom, 2),
            cohort_autonomy_mean=0.0,  # calculable desde epistemic_autonomy.py
            gaming_rate=0.0,           # calculable desde interaction_semiotics.py
            epistemic_silences_today=stats.get("epistemic_silences", 0),
            current_config_fingerprint=None,  # calculable desde config_genome
            pending_config_suggestions=len(self.temporal_advisor.pending_suggestions),
            pending_cross_node_alerts=len(self.cross_node.get_pending_alerts()),
            teacher_fidelity_mean=None,
            pressure_index=pressure.pressure_index,
            academic_moment=pressure.moment_type.value,
        )

    def get_research_export(self) -> Dict:
        """
        Export estructurado para análisis académico.
        
        El dataset que alimenta los papers de GENIE Learn:
        eventos + métricas + fingerprints + señales inter-nodo.
        """
        stats = self.event_logger.get_stats()
        attributional = self.config_genome.get_attributional_analysis()
        rag_report = self.rag_sensor.generate_quality_report()
        bloom_timeline = self.event_logger.get_cohort_bloom_over_time()
        decision_analytics = self.temporal_advisor.get_decision_analytics()
        silences = self.event_logger.get_epistemic_silences()

        return {
            "export_at": datetime.now().isoformat(),
            "node_id": self.node_id,
            "ecosystem_stats": stats,
            "attributional_analysis": attributional,
            "rag_quality": {
                "rephrase_rate": rag_report.rephrase_rate,
                "corpus_gaps": rag_report.corpus_gaps,
                "rephrase_pairs": len(rag_report.rephrase_pairs),
                "recommendations": rag_report.recommendations,
            },
            "bloom_trajectory": bloom_timeline,
            "teacher_decision_analytics": decision_analytics,
            "epistemic_silences": silences[:20],  # muestra
            "cross_node_signals_received": len(self.cross_node.received_signals),
        }


# ──────────────────────────────────────────────
# DEMO AUTOEJECTABLE — Integración completa del ecosistema
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import os, time

    # Limpiar BD de demo
    db_path = "ecosystem_demo.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    print("═" * 70)
    print("ECOSYSTEM DASHBOARD — Demo de integración completa GENIE Learn")
    print("═" * 70)
    print("Inicializando 6 módulos del ecosistema...")

    # Calendario académico de demo
    today = date.today()
    calendar = [
        AcademicEvent(
            event_id="sem_actual",
            event_type=AcademicMomentType.EXPLORATION,
            event_name="Semana normal – Tema Recursión",
            start_date=(today - timedelta(days=2)).isoformat(),
            end_date=(today + timedelta(days=3)).isoformat(),
            topics_covered=["recursión"],
            pressure_weight=0.3,
        ),
        AcademicEvent(
            event_id="entrega_pr2",
            event_type=AcademicMomentType.PRE_DELIVERY,
            event_name="Pre-entrega Práctica 2",
            start_date=(today + timedelta(days=9)).isoformat(),
            end_date=(today + timedelta(days=11)).isoformat(),
            topics_covered=["recursión", "arrays"],
            pressure_weight=0.75,
        ),
    ]

    # Inicializar ecosistema
    dashboard = EcosystemDashboard(
        node_id="uva",
        db_path=db_path,
        calendar=calendar,
    )

    # Configurar con config activa del docente
    config = {
        "scaffolding_mode": "direct",
        "max_daily_prompts": 20,
        "block_direct_solutions": False,
        "forced_hallucination_pct": 0.0,
        "use_rag": True,
    }
    dashboard.configure(config=config, teacher_id="prof_demo")

    print("✓ Ecosistema inicializado.\n")
    print("Simulando interacciones del cohorte...")

    # Simular 6 interacciones de 3 estudiantes
    interactions = [
        ("est_01", "¿qué es la recursión?", ["recursión"], 0.05, 2, 2, [0.82, 0.71], 2),
        ("est_01", "¿por qué necesita caso base?", ["recursión"], 0.03, 3, 2, [0.88, 0.76], 2),
        ("est_02", "explain recursion step by step", ["recursión"], 0.08, 1, 1, [0.31], 0),
        ("est_02", "recursion explained simply", ["recursión"], 0.07, 1, 1, [0.28], 0),  # rephrase
        ("est_03", "bucles for en python", ["bucles"], 0.04, 2, 3, [0.91, 0.85, 0.72], 3),
        ("est_01", "me doy cuenta que el caso base evita el stack overflow", ["recursión"], 0.02, 4, 2, [0.79], 1),
    ]

    for student, prompt, topics, cp, bloom, chunks, scores, above in interactions:
        result = dashboard.process_student_interaction(
            student_id=student,
            prompt=prompt,
            topics=topics,
            copy_paste_score=cp,
            bloom_estimate=bloom,
            rag_chunks=chunks,
            rag_scores=scores,
            rag_above_threshold=above,
        )
        if result["rag_degradation_detected"]:
            print(f"  ⚠ RAG degraded detectado: {student} — {result['rag_degradation_type']}")
        time.sleep(0.02)

    # Registrar silencio epistémico para est_03
    from system_event_logger import create_epistemic_silence_event
    silence_evt = create_epistemic_silence_event(
        student_id="est_03",
        topic="recursión",
        expected_question_density=0.78,
        observed_question_density=0.0,
        silence_duration_minutes=52.0,
        config_snapshot=config,
        bloom_estimate=1,
    )
    dashboard.event_logger.log_event(silence_evt)

    print("\n─" * 70)
    print("VISTA DEL DOCENTE (teacher dashboard)")
    print("─" * 70)
    teacher_view = dashboard.get_teacher_dashboard()
    
    if teacher_view.urgent_alerts:
        print(f"\n🚨 ALERTAS URGENTES:")
        for alert in teacher_view.urgent_alerts:
            print(f"   {alert}")
    
    if teacher_view.config_suggestions:
        print(f"\n💡 SUGERENCIAS DE CONFIGURACIÓN:")
        for sug in teacher_view.config_suggestions:
            print(f"   Momento: {sug['moment']}")
            print(f"   Cambios: {sug['changes']}")
    
    print(f"\n📊 COHORTE: {teacher_view.cohort_summary}")
    print(f"   Estado RAG: {teacher_view.rag_status.upper()}")
    
    if teacher_view.students_at_risk:
        print(f"   Estudiantes en riesgo: {teacher_view.students_at_risk}")
    
    if teacher_view.epistemic_silences:
        print(f"\n🔇 SILENCIOS EPISTÉMICOS:")
        for s in teacher_view.epistemic_silences:
            print(f"   {s['student_id']} — Topic: {s['topic']} — Anomalía: {s.get('anomaly_score', 0):.0%}")

    print("\n─" * 70)
    print("SNAPSHOT DEL ECOSISTEMA (vista investigador)")
    print("─" * 70)
    snapshot = dashboard.get_ecosystem_snapshot()
    print(f"   Eventos totales: {snapshot.total_events}")
    print(f"   Estudiantes únicos: {snapshot.unique_students}")
    print(f"   Bloom medio cohorte: {snapshot.cohort_bloom_mean}")
    print(f"   Rephrase rate RAG: {snapshot.rephrase_rate:.0%}")
    print(f"   Silencios epistémicos: {snapshot.epistemic_silences_today}")
    print(f"   Presión académica: {snapshot.pressure_index:.0%} ({snapshot.academic_moment})")
    print(f"   Sugerencias config pendientes: {snapshot.pending_config_suggestions}")
    print(f"   Alertas inter-nodo pendientes: {snapshot.pending_cross_node_alerts}")

    print("\n─" * 70)
    print("EXPORT DE INVESTIGACIÓN")
    print("─" * 70)
    research_data = dashboard.get_research_export()
    print(f"   Módulos exportados: {list(research_data.keys())}")
    print(f"   Listo para BERTopic, K-Means, Spearman.")

    # Limpiar
    if os.path.exists(db_path):
        os.remove(db_path)

    print("\n✅ DEMO COMPLETADO.")
    print("═" * 70)
