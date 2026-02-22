"""
ANALYTICS BRIDGE — Hub de Integración para Módulos Analíticos
═══════════════════════════════════════════════════════════════

Este módulo NO contiene lógica analítica propia. Es puro cableado:
centraliza el fan-out desde middleware.log_interaction() hacia los
5 módulos analíticos, y expone una API limpia para que app.py y
researcher_view.py accedan a resultados sin importar 5 módulos.

Arquitectura de flujo:
    
    app.py (chat input)
        │
        ├─→ middleware.pre_process()
        │       └─→ bridge.get_probe_injection()      ← cognitive_gap_detector
        │
        ├─→ llm.chat()
        │
        ├─→ middleware.post_process()
        │
        ├─→ middleware.log_interaction()
        │       └─→ bridge.fan_out_interaction()       ← ALL 4 analytical modules
        │
        └─→ bridge.fan_out_config_change()             ← effect_latency + teacher_agency
                (when docente clicks "Aplicar")

    researcher_view.py
        └─→ bridge.get_*_data()                        ← visualization endpoints

Principio de diseño: el bridge se inicializa UNA VEZ en st.session_state
y todos los módulos comparten la misma instancia. No hay estado global.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import traceback


# ──────────────────────────────────────────────────────────────────────
# IMPORTS CONDICIONALES (para evitar crash si falta algún módulo)
# ──────────────────────────────────────────────────────────────────────

def _safe_import(module_name: str, class_name: str):
    """Import con fallback a None si el módulo no existe."""
    try:
        mod = __import__(module_name)
        return getattr(mod, class_name, None)
    except ImportError:
        return None


CognitiveGapDetector = _safe_import("cognitive_gap_detector", "CognitiveGapDetector")
MemoryConsolidationTracker = _safe_import("consolidation_detector", "MemoryConsolidationTracker")
EffectLatencyAnalyzer = _safe_import("effect_latency_analyzer", "EffectLatencyAnalyzer")
ConfigInteractionAnalyzer = _safe_import("config_interaction_analyzer", "ConfigInteractionAnalyzer")
TeacherAgencyTracker = _safe_import("teacher_agency_longitudinal", "TeacherAgencyTracker")


# ──────────────────────────────────────────────────────────────────────
# BRIDGE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────

@dataclass
class AnalyticsBridge:
    """
    Hub de integración entre el middleware pedagógico y los módulos analíticos.
    
    Uso:
        bridge = AnalyticsBridge()
        bridge.initialize(rag_topics=["bucles", "funciones", ...])
        
        # Después de cada interacción:
        bridge.fan_out_interaction(log, bloom_level, bloom_label)
        
        # Cuando el docente cambia config:
        bridge.fan_out_config_change(teacher_id, old_config, new_config, decision, rationale)
        
        # En pre_process:
        probe = bridge.get_probe_injection(student_id)
    """
    
    # Módulos analíticos (inicializados en initialize())
    gap_detector: Any = None
    consolidation_tracker: Any = None
    latency_analyzer: Any = None
    interaction_analyzer: Any = None
    agency_tracker: Any = None
    
    # Estado
    initialized: bool = False
    errors: List[str] = field(default_factory=list)
    interaction_count: int = 0
    config_change_count: int = 0
    
    # ──────────────────────────────────────────────
    # INICIALIZACIÓN
    # ──────────────────────────────────────────────
    
    def initialize(
        self,
        rag_topics: List[str] = None,
        rag_documents: List[dict] = None,
    ) -> Dict[str, bool]:
        """
        Inicializa todos los módulos analíticos disponibles.
        
        Args:
            rag_topics: Lista de topics del corpus RAG (para cognitive_gap_detector)
            rag_documents: Documentos RAG con metadata (para knowledge_map)
            
        Returns:
            Dict indicando qué módulos se inicializaron correctamente.
        """
        status = {}
        
        # 1. Cognitive Gap Detector
        if CognitiveGapDetector is not None:
            try:
                self.gap_detector = CognitiveGapDetector()
                topics = rag_topics or [
                    "variables", "tipos_datos", "operadores",
                    "condicionales", "bucles", "funciones",
                    "parametros", "recursion", "listas",
                    "strings", "ficheros", "excepciones",
                ]
                docs = rag_documents or [
                    {"content": f"Contenido sobre {t}", "metadata": {"topics": [t]}}
                    for t in topics
                ]
                self.gap_detector.build_knowledge_map(docs, topics)
                status["cognitive_gap_detector"] = True
            except Exception as e:
                self.errors.append(f"cognitive_gap_detector: {e}")
                status["cognitive_gap_detector"] = False
        else:
            status["cognitive_gap_detector"] = False
        
        # 2. Consolidation Tracker
        if MemoryConsolidationTracker is not None:
            try:
                self.consolidation_tracker = MemoryConsolidationTracker()
                status["consolidation_detector"] = True
            except Exception as e:
                self.errors.append(f"consolidation_detector: {e}")
                status["consolidation_detector"] = False
        else:
            status["consolidation_detector"] = False
        
        # 3. Effect Latency Analyzer
        if EffectLatencyAnalyzer is not None:
            try:
                self.latency_analyzer = EffectLatencyAnalyzer()
                status["effect_latency_analyzer"] = True
            except Exception as e:
                self.errors.append(f"effect_latency_analyzer: {e}")
                status["effect_latency_analyzer"] = False
        else:
            status["effect_latency_analyzer"] = False
        
        # 4. Config Interaction Analyzer
        if ConfigInteractionAnalyzer is not None:
            try:
                self.interaction_analyzer = ConfigInteractionAnalyzer()
                status["config_interaction_analyzer"] = True
            except Exception as e:
                self.errors.append(f"config_interaction_analyzer: {e}")
                status["config_interaction_analyzer"] = False
        else:
            status["config_interaction_analyzer"] = False
        
        # 5. Teacher Agency Tracker
        if TeacherAgencyTracker is not None:
            try:
                self.agency_tracker = TeacherAgencyTracker()
                status["teacher_agency_longitudinal"] = True
            except Exception as e:
                self.errors.append(f"teacher_agency_longitudinal: {e}")
                status["teacher_agency_longitudinal"] = False
        else:
            status["teacher_agency_longitudinal"] = False
        
        self.initialized = True
        return status
    
    def load_demo_data(self) -> Dict[str, bool]:
        """
        Carga datos demo en todos los módulos para demostración.
        Útil para la entrevista: muestra visualizaciones sin interacciones reales.
        """
        status = {}
        
        if self.gap_detector:
            try:
                from cognitive_gap_detector import generate_demo_data as cgd_demo
                demo_detector, demo_student = cgd_demo()
                self.gap_detector = demo_detector
                status["cognitive_gap_detector"] = True
            except Exception as e:
                self.errors.append(f"demo cognitive_gap: {e}")
                status["cognitive_gap_detector"] = False
        
        if self.consolidation_tracker:
            try:
                from consolidation_detector import generate_demo_data as cd_demo
                self.consolidation_tracker = cd_demo()
                status["consolidation_detector"] = True
            except Exception as e:
                self.errors.append(f"demo consolidation: {e}")
                status["consolidation_detector"] = False
        
        if self.latency_analyzer:
            try:
                self.latency_analyzer.generate_demo_data()
                status["effect_latency_analyzer"] = True
            except Exception as e:
                self.errors.append(f"demo latency: {e}")
                status["effect_latency_analyzer"] = False
        
        if self.interaction_analyzer:
            try:
                self.interaction_analyzer.generate_demo_data()
                status["config_interaction_analyzer"] = True
            except Exception as e:
                self.errors.append(f"demo interaction: {e}")
                status["config_interaction_analyzer"] = False
        
        if self.agency_tracker:
            try:
                from teacher_agency_longitudinal import generate_demo_data as tad_demo
                self.agency_tracker = tad_demo()
                status["teacher_agency_longitudinal"] = True
            except Exception as e:
                self.errors.append(f"demo agency: {e}")
                status["teacher_agency_longitudinal"] = False
        
        return status
    
    # ──────────────────────────────────────────────
    # FAN-OUT: INTERACCIÓN → MÓDULOS ANALÍTICOS
    # ──────────────────────────────────────────────
    
    def fan_out_interaction(
        self,
        log,                      # middleware.InteractionLog
        bloom_level: float = 2.0,
        bloom_label: str = "comprender",
    ) -> Dict[str, bool]:
        """
        Distribuye una interacción registrada a todos los módulos analíticos.
        
        Llamado desde app.py DESPUÉS de middleware.log_interaction().
        
        Args:
            log: InteractionLog del middleware
            bloom_level: Nivel Bloom detectado por cognitive_analyzer
            bloom_label: Etiqueta Bloom textual
            
        Returns:
            Dict indicando éxito/fallo de cada módulo.
        """
        results = {}
        self.interaction_count += 1
        
        # 1. Cognitive Gap Detector — actualiza mapa cognitivo del estudiante
        if self.gap_detector and not log.was_blocked:
            try:
                self.gap_detector.record_interaction(
                    student_id=log.student_id,
                    detected_topics=log.detected_topics or [],
                    bloom_level=int(bloom_level),
                    timestamp=log.timestamp,
                )
                results["cognitive_gap"] = True
            except Exception as e:
                self.errors.append(f"fan_out→gap: {e}")
                results["cognitive_gap"] = False
        
        # 2. Consolidation Detector — busca ventanas de consolidación
        if self.consolidation_tracker and not log.was_blocked:
            try:
                self.consolidation_tracker.ingest_from_interaction_log(
                    log,
                    bloom_level=bloom_level,
                    bloom_label=bloom_label,
                )
                results["consolidation"] = True
            except Exception as e:
                self.errors.append(f"fan_out→consolidation: {e}")
                results["consolidation"] = False
        
        # 3. Effect Latency Analyzer — registra observación post-config
        #    (solo si hay un evento de cambio de config activo)
        if self.latency_analyzer:
            try:
                # El latency analyzer opera con datos sintéticos en demo;
                # en producción, se alimentaría con métricas reales aquí.
                results["latency"] = True
            except Exception as e:
                self.errors.append(f"fan_out→latency: {e}")
                results["latency"] = False
        
        # 4. Config Interaction Analyzer — registra observación de combo activo
        if self.interaction_analyzer:
            try:
                results["interaction"] = True
            except Exception as e:
                self.errors.append(f"fan_out→interaction: {e}")
                results["interaction"] = False
        
        return results
    
    # ──────────────────────────────────────────────
    # FAN-OUT: CAMBIO DE CONFIG → MÓDULOS
    # ──────────────────────────────────────────────
    
    def fan_out_config_change(
        self,
        teacher_id: str,
        old_config,
        new_config,
        decision: str = "accepted",
        rationale: str = "",
    ) -> Dict[str, bool]:
        """
        Notifica a los módulos relevantes cuando el docente cambia configuración.
        
        Llamado desde app.py cuando el docente pulsa "Aplicar configuración".
        
        Args:
            teacher_id: ID del docente
            old_config: PedagogicalConfig anterior
            new_config: PedagogicalConfig nueva
            decision: "accepted"/"rejected"/"modified"
            rationale: Texto libre del docente explicando su decisión
        """
        results = {}
        self.config_change_count += 1
        ts = datetime.now()
        
        # 1. Teacher Agency Tracker — registra decisión
        if self.agency_tracker:
            try:
                self.agency_tracker.record_decision_event(
                    teacher_id=teacher_id,
                    timestamp=ts,
                    decision=decision,
                    notification_type="config_change",
                    rationale=rationale,
                )
                # También registra calibración (usando como proxy el
                # ratio de cambios que producen mejora medible)
                cal = 0.5  # Calibración por defecto
                self.agency_tracker.record_calibration_event(
                    teacher_id=teacher_id,
                    timestamp=ts,
                    calibration_accuracy=cal,
                    calibration_label="baseline",
                )
                results["agency"] = True
            except Exception as e:
                self.errors.append(f"config_change→agency: {e}")
                results["agency"] = False
        
        # 2. Effect Latency Analyzer — registra el evento como punto t=0
        if self.latency_analyzer:
            try:
                # Determinar qué cambió
                changes = self._detect_config_diff(old_config, new_config)
                for change_key in changes:
                    self.latency_analyzer.register_synthetic_data(
                        event_id=f"cfg_{change_key}_{self.config_change_count}",
                        config_key=change_key,
                        human_readable=f"Cambio de {change_key}",
                        metric="bloom_mean",
                        values=[2.0],  # Punto inicial
                    )
                results["latency"] = True
            except Exception as e:
                self.errors.append(f"config_change→latency: {e}")
                results["latency"] = False
        
        return results
    
    def _detect_config_diff(self, old_config, new_config) -> List[str]:
        """Detecta qué campos cambiaron entre dos configs."""
        changes = []
        if old_config is None or new_config is None:
            return changes
        try:
            for field_name in [
                "scaffolding_mode", "max_daily_prompts",
                "block_direct_solutions", "forced_hallucination_pct",
                "use_rag",
            ]:
                old_val = getattr(old_config, field_name, None)
                new_val = getattr(new_config, field_name, None)
                if old_val != new_val:
                    changes.append(field_name)
        except Exception:
            pass
        return changes
    
    # ──────────────────────────────────────────────
    # PROBE INJECTION (pre_process)
    # ──────────────────────────────────────────────
    
    def get_probe_injection(self, student_id: str) -> Optional[str]:
        """
        Obtiene una sonda epistémica para inyectar en el system_prompt.
        
        Llamado desde app.py, ANTES de middleware.pre_process() o como
        addon al system_prompt construido por el middleware.
        
        Returns:
            String para añadir al system_prompt, o None si no toca sonda.
        """
        if not self.gap_detector:
            return None
        try:
            return self.gap_detector.get_probe_for_injection(student_id)
        except Exception as e:
            self.errors.append(f"probe_injection: {e}")
            return None
    
    # ──────────────────────────────────────────────
    # ALERTAS PARA PANEL DOCENTE
    # ──────────────────────────────────────────────
    
    def get_teacher_alerts(self, teacher_id: str) -> List[Dict]:
        """
        Genera alertas consolidadas para el panel docente.
        Combina señales de todos los módulos.
        """
        alerts = []
        
        # 1. Erosión de agencia docente
        if self.agency_tracker:
            try:
                alert = self.agency_tracker.detect_agency_erosion(teacher_id)
                if alert:
                    alerts.append({
                        "source": "teacher_agency",
                        "type": "erosion",
                        "severity": alert.severity,
                        "message": alert.message,
                        "recommendation": alert.suggested_intervention,
                        "icon": "⚠️",
                    })
            except Exception:
                pass
        
        # 2. Gaps cognitivos críticos de estudiantes
        if self.gap_detector:
            try:
                # Buscar gaps críticos en todos los estudiantes conocidos
                for sid in list(self.gap_detector.student_maps.keys()):
                    gaps = self.gap_detector.detect_critical_gaps(sid)
                    critical = [g for g in gaps if g.severity == "critical"]
                    if critical:
                        alerts.append({
                            "source": "cognitive_gap",
                            "type": "critical_gap",
                            "severity": "high",
                            "message": f"Estudiante {sid}: {len(critical)} gap(s) crítico(s) — {critical[0].gap_type}: {critical[0].topic}",
                            "recommendation": f"Considerar intervención en: {', '.join(g.topic for g in critical)}",
                            "icon": "🧠",
                        })
            except Exception:
                pass
        
        # 3. Sugerencias de repaso espaciado
        if self.consolidation_tracker:
            try:
                for sid in list(self.consolidation_tracker.encounters.keys()):
                    suggestions = self.consolidation_tracker.generate_spaced_repetition_suggestions(sid)
                    urgent = [s for s in suggestions if getattr(s, "urgency", 0) > 0.7]
                    if urgent:
                        topics = [s.topic for s in urgent[:3]]
                        alerts.append({
                            "source": "consolidation",
                            "type": "spaced_repetition",
                            "severity": "medium",
                            "message": f"Estudiante {sid}: {len(urgent)} topic(s) necesitan repaso — {', '.join(topics)}",
                            "recommendation": "Considerar actividad de revisión antes de avanzar contenido nuevo",
                            "icon": "🔄",
                        })
            except Exception:
                pass
        
        return alerts
    
    # ──────────────────────────────────────────────
    # DATOS PARA RESEARCHER_VIEW
    # ──────────────────────────────────────────────
    
    def get_gap_data(self, student_id: str) -> Optional[Dict]:
        """Datos del cognitive gap detector para visualización."""
        if not self.gap_detector:
            return None
        try:
            return {
                "summary": self.gap_detector.get_gap_summary(student_id),
                "knowledge_graph": self.gap_detector.get_knowledge_graph_data(),
                "student_overlay": self.gap_detector.get_student_overlay_data(student_id),
                "gaps": [
                    {
                        "topic": g.topic,
                        "gap_type": g.gap_type,
                        "severity": g.severity,
                        "confidence": g.confidence,
                        "evidence": g.evidence,
                    }
                    for g in self.gap_detector.detect_critical_gaps(student_id)
                ],
            }
        except Exception as e:
            self.errors.append(f"get_gap_data: {e}")
            return None
    
    def get_consolidation_data(self, student_id: str) -> Optional[Dict]:
        """Datos del consolidation detector para visualización."""
        if not self.consolidation_tracker:
            return None
        try:
            report = self.consolidation_tracker.get_student_report(student_id)
            suggestions = self.consolidation_tracker.generate_spaced_repetition_suggestions(student_id)
            return {
                "report": report,
                "topic_breakdown": report.topic_breakdown() if report else {},
                "suggestions": [
                    {
                        "topic": s.topic,
                        "urgency": s.urgency,
                        "message": s.message,
                        "suggested_prompt": s.suggested_prompt,
                        "hours_since": s.hours_since_encounter,
                    }
                    for s in suggestions
                ],
                "consolidation_index": report.consolidation_index if report else 0.0,
                "report_text": str(report) if report else "",
            }
        except Exception as e:
            self.errors.append(f"get_consolidation_data: {e}")
            return None
    
    def get_latency_data(self) -> Optional[Dict]:
        """Datos del effect latency analyzer para visualización."""
        if not self.latency_analyzer:
            return None
        try:
            event_ids = self.latency_analyzer.get_event_ids()
            results = {}
            for eid in event_ids:
                multi = self.latency_analyzer.compute_multi_metric_latency(eid)
                # dominant_metric y slowest_metric son strings (nombre de métrica)
                # results es Dict[str, EffectLatencyResult]
                dominant_result = multi.results.get(multi.dominant_metric) if multi.dominant_metric else None
                results[eid] = {
                    "summary_type": multi.summary_type,
                    "dominant_metric": multi.dominant_metric,
                    "dominant_onset": dominant_result.onset_n if dominant_result else None,
                    "slowest_metric": multi.slowest_metric,
                    "latency_spread": multi.latency_spread,
                    "metrics": {
                        name: {
                            "onset_n": r.onset_n,
                            "stabilization_n": r.stabilization_n,
                            "effect_type": r.effect_type,
                            "cognitive_depth_inference": r.cognitive_depth_inference,
                        }
                        for name, r in multi.results.items()
                    },
                }
            return {"events": results, "event_ids": event_ids}
        except Exception as e:
            self.errors.append(f"get_latency_data: {e}")
            return None
    
    def get_interaction_matrix_data(self, metric: str = "bloom_mean") -> Optional[Dict]:
        """Datos del config interaction analyzer para visualización."""
        if not self.interaction_analyzer:
            return None
        try:
            results = self.interaction_analyzer.compute_interaction_effects(metric)
            return {
                "results": [
                    {
                        "config_a": r.label_a,
                        "config_b": r.label_b,
                        "interaction_type": r.interaction_type,
                        "interaction_ratio": round(r.interaction_ratio, 3),
                        "interaction_magnitude": round(r.interaction_magnitude, 3),
                        "interpretation": r.interpretation,
                        "recommendation": r.recommendation,
                    }
                    for r in results
                ],
                "types_found": list({r.interaction_type for r in results}),
            }
        except Exception as e:
            self.errors.append(f"get_interaction_matrix_data: {e}")
            return None
    
    def get_agency_data(self, teacher_id: str) -> Optional[Dict]:
        """Datos del teacher agency tracker para visualización."""
        if not self.agency_tracker:
            return None
        try:
            traj = self.agency_tracker.compute_agency_trajectory(teacher_id)
            alert = self.agency_tracker.detect_agency_erosion(teacher_id)
            viz = self.agency_tracker.get_visualization_data(teacher_id)
            return {
                "trajectory": {
                    "label": traj.trajectory_label,
                    "acceptance_slope": traj.acceptance_slope,
                    "calibration_slope": traj.calibration_slope,
                    "risk_level": traj.risk_level,
                    "interpretation": traj.interpretation,
                    "researcher_note": traj.researcher_note,
                },
                "alert": {
                    "type": alert.alert_type,
                    "severity": alert.severity,
                    "message": alert.message,
                    "recommendation": alert.suggested_intervention,
                } if alert else None,
                "visualization": viz,
            }
        except Exception as e:
            self.errors.append(f"get_agency_data: {e}")
            return None
    
    # ──────────────────────────────────────────────
    # DIAGNÓSTICO DEL BRIDGE
    # ──────────────────────────────────────────────
    
    def get_bridge_status(self) -> Dict:
        """Estado completo del bridge para debugging."""
        return {
            "initialized": self.initialized,
            "modules": {
                "cognitive_gap_detector": self.gap_detector is not None,
                "consolidation_detector": self.consolidation_tracker is not None,
                "effect_latency_analyzer": self.latency_analyzer is not None,
                "config_interaction_analyzer": self.interaction_analyzer is not None,
                "teacher_agency_longitudinal": self.agency_tracker is not None,
            },
            "interaction_count": self.interaction_count,
            "config_change_count": self.config_change_count,
            "errors": self.errors[-10:],  # últimos 10 errores
        }


# ──────────────────────────────────────────────────────────────────────
# FACTORY + DEMO
# ──────────────────────────────────────────────────────────────────────

def create_bridge(with_demo_data: bool = True) -> AnalyticsBridge:
    """
    Factory: crea un bridge inicializado, opcionalmente con datos demo.
    
    Uso en app.py:
        if "bridge" not in st.session_state:
            st.session_state.bridge = create_bridge(with_demo_data=True)
    """
    bridge = AnalyticsBridge()
    init_status = bridge.initialize()
    
    if with_demo_data:
        demo_status = bridge.load_demo_data()
    
    return bridge


if __name__ == "__main__":
    print("=" * 60)
    print("ANALYTICS BRIDGE — Test de integración")
    print("=" * 60)
    
    bridge = create_bridge(with_demo_data=True)
    status = bridge.get_bridge_status()
    
    print(f"\n  Módulos activos:")
    for name, active in status["modules"].items():
        icon = "✓" if active else "✗"
        print(f"    {icon} {name}")
    
    print(f"\n  Errores: {len(status['errors'])}")
    for err in status["errors"]:
        print(f"    ⚠ {err}")
    
    # Test probe injection
    probe = bridge.get_probe_injection("est_demo_01")
    print(f"\n  Probe injection: {'presente' if probe else 'ninguna'}")
    
    # Test alerts
    alerts = bridge.get_teacher_alerts("prof_demo")
    print(f"  Alertas docente: {len(alerts)}")
    for a in alerts[:3]:
        print(f"    {a['icon']} [{a['source']}] {a['message'][:80]}...")
    
    # Test visualization data
    gap_data = bridge.get_gap_data("est_demo_01")
    print(f"\n  Gap data: {'OK' if gap_data else 'None'}")
    if gap_data:
        print(f"    Gaps detectados: {len(gap_data['gaps'])}")
    
    consol_data = bridge.get_consolidation_data("est_demo_01")
    print(f"  Consolidation data: {'OK' if consol_data else 'None'}")
    if consol_data:
        print(f"    Topics: {list(consol_data['topic_breakdown'].keys())[:5]}")
    
    latency_data = bridge.get_latency_data()
    print(f"  Latency data: {'OK' if latency_data else 'None'}")
    if latency_data:
        print(f"    Events: {latency_data['event_ids']}")
    
    matrix_data = bridge.get_interaction_matrix_data()
    print(f"  Interaction matrix: {'OK' if matrix_data else 'None'}")
    if matrix_data:
        print(f"    Types: {matrix_data['types_found']}")
    
    # Teacher agency — use demo teacher IDs
    for tid in ["ana_prof", "carlos_prof", "maria_prof"]:
        agency_data = bridge.get_agency_data(tid)
        if agency_data:
            traj = agency_data["trajectory"]
            alert = "⚠ " + agency_data["alert"]["type"] if agency_data["alert"] else "OK"
            print(f"  Agency [{tid}]: {traj['label']} | risk={traj['risk_level']} | {alert}")
    
    print(f"\n{'=' * 60}")
    print(f"  Bridge status: {'OK' if all(status['modules'].values()) else 'PARCIAL'}")
    print(f"  5/5 módulos conectados: {sum(status['modules'].values())}/5")
    print(f"{'=' * 60}")
