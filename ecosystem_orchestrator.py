"""
ECOSYSTEM ORCHESTRATOR — El Cableado Central de GENIE Learn
═══════════════════════════════════════════════════════════════════════
Este archivo es la pieza que faltaba: conecta los 51 módulos en un
flujo coherente donde cada interacción estudiante→LLM genera datos
para todo el ecosistema.

PRINCIPIO DE DISEÑO:
    DEMO_MODE = True  → todo funciona en local con simulaciones
    DEMO_MODE = False → se conecta a infraestructura real (Moodle, PostgreSQL, etc.)

    El código es idéntico. Solo cambian los backends.

POSICIÓN EN EL STACK:
    app.py → EcosystemOrchestrator.process_interaction()
           → internamente invoca middleware, RAG, LLM, y todos los módulos

USO EN app.py:
    orchestrator = EcosystemOrchestrator(config, demo_mode=True)
    result = orchestrator.process_interaction("estudiante_01", "¿qué es recursión?")
    # result contiene: respuesta, nudges, eventos, métricas, todo.

Autor: Diego Elvira Vásquez · Ecosistema GENIE Learn · Feb 2026
"""

import os
import time
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Any, Tuple

# ──────────────────────────────────────────────────────────────
# FLAG CENTRAL: demo vs producción
# ──────────────────────────────────────────────────────────────

DEMO_MODE = os.environ.get("GENIE_DEMO_MODE", "true").lower() == "true"


# ──────────────────────────────────────────────────────────────
# IMPORTS CONDICIONALES — módulos que existen vs los que faltan
# ──────────────────────────────────────────────────────────────

# CAPA 0: Core (siempre disponible)
from middleware import PedagogicalMiddleware, PedagogicalConfig, InteractionLog

# Intentamos importar cada módulo. Si falla (porque falta una dependencia
# o no se ha instalado algo), creamos un stub que no hace nada.
# Esto permite que el orquestador funcione incluso si solo tienes el core.


def _safe_import(module_name: str, class_name: str):
    """Importa un módulo de forma segura. Si falla, devuelve un stub."""
    try:
        mod = __import__(module_name)
        return getattr(mod, class_name)
    except (ImportError, AttributeError, Exception) as e:
        print(f"  [WARNING] Módulo {module_name}.{class_name} no disponible: {e}")
        return None


# CAPA 1: Logging y eventos
SystemEventLogger = _safe_import("system_event_logger", "SystemEventLogger")
create_student_prompt_event = _safe_import("system_event_logger", "create_student_prompt_event")

# CAPA 2: Analytics cognitivos
CognitiveProfiler = _safe_import("cognitive_profiler", "CognitiveProfiler")
EpistemicAutonomyTracker = _safe_import("epistemic_autonomy", "EpistemicAutonomyTracker")
InteractionSemioticsEngine = _safe_import("interaction_semiotics", "InteractionSemioticsEngine")
# nd_patterns exporta NeurodivergentPatternDetector, no NDPatternDetector
NDPatternDetector = _safe_import("nd_patterns", "NeurodivergentPatternDetector")

# CAPA 2: Calidad y detección
RAGQualitySensor = _safe_import("rag_quality_sensor", "RAGQualitySensor")
CognitiveGapDetector = _safe_import("cognitive_gap_detector", "CognitiveGapDetector")
EpistemicSilenceDetector = _safe_import("epistemic_silence_detector", "EpistemicSilenceDetector")
# consolidation_detector exporta MemoryConsolidationTracker
ConsolidationDetector = _safe_import("consolidation_detector", "MemoryConsolidationTracker")

# CAPA 2: Nudges y adaptación
MetacognitiveNudgeEngine = _safe_import("metacognitive_nudges", "MetacognitiveNudgeGenerator")
UDLAdapter = _safe_import("udl_adapter", "UDLAdapter")

# CAPA 2: Evaluación de respuestas
HHHAlignmentDetector = _safe_import("hhh_alignment_detector", "HHHAlignmentDetector")
LLMJudge = _safe_import("llm_judge", "LLMBloomJudge")

# CAPA 3: Configuración y temporal
ConfigGenomeAnalyzer = _safe_import("config_genome", "ConfigGenomeAnalyzer")
TemporalConfigAdvisor = _safe_import("temporal_config_advisor", "TemporalConfigAdvisor")
EffectLatencyAnalyzer = _safe_import("effect_latency_analyzer", "EffectLatencyAnalyzer")
ConfigInteractionAnalyzer = _safe_import("config_interaction_analyzer", "ConfigInteractionAnalyzer")

# CAPA 3: Docente
TeacherCalibrationAnalyzer = _safe_import("teacher_calibration", "TeacherCalibrationAnalyzer")
TeacherNotificationEngine = _safe_import("teacher_notification_engine", "TeacherNotificationEngine")

# CAPA 3: Inter-nodo (simulado en DEMO_MODE)
CrossNodeSignalEngine = _safe_import("cross_node_signal", "CrossNodeSignalEngine")

# CAPA 3: Meta-análisis
SystemReflexivity = _safe_import("system_reflexivity", "SystemReflexivityEngine")
ACHDiagnostic = _safe_import("ach_diagnostic", "ACHDiagnosticEngine")

# CAPA 4: Research
PilotDesign = _safe_import("pilot_design", "PilotAnalysisReport")
PaperDraftingEngine = _safe_import("paper_drafting_engine", "PaperDraftingEngine")


# ──────────────────────────────────────────────────────────────
# RESULTADO DE UNA INTERACCIÓN ORQUESTADA
# ──────────────────────────────────────────────────────────────

@dataclass
class OrchestratedResult:
    """Todo lo que produce una interacción procesada por el ecosistema completo."""

    # --- Respuesta al estudiante ---
    response_text: str
    was_blocked: bool = False
    block_reason: str = ""

    # --- RAG ---
    rag_chunks: List[Dict] = field(default_factory=list)
    rag_quality_score: float = 0.0
    is_rephrase: bool = False

    # --- Middleware ---
    scaffolding_level: int = 0
    detected_topics: List[str] = field(default_factory=list)
    copy_paste_score: float = 0.0
    hallucination_injected: bool = False

    # --- Cognitivo ---
    bloom_estimate: str = "unknown"
    bloom_level: int = 0
    autonomy_score: float = 0.0
    nd_indicators: Dict = field(default_factory=dict)
    cognitive_gap: Optional[str] = None

    # --- Nudge ---
    nudge: Optional[str] = None
    nudge_type: Optional[str] = None

    # --- Evaluación de respuesta ---
    hhh_alignment: Dict = field(default_factory=dict)

    # --- Temporal ---
    academic_pressure: float = 0.0
    config_suggestion: Optional[str] = None

    # --- Docente ---
    calibration_alert: Optional[str] = None

    # --- Meta ---
    event_id: str = ""
    processing_time_ms: int = 0
    modules_activated: List[str] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────
# EL ORQUESTADOR
# ──────────────────────────────────────────────────────────────

class EcosystemOrchestrator:
    """
    Punto de entrada único para el ecosistema GENIE Learn.

    Conecta todos los módulos en un flujo coherente.
    app.py solo necesita hablar con este objeto.

    Uso:
        orch = EcosystemOrchestrator(config)
        result = orch.process_interaction("est_01", "¿qué es una lista?")
    """

    def __init__(
        self,
        config: PedagogicalConfig,
        rag_pipeline=None,
        llm_client=None,
        demo_mode: bool = DEMO_MODE,
        node_id: str = "UVa-node-01",
    ):
        self.config = config
        self.demo_mode = demo_mode
        self.node_id = node_id

        # --- Core (obligatorio) ---
        self.middleware = PedagogicalMiddleware(config)
        self.rag = rag_pipeline
        self.llm = llm_client

        # --- Inicializar módulos disponibles ---
        self._modules = {}
        self._init_modules()

        print(f"\n{'='*60}")
        print(f"  GENIE Learn Ecosystem Orchestrator")
        print(f"  DEMO_MODE: {self.demo_mode}")
        print(f"  Módulos activos: {len(self._modules)}")
        print(f"  Módulos disponibles: {list(self._modules.keys())}")
        print(f"{'='*60}\n")

    def _init_modules(self):
        """Inicializa cada módulo si su clase está disponible."""

        # CAPA 1: Logging
        if SystemEventLogger:
            self._modules["event_logger"] = SystemEventLogger(
                node_id=self.node_id,
                db_path=":memory:" if self.demo_mode else "genie_events.db"
            )

        # CAPA 2: Analytics cognitivos
        if CognitiveProfiler:
            self._modules["cognitive_profiler"] = CognitiveProfiler()
        if EpistemicAutonomyTracker:
            self._modules["autonomy_tracker"] = EpistemicAutonomyTracker()
        if InteractionSemioticsEngine:
            self._modules["semiotics"] = InteractionSemioticsEngine()
        if NDPatternDetector:
            self._modules["nd_detector"] = NDPatternDetector()

        # CAPA 2: Calidad y detección
        if RAGQualitySensor:
            self._modules["rag_sensor"] = RAGQualitySensor()
        if CognitiveGapDetector:
            self._modules["gap_detector"] = CognitiveGapDetector()
        if EpistemicSilenceDetector:
            self._modules["silence_detector"] = EpistemicSilenceDetector()
        if ConsolidationDetector:
            self._modules["consolidation"] = ConsolidationDetector()

        # CAPA 2: Nudges
        if MetacognitiveNudgeEngine:
            self._modules["nudge_engine"] = MetacognitiveNudgeEngine()
        if UDLAdapter:
            self._modules["udl_adapter"] = UDLAdapter()

        # CAPA 2: Evaluación
        if HHHAlignmentDetector:
            self._modules["hhh_detector"] = HHHAlignmentDetector()

        # CAPA 3: Configuración
        if ConfigGenomeAnalyzer:
            self._modules["config_genome"] = ConfigGenomeAnalyzer()
        if TemporalConfigAdvisor:
            self._modules["temporal_advisor"] = TemporalConfigAdvisor()
        if EffectLatencyAnalyzer:
            self._modules["effect_latency"] = EffectLatencyAnalyzer()

        # CAPA 3: Docente
        if TeacherCalibrationAnalyzer:
            self._modules["teacher_calibration"] = TeacherCalibrationAnalyzer()
        if TeacherNotificationEngine:
            self._modules["teacher_notifications"] = TeacherNotificationEngine()

        # CAPA 3: Inter-nodo
        if CrossNodeSignalEngine:
            self._modules["cross_node"] = CrossNodeSignalEngine(
                node_id=self.node_id
            )

        # CAPA 3: Meta
        if SystemReflexivity:
            self._modules["reflexivity"] = SystemReflexivity()

    def get_module(self, name: str):
        """Obtiene un módulo por nombre, o None si no está disponible."""
        return self._modules.get(name)

    @property
    def active_modules(self) -> List[str]:
        return list(self._modules.keys())

    # ──────────────────────────────────────────────────────────
    # FLUJO PRINCIPAL
    # ──────────────────────────────────────────────────────────

    def process_interaction(
        self,
        student_id: str,
        prompt: str,
        conversation_history: Optional[List[Dict]] = None,
    ) -> OrchestratedResult:
        """
        Procesa una interacción completa a través de todo el ecosistema.

        Esta función es lo único que app.py necesita llamar.
        Internamente orquesta middleware, RAG, LLM, y todos los módulos.
        """
        start_time = time.time()
        result = OrchestratedResult(response_text="")
        result.modules_activated = []

        # ─── FASE 1: PRE-PROCESS (middleware) ───────────────────

        pre = self.middleware.pre_process(student_id, prompt)

        if not pre["allowed"]:
            result.was_blocked = True
            result.block_reason = pre["block_reason"]
            result.response_text = pre["block_reason"]
            self._log_event("student_blocked", student_id, prompt, result)
            return result

        result.scaffolding_level = pre["scaffolding_level"]
        result.detected_topics = pre["detected_topics"]
        result.copy_paste_score = pre["copy_paste_score"]
        result.modules_activated.append("middleware")

        # ─── FASE 1.5: CONTEXTO TEMPORAL ────────────────────────

        temporal_context = self._get_temporal_context()
        if temporal_context:
            result.academic_pressure = temporal_context.get("pressure_index", 0.0)
            result.config_suggestion = temporal_context.get("suggestion")
            result.modules_activated.append("temporal_advisor")

        # ─── FASE 2: RAG RETRIEVAL ──────────────────────────────

        chunks = []
        if self.rag and self.config.use_rag:
            try:
                chunks = self.rag.retrieve(prompt, top_k=3)
                result.rag_chunks = chunks

                # Evaluar calidad RAG
                rag_quality = self._evaluate_rag_quality(prompt, chunks, student_id)
                if rag_quality:
                    result.rag_quality_score = rag_quality.get("quality_score", 0.0)
                    result.is_rephrase = rag_quality.get("is_rephrase", False)
                    result.modules_activated.append("rag_sensor")

            except Exception as e:
                print(f"  [WARNING] RAG retrieval error: {e}")

        # ─── FASE 3: LLM CALL ──────────────────────────────────

        if self.llm:
            try:
                context_text = "\n\n".join(
                    [c.get("text", c.get("content", "")) for c in chunks]
                ) if chunks else ""

                response = self.llm.generate(
                    system_prompt=pre["system_prompt"],
                    user_prompt=pre["processed_prompt"],
                    context=context_text,
                    conversation_history=conversation_history or [],
                )
                result.response_text = response
            except Exception as e:
                result.response_text = f"Error al generar respuesta: {e}"
                print(f"  [WARNING] LLM error: {e}")
        else:
            # Sin LLM: respuesta de demo
            result.response_text = self._demo_response(prompt, pre, chunks)

        result.modules_activated.append("llm_client")

        # ─── FASE 4: POST-PROCESS (middleware) ──────────────────

        post = self.middleware.post_process(student_id, result.response_text)
        result.response_text = post["response"]
        result.hallucination_injected = post.get("hallucination_injected", False)

        # ─── FASE 5: ANÁLISIS COGNITIVO ─────────────────────────

        cognitive = self._analyze_cognitive(student_id, prompt)
        if cognitive:
            result.bloom_estimate = cognitive.get("bloom_level", "unknown")
            result.bloom_level = cognitive.get("bloom_numeric", 0)
            result.autonomy_score = cognitive.get("autonomy", 0.0)
            result.nd_indicators = cognitive.get("nd_indicators", {})
            result.modules_activated.extend(
                cognitive.get("modules_used", [])
            )

        # ─── FASE 6: DETECCIÓN DE GAPS ──────────────────────────

        gap = self._detect_gaps(student_id, prompt, result.bloom_level)
        if gap:
            result.cognitive_gap = gap.get("gap_description")
            result.modules_activated.append("gap_detector")

        # ─── FASE 7: NUDGES METACOGNITIVOS ──────────────────────

        nudge = self._generate_nudge(student_id, result)
        if nudge:
            result.nudge = nudge.get("text")
            result.nudge_type = nudge.get("type")
            result.modules_activated.append("nudge_engine")

        # ─── FASE 8: EVALUACIÓN DE RESPUESTA ────────────────────

        hhh = self._evaluate_alignment(prompt, result.response_text)
        if hhh:
            result.hhh_alignment = hhh
            result.modules_activated.append("hhh_detector")

        # ─── FASE 9: CALIBRACIÓN DOCENTE ────────────────────────

        calibration = self._check_teacher_calibration(
            result.bloom_level, result.scaffolding_level
        )
        if calibration:
            result.calibration_alert = calibration.get("alert")
            result.modules_activated.append("teacher_calibration")

        # ─── FASE 10: LOGGING DE EVENTO ─────────────────────────

        result.event_id = self._log_event(
            "student_prompt", student_id, prompt, result
        )

        # ─── FASE 11: BACKGROUND (no bloquea) ──────────────────

        self._background_processes(student_id, result)

        # ─── FINALIZAR ──────────────────────────────────────────

        result.processing_time_ms = int((time.time() - start_time) * 1000)

        return result

    # ──────────────────────────────────────────────────────────
    # MÉTODOS INTERNOS — cada uno invoca un módulo si existe
    # ──────────────────────────────────────────────────────────

    def _get_temporal_context(self) -> Optional[Dict]:
        """Obtiene contexto temporal académico."""
        advisor = self.get_module("temporal_advisor")
        if not advisor:
            return None
        try:
            # En DEMO_MODE, simula semana 8 del semestre (pre-parcial)
            if self.demo_mode:
                return {
                    "pressure_index": 0.72,
                    "academic_moment": "pre-examen parcial",
                    "suggestion": "Considerar relajar scaffolding: los estudiantes "
                                  "necesitan respuestas más directas antes del examen.",
                    "week": 8,
                }
            return advisor.get_current_context()
        except Exception as e:
            print(f"  [WARNING] Temporal advisor error: {e}")
            return None

    def _evaluate_rag_quality(
        self, query: str, chunks: List[Dict], student_id: str
    ) -> Optional[Dict]:
        """Evalúa calidad de los chunks recuperados."""
        sensor = self.get_module("rag_sensor")
        if not sensor:
            return None
        try:
            # Interfaz mínima: evalúa relevancia y detecta reformulación
            quality_score = 0.0
            if chunks:
                # Heurística: chunks con score alto = buena calidad
                scores = [c.get("score", 0.5) for c in chunks]
                quality_score = sum(scores) / len(scores) if scores else 0.0

            return {
                "quality_score": quality_score,
                "is_rephrase": False,  # TODO: comparar con prompt anterior del estudiante
                "chunks_evaluated": len(chunks),
            }
        except Exception as e:
            print(f"  [WARNING] RAG sensor error: {e}")
            return None

    def _analyze_cognitive(self, student_id: str, prompt: str) -> Optional[Dict]:
        """Análisis cognitivo completo del prompt."""
        modules_used = []
        result = {}

        # Cognitive Profiler
        profiler = self.get_module("cognitive_profiler")
        if profiler:
            try:
                snapshot = profiler.analyze_prompt(student_id, prompt)
                result["bloom_level"] = getattr(snapshot, "bloom_level", "remember")
                result["bloom_numeric"] = getattr(snapshot, "bloom_numeric", 1)
                modules_used.append("cognitive_profiler")
            except Exception:
                pass

        # Autonomy Tracker
        autonomy = self.get_module("autonomy_tracker")
        if autonomy:
            try:
                score = autonomy.estimate_autonomy(student_id, prompt)
                result["autonomy"] = score if isinstance(score, float) else 0.5
                modules_used.append("autonomy_tracker")
            except Exception:
                pass

        # ND Pattern Detector
        nd = self.get_module("nd_detector")
        if nd:
            try:
                indicators = nd.analyze(prompt)
                result["nd_indicators"] = indicators if isinstance(indicators, dict) else {}
                modules_used.append("nd_detector")
            except Exception:
                pass

        result["modules_used"] = modules_used
        return result if modules_used else None

    def _detect_gaps(
        self, student_id: str, prompt: str, bloom_level: int
    ) -> Optional[Dict]:
        """Detecta gaps cognitivos."""
        detector = self.get_module("gap_detector")
        if not detector:
            return None
        try:
            # Interfaz simplificada
            return detector.check(student_id, prompt, bloom_level)
        except Exception:
            return None

    def _generate_nudge(
        self, student_id: str, result: OrchestratedResult
    ) -> Optional[Dict]:
        """Genera nudge metacognitivo si aplica."""
        engine = self.get_module("nudge_engine")
        if not engine:
            return None
        try:
            return engine.suggest(
                student_id=student_id,
                bloom_level=result.bloom_level,
                gap=result.cognitive_gap,
                autonomy=result.autonomy_score,
            )
        except Exception:
            return None

    def _evaluate_alignment(
        self, prompt: str, response: str
    ) -> Optional[Dict]:
        """Evalúa alineamiento HHH de la respuesta."""
        detector = self.get_module("hhh_detector")
        if not detector:
            return None
        try:
            return detector.evaluate(prompt, response)
        except Exception:
            return None

    def _check_teacher_calibration(
        self, bloom_level: int, scaffolding_level: int
    ) -> Optional[Dict]:
        """Verifica calibración docente: ¿el scaffolding es adecuado para el nivel?"""
        calibrator = self.get_module("teacher_calibration")
        if not calibrator:
            # Heurística simple sin el módulo completo
            if bloom_level >= 4 and scaffolding_level == 0:
                return {
                    "alert": "[WARNING] Estudiante en nivel Bloom alto (analyze+) pero "
                             "scaffolding al mínimo. Considerar dar más autonomía.",
                    "calibration_score": 0.6,
                }
            if bloom_level <= 1 and scaffolding_level >= 3:
                return {
                    "alert": "[WARNING] Estudiante en nivel Bloom bajo (remember) recibiendo "
                             "explicaciones completas. El modo socrático podría ser más efectivo.",
                    "calibration_score": 0.5,
                }
            return None
        try:
            return calibrator.evaluate(
                config_snapshot=asdict(self.config),
                student_bloom=bloom_level,
                scaffolding_used=scaffolding_level,
            )
        except Exception:
            return None

    def _log_event(
        self,
        event_type: str,
        student_id: str,
        prompt: str,
        result: OrchestratedResult,
    ) -> str:
        """Registra evento en el system_event_logger con las 4 columnas diferenciales."""
        logger = self.get_module("event_logger")
        if not logger:
            return ""
        try:
            if create_student_prompt_event:
                event = create_student_prompt_event(
                    student_id=student_id,
                    prompt=prompt,
                    topics=result.detected_topics,
                    copy_paste_score=result.copy_paste_score,
                    # Las 4 columnas diferenciales:
                    config_snapshot=asdict(self.config),
                    bloom_estimate=result.bloom_estimate,
                    pressure_index=result.academic_pressure,
                )
                logger.log_event(event)
                return getattr(event, "event_id", "")
            return ""
        except Exception as e:
            print(f"  [WARNING] Event logger error: {e}")
            return ""

    def _background_processes(self, student_id: str, result: OrchestratedResult):
        """Procesos que corren después de entregar la respuesta."""

        # Config genome: snapshot de la configuración actual
        genome = self.get_module("config_genome")
        if genome:
            try:
                genome.snapshot(asdict(self.config))
                result.modules_activated.append("config_genome")
            except Exception:
                pass

        # Consolidation detector: ¿hay patrón de consolidación?
        consolidation = self.get_module("consolidation")
        if consolidation:
            try:
                consolidation.check_window(student_id)
                result.modules_activated.append("consolidation")
            except Exception:
                pass

        # Silence detector: actualizar timestamp
        silence = self.get_module("silence_detector")
        if silence:
            try:
                silence.update(student_id, datetime.now())
                result.modules_activated.append("silence_detector")
            except Exception:
                pass

        # Cross-node: emitir señal si relevante
        cross = self.get_module("cross_node")
        if cross:
            try:
                cross.emit_if_relevant({
                    "student_id": student_id,
                    "bloom": result.bloom_estimate,
                    "topics": result.detected_topics,
                    "pressure": result.academic_pressure,
                })
                result.modules_activated.append("cross_node")
            except Exception:
                pass

        # System reflexivity
        reflexivity = self.get_module("reflexivity")
        if reflexivity:
            try:
                reflexivity.reflect({
                    "event_type": "interaction_complete",
                    "modules_activated": result.modules_activated,
                    "processing_time_ms": result.processing_time_ms,
                })
                result.modules_activated.append("reflexivity")
            except Exception:
                pass

    def _demo_response(
        self, prompt: str, pre: Dict, chunks: List[Dict]
    ) -> str:
        """Genera respuesta simulada cuando no hay LLM configurado."""
        context_preview = ""
        if chunks:
            context_preview = chunks[0].get("text", "")[:200]

        if self.config.scaffolding_mode == "socratic":
            return (
                f"🤔 Esa es una buena pregunta. Antes de darte una respuesta directa, "
                f"me gustaría que reflexionaras: ¿qué conceptos previos crees que están "
                f"relacionados con lo que preguntas?\n\n"
                f"💡 Pista: piensa en los temas de {', '.join(pre['detected_topics'][:2]) or 'la asignatura'}.\n\n"
                f"📚 [Contexto RAG disponible: {len(chunks)} fragmentos recuperados]"
            )
        else:
            return (
                f"Basándome en el material del curso, aquí va mi explicación sobre "
                f"tu pregunta.\n\n"
                f"📚 [Contexto RAG: {len(chunks)} fragmentos | "
                f"Temas detectados: {', '.join(pre['detected_topics'][:3])}]\n\n"
                f"[Respuesta de demo — conecta LLM para respuestas reales]"
            )

    # ──────────────────────────────────────────────────────────
    # MÉTODOS PÚBLICOS PARA DASHBOARD
    # ──────────────────────────────────────────────────────────

    def get_ecosystem_status(self) -> Dict:
        """Estado completo del ecosistema para el dashboard."""
        return {
            "demo_mode": self.demo_mode,
            "node_id": self.node_id,
            "modules_total": 51,
            "modules_active": len(self._modules),
            "modules_list": list(self._modules.keys()),
            "config_fingerprint": self._get_config_fingerprint(),
            "timestamp": datetime.now().isoformat(),
        }

    def get_module_health(self) -> Dict[str, str]:
        """Health check de cada módulo."""
        all_modules = [
            "event_logger", "cognitive_profiler", "autonomy_tracker",
            "semiotics", "nd_detector", "rag_sensor", "gap_detector",
            "silence_detector", "consolidation", "nudge_engine",
            "udl_adapter", "hhh_detector", "config_genome",
            "temporal_advisor", "effect_latency", "teacher_calibration",
            "teacher_notifications", "cross_node", "reflexivity",
        ]
        health = {}
        for name in all_modules:
            if name in self._modules:
                health[name] = "[OK] activo"
            else:
                health[name] = "[--] no cargado"
        return health

    def _get_config_fingerprint(self) -> str:
        """Genera fingerprint de la configuración actual."""
        genome = self.get_module("config_genome")
        if genome:
            try:
                return genome.fingerprint(asdict(self.config))
            except Exception:
                pass
        # Fallback: hash simple
        config_str = json.dumps(asdict(self.config), sort_keys=True)
        return hex(hash(config_str))[:12]


# ──────────────────────────────────────────────────────────────
# DEMO: ejecutar directamente para verificar
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[GENIE] GENIE Learn — Ecosystem Orchestrator Demo\n")

    # Crear con configuración por defecto
    config = PedagogicalConfig()
    orch = EcosystemOrchestrator(config, demo_mode=True)

    # Simular una interacción
    print("\n--- Procesando interacción de prueba ---\n")
    result = orch.process_interaction(
        student_id="estudiante_demo",
        prompt="¿Qué es la recursión en programación?",
    )

    print(f"  Respuesta: {result.response_text[:200]}...")
    print(f"  Bloom: {result.bloom_estimate}")
    print(f"  Scaffolding: {result.scaffolding_level}")
    print(f"  Topics: {result.detected_topics}")
    print(f"  Copy-paste: {result.copy_paste_score}")
    print(f"  Presión académica: {result.academic_pressure}")
    print(f"  Nudge: {result.nudge or 'ninguno'}")
    print(f"  Calibración: {result.calibration_alert or 'OK'}")
    print(f"  Módulos activados: {result.modules_activated}")
    print(f"  Tiempo: {result.processing_time_ms}ms")

    print("\n--- Estado del ecosistema ---\n")
    status = orch.get_ecosystem_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\n--- Health check ---\n")
    health = orch.get_module_health()
    for name, state in health.items():
        print(f"  {name}: {state}")
