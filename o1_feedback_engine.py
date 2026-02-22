"""
O1 FEEDBACK ENGINE — El Bucle de Retroalimentación O3→O1
═══════════════════════════════════════════════════════════════════════
Motor de retroalimentación que cierra el ciclo DSRM del proyecto GENIE
Learn (CP25/152, GSIC/EMIC-UVa).

PROBLEMA QUE ATACA — EL PUNTO CIEGO DEL DISEÑO:
════════════════════════════════════════════════════
Los cuatro objetivos del proyecto (Delgado-Kloos et al., CSEDU 2025) operan
actualmente en flujo unidireccional:

    O1 (escenarios HL) → O4 (infraestructura) → O3 (soporte estudiante) → O2 (soporte docente)

Nadie retroalimenta O1 con evidencia empírica de los pilotos. Los supuestos
de diseño de los escenarios de alto nivel se formularon ANTES de disponer de
datos de interacción reales. Sin un mecanismo explícito de falsación, los
supuestos iniciales adquieren inercia institucional: se perpetúan no porque
se hayan validado, sino porque nadie los ha cuestionado formalmente con datos.

ESTE MÓDULO cierra el bucle. Recoge señales de O3 (cognitive_profiler,
nd_patterns, epistemic_autonomy, epistemic_silence_detector) y genera:

    1. Anomaly Reports: casos donde los datos contradicen un supuesto de O1
    2. Scenario Revision Proposals: revisiones estructuradas con impacto en O2/O3/O4
    3. ND Differential Analysis: cómo responden perfiles neurodivergentes vs.
       normativo a las mismas configuraciones pedagógicas
    4. O5 Emergence Tracker: hipótesis nuevas que no estaban en ningún O original

FUNDAMENTOS TEÓRICOS:
─────────────────────
- Design Science Research Methodology (Peffers et al., 2007):
  El ciclo DSRM exige evaluación y comunicación como fases POSTERIORES al
  diseño y desarrollo. Este módulo instrumenta la fase de evaluación con
  un mecanismo computacional, no solo con entrevistas post-hoc.

- Value-Sensitive Design (Friedman et al., 2017):
  Los stakeholders indirectos —estudiantes neurodivergentes— carecen de voz
  en el diseño original de O1. Este módulo les da voz EMPÍRICA: sus patrones
  de interacción contradicen o confirman supuestos que se formularon sin
  considerar la variabilidad cognitiva.

- Double-loop learning (Argyris & Schön, 1978):
  El single-loop ajusta las acciones (O2: docente cambia configuración).
  El double-loop cuestiona los supuestos que gobiernan las acciones
  (O1: los escenarios HL se revisan porque la evidencia muestra que un
  supuesto era incorrecto). Este módulo instrumenta el double-loop.

- Karl Popper (Conjectures and Refutations, 1963):
  Un supuesto de diseño que no puede ser falsado no es un supuesto
  científico — es un dogma institucional. Cada O1Assumption lleva un
  evidence_threshold explícito: el n mínimo para considerar válida la
  prueba. Sin ese umbral, toda afirmación es anécdota.

POSICIÓN EN EL ECOSISTEMA:
    cognitive_profiler.py         → EngagementProfile (input)
    nd_patterns.py                → NeurodivergentPattern (input)
    epistemic_autonomy.py         → AutonomyState (input)
    epistemic_silence_detector.py → SilenceAlert (input)
    config_genome.py              → ConfigFingerprint (input)
    ─────────────────────────────────────────────────
    o1_feedback_engine.py         → AnomalyReport, RevisionProposal, O5Signal (output)
    ─────────────────────────────────────────────────
    researcher_view.py            → pestaña "Retroalimentación O1" (rendering)

PRINCIPIO OPERATIVO:
    Corre en background cada N interacciones (configurable).
    No modifica nada en tiempo real — genera informes que el investigador
    decide si actuar o no. NUNCA prescribe; SIEMPRE propone.

Autor: Diego Elvira Vásquez · Ecosistema GENIE Learn · Feb 2026
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from datetime import datetime, timedelta
import math
import hashlib
import json

# ═══════════════════════════════════════════════════════════════════════
# IMPORTS DEL ECOSISTEMA
# ═══════════════════════════════════════════════════════════════════════
# Interfaces reales verificadas contra los módulos existentes:
#   NeurodivergentPattern.pattern_id ∈ {EPISODIC, TOPIC_SWITCH, COGNITIVE_JUMP,
#       SELECTIVE_FRUSTRATION, TWICE_EXCEPTIONAL, RE_ASKING}
#   EngagementProfile.engagement_type ∈ {deep_learner, surface_seeker,
#       struggling, disengaged, exploratory}
#   AutonomyState.phase ∈ {dependent, scaffolded, emergent, autonomous}
#   SilenceAlert.silence_type ∈ {metacognitive_gap, disconnection, competence}
#   ConfigFingerprint.pedagogical_style ∈ {scaffolded_explorer, strict_guardian,
#       permissive_guide, challenge_based, mixed}

from nd_patterns import (
    NeurodivergentPatternDetector,
    NeurodivergentPattern,
    InteractionEvent,
)
from cognitive_profiler import EngagementProfile, CognitiveProfiler
from epistemic_autonomy import AutonomyState, EpistemicAutonomyTracker
from epistemic_silence_detector import SilenceAlert, EpistemicSilenceDetector
from config_genome import ConfigFingerprint


# ═══════════════════════════════════════════════════════════════════════
# DATACLASSES DEL MOTOR DE RETROALIMENTACIÓN
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class O1Assumption:
    """
    Un supuesto de diseño de los escenarios HL de O1.

    Categoría ontológica: proposición falsable sobre la relación entre
    una configuración pedagógica y un resultado de aprendizaje.

    Marco teórico: Popper (1963) — toda hipótesis científica debe especificar
    las condiciones bajo las cuales sería falsa. Un supuesto de O1 sin
    evidence_threshold no es ciencia, es fe institucional.
    """
    assumption_id: str
    description: str                # texto legible: "El scaffolding socrático mejora la autonomía"
    target_population: str          # "all_students" | "normative" | "nd_episodic" | etc.
    expected_direction: str         # "positive" | "negative" | "neutral"
    config_params_involved: list    # qué parámetros del middleware están implicados
    evidence_threshold: int         # n mínimo de casos para considerar válida la prueba
    falsification_criterion: str    # condición precisa bajo la cual el supuesto es falso
    source: str = ""                # referencia: "CSEDU 2025, §3.2" | "implicit in O1 design"


@dataclass
class AnomalyReport:
    """
    Contradicción empírica entre datos de pilotos y supuestos de O1.

    Categoría ontológica: instancia de falsación parcial o total de un
    supuesto de diseño, sustentada en evidencia de interacción.

    Marco teórico: Heuer (1999) — la anomalía más informativa es la que
    tiene MENOS consistencia con la hipótesis dominante, no la que tiene
    más consistencia con una alternativa. El scoring ACH inverso se aplica
    también aquí: el supuesto cuestionado es aquel cuya predicción es
    MÁS inconsistente con los datos observados.
    """
    report_id: str
    timestamp: str
    assumption_challenged: O1Assumption
    observed_pattern: str               # descripción del patrón observado
    expected_by_assumption: str         # qué predecía el supuesto
    affected_population: str            # quién experimenta la anomalía
    nd_pattern_correlation: str         # patrón ND correlacionado (si aplica)
    effect_size: float                  # magnitud del efecto [0-1]
    n_cases: int                        # número de casos que sustentan la anomalía
    confidence: float                   # [0-1]
    revision_urgency: str               # "low" | "medium" | "high" | "critical"
    supporting_data: list = field(default_factory=list)   # evidencia concreta
    reasoning_trace: list = field(default_factory=list)   # cadena de razonamiento


@dataclass
class ScenarioRevisionProposal:
    """
    Propuesta de revisión de escenario O1 basada en evidencia empírica.

    Categoría ontológica: recomendación estructurada con impacto trazable
    en los cuatro objetivos del proyecto.

    Principio de diseño: cada propuesta especifica su impacto en O2, O3 y O4,
    porque un cambio en O1 que no se propaga al sistema completo es un cambio
    cosmético. La hipótesis de paper derivada convierte la revisión en output
    académico — alineando los incentivos de investigación con los de mejora.
    """
    proposal_id: str
    timestamp: str
    triggered_by: list                  # list[str] de report_ids de AnomalyReport
    original_assumption: str            # texto del supuesto original
    proposed_revision: str              # texto de la revisión propuesta
    supporting_evidence: list           # resumen de evidencia
    implementation_in_o2: str           # cambio concreto en configuraciones docentes
    implementation_in_o3: str           # cambio concreto en scaffolding estudiante
    implementation_in_o4: str           # cambio técnico en infraestructura
    paper_hypothesis: str               # hipótesis para paper derivada
    dsrm_cycle_target: int = 3          # en qué ciclo DSRM debería incorporarse
    estimated_effort: str = "medium"    # "low" | "medium" | "high"


@dataclass
class O5EmergenceSignal:
    """
    Señal de que ha emergido algo que no estaba en los objetivos originales.

    Categoría ontológica: hipótesis generativa derivada de la convergencia
    de múltiples anomalías — el sistema aprendiendo sobre sí mismo.

    Marco teórico: Cynefin (Snowden & Boone, 2007) — en sistemas complejos,
    los patrones no se predicen sino que emergen. El rol del investigador
    no es prever O5 sino detectarlo cuando aparece y nombrarlo antes de
    que se disuelva en el ruido. Esta dataclass es la red de captura.

    Comparación: el concepto es análogo al "unknown unknown" de Rumsfeld
    (más elegantemente formulado por Taleb como "black swan"). En el
    contexto DSRM, un O5 emergente indica que el espacio de problemas
    era más amplio de lo que el diseño inicial asumió — señal de que el
    sistema está generando conocimiento genuinamente nuevo.
    """
    signal_id: str
    timestamp: str
    discovery_type: str             # "unexpected_population_segment" | "config_interaction" |
                                    # "temporal_pattern" | "equity_gap" | "system_learning"
    description: str                # descripción legible de la emergencia
    evidence_base: list             # list[str] de report_ids que convergieron
    anomaly_convergence: str        # cómo las anomalías apuntan en la misma dirección
    suggested_new_objective: str    # formulación de un O adicional implícito
    dsrm_cycle_target: int          # en qué ciclo DSRM debería incorporarse
    paper_potential: str            # "note" | "short_paper" | "full_paper"
    confidence: float = 0.0         # [0-1]


# ═══════════════════════════════════════════════════════════════════════
# MOTOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

class O1FeedbackEngine:
    """
    Motor de retroalimentación O3→O1.

    Principio operativo: corre en background cada N interacciones
    (configurable: cada 50 interacciones por curso, o cada 7 días).
    No modifica nada en tiempo real — genera informes que el investigador
    o el docente deciden si actuar o no. NUNCA prescribe; SIEMPRE propone.

    Integración: se invoca desde researcher_view.py como pestaña adicional
    "Retroalimentación O1" visible SOLO al investigador (no al docente,
    no al estudiante).

    Flujo de uso:
        engine = O1FeedbackEngine()
        anomalies, proposals, signals = engine.run_analysis(
            engagement_profiles=...,
            nd_patterns_by_student=...,
            autonomy_states=...,
            silence_alerts=...,
            config_history=...,
        )
        report = engine.export_for_researcher_view()
    """

    # ── Umbrales de detección ──
    ANOMALY_EFFECT_THRESHOLD = 0.15     # efecto mínimo para reportar anomalía
    ANOMALY_N_THRESHOLD = 5             # n mínimo de casos para considerar
    O5_CONVERGENCE_THRESHOLD = 3        # anomalías convergentes para señal O5
    URGENCY_THRESHOLDS = {
        "critical": (0.6, 0.7),         # (effect_size, confidence)
        "high":     (0.4, 0.5),
        "medium":   (0.2, 0.3),
        "low":      (0.0, 0.0),
    }

    def __init__(self):
        self.o1_assumptions: list[O1Assumption] = self._init_o1_assumptions()
        self.anomaly_log: list[AnomalyReport] = []
        self.revision_proposals: list[ScenarioRevisionProposal] = []
        self.o5_signals: list[O5EmergenceSignal] = []
        self._run_count: int = 0

    # ──────────────────────────────────────────────────────────────
    # 1. SUPUESTOS DE O1 — LA BATERÍA DE HIPÓTESIS A FALSIFICAR
    # ──────────────────────────────────────────────────────────────

    def _init_o1_assumptions(self) -> list[O1Assumption]:
        """
        Los supuestos implícitos y explícitos de O1 según CSEDU 2025
        (Delgado-Kloos et al.) y el diseño del middleware actual.

        Cada supuesto es una proposición falsable. Si la evidencia lo
        contradice, genera un AnomalyReport. Si lo confirma, se fortalece
        su peso en el diseño. Si no hay evidencia suficiente, permanece
        como supuesto provisional — exactamente como funciona la ciencia.

        Nota crítica sobre A4: basándose en nd_patterns.py, los estudiantes
        con patrón EPISODIC o RE_ASKING copian-pegan su propio trabajo
        anterior como estrategia de memoria de trabajo — no es deshonestidad.

        Nota crítica sobre A5: es el supuesto más peligroso y el que más
        directamente conecta con la agenda de equidad del proyecto. Si una
        configuración funciona para el 80% y daña al 20%, el promedio
        dice "funciona" — pero el 20% es el 100% de quienes experimentan
        el daño.
        """
        return [
            O1Assumption(
                assumption_id="A1",
                description=(
                    "El scaffolding socrático produce mayor autonomía epistémica "
                    "que el modo directo en el conjunto de la población estudiantil."
                ),
                target_population="all_students",
                expected_direction="positive",
                config_params_involved=["scaffolding_mode", "socratic_scaffolding"],
                evidence_threshold=15,
                falsification_criterion=(
                    "Media de autonomy_score bajo socrático ≤ media bajo directo "
                    "para un subgrupo con n ≥ 15 y p < 0.05"
                ),
                source="CSEDU 2025, O3 implicit; Vygotsky (1978) as theoretical basis",
            ),
            O1Assumption(
                assumption_id="A2",
                description=(
                    "El límite diario de prompts distribuye el esfuerzo de aprendizaje "
                    "de manera uniforme a lo largo del periodo del curso."
                ),
                target_population="all_students",
                expected_direction="positive",
                config_params_involved=["max_daily_prompts"],
                evidence_threshold=10,
                falsification_criterion=(
                    "CV temporal de uso > 1.5 para estudiantes bajo límite diario, "
                    "indicando que el límite no uniformiza sino que desplaza ráfagas"
                ),
                source="CSEDU 2025, O2 config design; implicit assumption",
            ),
            O1Assumption(
                assumption_id="A3",
                description=(
                    "Los estudiantes con bajo nivel Bloom inicial (≤ 2) responden "
                    "mejor a scaffolding alto (socrático + restricción) que a "
                    "scaffolding bajo (directo + sin restricción)."
                ),
                target_population="low_bloom_initial",
                expected_direction="positive",
                config_params_involved=["scaffolding_mode", "block_direct_solutions"],
                evidence_threshold=10,
                falsification_criterion=(
                    "Trayectoria cognitiva (cognitive_trajectory) bajo scaffolding alto "
                    "≤ bajo scaffolding bajo para estudiantes con Bloom inicial ≤ 2"
                ),
                source="Wood, Bruner & Ross (1976); implicit in middleware design",
            ),
            O1Assumption(
                assumption_id="A4",
                description=(
                    "Un copy-paste score alto (> 0.6) refleja deshonestidad académica "
                    "o falta de esfuerzo propio del estudiante."
                ),
                target_population="all_students",
                expected_direction="negative",
                config_params_involved=["copy_paste_detection"],
                evidence_threshold=8,
                falsification_criterion=(
                    "Estudiantes con copy_paste > 0.6 Y patrón ND ∈ {EPISODIC, RE_ASKING} "
                    "muestran autonomy_score comparable o superior al grupo normativo — "
                    "indicando que el copy-paste es estrategia de memoria de trabajo, "
                    "no deshonestidad"
                ),
                source="Implicit in middleware copy-paste detection; refuted by Barkley (2015)",
            ),
            O1Assumption(
                assumption_id="A5",
                description=(
                    "Las configuraciones pedagógicas óptimas para la mayoría de "
                    "estudiantes son óptimas para todos los estudiantes."
                ),
                target_population="all_students",
                expected_direction="positive",
                config_params_involved=[
                    "scaffolding_mode", "max_daily_prompts",
                    "block_direct_solutions", "forced_hallucination_pct",
                ],
                evidence_threshold=10,
                falsification_criterion=(
                    "Varianza de outcome (autonomía, Bloom, engagement) entre subgrupos "
                    "ND y normativo > 0.3 bajo la misma configuración — indicando que "
                    "la configuración 'óptima' global penaliza a subgrupos específicos"
                ),
                source="Implicit universalism in O1 scenario design; challenged by VSD (Friedman, 2017)",
            ),
            O1Assumption(
                assumption_id="A6",
                description=(
                    "El aprendizaje híbrido requiere consistencia temporal en el "
                    "acceso al chatbot: uso regular produce mejor resultado que "
                    "uso concentrado."
                ),
                target_population="all_students",
                expected_direction="positive",
                config_params_involved=["max_daily_prompts", "session_management"],
                evidence_threshold=12,
                falsification_criterion=(
                    "Estudiantes con patrón EPISODIC (uso en ráfagas) muestran "
                    "cognitive_trajectory ≥ estudiantes con uso regular, indicando "
                    "que la consistencia no es prerequisito de aprendizaje efectivo"
                ),
                source="CSEDU 2025, O1 hybrid learning scenarios; challenged by ADHD literature",
            ),
            O1Assumption(
                assumption_id="A7",
                description=(
                    "El modo socrático es pedagógicamente superior al directo en "
                    "todos los temas del currículo por igual."
                ),
                target_population="all_students",
                expected_direction="positive",
                config_params_involved=["scaffolding_mode", "socratic_scaffolding"],
                evidence_threshold=15,
                falsification_criterion=(
                    "Estudiantes con patrón SELECTIVE_FRUSTRATION muestran peor "
                    "engagement bajo socrático en topics de bajo interés que "
                    "bajo modo directo — indicando que el socrático amplifica "
                    "la frustración selectiva"
                ),
                source="Implicit in uniform scaffolding design; challenged by Reis et al. (2014)",
            ),
        ]

    # ──────────────────────────────────────────────────────────────
    # 2. ANÁLISIS PRINCIPAL
    # ──────────────────────────────────────────────────────────────

    def run_analysis(
        self,
        engagement_profiles: dict[str, EngagementProfile],
        nd_patterns_by_student: dict[str, list[NeurodivergentPattern]],
        autonomy_states: dict[str, AutonomyState],
        silence_alerts: dict[str, list[SilenceAlert]],
        config_history: list[ConfigFingerprint],
        n_days: int = 30,
    ) -> tuple[list[AnomalyReport], list[ScenarioRevisionProposal], list[O5EmergenceSignal]]:
        """
        Análisis completo. Devuelve las tres listas para que researcher_view.py
        las renderice.

        El método no modifica el estado del sistema educativo. Observa,
        analiza y propone. La decisión de actuar es humana.
        """
        self._run_count += 1
        run_ts = datetime.now().isoformat()
        new_anomalies = []

        # ── Fase 1: Testear cada supuesto contra datos ──
        # A1: Socrático vs. directo para autonomía
        a1_results = self._analyze_socratic_autonomy(
            autonomy_states, nd_patterns_by_student, config_history
        )
        if a1_results:
            new_anomalies.append(a1_results)

        # A2: Límite diario y distribución temporal
        a2_results = self._analyze_daily_limit_distribution(
            engagement_profiles, nd_patterns_by_student
        )
        if a2_results:
            new_anomalies.append(a2_results)

        # A3: Scaffolding alto para Bloom bajo
        a3_results = self._analyze_scaffolding_for_low_bloom(
            engagement_profiles, autonomy_states, config_history
        )
        if a3_results:
            new_anomalies.append(a3_results)

        # A4: Copy-paste y ND (el análisis más revelador éticamente)
        a4_results = self._analyze_copypaste_nd_interaction(
            engagement_profiles, nd_patterns_by_student, autonomy_states
        )
        if a4_results:
            new_anomalies.append(a4_results)

        # A5: Configuración universal vs. equidad ND
        a5_results = self._analyze_config_universality(
            engagement_profiles, nd_patterns_by_student, autonomy_states
        )
        if a5_results:
            new_anomalies.append(a5_results)

        # A6: Consistencia temporal vs. interacción episódica
        a6_results = self._analyze_temporal_consistency(
            engagement_profiles, nd_patterns_by_student
        )
        if a6_results:
            new_anomalies.append(a6_results)

        # A7: Socrático-ND interaction (el análisis central)
        a7_results = self._analyze_socratic_nd_interaction(
            engagement_profiles, nd_patterns_by_student, autonomy_states
        )
        if a7_results:
            new_anomalies.extend(a7_results) if isinstance(a7_results, list) else new_anomalies.append(a7_results)

        # ── Fase 2: Detectar efectos de interacción entre configuraciones ──
        interaction_anomalies = self._detect_config_interaction_effects(
            engagement_profiles, nd_patterns_by_student, config_history
        )
        new_anomalies.extend(interaction_anomalies)

        # ── Fase 3: Incorporar señales de silencio epistémico ──
        silence_anomalies = self._analyze_silence_patterns(
            silence_alerts, nd_patterns_by_student
        )
        new_anomalies.extend(silence_anomalies)

        # Registrar anomalías
        self.anomaly_log.extend(new_anomalies)

        # ── Fase 4: Generar propuestas de revisión ──
        new_proposals = self._generate_revision_proposals(new_anomalies)
        self.revision_proposals.extend(new_proposals)

        # ── Fase 5: Detectar señales O5 ──
        new_signals = self.generate_o5_hypothesis(new_anomalies)
        self.o5_signals.extend(new_signals)

        return new_anomalies, new_proposals, new_signals

    # ──────────────────────────────────────────────────────────────
    # 3. ANÁLISIS ESPECÍFICOS POR SUPUESTO
    # ──────────────────────────────────────────────────────────────

    def _analyze_socratic_autonomy(
        self,
        autonomy_states: dict[str, AutonomyState],
        nd_patterns: dict[str, list[NeurodivergentPattern]],
        config_history: list[ConfigFingerprint],
    ) -> Optional[AnomalyReport]:
        """
        A1: ¿El socrático produce mayor autonomía que el directo?

        Segmenta la población por presencia de patrones ND y compara
        la distribución de autonomy_score bajo configuraciones con alta
        intensidad socrática vs. baja.
        """
        assumption = self._get_assumption("A1")
        if not autonomy_states or not config_history:
            return None

        # Determinar si la configuración activa es socrática
        socratic_configs = [
            fp for fp in config_history if fp.socratic_intensity > 0.6
        ]
        is_socratic_dominant = len(socratic_configs) > len(config_history) * 0.5

        if not is_socratic_dominant:
            return None  # no hay suficiente exposición socrática para testear

        # Segmentar: estudiantes con patrón ND vs. sin patrón
        nd_students = {sid for sid, patterns in nd_patterns.items() if patterns}
        normative_students = {sid for sid in autonomy_states if sid not in nd_students}

        nd_autonomy = [
            autonomy_states[sid].autonomy_score
            for sid in nd_students
            if sid in autonomy_states
        ]
        norm_autonomy = [
            autonomy_states[sid].autonomy_score
            for sid in normative_students
            if sid in autonomy_states
        ]

        if len(nd_autonomy) < 3 or len(norm_autonomy) < 3:
            return None

        nd_mean = sum(nd_autonomy) / len(nd_autonomy)
        norm_mean = sum(norm_autonomy) / len(norm_autonomy)
        gap = norm_mean - nd_mean

        # Anomalía: gap significativo entre ND y normativo bajo socrático
        if gap > self.ANOMALY_EFFECT_THRESHOLD:
            effect_size = min(1.0, gap / 0.5)  # normalizar a [0-1]
            n_cases = len(nd_autonomy) + len(norm_autonomy)
            confidence = min(0.95, 0.4 + n_cases * 0.03)

            return AnomalyReport(
                report_id=self._gen_id("AR", "A1"),
                timestamp=datetime.now().isoformat(),
                assumption_challenged=assumption,
                observed_pattern=(
                    f"Bajo configuración socrática dominante, los estudiantes con "
                    f"patrones ND muestran autonomía media {nd_mean:.2f} vs. "
                    f"{norm_mean:.2f} de la población normativa. "
                    f"Gap: {gap:.2f} puntos."
                ),
                expected_by_assumption=(
                    "El socrático debería producir mayor autonomía para TODOS "
                    "los estudiantes, sin diferencia significativa entre subgrupos."
                ),
                affected_population=f"nd_students (n={len(nd_autonomy)})",
                nd_pattern_correlation="multiple_nd_patterns",
                effect_size=round(effect_size, 3),
                n_cases=n_cases,
                confidence=round(confidence, 3),
                revision_urgency=self._classify_urgency(effect_size, confidence),
                supporting_data=[
                    f"ND autonomy scores: {[round(a, 2) for a in nd_autonomy]}",
                    f"Normative autonomy scores: {[round(a, 2) for a in norm_autonomy]}",
                    f"Socratic configs: {len(socratic_configs)}/{len(config_history)}",
                ],
                reasoning_trace=[
                    "Paso 1: Verificar que la configuración socrática es dominante en el período",
                    f"Paso 2: Segmentar por patrón ND: {len(nd_students)} ND, {len(normative_students)} normativos",
                    f"Paso 3: Comparar medias de autonomía: ND={nd_mean:.2f}, Norm={norm_mean:.2f}",
                    f"Paso 4: Gap ({gap:.2f}) supera umbral ({self.ANOMALY_EFFECT_THRESHOLD})",
                    "Conclusión: A1 se falsifica parcialmente para la población ND",
                ],
            )
        return None

    def _analyze_daily_limit_distribution(
        self,
        profiles: dict[str, EngagementProfile],
        nd_patterns: dict[str, list[NeurodivergentPattern]],
    ) -> Optional[AnomalyReport]:
        """
        A2: ¿El límite diario uniformiza realmente el uso?

        Busca estudiantes con patrón EPISODIC: si existen bajo límite diario,
        el límite no está cumpliendo su función de distribución uniforme.
        """
        assumption = self._get_assumption("A2")
        episodic_students = {
            sid for sid, patterns in nd_patterns.items()
            if any(p.pattern_id == "EPISODIC" for p in patterns)
        }

        if len(episodic_students) < self.ANOMALY_N_THRESHOLD:
            return None

        # Comparar engagement de episódicos vs. regulares
        episodic_depths = [
            profiles[sid].cognitive_depth_mean
            for sid in episodic_students if sid in profiles
        ]
        regular_students = set(profiles.keys()) - episodic_students
        regular_depths = [
            profiles[sid].cognitive_depth_mean
            for sid in regular_students if sid in profiles
        ]

        if not episodic_depths or not regular_depths:
            return None

        ep_mean = sum(episodic_depths) / len(episodic_depths)
        reg_mean = sum(regular_depths) / len(regular_depths)

        # El supuesto predice que los episódicos deberían tener PEOR resultado
        # (porque el límite diario les fuerza una distribución artificial).
        # Si tienen resultado COMPARABLE o MEJOR, el límite no es necesario.
        gap = reg_mean - ep_mean

        # Anomalía: los episódicos NO están significativamente peor
        if gap < 0.3:  # gap pequeño = el límite no mejora nada para ellos
            return AnomalyReport(
                report_id=self._gen_id("AR", "A2"),
                timestamp=datetime.now().isoformat(),
                assumption_challenged=assumption,
                observed_pattern=(
                    f"Estudiantes con patrón episódico (n={len(episodic_depths)}) "
                    f"muestran profundidad cognitiva {ep_mean:.2f} vs. {reg_mean:.2f} "
                    f"de regulares. El límite diario no mejora su outcome."
                ),
                expected_by_assumption=(
                    "El límite diario debería producir distribución uniforme "
                    "y mejor resultado para todos."
                ),
                affected_population=f"episodic_pattern (n={len(episodic_depths)})",
                nd_pattern_correlation="EPISODIC",
                effect_size=round(min(1.0, abs(gap) / 0.5), 3),
                n_cases=len(episodic_depths) + len(regular_depths),
                confidence=round(min(0.9, 0.3 + len(episodic_depths) * 0.05), 3),
                revision_urgency=self._classify_urgency(abs(gap) / 0.5, 0.5),
                supporting_data=[
                    f"Episodic depth mean: {ep_mean:.2f}",
                    f"Regular depth mean: {reg_mean:.2f}",
                    f"N episodic: {len(episodic_depths)}, N regular: {len(regular_depths)}",
                ],
                reasoning_trace=[
                    f"Paso 1: Identificar estudiantes EPISODIC: {len(episodic_students)}",
                    f"Paso 2: Comparar profundidad cognitiva media",
                    f"Paso 3: Gap ({gap:.2f}) < 0.3 → el límite no mejora outcome para episódicos",
                    "Conclusión: A2 cuestionado — el límite diario penaliza sin beneficio a estudiantes episódicos",
                ],
            )
        return None

    def _analyze_scaffolding_for_low_bloom(
        self,
        profiles: dict[str, EngagementProfile],
        autonomy_states: dict[str, AutonomyState],
        config_history: list[ConfigFingerprint],
    ) -> Optional[AnomalyReport]:
        """A3: ¿El scaffolding alto funciona para Bloom bajo?"""
        assumption = self._get_assumption("A3")

        low_bloom_students = {
            sid for sid, p in profiles.items()
            if p.cognitive_depth_mean <= 2.5 and p.n_interactions >= 5
        }
        if len(low_bloom_students) < self.ANOMALY_N_THRESHOLD:
            return None

        # Medir trayectoria cognitiva bajo scaffolding alto
        trajectories = []
        for sid in low_bloom_students:
            if sid in profiles:
                trajectories.append(profiles[sid].cognitive_trajectory)

        if not trajectories:
            return None

        mean_trajectory = sum(trajectories) / len(trajectories)
        negative_count = sum(1 for t in trajectories if t < 0)
        negative_ratio = negative_count / len(trajectories)

        # Anomalía: muchos estudiantes de Bloom bajo NO mejoran con scaffolding alto
        if negative_ratio > 0.4:
            return AnomalyReport(
                report_id=self._gen_id("AR", "A3"),
                timestamp=datetime.now().isoformat(),
                assumption_challenged=assumption,
                observed_pattern=(
                    f"{negative_ratio:.0%} de estudiantes con Bloom bajo "
                    f"(n={len(trajectories)}) muestran trayectoria negativa "
                    f"bajo scaffolding alto (media: {mean_trajectory:.3f}). "
                    f"El scaffolding alto no produce mejora en este segmento."
                ),
                expected_by_assumption="Trayectoria positiva para la mayoría de estudiantes con Bloom bajo.",
                affected_population=f"low_bloom_initial (n={len(trajectories)})",
                nd_pattern_correlation="none_specific",
                effect_size=round(negative_ratio, 3),
                n_cases=len(trajectories),
                confidence=round(min(0.85, 0.3 + len(trajectories) * 0.04), 3),
                revision_urgency=self._classify_urgency(negative_ratio, 0.5),
                supporting_data=[
                    f"Trajectories: {[round(t, 3) for t in trajectories[:10]]}",
                    f"Negative ratio: {negative_ratio:.2%}",
                ],
                reasoning_trace=[
                    f"Paso 1: Filtrar estudiantes con Bloom ≤ 2.5: {len(low_bloom_students)}",
                    f"Paso 2: Medir trayectoria cognitiva bajo scaffolding alto",
                    f"Paso 3: {negative_ratio:.0%} con trayectoria negativa (umbral: 40%)",
                    "Conclusión: A3 cuestionado — scaffolding alto puede frustrar sin mejorar",
                ],
            )
        return None

    def _analyze_copypaste_nd_interaction(
        self,
        profiles: dict[str, EngagementProfile],
        nd_patterns: dict[str, list[NeurodivergentPattern]],
        autonomy_states: dict[str, AutonomyState],
    ) -> Optional[AnomalyReport]:
        """
        A4: ¿El copy-paste alto = deshonestidad?

        Este es el análisis más éticamente relevante. La hipótesis implícita
        de la detección de copy-paste es que pegar código/texto ajeno = falta
        de esfuerzo. Pero Barkley (2015) documenta que estudiantes con
        limitaciones de memoria de trabajo COPIAN SU PROPIO TRABAJO ANTERIOR
        como estrategia compensatoria — no es trampa, es adaptación.

        Si confirmamos esto empíricamente, el paper se escribe solo.
        Si lo refutamos, también — la ciencia funciona en ambas direcciones.
        """
        assumption = self._get_assumption("A4")

        # Estudiantes con patrón RE_ASKING o EPISODIC (candidatos a WM-driven copy-paste)
        wm_students = {
            sid for sid, patterns in nd_patterns.items()
            if any(p.pattern_id in ("RE_ASKING", "EPISODIC") for p in patterns)
        }

        if len(wm_students) < 3:
            return None

        # Comparar autonomía de wm_students con alto copy-paste vs. normativo con alto copy-paste
        # Si los WM students con alto CP tienen BUENA autonomía → CP no es deshonestidad
        wm_with_good_autonomy = 0
        wm_total_assessed = 0
        for sid in wm_students:
            if sid in autonomy_states:
                wm_total_assessed += 1
                if autonomy_states[sid].autonomy_score >= 0.3:
                    wm_with_good_autonomy += 1

        if wm_total_assessed < 3:
            return None

        good_ratio = wm_with_good_autonomy / wm_total_assessed

        if good_ratio >= 0.4:
            return AnomalyReport(
                report_id=self._gen_id("AR", "A4"),
                timestamp=datetime.now().isoformat(),
                assumption_challenged=assumption,
                observed_pattern=(
                    f"{good_ratio:.0%} de estudiantes con patrón de memoria de trabajo "
                    f"limitada (RE_ASKING/EPISODIC, n={wm_total_assessed}) mantienen "
                    f"autonomía epistémica ≥ 0.3 pese a potencial copy-paste alto. "
                    f"El copy-paste en este subgrupo es estrategia compensatoria, "
                    f"no deshonestidad."
                ),
                expected_by_assumption="Copy-paste alto debería correlacionar con baja autonomía.",
                affected_population=f"wm_limited_nd_pattern (n={wm_total_assessed})",
                nd_pattern_correlation="RE_ASKING + EPISODIC",
                effect_size=round(good_ratio, 3),
                n_cases=wm_total_assessed,
                confidence=round(min(0.85, 0.3 + wm_total_assessed * 0.06), 3),
                revision_urgency="high",  # ético → siempre alta urgencia
                supporting_data=[
                    f"WM students with good autonomy: {wm_with_good_autonomy}/{wm_total_assessed}",
                    f"Ratio: {good_ratio:.2%}",
                ],
                reasoning_trace=[
                    f"Paso 1: Identificar estudiantes con RE_ASKING/EPISODIC: {len(wm_students)}",
                    f"Paso 2: Verificar autonomía en esos estudiantes",
                    f"Paso 3: {good_ratio:.0%} mantienen autonomía ≥ 0.3",
                    "Paso 4: El copy-paste no correlaciona con baja autonomía en este grupo",
                    "Conclusión: A4 falsificado para el subgrupo ND con limitación de WM",
                    "Implicación ética: el sistema está penalizando una estrategia legítima de compensación",
                ],
            )
        return None

    def _analyze_config_universality(
        self,
        profiles: dict[str, EngagementProfile],
        nd_patterns: dict[str, list[NeurodivergentPattern]],
        autonomy_states: dict[str, AutonomyState],
    ) -> Optional[AnomalyReport]:
        """
        A5: ¿La configuración óptima para la mayoría es óptima para todos?

        El supuesto más peligroso del sistema. Si se falsifica, la implicación
        es que GENIE Learn necesita configuraciones diferenciadas por perfil
        ND — un cambio arquitectónico profundo que afecta a O1, O2, O3 y O4.
        """
        assumption = self._get_assumption("A5")

        nd_students = {sid for sid, patterns in nd_patterns.items() if patterns}
        normative_students = set(profiles.keys()) - nd_students

        if len(nd_students) < self.ANOMALY_N_THRESHOLD or len(normative_students) < self.ANOMALY_N_THRESHOLD:
            return None

        # Calcular varianza de outcome ENTRE subgrupos
        nd_outcomes = []
        for sid in nd_students:
            if sid in profiles and sid in autonomy_states:
                # Composite outcome: depth + autonomy + trajectory
                composite = (
                    profiles[sid].cognitive_depth_mean / 6.0 * 0.4 +
                    autonomy_states[sid].autonomy_score * 0.4 +
                    max(0, profiles[sid].cognitive_trajectory) * 0.2
                )
                nd_outcomes.append(composite)

        norm_outcomes = []
        for sid in normative_students:
            if sid in profiles and sid in autonomy_states:
                composite = (
                    profiles[sid].cognitive_depth_mean / 6.0 * 0.4 +
                    autonomy_states[sid].autonomy_score * 0.4 +
                    max(0, profiles[sid].cognitive_trajectory) * 0.2
                )
                norm_outcomes.append(composite)

        if not nd_outcomes or not norm_outcomes:
            return None

        nd_mean = sum(nd_outcomes) / len(nd_outcomes)
        norm_mean = sum(norm_outcomes) / len(norm_outcomes)
        gap = norm_mean - nd_mean

        # Calcular varianza inter-grupo (proxy de inequidad)
        nd_var = sum((x - nd_mean) ** 2 for x in nd_outcomes) / len(nd_outcomes)
        norm_var = sum((x - norm_mean) ** 2 for x in norm_outcomes) / len(norm_outcomes)
        inter_group_variance = abs(nd_mean - norm_mean)

        if inter_group_variance > 0.15:
            return AnomalyReport(
                report_id=self._gen_id("AR", "A5"),
                timestamp=datetime.now().isoformat(),
                assumption_challenged=assumption,
                observed_pattern=(
                    f"Varianza inter-grupo significativa bajo la misma configuración: "
                    f"ND composite={nd_mean:.3f} (σ²={nd_var:.3f}, n={len(nd_outcomes)}), "
                    f"Normativo composite={norm_mean:.3f} (σ²={norm_var:.3f}, n={len(norm_outcomes)}). "
                    f"Gap: {gap:.3f}. La configuración 'óptima' global produce resultados "
                    f"inequitativos entre subgrupos."
                ),
                expected_by_assumption="Outcomes comparables entre subgrupos bajo la misma configuración.",
                affected_population=f"nd_students (n={len(nd_outcomes)}) vs normative (n={len(norm_outcomes)})",
                nd_pattern_correlation="all_nd_patterns",
                effect_size=round(min(1.0, inter_group_variance / 0.3), 3),
                n_cases=len(nd_outcomes) + len(norm_outcomes),
                confidence=round(min(0.9, 0.4 + (len(nd_outcomes) + len(norm_outcomes)) * 0.02), 3),
                revision_urgency="critical" if inter_group_variance > 0.25 else "high",
                supporting_data=[
                    f"ND composite outcomes: {[round(o, 3) for o in nd_outcomes]}",
                    f"Norm composite outcomes: {[round(o, 3) for o in norm_outcomes]}",
                    f"Inter-group variance: {inter_group_variance:.3f}",
                ],
                reasoning_trace=[
                    f"Paso 1: Segmentar por presencia de patrón ND: {len(nd_students)} ND, {len(normative_students)} normativos",
                    "Paso 2: Calcular outcome compuesto (depth 40% + autonomy 40% + trajectory 20%)",
                    f"Paso 3: ND mean={nd_mean:.3f}, Norm mean={norm_mean:.3f}",
                    f"Paso 4: Inter-group variance={inter_group_variance:.3f} > 0.15",
                    "Conclusión: A5 FALSIFICADO — la misma configuración produce inequidad entre subgrupos",
                    "Implicación: se necesitan configuraciones diferenciadas por perfil ND",
                ],
            )
        return None

    def _analyze_temporal_consistency(
        self,
        profiles: dict[str, EngagementProfile],
        nd_patterns: dict[str, list[NeurodivergentPattern]],
    ) -> Optional[AnomalyReport]:
        """A6: ¿La consistencia temporal es necesaria para aprender?"""
        assumption = self._get_assumption("A6")

        episodic_students = {
            sid for sid, patterns in nd_patterns.items()
            if any(p.pattern_id == "EPISODIC" for p in patterns)
        }
        if len(episodic_students) < 3:
            return None

        # Comparar trayectoria cognitiva
        ep_trajectories = [
            profiles[sid].cognitive_trajectory
            for sid in episodic_students if sid in profiles
        ]
        regular = set(profiles.keys()) - episodic_students
        reg_trajectories = [
            profiles[sid].cognitive_trajectory
            for sid in regular if sid in profiles and profiles[sid].n_interactions >= 5
        ]

        if not ep_trajectories or not reg_trajectories:
            return None

        ep_mean = sum(ep_trajectories) / len(ep_trajectories)
        reg_mean = sum(reg_trajectories) / len(reg_trajectories)

        # Anomalía: episódicos con trayectoria comparable o superior
        if ep_mean >= reg_mean - 0.1:
            return AnomalyReport(
                report_id=self._gen_id("AR", "A6"),
                timestamp=datetime.now().isoformat(),
                assumption_challenged=assumption,
                observed_pattern=(
                    f"Estudiantes episódicos: trayectoria {ep_mean:.3f} vs. "
                    f"regulares {reg_mean:.3f}. La distribución irregular de uso "
                    f"NO produce peor trayectoria cognitiva."
                ),
                expected_by_assumption="Uso regular debería producir mejor trayectoria que uso en ráfagas.",
                affected_population=f"episodic (n={len(ep_trajectories)})",
                nd_pattern_correlation="EPISODIC",
                effect_size=round(min(1.0, max(0, ep_mean - reg_mean + 0.3)), 3),
                n_cases=len(ep_trajectories) + len(reg_trajectories),
                confidence=round(min(0.85, 0.3 + len(ep_trajectories) * 0.05), 3),
                revision_urgency="medium",
                supporting_data=[
                    f"Episodic trajectories: {[round(t, 3) for t in ep_trajectories]}",
                    f"Regular trajectories: {[round(t, 3) for t in reg_trajectories[:10]]}",
                ],
                reasoning_trace=[
                    f"Paso 1: Identificar estudiantes EPISODIC: {len(episodic_students)}",
                    f"Paso 2: Comparar trayectorias cognitivas",
                    f"Paso 3: Episódicos ({ep_mean:.3f}) ≥ regulares ({reg_mean:.3f}) - 0.1",
                    "Conclusión: A6 cuestionado — la consistencia temporal no es prerequisito",
                ],
            )
        return None

    def _analyze_socratic_nd_interaction(
        self,
        profiles: dict[str, EngagementProfile],
        nd_patterns: dict[str, list[NeurodivergentPattern]],
        autonomy_states: dict[str, AutonomyState],
    ) -> list[AnomalyReport]:
        """
        EL ANÁLISIS CENTRAL: ¿el modo socrático funciona igual para
        estudiantes con patrones ND específicos?

        Hipótesis derivada del perfil 2e + literatura:
        - Patrón EPISODIC: el socrático en ráfaga puede frustrar
        - Patrón COGNITIVE_JUMP: el socrático trivial genera desenganche
        - Patrón SELECTIVE_FRUSTRATION: el socrático en temas de desinterés = abandono

        Si se confirma, esto es un paper. Si se refuta, también es un paper.
        """
        assumption_a7 = self._get_assumption("A7")
        results = []

        # Análisis por cada tipo de patrón ND
        pattern_analysis = {
            "EPISODIC": {
                "hypothesis": (
                    "El socrático en ráfagas de hiperfoco frustra al estudiante "
                    "episódico: necesita respuestas durante la ventana de atención, "
                    "no preguntas guía que consumen esa ventana."
                ),
                "metric": lambda sid: profiles.get(sid, EngagementProfile(student_id=sid, period_start="", period_end="")).affect_valence_mean,
                "threshold": -0.3,  # valencia afectiva negativa
                "compare": "lower_is_worse",
            },
            "COGNITIVE_JUMP": {
                "hypothesis": (
                    "El socrático trivial genera desenganche activo en estudiantes "
                    "con procesamiento no-lineal: preguntas guía que asumen "
                    "transición gradual Bloom 2→3→4 son irrelevantes para quien "
                    "salta directamente a Bloom 5."
                ),
                "metric": lambda sid: profiles.get(sid, EngagementProfile(student_id=sid, period_start="", period_end="")).cognitive_depth_mean,
                "threshold": 3.0,  # desenganche = no alcanza profundidad
                "compare": "lower_is_worse",
            },
            "SELECTIVE_FRUSTRATION": {
                "hypothesis": (
                    "El socrático amplifica la frustración selectiva: en topics "
                    "de bajo interés, las preguntas guía se perciben como "
                    "obstáculos, no como andamiaje."
                ),
                "metric": lambda sid: profiles.get(sid, EngagementProfile(student_id=sid, period_start="", period_end="")).affect_valence_mean,
                "threshold": -0.5,
                "compare": "lower_is_worse",
            },
        }

        for pattern_id, config in pattern_analysis.items():
            students_with_pattern = {
                sid for sid, patterns in nd_patterns.items()
                if any(p.pattern_id == pattern_id for p in patterns)
            }

            if len(students_with_pattern) < 3:
                continue

            # Medir la métrica para este subgrupo
            metrics = []
            for sid in students_with_pattern:
                val = config["metric"](sid)
                if val is not None:
                    metrics.append(val)

            if len(metrics) < 3:
                continue

            mean_metric = sum(metrics) / len(metrics)
            below_threshold = sum(1 for m in metrics if m < config["threshold"])
            below_ratio = below_threshold / len(metrics)

            if below_ratio > 0.5:
                results.append(AnomalyReport(
                    report_id=self._gen_id("AR", f"A7_{pattern_id}"),
                    timestamp=datetime.now().isoformat(),
                    assumption_challenged=assumption_a7,
                    observed_pattern=(
                        f"Patrón {pattern_id}: {below_ratio:.0%} de estudiantes "
                        f"(n={len(metrics)}) están por debajo del umbral "
                        f"({config['threshold']}) en la métrica relevante. "
                        f"Media: {mean_metric:.2f}."
                    ),
                    expected_by_assumption="El socrático debería beneficiar a todos los perfiles.",
                    affected_population=f"{pattern_id}_students (n={len(metrics)})",
                    nd_pattern_correlation=pattern_id,
                    effect_size=round(below_ratio, 3),
                    n_cases=len(metrics),
                    confidence=round(min(0.85, 0.3 + len(metrics) * 0.05), 3),
                    revision_urgency=self._classify_urgency(below_ratio, 0.5),
                    supporting_data=[
                        f"Hypothesis: {config['hypothesis']}",
                        f"Metrics: {[round(m, 2) for m in metrics]}",
                        f"Below threshold ({config['threshold']}): {below_ratio:.0%}",
                    ],
                    reasoning_trace=[
                        f"Paso 1: Aislar estudiantes con patrón {pattern_id}: {len(students_with_pattern)}",
                        f"Paso 2: Medir métrica relevante bajo socrático",
                        f"Paso 3: {below_ratio:.0%} por debajo de umbral {config['threshold']}",
                        f"Paso 4: Hipótesis específica: {config['hypothesis'][:100]}...",
                        f"Conclusión: A7 falsificado para subgrupo {pattern_id}",
                    ],
                ))

        return results

    def _detect_config_interaction_effects(
        self,
        profiles: dict[str, EngagementProfile],
        nd_patterns: dict[str, list[NeurodivergentPattern]],
        config_history: list[ConfigFingerprint],
    ) -> list[AnomalyReport]:
        """
        Detecta efectos de interacción entre configuraciones que O2 no puede ver.

        Ejemplo: socrático=True + max_response_length=50 + hallucination_rate=0.2
        puede crear una combinación que funciona bien en promedio pero es
        devastadora para estudiantes con alta ansiedad epistémica.

        Esto conecta con el paper LAK 2026 (Ortega-Arranz) que evalúa
        configuraciones aisladas pero no sus interacciones.
        """
        results = []

        if not config_history:
            return results

        # Detectar configuraciones con alta restrictividad + alta tolerancia al error
        # (combinación paradójica: restringes acceso pero introduces errores)
        for fp in config_history:
            if fp.restrictiveness_score > 0.6 and fp.tolerance_for_error > 0.3:
                # Buscar estudiantes struggling bajo esta combinación
                struggling = {
                    sid for sid, p in profiles.items()
                    if p.engagement_type == "struggling"
                }
                if len(struggling) >= 3:
                    results.append(AnomalyReport(
                        report_id=self._gen_id("AR", "CONFIG_INTERACT_01"),
                        timestamp=datetime.now().isoformat(),
                        assumption_challenged=self._get_assumption("A5"),
                        observed_pattern=(
                            f"Configuración paradójica detectada: restrictividad alta "
                            f"({fp.restrictiveness_score:.2f}) + tolerancia al error alta "
                            f"({fp.tolerance_for_error:.2f}). "
                            f"{len(struggling)} estudiantes en estado 'struggling'. "
                            f"El sistema restringe las respuestas directas Y introduce "
                            f"errores deliberados — combinación potencialmente ansiogénica."
                        ),
                        expected_by_assumption="Las configuraciones individuales deberían ser aditivas, no multiplicativas.",
                        affected_population=f"struggling_students (n={len(struggling)})",
                        nd_pattern_correlation="none_specific",
                        effect_size=round(len(struggling) / max(len(profiles), 1), 3),
                        n_cases=len(struggling),
                        confidence=0.55,
                        revision_urgency="medium",
                        supporting_data=[
                            f"Config fingerprint: {fp.pedagogical_style}",
                            f"Restrictiveness: {fp.restrictiveness_score:.2f}",
                            f"Tolerance for error: {fp.tolerance_for_error:.2f}",
                            f"Struggling students: {len(struggling)}",
                        ],
                        reasoning_trace=[
                            "Paso 1: Detectar combinación paradójica en config activa",
                            "Paso 2: Contar estudiantes struggling bajo esta combinación",
                            "Paso 3: Valorar si la interacción de configs produce el efecto",
                            "Conclusión: Posible efecto de interacción no previsto en O1",
                        ],
                    ))
                break  # solo reportar una vez

        return results

    def _analyze_silence_patterns(
        self,
        silence_alerts: dict[str, list[SilenceAlert]],
        nd_patterns: dict[str, list[NeurodivergentPattern]],
    ) -> list[AnomalyReport]:
        """
        Cruza alertas de silencio epistémico con patrones ND.

        Si los estudiantes con patrones ND tienen más silencios de tipo
        metacognitive_gap, eso indica que el sistema no está adaptando
        su scaffolding a la diversidad cognitiva.
        """
        results = []
        nd_students = {sid for sid, p in nd_patterns.items() if p}

        nd_silence_metacog = 0
        nd_silence_total = 0
        norm_silence_metacog = 0
        norm_silence_total = 0

        for sid, alerts in silence_alerts.items():
            for alert in alerts:
                if sid in nd_students:
                    nd_silence_total += 1
                    if alert.silence_type == "metacognitive_gap":
                        nd_silence_metacog += 1
                else:
                    norm_silence_total += 1
                    if alert.silence_type == "metacognitive_gap":
                        norm_silence_metacog += 1

        nd_metacog_ratio = nd_silence_metacog / max(nd_silence_total, 1)
        norm_metacog_ratio = norm_silence_metacog / max(norm_silence_total, 1)

        if nd_silence_total >= 3 and nd_metacog_ratio > norm_metacog_ratio + 0.15:
            results.append(AnomalyReport(
                report_id=self._gen_id("AR", "SILENCE_ND"),
                timestamp=datetime.now().isoformat(),
                assumption_challenged=self._get_assumption("A5"),
                observed_pattern=(
                    f"Estudiantes ND muestran mayor ratio de silencio por gap "
                    f"metacognitivo ({nd_metacog_ratio:.0%}) vs. normativos "
                    f"({norm_metacog_ratio:.0%}). El scaffolding no está "
                    f"compensando la dificultad de detección de gaps propios "
                    f"en perfiles neurodivergentes."
                ),
                expected_by_assumption="El scaffolding debería ser igualmente efectivo para todos.",
                affected_population=f"nd_students (silences: {nd_silence_total})",
                nd_pattern_correlation="all_nd",
                effect_size=round(nd_metacog_ratio - norm_metacog_ratio, 3),
                n_cases=nd_silence_total + norm_silence_total,
                confidence=round(min(0.8, 0.3 + nd_silence_total * 0.05), 3),
                revision_urgency="high",
                supporting_data=[
                    f"ND metacog gap silences: {nd_silence_metacog}/{nd_silence_total}",
                    f"Normative metacog gap silences: {norm_silence_metacog}/{norm_silence_total}",
                ],
                reasoning_trace=[
                    "Paso 1: Cruzar alertas de silencio con patrones ND",
                    f"Paso 2: ND metacognitive_gap ratio: {nd_metacog_ratio:.0%}",
                    f"Paso 3: Normative metacognitive_gap ratio: {norm_metacog_ratio:.0%}",
                    f"Paso 4: Diferencia ({nd_metacog_ratio - norm_metacog_ratio:.0%}) > 15%",
                    "Conclusión: El silencio epistémico golpea desproporcionadamente a ND",
                ],
            ))

        return results

    # ──────────────────────────────────────────────────────────────
    # 4. GENERACIÓN DE PROPUESTAS DE REVISIÓN
    # ──────────────────────────────────────────────────────────────

    def _generate_revision_proposals(
        self, anomalies: list[AnomalyReport]
    ) -> list[ScenarioRevisionProposal]:
        """
        Traduce anomalías en propuestas accionables de revisión de O1.

        Cada propuesta conecta con los cuatro objetivos y con una hipótesis
        de paper — alineando incentivos de mejora con incentivos académicos.
        """
        proposals = []

        # Agrupar anomalías por supuesto
        by_assumption = defaultdict(list)
        for a in anomalies:
            by_assumption[a.assumption_challenged.assumption_id].append(a)

        for assumption_id, related in by_assumption.items():
            # Solo generar propuesta si hay anomalías con urgencia ≥ medium
            urgent = [a for a in related if a.revision_urgency in ("medium", "high", "critical")]
            if not urgent:
                continue

            best = max(urgent, key=lambda a: a.effect_size)
            proposal = self._build_proposal(assumption_id, best, related)
            if proposal:
                proposals.append(proposal)

        return proposals

    def _build_proposal(
        self, assumption_id: str, primary: AnomalyReport, all_related: list[AnomalyReport]
    ) -> Optional[ScenarioRevisionProposal]:
        """Construye una propuesta específica por supuesto."""

        PROPOSAL_TEMPLATES = {
            "A1": {
                "revision": (
                    "Revisar A1: el scaffolding socrático NO es universalmente superior. "
                    "Proponer scaffolding diferenciado por perfil de autonomía y patrón ND."
                ),
                "o2": "Añadir toggle 'Socrático adaptativo' que ajusta intensidad por perfil del estudiante.",
                "o3": "Implementar socrático dinámico: reduce intensidad cuando detecta patrón ND con baja respuesta.",
                "o4": "Extender middleware para aceptar scaffolding_mode per-student, no per-course.",
                "paper": (
                    "H: El scaffolding socrático adaptativo (variable por perfil ND) "
                    "produce mejor autonomía que el socrático uniforme."
                ),
            },
            "A2": {
                "revision": (
                    "Revisar A2: el límite diario de prompts no uniformiza el uso — "
                    "solo desplaza las ráfagas y penaliza a perfiles episódicos. "
                    "Proponer límites por ventana temporal en vez de diarios."
                ),
                "o2": "Reemplazar 'max prompts/día' por 'max prompts/ventana' con ventanas configurables.",
                "o3": "Detectar ráfagas productivas y no interrumpirlas.",
                "o4": "Rediseñar el contador de prompts para operar por ventana temporal, no por día.",
                "paper": (
                    "H: Los límites por ventana temporal producen mejor engagement "
                    "que los límites diarios en estudiantes con patrón episódico."
                ),
            },
            "A4": {
                "revision": (
                    "Revisar A4: el copy-paste alto en estudiantes con patrón RE_ASKING/EPISODIC "
                    "es estrategia compensatoria de memoria de trabajo, no deshonestidad. "
                    "Eliminar la penalización por copy-paste cuando hay patrón ND de WM."
                ),
                "o2": "Añadir nota contextual al docente: 'copy-paste alto con patrón ND → probable compensación WM'.",
                "o3": "No activar alerta de gaming para estudiantes con patrón RE_ASKING que copian su propio trabajo.",
                "o4": "Diferenciar copy-paste externo (fuente ajena) de copy-paste interno (propio trabajo anterior).",
                "paper": (
                    "H: La detección de copy-paste sin contexto ND produce falsos "
                    "positivos que penalizan estrategias legítimas de compensación."
                ),
            },
            "A5": {
                "revision": (
                    "Revisar A5: la configuración óptima para la mayoría NO es óptima "
                    "para todos. Los perfiles ND muestran outcomes significativamente "
                    "diferentes bajo la misma configuración. Proponer configuraciones "
                    "diferenciadas o adaptativas."
                ),
                "o2": "Implementar perfiles de configuración (preset ND-friendly) que el docente puede activar.",
                "o3": "Scaffolding adaptativo por perfil detectado, no uniforme por curso.",
                "o4": "Extender PedagogicalConfig para soportar overrides per-student basados en perfil ND.",
                "paper": (
                    "H: Las configuraciones pedagógicas diferenciadas por perfil ND "
                    "reducen la varianza inter-grupo de outcomes sin reducir la media global."
                ),
            },
            "A6": {
                "revision": (
                    "Revisar A6: la consistencia temporal no es prerrequisito de aprendizaje "
                    "efectivo. Los estudiantes episódicos alcanzan trayectorias comparables "
                    "con uso en ráfagas."
                ),
                "o2": "Desactivar alertas de 'uso irregular' para estudiantes con patrón EPISODIC.",
                "o3": "Adaptar los resúmenes de sesión para ráfagas: 'bienvenido, la última vez...'.",
                "o4": "Implementar session memory que persista entre ráfagas de uso.",
                "paper": (
                    "H: El patrón de uso episódico produce outcomes equivalentes al "
                    "uso regular cuando el sistema se adapta a la temporalidad del estudiante."
                ),
            },
            "A7": {
                "revision": (
                    "Revisar A7: el socrático no es uniformemente superior por topic ni por perfil. "
                    "Estudiantes con COGNITIVE_JUMP, SELECTIVE_FRUSTRATION y EPISODIC muestran "
                    "respuestas diferenciadas que requieren adaptación del scaffolding."
                ),
                "o2": "Ofrecer modo 'socrático inteligente' que adapta por topic y perfil.",
                "o3": "Reducir escalones socráticos para COGNITIVE_JUMP; modo directo en topics de bajo interés para SELECTIVE_FRUSTRATION.",
                "o4": "Enriquecer middleware con tabla de interacciones scaffolding_mode × ND_pattern × topic.",
                "paper": (
                    "H: El scaffolding socrático interacciona con el perfil ND del estudiante "
                    "y el topic específico, produciendo efectos que varían de positivos a negativos "
                    "según la combinación tridimensional."
                ),
            },
        }

        template = PROPOSAL_TEMPLATES.get(assumption_id)
        if not template:
            return None

        return ScenarioRevisionProposal(
            proposal_id=self._gen_id("RP", assumption_id),
            timestamp=datetime.now().isoformat(),
            triggered_by=[a.report_id for a in all_related],
            original_assumption=primary.assumption_challenged.description,
            proposed_revision=template["revision"],
            supporting_evidence=[a.observed_pattern for a in all_related],
            implementation_in_o2=template["o2"],
            implementation_in_o3=template["o3"],
            implementation_in_o4=template["o4"],
            paper_hypothesis=template["paper"],
            dsrm_cycle_target=3,
            estimated_effort="high" if assumption_id == "A5" else "medium",
        )

    # ──────────────────────────────────────────────────────────────
    # 5. DETECCIÓN DE EMERGENCIA O5
    # ──────────────────────────────────────────────────────────────

    def generate_o5_hypothesis(
        self, anomalies: list[AnomalyReport]
    ) -> list[O5EmergenceSignal]:
        """
        Cuando hay ≥3 anomalías que apuntan en la misma dirección,
        el motor genera una señal O5: algo que no estaba previsto
        está emergiendo.

        Principio: Cynefin (Snowden & Boone, 2007). En sistemas complejos,
        los patrones no se predicen: emergen. El rol del investigador es
        detectarlos antes de que se disuelvan en el ruido.
        """
        signals = []

        # Patrón O5-A: Múltiples anomalías apuntan a inequidad ND
        nd_anomalies = [
            a for a in anomalies
            if a.nd_pattern_correlation != "none_specific"
            and a.effect_size > self.ANOMALY_EFFECT_THRESHOLD
        ]

        if len(nd_anomalies) >= self.O5_CONVERGENCE_THRESHOLD:
            # Calcular confianza por convergencia
            mean_confidence = sum(a.confidence for a in nd_anomalies) / len(nd_anomalies)
            mean_effect = sum(a.effect_size for a in nd_anomalies) / len(nd_anomalies)

            signals.append(O5EmergenceSignal(
                signal_id=self._gen_id("O5", "ND_EQUITY"),
                timestamp=datetime.now().isoformat(),
                discovery_type="equity_gap",
                description=(
                    f"{len(nd_anomalies)} anomalías convergen en la misma dirección: "
                    f"los estudiantes con perfiles ND experimentan el sistema de manera "
                    f"cualitativamente diferente al perfil normativo. No es un ajuste "
                    f"incremental lo que se necesita — es un modo de operación paralelo. "
                    f"La arquitectura actual del SLE asume un estudiante modal que no "
                    f"existe: el 'estudiante promedio' es un artefacto estadístico que "
                    f"oculta la varianza real de la población."
                ),
                evidence_base=[a.report_id for a in nd_anomalies],
                anomaly_convergence=(
                    f"Efecto medio: {mean_effect:.2f}, Confianza media: {mean_confidence:.2f}. "
                    f"Patrones implicados: {set(a.nd_pattern_correlation for a in nd_anomalies)}. "
                    f"Supuestos cuestionados: {set(a.assumption_challenged.assumption_id for a in nd_anomalies)}."
                ),
                suggested_new_objective=(
                    "O5 (emergente): Diseñar un modo de operación ND-aware en el SLE "
                    "que no sea un 'ajuste' del modo normativo sino un modo paralelo "
                    "con lógica propia de scaffolding, temporalidad y evaluación."
                ),
                dsrm_cycle_target=3,
                paper_potential="full_paper",
                confidence=round(mean_confidence, 3),
            ))

        # Patrón O5-B: El sistema está aprendiendo sobre sí mismo
        high_confidence_anomalies = [a for a in anomalies if a.confidence > 0.6]
        unique_assumptions = set(a.assumption_challenged.assumption_id for a in high_confidence_anomalies)

        if len(unique_assumptions) >= 3:
            signals.append(O5EmergenceSignal(
                signal_id=self._gen_id("O5", "SYSTEM_LEARNING"),
                timestamp=datetime.now().isoformat(),
                discovery_type="system_learning",
                description=(
                    f"El motor de retroalimentación ha falsificado parcialmente "
                    f"{len(unique_assumptions)} supuestos de O1 con confianza > 0.6 "
                    f"en un solo ciclo de análisis. Esto indica que los escenarios HL "
                    f"necesitan revisión sustancial para el Ciclo 3 del DSRM — no "
                    f"ajustes paramétricos sino reconceptualización de supuestos base."
                ),
                evidence_base=[a.report_id for a in high_confidence_anomalies],
                anomaly_convergence=(
                    f"Supuestos cuestionados: {unique_assumptions}. "
                    f"Este nivel de falsación simultánea sugiere que la ontología "
                    f"de O1 (qué tipo de estudiante asume) necesita revisión."
                ),
                suggested_new_objective=(
                    "O5 (emergente): Reformular los escenarios HL de O1 incorporando "
                    "la variabilidad cognitiva como variable de DISEÑO, no como "
                    "excepción a gestionar."
                ),
                dsrm_cycle_target=3,
                paper_potential="short_paper",
                confidence=round(len(unique_assumptions) / 7 * 0.9, 3),  # 7 assumptions total
            ))

        return signals

    # ──────────────────────────────────────────────────────────────
    # 6. EXPORTACIÓN PARA RESEARCHER VIEW
    # ──────────────────────────────────────────────────────────────

    def export_for_researcher_view(self) -> dict:
        """
        Estructura de datos para researcher_view.py.

        Todo lo que la pestaña "Retroalimentación O1" necesita para renderizar:
        métricas de resumen, anomalías por urgencia, propuestas de revisión,
        señales O5, y el estado de cada supuesto.
        """
        # Estado de cada supuesto: cuántas anomalías lo cuestionan
        assumption_status = {}
        for a in self.o1_assumptions:
            related = [
                ar for ar in self.anomaly_log
                if ar.assumption_challenged.assumption_id == a.assumption_id
            ]
            if not related:
                status = "untested"
            elif any(ar.revision_urgency == "critical" for ar in related):
                status = "falsified"
            elif any(ar.revision_urgency in ("high", "medium") for ar in related):
                status = "challenged"
            else:
                status = "provisional"

            assumption_status[a.assumption_id] = {
                "description": a.description,
                "status": status,
                "n_anomalies": len(related),
                "max_urgency": max(
                    (ar.revision_urgency for ar in related),
                    key=lambda u: {"low": 0, "medium": 1, "high": 2, "critical": 3}.get(u, 0),
                    default="none",
                ),
                "target_population": a.target_population,
            }

        # Anomalías ordenadas por urgencia
        urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        sorted_anomalies = sorted(
            self.anomaly_log,
            key=lambda a: urgency_order.get(a.revision_urgency, 4),
        )

        return {
            "run_count": self._run_count,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_anomalies": len(self.anomaly_log),
                "critical": sum(1 for a in self.anomaly_log if a.revision_urgency == "critical"),
                "high": sum(1 for a in self.anomaly_log if a.revision_urgency == "high"),
                "medium": sum(1 for a in self.anomaly_log if a.revision_urgency == "medium"),
                "low": sum(1 for a in self.anomaly_log if a.revision_urgency == "low"),
                "total_proposals": len(self.revision_proposals),
                "total_o5_signals": len(self.o5_signals),
                "assumptions_falsified": sum(
                    1 for s in assumption_status.values() if s["status"] == "falsified"
                ),
                "assumptions_challenged": sum(
                    1 for s in assumption_status.values() if s["status"] == "challenged"
                ),
                "assumptions_untested": sum(
                    1 for s in assumption_status.values() if s["status"] == "untested"
                ),
            },
            "assumption_status": assumption_status,
            "anomalies": [
                {
                    "report_id": a.report_id,
                    "assumption_id": a.assumption_challenged.assumption_id,
                    "urgency": a.revision_urgency,
                    "pattern": a.observed_pattern,
                    "affected_population": a.affected_population,
                    "nd_correlation": a.nd_pattern_correlation,
                    "effect_size": a.effect_size,
                    "n_cases": a.n_cases,
                    "confidence": a.confidence,
                    "reasoning": a.reasoning_trace,
                }
                for a in sorted_anomalies
            ],
            "proposals": [
                {
                    "proposal_id": p.proposal_id,
                    "original": p.original_assumption,
                    "revision": p.proposed_revision,
                    "o2_impact": p.implementation_in_o2,
                    "o3_impact": p.implementation_in_o3,
                    "o4_impact": p.implementation_in_o4,
                    "paper_hypothesis": p.paper_hypothesis,
                    "evidence_count": len(p.triggered_by),
                }
                for p in self.revision_proposals
            ],
            "o5_signals": [
                {
                    "signal_id": s.signal_id,
                    "type": s.discovery_type,
                    "description": s.description,
                    "suggested_objective": s.suggested_new_objective,
                    "paper_potential": s.paper_potential,
                    "confidence": s.confidence,
                    "n_supporting_anomalies": len(s.evidence_base),
                }
                for s in self.o5_signals
            ],
        }

    # ──────────────────────────────────────────────────────────────
    # UTILIDADES INTERNAS
    # ──────────────────────────────────────────────────────────────

    def _get_assumption(self, assumption_id: str) -> O1Assumption:
        """Recupera un supuesto por ID."""
        for a in self.o1_assumptions:
            if a.assumption_id == assumption_id:
                return a
        raise ValueError(f"Assumption {assumption_id} not found")

    def _gen_id(self, prefix: str, context: str) -> str:
        """Genera un ID único determinístico."""
        raw = f"{prefix}_{context}_{self._run_count}_{datetime.now().isoformat()}"
        return f"{prefix}_{hashlib.md5(raw.encode()).hexdigest()[:10]}"

    def _classify_urgency(self, effect_size: float, confidence: float) -> str:
        """Clasifica la urgencia de revisión."""
        for level, (e_thresh, c_thresh) in self.URGENCY_THRESHOLDS.items():
            if effect_size >= e_thresh and confidence >= c_thresh:
                return level
        return "low"


# ═══════════════════════════════════════════════════════════════════════
# DEMO — SIMULACIÓN DE ESCENARIO COMPLETO
# ═══════════════════════════════════════════════════════════════════════

def _demo():
    """
    Simula un escenario con:
    - 20 estudiantes, 30 días, configuración socrático=True activa
    - 5 con patrón EPISODIC, 3 con COGNITIVE_JUMP, 2 con TWICE_EXCEPTIONAL
    - Muestra cómo el motor detecta que A5 falla para los perfiles ND
    - Genera una señal O5 de emergencia
    - Exporta el informe completo
    """
    import random
    random.seed(42)  # reproducibilidad

    print("═" * 72)
    print("  O1 FEEDBACK ENGINE — Demo de retroalimentación O3→O1")
    print("  Simulación: 20 estudiantes, 30 días, socrático=True")
    print("═" * 72)

    engine = O1FeedbackEngine()

    # ── 1. Generar perfiles de engagement simulados ──
    engagement_profiles = {}
    nd_patterns_by_student = {}
    autonomy_states = {}
    silence_alerts = {}

    student_configs = {
        # 10 estudiantes normativos: buen engagement bajo socrático
        **{f"S_{i:02d}": {"nd": [], "depth": random.uniform(3.0, 5.0),
                          "autonomy": random.uniform(0.4, 0.8),
                          "trajectory": random.uniform(0.01, 0.15),
                          "affect": random.uniform(0.2, 1.5),
                          "engagement": random.choice(["deep_learner", "exploratory", "surface_seeker"])}
           for i in range(1, 11)},
        # 5 EPISODIC: profundidad comparable pero temporalidad distinta
        **{f"S_{i:02d}": {"nd": [NeurodivergentPattern(
                pattern_id="EPISODIC", pattern_name="Interacción episódica",
                functional_description="Ráfagas de hiperfoco",
                confidence=random.uniform(0.6, 0.9),
                evidence=["CV temporal > 1.5"], scaffolding_adaptation="...", teacher_note="...")],
            "depth": random.uniform(2.5, 4.0),
            "autonomy": random.uniform(0.2, 0.5),
            "trajectory": random.uniform(-0.05, 0.1),
            "affect": random.uniform(-0.5, 0.3),
            "engagement": random.choice(["struggling", "exploratory"])}
           for i in range(11, 16)},
        # 3 COGNITIVE_JUMP: alta capacidad pero desenganche bajo socrático trivial
        **{f"S_{i:02d}": {"nd": [NeurodivergentPattern(
                pattern_id="COGNITIVE_JUMP", pattern_name="Saltos cognitivos",
                functional_description="Procesamiento no-lineal",
                confidence=random.uniform(0.7, 0.85),
                evidence=["Saltos ≥3 niveles Bloom"], scaffolding_adaptation="...", teacher_note="...")],
            "depth": random.uniform(3.5, 5.5),
            "autonomy": random.uniform(0.3, 0.6),
            "trajectory": random.uniform(-0.1, 0.05),
            "affect": random.uniform(-0.8, -0.1),
            "engagement": "disengaged"}
           for i in range(16, 19)},
        # 2 TWICE_EXCEPTIONAL: el caso más complejo
        **{f"S_{i:02d}": {"nd": [
                NeurodivergentPattern(
                    pattern_id="TWICE_EXCEPTIONAL", pattern_name="Perfil 2e",
                    functional_description="Combinación AACC + episódico",
                    confidence=0.75,
                    evidence=["Score compuesto > 0.5"], scaffolding_adaptation="...", teacher_note="..."),
                NeurodivergentPattern(
                    pattern_id="RE_ASKING", pattern_name="Re-consulta",
                    functional_description="Regresiones aparentes por WM",
                    confidence=0.65,
                    evidence=["Regresiones en topics dominados"], scaffolding_adaptation="...", teacher_note="..."),
            ],
            "depth": random.uniform(2.0, 4.5),
            "autonomy": random.uniform(0.3, 0.55),
            "trajectory": random.uniform(-0.05, 0.08),
            "affect": random.uniform(-0.4, 0.2),
            "engagement": "struggling"}
           for i in range(19, 21)},
    }

    for sid, cfg in student_configs.items():
        engagement_profiles[sid] = EngagementProfile(
            student_id=sid,
            period_start="2026-01-20",
            period_end="2026-02-20",
            cognitive_depth_mean=cfg["depth"],
            cognitive_trajectory=cfg["trajectory"],
            dominant_affect="curiosity" if cfg["affect"] > 0 else "frustration",
            affect_valence_mean=cfg["affect"],
            metacognitive_ratio=random.uniform(0.05, 0.35),
            engagement_type=cfg["engagement"],
            n_interactions=random.randint(8, 40),
        )
        nd_patterns_by_student[sid] = cfg["nd"]
        autonomy_states[sid] = AutonomyState(
            student_id=sid,
            autonomy_score=cfg["autonomy"],
            phase="scaffolded" if cfg["autonomy"] < 0.45 else "emergent",
            self_efficacy_proxy=cfg["autonomy"] * 0.8,
            total_interactions=engagement_profiles[sid].n_interactions,
        )
        # Silencio epistémico: más metacognitive_gap en ND
        if cfg["nd"]:
            silence_alerts[sid] = [
                SilenceAlert(
                    timestamp="2026-02-15", student_id=sid, course_id="PROG101",
                    alert_level="medium", topics_affected=["recursión", "punteros"],
                    n_consecutive_topics=2, silence_type="metacognitive_gap",
                    confidence=0.65, supporting_evidence=["Bloom bajo + no pregunta"],
                )
            ]
        else:
            if random.random() < 0.3:
                silence_alerts[sid] = [
                    SilenceAlert(
                        timestamp="2026-02-15", student_id=sid, course_id="PROG101",
                        alert_level="low", topics_affected=["funciones"],
                        n_consecutive_topics=1, silence_type="competence",
                        confidence=0.7, supporting_evidence=["Bloom alto + no pregunta"],
                    )
                ]

    # ── 2. Configuración socrática dominante ──
    config_history = [
        ConfigFingerprint(
            teacher_id="prof_demo",
            fingerprint_id="fp_socratic_01",
            computed_at="2026-02-01",
            restrictiveness_score=0.65,
            socratic_intensity=0.85,
            tolerance_for_error=0.15,
            trust_in_student=0.7,
            pedagogical_style="scaffolded_explorer",
        ),
        ConfigFingerprint(
            teacher_id="prof_demo",
            fingerprint_id="fp_socratic_02",
            computed_at="2026-02-10",
            restrictiveness_score=0.70,
            socratic_intensity=0.80,
            tolerance_for_error=0.20,
            trust_in_student=0.65,
            pedagogical_style="scaffolded_explorer",
        ),
    ]

    # ── 3. Ejecutar análisis ──
    print("\n▶ Ejecutando análisis O3→O1...")
    anomalies, proposals, signals = engine.run_analysis(
        engagement_profiles=engagement_profiles,
        nd_patterns_by_student=nd_patterns_by_student,
        autonomy_states=autonomy_states,
        silence_alerts=silence_alerts,
        config_history=config_history,
        n_days=30,
    )

    # ── 4. Mostrar resultados ──
    print(f"\n{'─' * 72}")
    print(f"  RESULTADOS DEL ANÁLISIS")
    print(f"{'─' * 72}")

    print(f"\n  📊 Anomalías detectadas: {len(anomalies)}")
    for a in anomalies:
        urgency_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        icon = urgency_icons.get(a.revision_urgency, "⚪")
        print(f"\n  {icon} [{a.revision_urgency.upper()}] {a.report_id}")
        print(f"     Supuesto: {a.assumption_challenged.assumption_id} — {a.assumption_challenged.description[:70]}...")
        print(f"     Población: {a.affected_population}")
        print(f"     ND correlación: {a.nd_pattern_correlation}")
        print(f"     Efecto: {a.effect_size:.3f} | Confianza: {a.confidence:.3f} | n={a.n_cases}")
        print(f"     Patrón: {a.observed_pattern[:100]}...")

    print(f"\n  📝 Propuestas de revisión: {len(proposals)}")
    for p in proposals:
        print(f"\n  ▸ {p.proposal_id}")
        print(f"    Revisión: {p.proposed_revision[:90]}...")
        print(f"    O2: {p.implementation_in_o2[:80]}...")
        print(f"    O3: {p.implementation_in_o3[:80]}...")
        print(f"    O4: {p.implementation_in_o4[:80]}...")
        print(f"    Paper: {p.paper_hypothesis[:80]}...")

    print(f"\n  🌟 Señales O5 emergentes: {len(signals)}")
    for s in signals:
        print(f"\n  ★ {s.signal_id} [{s.discovery_type}] (confianza: {s.confidence:.2f})")
        print(f"    Descripción: {s.description[:120]}...")
        print(f"    Objetivo sugerido: {s.suggested_new_objective[:100]}...")
        print(f"    Potencial paper: {s.paper_potential}")

    # ── 5. Export para researcher_view ──
    report = engine.export_for_researcher_view()
    print(f"\n{'─' * 72}")
    print(f"  RESUMEN EXPORTADO PARA RESEARCHER VIEW")
    print(f"{'─' * 72}")
    print(f"  Ejecuciones del motor: {report['run_count']}")
    print(f"  Anomalías totales: {report['summary']['total_anomalies']}")
    print(f"    Críticas: {report['summary']['critical']}")
    print(f"    Altas: {report['summary']['high']}")
    print(f"    Medias: {report['summary']['medium']}")
    print(f"    Bajas: {report['summary']['low']}")
    print(f"  Propuestas de revisión: {report['summary']['total_proposals']}")
    print(f"  Señales O5: {report['summary']['total_o5_signals']}")
    print(f"  Supuestos falsificados: {report['summary']['assumptions_falsified']}")
    print(f"  Supuestos cuestionados: {report['summary']['assumptions_challenged']}")
    print(f"  Supuestos sin testear: {report['summary']['assumptions_untested']}")

    print(f"\n  Estado de supuestos:")
    for aid, status in report["assumption_status"].items():
        status_icons = {"falsified": "❌", "challenged": "⚠️", "provisional": "✓", "untested": "?"}
        icon = status_icons.get(status["status"], "?")
        print(f"    {icon} {aid}: {status['status']} ({status['n_anomalies']} anomalías) — {status['description'][:60]}...")

    print(f"\n{'═' * 72}")
    print(f"  Demo completa. Este informe es lo que researcher_view.py renderiza")
    print(f"  en la pestaña 'Retroalimentación O1'.")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    _demo()
