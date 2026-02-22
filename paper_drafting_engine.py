"""
PAPER DRAFTING ENGINE — El Sistema Que Escribe Su Propio Paper
═══════════════════════════════════════════════════════════════════════
El ecosistema GENIE Learn genera datos. Los datos alimentan papers.
Este módulo extrae datos de todos los módulos analíticos y genera
borradores de secciones académicas listas para editar.

NO escribe el paper por ti. Genera:
    - Tablas de resultados formateadas (LaTeX + Markdown)
    - Secciones de metodología con métricas reales
    - Figuras descriptivas en formato publicable
    - Análisis estadístico básico pre-formateado
    - Templates por venue (LAK, CSEDU, EC-TEL, C&E journal)

EL PRINCIPIO: separar EXTRACCIÓN (automática) de INTERPRETACIÓN (humana).
La máquina extrae, tabula y formatea. El investigador interpreta, argumenta
y construye la narrativa. Esa frontera es sagrada.

VENUES TARGET:
    LAK 2027     → ACM format, 10 pages, emphasis on analytics
    CSEDU 2026   → SCITEPRESS, 12 pages, emphasis on system design
    EC-TEL 2026  → Springer LNCS, 15 pages, emphasis on pedagogy
    C&E journal  → Elsevier, no page limit, emphasis on empirical results

Autor: Diego Elvira Vásquez · CP25/152 GSIC/EMIC · Feb 2026
"""

import json
import statistics
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from collections import Counter, defaultdict


# ═══════════════════════════════════════════════════════════════
# VENUE TEMPLATES
# ═══════════════════════════════════════════════════════════════

VENUE_CONFIGS = {
    "lak": {
        "name": "Learning Analytics & Knowledge (LAK 2027)",
        "format": "ACM sigconf",
        "max_pages": 10,
        "emphasis": "analytics_results",
        "sections": ["abstract", "introduction", "related_work", "system_design",
                     "methodology", "results", "discussion", "conclusion"],
        "bib_style": "ACM-Reference-Format",
    },
    "csedu": {
        "name": "Computer Supported Education (CSEDU 2026)",
        "format": "SCITEPRESS",
        "max_pages": 12,
        "emphasis": "system_architecture",
        "sections": ["abstract", "introduction", "related_work", "system_design",
                     "evaluation", "results", "discussion", "conclusion"],
        "bib_style": "plain",
    },
    "ectel": {
        "name": "European Conference on TEL (EC-TEL 2026)",
        "format": "Springer LNCS",
        "max_pages": 15,
        "emphasis": "pedagogical_framework",
        "sections": ["abstract", "introduction", "theoretical_framework",
                     "system_design", "methodology", "results",
                     "discussion", "limitations", "conclusion"],
        "bib_style": "splncs04",
    },
    "ce_journal": {
        "name": "Computers & Education (Elsevier)",
        "format": "Elsevier",
        "max_pages": None,
        "emphasis": "empirical_evidence",
        "sections": ["abstract", "introduction", "literature_review",
                     "theoretical_framework", "methodology", "system_design",
                     "results", "discussion", "implications",
                     "limitations", "conclusion"],
        "bib_style": "elsarticle-num",
    },
}


@dataclass
class PaperDataPackage:
    """Paquete de datos extraídos del sistema para un paper."""
    extraction_date: str = ""
    venue: str = ""
    pilot_id: str = ""

    # Descriptive stats
    n_students: int = 0
    n_teachers: int = 0
    n_interactions: int = 0
    n_sessions: int = 0
    duration_days: int = 0
    courses: List[str] = field(default_factory=list)

    # Bloom
    bloom_distribution: Dict[str, int] = field(default_factory=dict)
    bloom_mean: float = 0.0
    bloom_std: float = 0.0
    bloom_trajectory_slope: float = 0.0
    bloom_by_scaffolding: Dict[str, float] = field(default_factory=dict)

    # Autonomy
    autonomy_phase_distribution: Dict[str, int] = field(default_factory=dict)
    autonomy_mean: float = 0.0
    autonomy_trajectory: float = 0.0
    scaffolding_fading_rate: float = 0.0

    # HHH Alignment
    hhh_means: Dict[str, float] = field(default_factory=dict)
    hhh_trend: str = ""

    # RAGAS
    ragas_means: Dict[str, float] = field(default_factory=dict)
    ragas_quality_distribution: Dict[str, int] = field(default_factory=dict)

    # Engagement profiles (from clustering)
    engagement_profiles: List[Dict] = field(default_factory=list)

    # Gaming
    gaming_rate: float = 0.0
    gaming_by_hour: Dict[int, float] = field(default_factory=dict)

    # Config effectiveness
    config_changes: int = 0
    config_impact_data: List[Dict] = field(default_factory=list)

    # Teacher agency
    teacher_decisions: int = 0
    acceptance_rate: float = 0.0
    agency_index: float = 0.0

    # Reflexive insights
    reflexive_insights: int = 0
    critical_insights: List[str] = field(default_factory=list)

    # LLM Judge
    bloom_kappa: float = 0.0
    bloom_kappa_interpretation: str = ""

    # GDPR
    consent_coverage: float = 0.0

    # Statistical tests (pre-filled placeholders)
    stat_tests: List[Dict] = field(default_factory=list)


class PaperDraftingEngine:
    """
    Extrae datos del ecosistema y genera borradores de secciones académicas.
    
    Uso:
        engine = PaperDraftingEngine()
        engine.extract_from_system(modules_dict)
        draft = engine.generate_draft("lak")
        engine.export_all("paper_output/")
    """

    def __init__(self):
        self.data = PaperDataPackage()
        self.tables: List[Dict] = []
        self.figures: List[Dict] = []

    # ═══════════════════════════════════════════════════════════
    # EXTRACCIÓN — Recoger datos de todos los módulos
    # ═══════════════════════════════════════════════════════════

    def extract_from_system(self, modules: dict):
        """
        Extrae datos de los módulos del ecosistema.
        
        Args:
            modules: dict con instancias de los módulos:
                {
                    "profiler": CognitiveProfiler instance,
                    "autonomy": EpistemicAutonomyTracker instance,
                    "semiotics": InteractionSemioticsEngine instance,
                    "hhh": HHHAlignmentDetector instance,
                    "ragas": RAGASEvaluator instance,
                    "judge": LLMBloomJudge instance,
                    "notifications": TeacherNotificationEngine instance,
                    "reflexivity": SystemReflexivityEngine instance,
                    "anonymizer": GDPRAnonymizer instance,
                    "persistence": PersistenceEngine instance,
                    "middleware": PedagogicalMiddleware instance,
                }
        """
        self.data.extraction_date = datetime.now().isoformat()

        # ── Middleware stats ──
        mw = modules.get("middleware")
        if mw:
            stats = mw.get_analytics_summary() if hasattr(mw, "get_analytics_summary") else {}
            self.data.n_interactions = stats.get("total_interactions", 0)
            self.data.n_students = stats.get("unique_students", 0)

        # ── Cognitive Profiler ──
        profiler = modules.get("profiler")
        if profiler and hasattr(profiler, "get_cohort_summary"):
            cohort = profiler.get_cohort_summary()
            self.data.bloom_distribution = cohort.get("bloom_distribution", {})
            self.data.bloom_mean = cohort.get("bloom_mean", 0)

        # ── HHH ──
        hhh = modules.get("hhh")
        if hhh and hasattr(hhh, "get_session_report"):
            report = hhh.get_session_report()
            self.data.hhh_means = {
                "helpful": report.get("avg_helpful", 0),
                "honest": report.get("avg_honest", 0),
                "harmless": report.get("avg_harmless", 0),
            }

        # ── RAGAS ──
        ragas = modules.get("ragas")
        if ragas and hasattr(ragas, "get_session_report"):
            report = ragas.get_session_report()
            self.data.ragas_means = {
                "faithfulness": report.get("avg_faithfulness", 0),
                "answer_relevance": report.get("avg_answer_relevance", 0),
                "context_precision": report.get("avg_context_precision", 0),
                "context_recall": report.get("avg_context_recall", 0),
            }
            self.data.ragas_quality_distribution = report.get("quality_distribution", {})

        # ── LLM Judge ──
        judge = modules.get("judge")
        if judge and hasattr(judge, "compute_kappa"):
            kappa = judge.compute_kappa()
            self.data.bloom_kappa = kappa.get("kappa", 0) or 0
            self.data.bloom_kappa_interpretation = kappa.get("interpretation", "")

        # ── Teacher Notifications ──
        notifications = modules.get("notifications")
        if notifications and hasattr(notifications, "get_agency_metrics"):
            metrics = notifications.get_agency_metrics()
            self.data.teacher_decisions = metrics.get("total_decisions", 0)
            self.data.acceptance_rate = metrics.get("acceptance_rate", 0)
            self.data.agency_index = metrics.get("agency_index", 0)

        # ── Reflexivity ──
        reflexivity = modules.get("reflexivity")
        if reflexivity and hasattr(reflexivity, "get_insights_report"):
            report = reflexivity.get_insights_report()
            self.data.reflexive_insights = report.get("total_insights", 0)
            self.data.critical_insights = [
                i["title"] for i in report.get("critical_insights", [])
            ]

        # ── GDPR ──
        anonymizer = modules.get("anonymizer")
        if anonymizer and hasattr(anonymizer, "get_consent_summary"):
            summary = anonymizer.get_consent_summary()
            active = summary.get("active_consents", 0)
            total = summary.get("total_participants", 1)
            self.data.consent_coverage = round(active / max(total, 1), 2)

    def extract_from_raw(self, interactions: List[dict]):
        """Extrae directamente de una lista de interacciones (dict)."""
        if not interactions:
            return

        self.data.n_interactions = len(interactions)
        self.data.n_students = len(set(i.get("student_id", "") for i in interactions))
        self.data.extraction_date = datetime.now().isoformat()

        # Bloom
        blooms = [i.get("bloom_estimate", 1) for i in interactions]
        self.data.bloom_mean = round(statistics.mean(blooms), 2)
        self.data.bloom_std = round(statistics.stdev(blooms), 2) if len(blooms) > 1 else 0
        bloom_counter = Counter(blooms)
        bloom_labels = {1: "Recordar", 2: "Comprender", 3: "Aplicar",
                       4: "Analizar", 5: "Evaluar", 6: "Crear"}
        self.data.bloom_distribution = {
            bloom_labels.get(k, str(k)): v for k, v in sorted(bloom_counter.items())
        }

        # Bloom by scaffolding mode
        by_mode = defaultdict(list)
        for i in interactions:
            mode = i.get("scaffolding_mode", "unknown")
            by_mode[mode].append(i.get("bloom_estimate", 1))
        self.data.bloom_by_scaffolding = {
            mode: round(statistics.mean(vals), 2)
            for mode, vals in by_mode.items() if vals
        }

        # HHH
        hhh_vals = [i.get("hhh_overall", 0) for i in interactions if i.get("hhh_overall")]
        if hhh_vals:
            self.data.hhh_means = {
                "helpful": round(statistics.mean(i.get("hhh_helpful", 0) for i in interactions), 3),
                "honest": round(statistics.mean(i.get("hhh_honest", 0) for i in interactions), 3),
                "harmless": round(statistics.mean(i.get("hhh_harmless", 0) for i in interactions), 3),
                "overall": round(statistics.mean(hhh_vals), 3),
            }

        # Autonomy
        phases = [i.get("autonomy_phase", "unknown") for i in interactions]
        self.data.autonomy_phase_distribution = dict(Counter(phases))
        auto_scores = [i.get("autonomy_score", 0) for i in interactions if i.get("autonomy_score")]
        if auto_scores:
            self.data.autonomy_mean = round(statistics.mean(auto_scores), 3)

        # Gaming
        gaming = [i.get("gaming_suspicion", 0) for i in interactions]
        self.data.gaming_rate = round(
            sum(1 for g in gaming if g > 0.5) / max(len(gaming), 1), 3
        )

    # ═══════════════════════════════════════════════════════════
    # GENERACIÓN DE TABLAS
    # ═══════════════════════════════════════════════════════════

    def generate_tables(self) -> List[Dict]:
        """Genera todas las tablas del paper."""
        self.tables = []

        # Table 1: Descriptive stats
        self.tables.append({
            "id": "tab:descriptive",
            "caption": "Descriptive statistics of the pilot study",
            "columns": ["Metric", "Value"],
            "rows": [
                ["Participants (students)", str(self.data.n_students)],
                ["Total interactions", str(self.data.n_interactions)],
                ["Mean interactions/student",
                 str(round(self.data.n_interactions / max(self.data.n_students, 1), 1))],
                ["Duration (days)", str(self.data.duration_days)],
                ["GDPR consent coverage", f"{self.data.consent_coverage:.0%}"],
            ],
        })

        # Table 2: Bloom distribution
        if self.data.bloom_distribution:
            self.tables.append({
                "id": "tab:bloom",
                "caption": "Distribution of student prompts across Bloom's taxonomy levels",
                "columns": ["Bloom Level", "Count", "Percentage"],
                "rows": [
                    [level, str(count),
                     f"{count / max(self.data.n_interactions, 1):.1%}"]
                    for level, count in self.data.bloom_distribution.items()
                ],
            })

        # Table 3: Bloom by scaffolding mode
        if self.data.bloom_by_scaffolding:
            self.tables.append({
                "id": "tab:bloom_scaffolding",
                "caption": "Mean Bloom level by scaffolding mode",
                "columns": ["Scaffolding Mode", "Mean Bloom", "n"],
                "rows": [
                    [mode, str(val), "—"]
                    for mode, val in self.data.bloom_by_scaffolding.items()
                ],
            })

        # Table 4: HHH Alignment
        if self.data.hhh_means:
            self.tables.append({
                "id": "tab:hhh",
                "caption": "HHH alignment scores (Askell et al., 2021)",
                "columns": ["Dimension", "Mean Score"],
                "rows": [
                    [dim.capitalize(), f"{val:.3f}"]
                    for dim, val in self.data.hhh_means.items()
                ],
            })

        # Table 5: RAGAS
        if self.data.ragas_means:
            self.tables.append({
                "id": "tab:ragas",
                "caption": "RAG quality assessment (RAGAS framework)",
                "columns": ["Metric", "Mean Score"],
                "rows": [
                    [metric.replace("_", " ").title(), f"{val:.3f}"]
                    for metric, val in self.data.ragas_means.items()
                ],
            })

        # Table 6: Teacher Agency
        if self.data.teacher_decisions > 0:
            self.tables.append({
                "id": "tab:agency",
                "caption": "Teacher agency metrics derived from notification decisions",
                "columns": ["Metric", "Value"],
                "rows": [
                    ["Total decisions", str(self.data.teacher_decisions)],
                    ["Acceptance rate", f"{self.data.acceptance_rate:.1%}"],
                    ["Agency index", f"{self.data.agency_index:.3f}"],
                ],
            })

        return self.tables

    # ═══════════════════════════════════════════════════════════
    # GENERACIÓN DE SECCIONES
    # ═══════════════════════════════════════════════════════════

    def generate_draft(self, venue: str = "lak") -> Dict[str, str]:
        """
        Genera borrador de secciones para un venue específico.
        Returns dict: section_name → markdown text
        """
        config = VENUE_CONFIGS.get(venue, VENUE_CONFIGS["lak"])
        self.generate_tables()

        sections = {}

        sections["_meta"] = (
            f"% Paper draft for {config['name']}\n"
            f"% Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"% Data extracted from GENIE Learn ecosystem\n"
            f"% Format: {config['format']}\n"
            f"% WARNING: This is a DATA EXTRACTION, not a finished paper.\n"
            f"% The researcher must interpret, argue, and build the narrative.\n"
        )

        sections["abstract"] = self._draft_abstract(config)
        sections["methodology_participants"] = self._draft_participants()
        sections["methodology_instruments"] = self._draft_instruments()
        sections["results_descriptive"] = self._draft_results_descriptive()
        sections["results_bloom"] = self._draft_results_bloom()
        sections["results_hhh"] = self._draft_results_hhh()
        sections["results_autonomy"] = self._draft_results_autonomy()
        sections["results_teacher_agency"] = self._draft_results_agency()
        sections["results_reflexivity"] = self._draft_results_reflexivity()
        sections["tables_latex"] = self._generate_latex_tables()
        sections["tables_markdown"] = self._generate_markdown_tables()
        sections["limitations"] = self._draft_limitations()
        sections["references_needed"] = self._list_references_needed()

        return sections

    def _draft_abstract(self, config: dict) -> str:
        return (
            f"% ABSTRACT — Venue: {config['name']}\n"
            f"% [RESEARCHER: complete the narrative framing]\n\n"
            f"This study presents an evaluation of a GenAI-enhanced pedagogical chatbot\n"
            f"with configurable pedagogical parameters deployed in [CONTEXT].\n"
            f"The system incorporates {self.data.n_interactions} student-chatbot interactions\n"
            f"from {self.data.n_students} participants, analyzed through a multi-layered\n"
            f"analytics framework including Bloom's taxonomy classification\n"
            f"(M={self.data.bloom_mean}, SD={self.data.bloom_std}),\n"
            f"HHH alignment auditing (overall M={self.data.hhh_means.get('overall', 0):.3f}),\n"
            f"and epistemic autonomy tracking.\n"
            f"{'Results revealed ' + str(self.data.reflexive_insights) + ' emergent systemic patterns through reflexive meta-analytics. ' if self.data.reflexive_insights else ''}"
            f"Teacher agency was measured through {self.data.teacher_decisions} notification decisions\n"
            f"(agency index: {self.data.agency_index:.3f}).\n"
            f"% [RESEARCHER: add main finding and contribution statement]\n"
        )

    def _draft_participants(self) -> str:
        return (
            f"### Participants\n\n"
            f"A total of {self.data.n_students} students participated in the study,\n"
            f"generating {self.data.n_interactions} interactions with the chatbot over\n"
            f"{self.data.duration_days or '[X]'} days "
            f"(M={self.data.n_interactions / max(self.data.n_students, 1):.1f} interactions/student).\n"
            f"GDPR consent was obtained from {self.data.consent_coverage:.0%} of participants\n"
            f"through a granular consent protocol covering {5} data categories.\n"
            f"% [RESEARCHER: add demographics, course details, recruitment]\n"
        )

    def _draft_instruments(self) -> str:
        return (
            f"### Instruments and Measures\n\n"
            f"The following analytical instruments were deployed:\n\n"
            f"- **Cognitive level classification**: Bloom's revised taxonomy\n"
            f"  (Anderson & Krathwohl, 2001) operationalized through linguistic markers\n"
            f"  validated with LLM-as-Judge (Cohen's κ = {self.data.bloom_kappa:.3f},\n"
            f"  {self.data.bloom_kappa_interpretation}).\n"
            f"- **AI alignment auditing**: HHH framework (Askell et al., 2021)\n"
            f"  measuring helpfulness, honesty, and harmlessness per response.\n"
            f"- **RAG quality assessment**: Adapted RAGAS framework (Es et al., 2023)\n"
            f"  measuring faithfulness, answer relevance, context precision, and recall.\n"
            f"- **Epistemic autonomy**: 4-phase tracker based on scaffolding fading\n"
            f"  (Wood, Bruner & Ross, 1976) and self-efficacy (Bandura, 1997).\n"
            f"- **Interaction semiotics**: Speech Act Theory (Austin, 1962; Searle, 1969)\n"
            f"  and Gricean maxims for interaction quality.\n"
            f"- **Teacher agency**: Notification-decision protocol measuring acceptance,\n"
            f"  rejection, and modification rates with rationale capture.\n"
            f"- **System reflexivity**: 8 meta-analytic detectors for emergent patterns.\n"
            f"% [RESEARCHER: add SUS, NPS, TAM, interview protocols if applicable]\n"
        )

    def _draft_results_descriptive(self) -> str:
        return (
            f"### Descriptive Results\n\n"
            f"See Table {{\\ref{{tab:descriptive}}}} for an overview.\n"
            f"Over the study period, {self.data.n_students} students generated\n"
            f"{self.data.n_interactions} interactions. The mean Bloom level was\n"
            f"{self.data.bloom_mean} (SD={self.data.bloom_std}), with a\n"
            f"{'positive' if self.data.bloom_trajectory_slope > 0 else 'negative' if self.data.bloom_trajectory_slope < 0 else 'flat'}\n"
            f"trajectory (slope={self.data.bloom_trajectory_slope:.3f}).\n"
            f"Gaming behavior was detected in {self.data.gaming_rate:.1%} of interactions.\n"
        )

    def _draft_results_bloom(self) -> str:
        lines = [
            f"### Cognitive Level Analysis\n\n"
            f"Table {{\\ref{{tab:bloom}}}} shows the distribution across Bloom levels.\n"
        ]
        if self.data.bloom_by_scaffolding:
            lines.append(
                f"A comparison by scaffolding mode (Table {{\\ref{{tab:bloom_scaffolding}}}}) reveals:\n"
            )
            for mode, val in self.data.bloom_by_scaffolding.items():
                lines.append(f"- {mode}: M={val}")
            lines.append(
                f"\n% [RESEARCHER: run ANOVA/Kruskal-Wallis to test significance between modes]\n"
                f"% Suggested test: Kruskal-Wallis (non-parametric, Bloom is ordinal)\n"
                f"% If significant, post-hoc: Dunn's test with Bonferroni correction\n"
            )
        return "\n".join(lines)

    def _draft_results_hhh(self) -> str:
        if not self.data.hhh_means:
            return "### HHH Alignment\n\n% No HHH data available.\n"
        return (
            f"### AI Alignment Assessment\n\n"
            f"The HHH alignment audit (Table {{\\ref{{tab:hhh}}}}) yielded:\n"
            f"helpfulness M={self.data.hhh_means.get('helpful', 0):.3f},\n"
            f"honesty M={self.data.hhh_means.get('honest', 0):.3f},\n"
            f"harmlessness M={self.data.hhh_means.get('harmless', 0):.3f}.\n"
            f"% [RESEARCHER: interpret — is honesty lower than helpful? Why?]\n"
            f"% Possible narrative: RAG-grounded responses score higher on honesty\n"
            f"% than non-RAG responses (hypothesis to test with t-test on hhh_honest).\n"
        )

    def _draft_results_autonomy(self) -> str:
        if not self.data.autonomy_phase_distribution:
            return "### Autonomy\n\n% No autonomy data available.\n"
        return (
            f"### Epistemic Autonomy Trajectories\n\n"
            f"Phase distribution at end of study:\n"
            f"{json.dumps(self.data.autonomy_phase_distribution, indent=2)}\n\n"
            f"Mean autonomy score: {self.data.autonomy_mean:.3f}\n\n"
            f"% [RESEARCHER: plot the 'electrocardiogram' from autonomy_viz.py]\n"
            f"% Key question: does scaffolding fading correlate with phase progression?\n"
            f"% Test: Spearman correlation between scaffolding_level and autonomy_score\n"
        )

    def _draft_results_agency(self) -> str:
        if self.data.teacher_decisions == 0:
            return "### Teacher Agency\n\n% No decision data yet.\n"
        return (
            f"### Teacher Agency Through Notification Decisions\n\n"
            f"Teachers received system-generated suggestions based on cohort analytics.\n"
            f"Over {self.data.teacher_decisions} decisions:\n"
            f"- Acceptance rate: {self.data.acceptance_rate:.1%}\n"
            f"- Agency index (rejection + modification rate): {self.data.agency_index:.3f}\n\n"
            f"% [RESEARCHER: analyze rationale qualitatively]\n"
            f"% Use thematic analysis (Braun & Clarke, 2006) on decision_rationale field.\n"
            f"% Priestley et al. (2015) ecological model of teacher agency applies here.\n"
            f"% Key insight: high agency index = teacher exercises active pedagogical judgment\n"
            f"% over algorithmic suggestions. This is HCAI in action.\n"
        )

    def _draft_results_reflexivity(self) -> str:
        if not self.data.reflexive_insights:
            return "### Reflexive Meta-Analytics\n\n% No insights generated yet.\n"
        lines = [
            f"### Emergent Patterns from System Reflexivity\n\n"
            f"The reflexive meta-analytics engine identified {self.data.reflexive_insights}\n"
            f"emergent patterns not visible to individual analytical modules.\n"
        ]
        if self.data.critical_insights:
            lines.append("Critical findings:")
            for insight in self.data.critical_insights[:5]:
                lines.append(f"- {insight}")
        lines.append(
            f"\n% [RESEARCHER: these are the most novel findings. Build narrative around them.]\n"
            f"% Frame as: 'The system detected patterns that neither the designer\n"
            f"% nor the teacher anticipated' — Suchman's (2007) gap between plans and action.\n"
        )
        return "\n".join(lines)

    def _draft_limitations(self) -> str:
        limitations = [
            "### Limitations\n",
            f"- Sample size (n={self.data.n_students}) limits generalizability.",
            "- Single institution context (UVa) — cross-institutional validation pending.",
            f"- Bloom classification validated with LLM-as-Judge (κ={self.data.bloom_kappa:.3f})"
            f" — human expert validation planned for future work.",
            "- In-memory analytics during prototype phase — full persistence with PostgreSQL in production.",
            "- Self-reported teacher decisions may not capture implicit pedagogical reasoning.",
            "% [RESEARCHER: add specific limitations from your context]",
        ]
        return "\n".join(limitations)

    def _list_references_needed(self) -> str:
        refs = [
            "### References Needed (auto-detected from metrics used)\n",
            "- Anderson, L. W. & Krathwohl, D. R. (2001). Bloom's taxonomy revised.",
            "- Askell, A. et al. (2021). A General Language Assistant as a Laboratory for Alignment. [HHH]",
            "- Austin, J. L. (1962). How to Do Things with Words. [Speech Acts]",
            "- Baker, R. S. et al. (2008). Gaming the system. [Gaming detection]",
            "- Bandura, A. (1997). Self-Efficacy. [Autonomy tracking]",
            "- Bourdieu, P. (1990). The Logic of Practice. [Reflexivity]",
            "- Braun, V. & Clarke, V. (2006). Thematic Analysis. [Qualitative]",
            "- Es, S. et al. (2023). RAGAS. [RAG quality]",
            "- Ortega-Arranz, A. et al. (2026). GenAI Analytics. LAK 2026. [Prior work]",
            "- Priestley, M. et al. (2015). Teacher Agency. [Agency model]",
            "- Shneiderman, B. (2022). Human-Centered AI. [HCAI]",
            "- Suchman, L. (2007). Plans and Situated Actions. [Reflexivity]",
            "- Vygotsky, L. S. (1978). Mind in Society. [ZPD]",
            "- Wood, D., Bruner, J. S. & Ross, G. (1976). Tutoring. [Scaffolding]",
        ]
        return "\n".join(refs)

    # ═══════════════════════════════════════════════════════════
    # FORMAT: LaTeX y Markdown tables
    # ═══════════════════════════════════════════════════════════

    def _generate_latex_tables(self) -> str:
        """Genera todas las tablas en formato LaTeX."""
        output = []
        for table in self.tables:
            n_cols = len(table["columns"])
            col_spec = "l" + "r" * (n_cols - 1)
            lines = [
                f"\\begin{{table}}[ht]",
                f"\\centering",
                f"\\caption{{{table['caption']}}}",
                f"\\label{{{table['id']}}}",
                f"\\begin{{tabular}}{{{col_spec}}}",
                f"\\toprule",
                " & ".join(f"\\textbf{{{c}}}" for c in table["columns"]) + " \\\\",
                "\\midrule",
            ]
            for row in table["rows"]:
                lines.append(" & ".join(row) + " \\\\")
            lines.extend([
                "\\bottomrule",
                "\\end{tabular}",
                "\\end{table}",
                "",
            ])
            output.append("\n".join(lines))
        return "\n".join(output)

    def _generate_markdown_tables(self) -> str:
        """Genera todas las tablas en Markdown."""
        output = []
        for table in self.tables:
            output.append(f"**{table['caption']}** ({table['id']})\n")
            output.append("| " + " | ".join(table["columns"]) + " |")
            output.append("| " + " | ".join("---" for _ in table["columns"]) + " |")
            for row in table["rows"]:
                output.append("| " + " | ".join(row) + " |")
            output.append("")
        return "\n".join(output)

    # ═══════════════════════════════════════════════════════════
    # EXPORTAR TODO
    # ═══════════════════════════════════════════════════════════

    def export_all(self, output_dir: str = "paper_output", venue: str = "lak"):
        """Exporta todo el paquete de datos + borradores."""
        os.makedirs(output_dir, exist_ok=True)

        # 1. Data package (JSON)
        data_path = os.path.join(output_dir, "data_package.json")
        from dataclasses import asdict
        with open(data_path, "w") as f:
            json.dump(asdict(self.data), f, indent=2, default=str)

        # 2. Draft sections (Markdown)
        draft = self.generate_draft(venue)
        for section_name, content in draft.items():
            path = os.path.join(output_dir, f"{section_name}.md")
            with open(path, "w") as f:
                f.write(content)

        # 3. LaTeX tables (standalone)
        latex_path = os.path.join(output_dir, "tables.tex")
        with open(latex_path, "w") as f:
            f.write(self._generate_latex_tables())

        # 4. Summary
        summary = {
            "venue": venue,
            "config": VENUE_CONFIGS.get(venue, {}),
            "tables_generated": len(self.tables),
            "sections_generated": len(draft),
            "data_completeness": self._assess_completeness(),
            "export_date": datetime.now().isoformat(),
        }
        summary_path = os.path.join(output_dir, "export_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _assess_completeness(self) -> dict:
        """Evalúa qué datos faltan para el paper."""
        checks = {
            "bloom_data": self.data.bloom_mean > 0,
            "hhh_data": bool(self.data.hhh_means),
            "ragas_data": bool(self.data.ragas_means),
            "autonomy_data": self.data.autonomy_mean > 0,
            "teacher_agency_data": self.data.teacher_decisions > 0,
            "reflexivity_data": self.data.reflexive_insights > 0,
            "bloom_validation": self.data.bloom_kappa > 0,
            "gdpr_compliance": self.data.consent_coverage > 0,
            "sample_size_adequate": self.data.n_students >= 30,
            "interactions_adequate": self.data.n_interactions >= 200,
        }
        return {
            "checks": checks,
            "complete": sum(checks.values()),
            "total": len(checks),
            "missing": [k for k, v in checks.items() if not v],
        }


# ═══════════════════════════════════════════════════════════════
# DEMO — Generar un paper draft con datos simulados
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    engine = PaperDraftingEngine()

    # Simular extracción de datos
    import random
    random.seed(42)
    fake_interactions = []
    bloom_labels = {1: "Recordar", 2: "Comprender", 3: "Aplicar",
                    4: "Analizar", 5: "Evaluar", 6: "Crear"}
    modes = ["socratic", "hints", "direct"]

    for i in range(150):
        sid = f"est_{random.randint(1,25):02d}"
        mode = random.choice(modes)
        bloom = random.choices([1,2,3,4,5,6], weights=[15,25,25,20,10,5])[0]
        fake_interactions.append({
            "student_id": sid,
            "bloom_estimate": bloom,
            "scaffolding_mode": mode,
            "hhh_overall": random.uniform(0.5, 0.95),
            "hhh_helpful": random.uniform(0.5, 0.95),
            "hhh_honest": random.uniform(0.4, 0.9),
            "hhh_harmless": random.uniform(0.6, 0.98),
            "autonomy_score": random.uniform(0.2, 0.8),
            "autonomy_phase": random.choice(["dependent", "scaffolded", "emergent", "autonomous"]),
            "gaming_suspicion": random.uniform(0, 0.6),
        })

    engine.extract_from_raw(fake_interactions)

    # Añadir datos manuales
    engine.data.bloom_kappa = 0.67
    engine.data.bloom_kappa_interpretation = "substantial"
    engine.data.teacher_decisions = 12
    engine.data.acceptance_rate = 0.58
    engine.data.agency_index = 0.42
    engine.data.reflexive_insights = 6
    engine.data.critical_insights = [
        "Paradoja del scaffolding detectada",
        "Gaming como fenómeno social/temporal",
    ]
    engine.data.consent_coverage = 0.92
    engine.data.duration_days = 21
    engine.data.ragas_means = {
        "faithfulness": 0.73, "answer_relevance": 0.81,
        "context_precision": 0.68, "context_recall": 0.62,
    }

    # Generar draft para LAK
    draft = engine.generate_draft("lak")

    print("=" * 70)
    print("PAPER DRAFT — LAK 2027")
    print("=" * 70)
    for section, content in draft.items():
        if section.startswith("_"):
            continue
        print(f"\n{'─' * 50}")
        print(f"SECTION: {section}")
        print(f"{'─' * 50}")
        print(content[:500])
        if len(content) > 500:
            print(f"  [...{len(content)} chars total]")

    # Exportar
    result = engine.export_all("paper_output_demo", "lak")
    print(f"\n{'=' * 70}")
    print(f"Exported: {result['tables_generated']} tables, {result['sections_generated']} sections")
    completeness = result["data_completeness"]
    print(f"Completeness: {completeness['complete']}/{completeness['total']}")
    if completeness["missing"]:
        print(f"Missing: {', '.join(completeness['missing'])}")

    print("\n✓ Paper Drafting Engine operativo")
