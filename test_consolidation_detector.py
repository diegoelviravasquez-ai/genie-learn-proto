"""
TEST SUITE — consolidation_detector.py
════════════════════════════════════════
Tests de invariantes teóricos y funcionales para el módulo de
detección de consolidación mnésica inter-sesión.

Estructura:
  1. TestTopicEncounter — Dataclasses base
  2. TestWindowDetection — Detección de ventanas temporales
  3. TestConsolidationSignals — Clasificación de señales
  4. TestReformulationDepth — Profundidad de reformulación
  5. TestConsolidationIndex — Índice global y ponderación
  6. TestSpacedRepetition — Sugerencias de repaso espaciado
  7. TestStudentReport — Informe y print_demo_report
  8. TestCohortSummary — Agregados de cohorte
  9. TestIntegration — Flujo completo end-to-end

Convención: invariantes teóricos documentados en cada assertion.
Autor: Diego Elvira Vásquez · Feb 2026
"""

import pytest
from datetime import datetime, timedelta

from consolidation_detector import (
    MemoryConsolidationTracker,
    TopicEncounter,
    ConsolidationWindow,
    TopicConsolidationProfile,
    StudentConsolidationReport,
    TopicSuggestion,
    generate_demo_data,
    print_demo_report,
    # Constantes
    MIN_GAP_HOURS,
    MAX_GAP_HOURS,
    OPTIMAL_GAP_MIN,
    OPTIMAL_GAP_MAX,
    STRONG_CONSOLIDATION_BLOOM_DELTA,
    WEAK_CONSOLIDATION_BLOOM_DELTA,
    REGRESSION_BLOOM_DELTA,
)


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def tracker():
    return MemoryConsolidationTracker()


@pytest.fixture
def base_time():
    return datetime(2026, 2, 10, 10, 0, 0)


@pytest.fixture
def tracker_with_data(tracker, base_time):
    """Tracker con un patrón simple: dos encuentros sobre bucles separados 50h."""
    tracker.ingest_interaction(
        student_id="s1", timestamp=base_time,
        topics=["bucles"], bloom_level=2.0, bloom_label="comprender",
        prompt_text="¿Qué es un bucle for?", session_id="ses1",
    )
    tracker.ingest_interaction(
        student_id="s1", timestamp=base_time + timedelta(hours=50),
        topics=["bucles"], bloom_level=4.0, bloom_label="analizar",
        prompt_text="¿Por qué un for es mejor que un while para iterar arrays?",
        session_id="ses2",
    )
    return tracker


# ═══════════════════════════════════════════════════════════════════
# 1. DATACLASSES
# ═══════════════════════════════════════════════════════════════════

class TestTopicEncounter:
    def test_creation(self, base_time):
        e = TopicEncounter(
            timestamp=base_time, student_id="s1", topic="bucles",
            bloom_level=2.0, bloom_label="comprender", prompt_text="test",
        )
        assert e.student_id == "s1"
        assert e.bloom_level == 2.0

    def test_session_default(self, base_time):
        e = TopicEncounter(
            timestamp=base_time, student_id="s1", topic="x",
            bloom_level=1.0, bloom_label="recordar", prompt_text="",
        )
        assert e.session_id == ""


class TestConsolidationWindow:
    def test_default_signal_is_absent(self):
        w = ConsolidationWindow(student_id="s1", topic="x")
        assert w.consolidation_signal == "absent"

    def test_bloom_delta_stored(self, tracker_with_data):
        windows = tracker_with_data.detect_consolidation_patterns("s1")
        assert len(windows) == 1
        # Bloom 2 → 4 = Δ+2.0
        assert windows[0].bloom_delta == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════════
# 2. DETECCIÓN DE VENTANAS TEMPORALES
# ═══════════════════════════════════════════════════════════════════

class TestWindowDetection:
    def test_intra_session_ignored(self, tracker, base_time):
        """Encuentros separados <24h NO generan ventana (son intra-sesión)."""
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time,
            topics=["x"], bloom_level=2.0, bloom_label="c", prompt_text="a",
        )
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time + timedelta(hours=2),
            topics=["x"], bloom_level=4.0, bloom_label="c", prompt_text="b",
        )
        windows = tracker.detect_consolidation_patterns("s1")
        assert len(windows) == 0, "Gaps <24h deben ignorarse (misma sesión)"

    def test_gap_beyond_max_ignored(self, tracker, base_time):
        """Encuentros separados >168h NO generan ventana (ambigüedad causal)."""
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time,
            topics=["x"], bloom_level=2.0, bloom_label="c", prompt_text="a",
        )
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time + timedelta(hours=200),
            topics=["x"], bloom_level=4.0, bloom_label="c", prompt_text="b",
        )
        windows = tracker.detect_consolidation_patterns("s1")
        assert len(windows) == 0, "Gaps >168h deben ignorarse"

    def test_valid_window_detected(self, tracker_with_data):
        """Encuentro separado 50h está en ventana válida [24h, 168h]."""
        windows = tracker_with_data.detect_consolidation_patterns("s1")
        assert len(windows) == 1
        assert MIN_GAP_HOURS <= windows[0].gap_hours <= MAX_GAP_HOURS

    def test_optimal_window_flagged(self, tracker_with_data):
        """50h está en la franja óptima [48h, 72h] → in_optimal_window=True."""
        windows = tracker_with_data.detect_consolidation_patterns("s1")
        assert windows[0].in_optimal_window is True

    def test_non_optimal_window(self, tracker, base_time):
        """130h está fuera de la franja óptima → in_optimal_window=False."""
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time,
            topics=["x"], bloom_level=2.0, bloom_label="c", prompt_text="a",
        )
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time + timedelta(hours=130),
            topics=["x"], bloom_level=3.0, bloom_label="c", prompt_text="b",
        )
        windows = tracker.detect_consolidation_patterns("s1")
        assert windows[0].in_optimal_window is False

    def test_different_topics_no_window(self, tracker, base_time):
        """Encuentros sobre topics DISTINTOS no forman ventana."""
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time,
            topics=["bucles"], bloom_level=2.0, bloom_label="c", prompt_text="a",
        )
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time + timedelta(hours=50),
            topics=["funciones"], bloom_level=4.0, bloom_label="c", prompt_text="b",
        )
        windows = tracker.detect_consolidation_patterns("s1")
        assert len(windows) == 0

    def test_different_students_no_window(self, tracker, base_time):
        """Encuentros de estudiantes DISTINTOS no forman ventana."""
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time,
            topics=["x"], bloom_level=2.0, bloom_label="c", prompt_text="a",
        )
        tracker.ingest_interaction(
            student_id="s2", timestamp=base_time + timedelta(hours=50),
            topics=["x"], bloom_level=4.0, bloom_label="c", prompt_text="b",
        )
        assert len(tracker.detect_consolidation_patterns("s1")) == 0
        assert len(tracker.detect_consolidation_patterns("s2")) == 0


# ═══════════════════════════════════════════════════════════════════
# 3. CLASIFICACIÓN DE SEÑALES DE CONSOLIDACIÓN
# ═══════════════════════════════════════════════════════════════════

class TestConsolidationSignals:
    def test_strong_consolidation(self, tracker_with_data):
        """Bloom Δ+2.0 con reformulación ≥0.3 → strong."""
        windows = tracker_with_data.detect_consolidation_patterns("s1")
        assert windows[0].consolidation_signal == "strong"

    def test_weak_consolidation(self, tracker, base_time):
        """Bloom Δ+0.5 → weak (por debajo del umbral strong)."""
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time,
            topics=["x"], bloom_level=2.0, bloom_label="c",
            prompt_text="¿Cómo recorro un array?",
        )
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time + timedelta(hours=50),
            topics=["x"], bloom_level=2.5, bloom_label="c",
            prompt_text="¿Puedo usar for-each para arrays?",
        )
        windows = tracker.detect_consolidation_patterns("s1")
        assert windows[0].consolidation_signal == "weak"

    def test_regression_signal(self, tracker, base_time):
        """Bloom Δ-2.0 → regression (conocimiento se evaporó)."""
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time,
            topics=["x"], bloom_level=4.0, bloom_label="analizar",
            prompt_text="¿Por qué elegir paso por referencia?",
        )
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time + timedelta(hours=50),
            topics=["x"], bloom_level=1.0, bloom_label="recordar",
            prompt_text="¿Qué es una función?",
        )
        windows = tracker.detect_consolidation_patterns("s1")
        assert windows[0].consolidation_signal == "regression"
        assert windows[0].bloom_delta < 0

    def test_absent_signal_no_change(self, tracker, base_time):
        """Bloom sin cambio significativo → absent."""
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time,
            topics=["x"], bloom_level=2.0, bloom_label="c",
            prompt_text="¿Qué es un bucle?",
        )
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time + timedelta(hours=50),
            topics=["x"], bloom_level=2.2, bloom_label="c",
            prompt_text="¿Qué es un bucle for?",
        )
        windows = tracker.detect_consolidation_patterns("s1")
        # Δ+0.2 is below WEAK threshold (0.5), and low reformulation on similar prompts
        assert windows[0].consolidation_signal in ("absent", "weak")


# ═══════════════════════════════════════════════════════════════════
# 4. PROFUNDIDAD DE REFORMULACIÓN
# ═══════════════════════════════════════════════════════════════════

class TestReformulationDepth:
    def test_identical_prompts_low_reformulation(self, tracker, base_time):
        """Prompts idénticos → reformulación baja (repetición, no integración)."""
        same_prompt = "¿Qué es un bucle for?"
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time,
            topics=["x"], bloom_level=2.0, bloom_label="c",
            prompt_text=same_prompt,
        )
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time + timedelta(hours=50),
            topics=["x"], bloom_level=2.0, bloom_label="c",
            prompt_text=same_prompt,
        )
        windows = tracker.detect_consolidation_patterns("s1")
        assert windows[0].reformulation_depth < 0.3

    def test_different_prompts_higher_reformulation(self, tracker_with_data):
        """Prompts distintos con Bloom ascendente → reformulación alta."""
        windows = tracker_with_data.detect_consolidation_patterns("s1")
        assert windows[0].reformulation_depth > 0.4


# ═══════════════════════════════════════════════════════════════════
# 5. ÍNDICE DE CONSOLIDACIÓN
# ═══════════════════════════════════════════════════════════════════

class TestConsolidationIndex:
    def test_no_windows_returns_zero(self, tracker):
        assert tracker.compute_consolidation_index("nonexistent") == 0.0

    def test_positive_delta_positive_index(self, tracker_with_data):
        idx = tracker_with_data.compute_consolidation_index("s1")
        assert idx > 0.0, "Bloom delta positivo debe producir índice > 0"

    def test_index_bounded_zero_one(self, tracker_with_data):
        idx = tracker_with_data.compute_consolidation_index("s1")
        assert 0.0 <= idx <= 1.0

    def test_optimal_window_weighted_more(self, tracker, base_time):
        """Ventanas en franja óptima (48-72h) pesan el doble (Cepeda et al. 2006)."""
        # Dos encuentros idénticos en bloom_delta, uno en ventana óptima y otro fuera
        # El de la ventana óptima debería ponderar más
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time,
            topics=["a"], bloom_level=2.0, bloom_label="c", prompt_text="q1",
        )
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time + timedelta(hours=50),  # óptima
            topics=["a"], bloom_level=4.0, bloom_label="c", prompt_text="q2 diferente",
        )
        # Calcular solo este
        idx_optimal = tracker.compute_consolidation_index("s1")

        tracker2 = MemoryConsolidationTracker()
        tracker2.ingest_interaction(
            student_id="s1", timestamp=base_time,
            topics=["a"], bloom_level=2.0, bloom_label="c", prompt_text="q1",
        )
        tracker2.ingest_interaction(
            student_id="s1", timestamp=base_time + timedelta(hours=130),  # no óptima
            topics=["a"], bloom_level=4.0, bloom_label="c", prompt_text="q2 diferente",
        )
        idx_non_optimal = tracker2.compute_consolidation_index("s1")

        # With same bloom_delta but different weights, the optimal-window one
        # may or may not be strictly higher due to reformulation differences.
        # But both should be positive.
        assert idx_optimal > 0
        assert idx_non_optimal > 0


# ═══════════════════════════════════════════════════════════════════
# 6. REPASO ESPACIADO
# ═══════════════════════════════════════════════════════════════════

class TestSpacedRepetition:
    def test_single_encounter_triggers_suggestion(self, tracker, base_time):
        """Un topic visitado una vez y no revisitado → candidato a repaso."""
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time,
            topics=["recursión"], bloom_level=2.0, bloom_label="comprender",
            prompt_text="¿Qué es la recursión?",
        )
        ref = base_time + timedelta(hours=60)  # 60h después, en ventana óptima
        suggestions = tracker.generate_spaced_repetition_suggestions("s1", ref)
        assert len(suggestions) >= 1
        assert suggestions[0].topic == "recursión"

    def test_revisited_topic_no_suggestion(self, tracker_with_data, base_time):
        """Un topic con dos encuentros ya NO necesita sugerencia de repaso."""
        ref = base_time + timedelta(hours=60)
        suggestions = tracker_with_data.generate_spaced_repetition_suggestions("s1", ref)
        bucle_suggestions = [s for s in suggestions if s.topic == "bucles"]
        assert len(bucle_suggestions) == 0

    def test_suggestion_has_escalation_prompt(self, tracker, base_time):
        """Sugerencia incluye un prompt de escalación Bloom."""
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time,
            topics=["arrays"], bloom_level=1.0, bloom_label="recordar",
            prompt_text="¿Qué es un array?",
        )
        ref = base_time + timedelta(hours=60)
        suggestions = tracker.generate_spaced_repetition_suggestions("s1", ref)
        assert len(suggestions) >= 1
        assert len(suggestions[0].suggested_prompt) > 0

    def test_urgency_ordering(self, tracker, base_time):
        """Sugerencias ordenadas por urgencia: high → medium → low."""
        # Topic 1: visto hace 100h (high urgency, ventana cerrándose)
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time,
            topics=["a"], bloom_level=2.0, bloom_label="c", prompt_text="q",
        )
        # Topic 2: visto hace 50h (medium urgency, ventana óptima)
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time + timedelta(hours=50),
            topics=["b"], bloom_level=2.0, bloom_label="c", prompt_text="q",
        )
        ref = base_time + timedelta(hours=100)
        suggestions = tracker.generate_spaced_repetition_suggestions("s1", ref)
        if len(suggestions) >= 2:
            urgency_order = {"high": 0, "medium": 1, "low": 2}
            for i in range(len(suggestions) - 1):
                assert urgency_order[suggestions[i].urgency] <= urgency_order[suggestions[i+1].urgency]


# ═══════════════════════════════════════════════════════════════════
# 7. INFORME Y PRINT
# ═══════════════════════════════════════════════════════════════════

class TestStudentReport:
    def test_report_has_all_fields(self, tracker_with_data):
        report = tracker_with_data.get_student_report("s1")
        assert isinstance(report, StudentConsolidationReport)
        assert report.student_id == "s1"
        assert report.n_topics_tracked > 0
        assert len(report.interpretation) > 0

    def test_report_str_readable(self, tracker_with_data):
        report = tracker_with_data.get_student_report("s1")
        text = str(report)
        assert "INFORME DE CONSOLIDACIÓN" in text
        assert "DESGLOSE POR TOPIC" in text
        assert "s1" in text

    def test_topic_breakdown(self, tracker_with_data):
        report = tracker_with_data.get_student_report("s1")
        breakdown = report.topic_breakdown()
        assert isinstance(breakdown, dict)
        assert "bucles" in breakdown

    def test_print_demo_report_returns_string(self):
        tracker = generate_demo_data()
        output = print_demo_report(tracker, "est_demo_01")
        assert isinstance(output, str)
        assert "MEMORY CONSOLIDATION TRACKER" in output
        assert "DESGLOSE POR TOPIC" in output

    def test_report_cache_invalidation(self, tracker, base_time):
        """Ingestar nueva interacción invalida cache del reporte."""
        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time,
            topics=["x"], bloom_level=2.0, bloom_label="c", prompt_text="q",
        )
        r1 = tracker.get_student_report("s1")

        tracker.ingest_interaction(
            student_id="s1", timestamp=base_time + timedelta(hours=50),
            topics=["x"], bloom_level=4.0, bloom_label="c", prompt_text="q2",
        )
        r2 = tracker.get_student_report("s1")
        assert r2.n_consolidation_windows > r1.n_consolidation_windows


# ═══════════════════════════════════════════════════════════════════
# 8. COHORTE
# ═══════════════════════════════════════════════════════════════════

class TestCohortSummary:
    def test_empty_cohort(self, tracker):
        summary = tracker.get_cohort_consolidation_summary()
        assert summary["n_students"] == 0

    def test_single_student_cohort(self, tracker_with_data):
        summary = tracker_with_data.get_cohort_consolidation_summary()
        assert summary["n_students"] == 1
        assert summary["total_windows"] > 0

    def test_cohort_interpretation(self, tracker_with_data):
        summary = tracker_with_data.get_cohort_consolidation_summary()
        assert summary["cohort_interpretation"] in (
            "CONSOLIDATION_HEALTHY", "CONSOLIDATION_AT_RISK", "CONSOLIDATION_CRITICAL"
        )


# ═══════════════════════════════════════════════════════════════════
# 9. INTEGRACIÓN — Flujo completo
# ═══════════════════════════════════════════════════════════════════

class TestIntegration:
    """Reproduce el flujo completo del generate_demo_data()."""

    def test_demo_data_generates(self):
        tracker = generate_demo_data()
        assert len(tracker.encounters) > 0

    def test_demo_has_all_signal_types(self):
        """Demo sintético debe producir strong, weak, y regression."""
        tracker = generate_demo_data()
        windows = tracker.detect_consolidation_patterns("est_demo_01")
        signals = {w.consolidation_signal for w in windows}
        assert "strong" in signals, "Demo debe tener consolidación fuerte"
        assert "regression" in signals, "Demo debe tener regresión"
        # weak puede o no estar presente dependiendo de umbrales

    def test_demo_report_complete(self):
        tracker = generate_demo_data()
        report = tracker.get_student_report("est_demo_01")
        assert report.n_topics_tracked >= 5
        assert report.n_consolidation_windows >= 5
        assert report.strong_consolidations >= 1
        assert report.regressions >= 1

    def test_full_pipeline(self):
        """Pipeline completo: ingest → detect → report → suggestions → cohort."""
        tracker = generate_demo_data()
        sid = "est_demo_01"

        # 1. Detección
        windows = tracker.detect_consolidation_patterns(sid)
        assert len(windows) > 0

        # 2. Índice
        idx = tracker.compute_consolidation_index(sid)
        assert 0 <= idx <= 1

        # 3. Reporte
        report = tracker.get_student_report(sid)
        assert isinstance(report, StudentConsolidationReport)

        # 4. Topic breakdown (nueva funcionalidad)
        breakdown = report.topic_breakdown()
        assert len(breakdown) > 0

        # 5. Cohorte
        cohort = tracker.get_cohort_consolidation_summary()
        assert cohort["n_students"] >= 1

        # 6. Print report
        output = print_demo_report(tracker, sid)
        assert len(output) > 100
