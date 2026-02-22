"""
TEST SUITE — config_interaction_analyzer.py
════════════════════════════════════════════════
Tests de invariantes teóricos y funcionales para el módulo de
análisis de interacciones entre configuraciones pedagógicas.

Estructura:
  1. TestDataclasses — ConfigCombination, ConfigInteractionResult, InteractionMatrix
  2. TestRecording — Ingesta de observaciones y efectos individuales
  3. TestInteractionTypes — Clasificación synergistic/antagonistic/additive/emergent
  4. TestInteractionRatio — Invariantes del ratio actual/expected
  5. TestMatrix — Generación y acceso a la matriz de interacciones
  6. TestInterpretation — Interpretaciones narrativas y teóricas
  7. TestDemoData — Datos sintéticos canónicos
  8. TestIntegration — Flujo completo

Convenciones: cada test documenta el invariante teórico que verifica.
Autor: Diego Elvira Vásquez · Feb 2026
"""

import pytest
from config_interaction_analyzer import (
    ConfigInteractionAnalyzer,
    ConfigCombination,
    ConfigInteractionResult,
    InteractionMatrix,
)


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def analyzer():
    return ConfigInteractionAnalyzer()


@pytest.fixture
def analyzer_with_baseline(analyzer):
    """Analyzer con baseline y efectos individuales configurados."""
    analyzer.set_baseline({"bloom_mean": 2.0, "autonomy_score": 0.25})
    analyzer.set_individual_effect(
        ("socratic_scaffolding", True),
        {"bloom_mean": 0.8, "autonomy_score": 0.15}
    )
    analyzer.set_individual_effect(
        ("max_daily_prompts", 8),
        {"bloom_mean": 0.3, "autonomy_score": 0.08}
    )
    analyzer.set_individual_effect(
        ("hallucination_rate", 0.15),
        {"bloom_mean": 0.5, "autonomy_score": 0.10}
    )
    return analyzer


@pytest.fixture
def analyzer_with_demo():
    a = ConfigInteractionAnalyzer()
    a.generate_demo_data()
    return a


# ═══════════════════════════════════════════════════════════════════
# 1. DATACLASSES
# ═══════════════════════════════════════════════════════════════════

class TestDataclasses:
    def test_config_combination_label(self):
        combo = ConfigCombination(
            active_configs=frozenset([("socratic", True), ("limit", 8)])
        )
        label = combo.config_label()
        assert "socratic" in label
        assert "limit" in label

    def test_config_interaction_result_labels(self):
        r = ConfigInteractionResult(
            config_a=("socratic", True),
            config_b=("limit", 8),
            metric="bloom_mean",
        )
        assert "socratic" in r.label_a()
        assert "limit" in r.label_b()

    def test_interaction_matrix_get(self):
        m = InteractionMatrix(config_names=["A", "B"])
        result = ConfigInteractionResult(
            config_a=("x", 1), config_b=("y", 2), metric="m",
        )
        m.matrix[("A", "B")] = result
        assert m.get("A", "B") is not None
        # Bidirectional access
        assert m.get("B", "A") is not None

    def test_interaction_matrix_get_missing(self):
        m = InteractionMatrix(config_names=["A", "B"])
        assert m.get("A", "C") is None


# ═══════════════════════════════════════════════════════════════════
# 2. RECORDING
# ═══════════════════════════════════════════════════════════════════

class TestRecording:
    def test_set_baseline(self, analyzer):
        analyzer.set_baseline({"bloom_mean": 2.0})
        assert analyzer._baseline_metrics["bloom_mean"] == 2.0

    def test_set_individual_effect(self, analyzer):
        analyzer.set_individual_effect(
            ("socratic", True), {"bloom_mean": 0.8}
        )
        key = ("socratic", True)
        assert key in analyzer._individual_effects
        assert analyzer._individual_effects[key]["bloom_mean"] == 0.8

    def test_record_observation(self, analyzer):
        analyzer.record_observation(
            {"socratic": True, "limit": 8},
            {"bloom_mean": 3.0}
        )
        assert len(analyzer.combinations) > 0

    def test_observation_count_increments(self, analyzer):
        config = {"socratic": True}
        for _ in range(5):
            analyzer.record_observation(config, {"bloom_mean": 3.0})
        key = frozenset(config.items())
        assert analyzer.combinations[key].interaction_count == 5

    def test_metric_values_stored(self, analyzer):
        config = {"socratic": True}
        analyzer.record_observation(config, {"bloom_mean": 3.0})
        analyzer.record_observation(config, {"bloom_mean": 3.5})
        key = frozenset(config.items())
        vals = analyzer.combinations[key].metrics_observed["bloom_mean"]
        assert len(vals) == 2
        assert vals[0] == 3.0


# ═══════════════════════════════════════════════════════════════════
# 3. TIPOS DE INTERACCIÓN
# ═══════════════════════════════════════════════════════════════════

class TestInteractionTypes:
    def test_synergistic_detected(self, analyzer_with_demo):
        """
        Socrático × alucinación → SINÉRGICO (ratio > 1.2).
        Piensa + verifica = engagement más profundo que la suma de las partes.
        """
        results = analyzer_with_demo.compute_interaction_effects("autonomy_score")
        synergistic = [r for r in results if r.interaction_type == "synergistic"]
        assert len(synergistic) > 0, "Demo debe producir al menos una interacción sinérgica"

    def test_antagonistic_detected(self, analyzer_with_demo):
        """
        Socrático × límite bajo → ANTAGÓNICO (ratio < 0.8).
        El socrático genera ida-y-vuelta, el límite corta el diálogo.
        """
        results = analyzer_with_demo.compute_interaction_effects("bloom_mean")
        antagonistic = [r for r in results if r.interaction_type in ("antagonistic", "emergent")]
        assert len(antagonistic) > 0, "Demo debe producir interacción antagónica o emergente"

    def test_additive_detected(self, analyzer_with_demo):
        """
        Bloqueo × RAG → ADITIVO (ratio ≈ 1.0).
        Independientes, sin interferencia.
        """
        results = analyzer_with_demo.compute_interaction_effects("bloom_mean")
        additive = [r for r in results if r.interaction_type == "additive"]
        assert len(additive) > 0, "Demo debe producir interacción aditiva"

    def test_interaction_types_exhaustive(self, analyzer_with_demo):
        """Todos los resultados tienen tipo válido."""
        valid_types = {"synergistic", "antagonistic", "additive", "emergent", "unknown"}
        results = analyzer_with_demo.compute_interaction_effects("bloom_mean")
        for r in results:
            assert r.interaction_type in valid_types


# ═══════════════════════════════════════════════════════════════════
# 4. INTERACTION RATIO
# ═══════════════════════════════════════════════════════════════════

class TestInteractionRatio:
    def test_synergistic_ratio_above_threshold(self, analyzer_with_demo):
        results = analyzer_with_demo.compute_interaction_effects("autonomy_score")
        synergistic = [r for r in results if r.interaction_type == "synergistic"]
        for r in synergistic:
            assert r.interaction_ratio > 1.2

    def test_antagonistic_ratio_below_threshold(self, analyzer_with_demo):
        results = analyzer_with_demo.compute_interaction_effects("bloom_mean")
        antagonistic = [r for r in results if r.interaction_type == "antagonistic"]
        for r in antagonistic:
            assert r.interaction_ratio < 0.8

    def test_additive_ratio_near_one(self, analyzer_with_demo):
        results = analyzer_with_demo.compute_interaction_effects("bloom_mean")
        additive = [r for r in results if r.interaction_type == "additive"]
        for r in additive:
            assert 0.8 <= r.interaction_ratio <= 1.2

    def test_magnitude_non_negative(self, analyzer_with_demo):
        """interaction_magnitude = |actual - expected| ≥ 0."""
        results = analyzer_with_demo.compute_interaction_effects("bloom_mean")
        for r in results:
            assert r.interaction_magnitude >= 0


# ═══════════════════════════════════════════════════════════════════
# 5. MATRIX
# ═══════════════════════════════════════════════════════════════════

class TestMatrix:
    def test_matrix_is_dataclass(self, analyzer_with_demo):
        matrix = analyzer_with_demo.generate_interaction_matrix("bloom_mean")
        assert isinstance(matrix, InteractionMatrix)

    def test_matrix_has_config_names(self, analyzer_with_demo):
        matrix = analyzer_with_demo.generate_interaction_matrix("bloom_mean")
        assert len(matrix.config_names) >= 3

    def test_matrix_bidirectional_access(self, analyzer_with_demo):
        matrix = analyzer_with_demo.generate_interaction_matrix("bloom_mean")
        names = matrix.config_names
        if len(names) >= 2:
            # At least one pair should exist
            found = False
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    if matrix.get(names[i], names[j]) is not None:
                        found = True
                        # Test bidirectionality
                        assert matrix.get(names[j], names[i]) is not None
                        break
            assert found, "Matrix should contain at least one pair"


# ═══════════════════════════════════════════════════════════════════
# 6. INTERPRETATION
# ═══════════════════════════════════════════════════════════════════

class TestInterpretation:
    def test_results_have_interpretation(self, analyzer_with_demo):
        results = analyzer_with_demo.compute_interaction_effects("bloom_mean")
        for r in results:
            if r.sufficient_data:
                assert len(r.interpretation) > 0

    def test_results_have_theoretical_basis(self, analyzer_with_demo):
        results = analyzer_with_demo.compute_interaction_effects("bloom_mean")
        for r in results:
            if r.sufficient_data:
                assert len(r.theoretical_basis) > 0

    def test_results_have_recommendation(self, analyzer_with_demo):
        results = analyzer_with_demo.compute_interaction_effects("bloom_mean")
        for r in results:
            if r.sufficient_data:
                assert len(r.recommendation) > 0

    def test_sufficient_data_flag(self, analyzer_with_demo):
        """sufficient_data True cuando n_observations_ab >= MIN_OBSERVATIONS."""
        results = analyzer_with_demo.compute_interaction_effects("bloom_mean")
        for r in results:
            if r.n_observations_ab >= analyzer_with_demo.MIN_OBSERVATIONS:
                assert r.sufficient_data is True


# ═══════════════════════════════════════════════════════════════════
# 7. DEMO DATA
# ═══════════════════════════════════════════════════════════════════

class TestDemoData:
    def test_demo_generates(self):
        a = ConfigInteractionAnalyzer()
        matrix = a.generate_demo_data()
        assert isinstance(matrix, InteractionMatrix)

    def test_demo_has_multiple_pairs(self, analyzer_with_demo):
        results = analyzer_with_demo.compute_interaction_effects("bloom_mean")
        assert len(results) >= 4, "Demo debe generar ≥4 pares de interacción"

    def test_demo_has_diverse_types(self, analyzer_with_demo):
        """Demo debe producir al menos 2 tipos distintos de interacción."""
        results = analyzer_with_demo.compute_interaction_effects("bloom_mean")
        types = {r.interaction_type for r in results}
        assert len(types) >= 2, f"Expected ≥2 types, got {types}"


# ═══════════════════════════════════════════════════════════════════
# 8. INTEGRACIÓN
# ═══════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_full_pipeline(self):
        """Pipeline: baseline → individual effects → observations → matrix."""
        a = ConfigInteractionAnalyzer()

        # 1. Baseline
        a.set_baseline({"bloom_mean": 2.0})

        # 2. Individual effects
        a.set_individual_effect(("A", True), {"bloom_mean": 0.5})
        a.set_individual_effect(("B", True), {"bloom_mean": 0.3})

        # 3. Observations (synergistic: combined > sum)
        for _ in range(5):
            a.record_observation(
                {"A": True, "B": True},
                {"bloom_mean": 3.2}  # delta = 1.2, expected = 0.8, ratio = 1.5
            )

        # 4. Compute
        results = a.compute_interaction_effects("bloom_mean")
        assert len(results) == 1
        assert results[0].interaction_type == "synergistic"
        assert results[0].interaction_ratio > 1.2

        # 5. Matrix
        matrix = a.generate_interaction_matrix("bloom_mean")
        assert isinstance(matrix, InteractionMatrix)
        assert len(matrix.matrix) == 1

    def test_antagonistic_pipeline(self):
        """Pipeline verificando detección antagónica."""
        a = ConfigInteractionAnalyzer()
        a.set_baseline({"bloom_mean": 2.0})
        a.set_individual_effect(("X", True), {"bloom_mean": 0.8})
        a.set_individual_effect(("Y", True), {"bloom_mean": 0.3})
        # Expected additive: 0.8 + 0.3 = 1.1
        # Actual: 0.2 (much less than expected)
        for _ in range(5):
            a.record_observation(
                {"X": True, "Y": True},
                {"bloom_mean": 2.2}  # delta = 0.2
            )
        results = a.compute_interaction_effects("bloom_mean")
        assert results[0].interaction_type in ("antagonistic", "emergent")
        assert results[0].interaction_ratio < 0.8

    def test_emergent_detection(self):
        """
        Efecto emergente: ambos efectos individuales positivos,
        pero combinados producen efecto NEGATIVO → cambio de signo.
        """
        a = ConfigInteractionAnalyzer()
        a.set_baseline({"bloom_mean": 2.0})
        a.set_individual_effect(("P", True), {"bloom_mean": 0.5})
        a.set_individual_effect(("Q", True), {"bloom_mean": 0.3})
        # Both positive individually, but combined produces NEGATIVE delta
        for _ in range(5):
            a.record_observation(
                {"P": True, "Q": True},
                {"bloom_mean": 1.7}  # delta = -0.3 (sign flip!)
            )
        results = a.compute_interaction_effects("bloom_mean")
        assert results[0].interaction_type == "emergent", (
            "Sign flip (both positive individual, negative combined) → emergent"
        )
