"""
TEST SUITE — effect_latency_analyzer.py
═══════════════════════════════════════════
Tests de invariantes teóricos y funcionales para el módulo de
análisis de latencia de efecto de configuraciones pedagógicas.

Estructura:
  1. TestDataclasses — EffectLatencyResult y MultiMetricLatency
  2. TestEffectCurve — Extracción de curvas y datos sintéticos
  3. TestOnsetDetection — Detección del punto de inicio del efecto
  4. TestStabilization — Detección del plateau de estabilización
  5. TestEffectTypes — Clasificación immediate/gradual/delayed/null
  6. TestCognitiveDepth — Inferencia de profundidad cognitiva
  7. TestMultiMetric — Análisis multi-métrica y asimetría
  8. TestPublicAccessors — get_event_ids() y get_event_metrics()
  9. TestIntegration — Flujo completo con demo data

Convenciones: cada test documenta el invariante teórico que verifica.
Autor: Diego Elvira Vásquez · Feb 2026
"""

import pytest
from effect_latency_analyzer import (
    EffectLatencyAnalyzer,
    EffectLatencyResult,
    MultiMetricLatency,
)


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def analyzer():
    return EffectLatencyAnalyzer()


@pytest.fixture
def analyzer_with_data():
    a = EffectLatencyAnalyzer()
    a.generate_demo_data()
    return a


@pytest.fixture
def analyzer_custom():
    """Analyzer con datos sintéticos controlados para tests deterministas."""
    a = EffectLatencyAnalyzer()
    # Efecto inmediato: salto brusco en interacción 1
    a.register_synthetic_data(
        "evt_test_immediate", "test_config", "Test Inmediato",
        "test_metric", [0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    )
    # Efecto gradual: onset en interacción 3
    a.register_synthetic_data(
        "evt_test_gradual", "test_config2", "Test Gradual",
        "test_metric", [0.0, 0.05, 0.1, 0.3, 0.4, 0.4, 0.4, 0.4]
    )
    # Efecto retardado: onset en interacción 8+
    a.register_synthetic_data(
        "evt_test_delayed", "test_config3", "Test Retardado",
        "test_metric", [0.0, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.5, 0.6, 0.6, 0.6, 0.6]
    )
    # Sin efecto: variación pura por debajo del umbral
    a.register_synthetic_data(
        "evt_test_null", "test_config4", "Test Nulo",
        "test_metric", [0.0, 0.01, -0.02, 0.01, 0.0, -0.01, 0.02, 0.01]
    )
    return a


# ═══════════════════════════════════════════════════════════════════
# 1. DATACLASSES
# ═══════════════════════════════════════════════════════════════════

class TestDataclasses:
    def test_effect_latency_result_defaults(self):
        r = EffectLatencyResult()
        assert r.onset_n == -1
        assert r.effect_type == "unknown"
        assert r.sufficient_data is False

    def test_multi_metric_latency_defaults(self):
        m = MultiMetricLatency()
        assert m.dominant_metric == ""
        assert m.latency_spread == 0


# ═══════════════════════════════════════════════════════════════════
# 2. EXTRACCIÓN DE DATOS
# ═══════════════════════════════════════════════════════════════════

class TestEffectCurve:
    def test_nonexistent_event_returns_null(self, analyzer):
        result = analyzer.compute_effect_latency("nonexistent", "bloom_mean")
        assert result.effect_type == "null"
        assert result.sufficient_data is False

    def test_insufficient_data_flagged(self, analyzer):
        """Menos de 3 datapoints → insufficient data."""
        analyzer.register_synthetic_data(
            "short", "cfg", "Short", "m", [0.1, 0.2]
        )
        result = analyzer.compute_effect_latency("short", "m")
        assert result.sufficient_data is False
        assert "insuficientes" in result.cognitive_depth_inference.lower()


# ═══════════════════════════════════════════════════════════════════
# 3. DETECCIÓN DE ONSET
# ═══════════════════════════════════════════════════════════════════

class TestOnsetDetection:
    def test_immediate_onset(self, analyzer_custom):
        """Salto brusco en interacción 1 → onset_n=1."""
        r = analyzer_custom.compute_effect_latency("evt_test_immediate", "test_metric")
        assert r.onset_n == 1, "Salto de 0.5 en posición 1 debe detectar onset_n=1"

    def test_gradual_onset(self, analyzer_custom):
        """Efecto gradual: onset entre 2-4."""
        r = analyzer_custom.compute_effect_latency("evt_test_gradual", "test_metric")
        assert 2 <= r.onset_n <= 4, f"Onset gradual esperado 2-4, got {r.onset_n}"

    def test_delayed_onset(self, analyzer_custom):
        """Efecto retardado: onset >= 5."""
        r = analyzer_custom.compute_effect_latency("evt_test_delayed", "test_metric")
        assert r.onset_n >= 5, f"Onset retardado esperado >=5, got {r.onset_n}"

    def test_null_no_onset(self, analyzer_custom):
        """Sin efecto significativo → onset_n=-1."""
        r = analyzer_custom.compute_effect_latency("evt_test_null", "test_metric")
        assert r.onset_n == -1 or r.effect_type == "null"


# ═══════════════════════════════════════════════════════════════════
# 4. ESTABILIZACIÓN
# ═══════════════════════════════════════════════════════════════════

class TestStabilization:
    def test_immediate_stabilizes_quickly(self, analyzer_custom):
        """Efecto inmediato con plateau constante → estabilización temprana."""
        r = analyzer_custom.compute_effect_latency("evt_test_immediate", "test_metric")
        if r.stabilization_n > 0:
            assert r.stabilization_n <= 5

    def test_onset_before_stabilization(self, analyzer_custom):
        """Invariante: onset siempre precede estabilización."""
        r = analyzer_custom.compute_effect_latency("evt_test_gradual", "test_metric")
        if r.onset_n > 0 and r.stabilization_n > 0:
            assert r.onset_n <= r.stabilization_n


# ═══════════════════════════════════════════════════════════════════
# 5. CLASIFICACIÓN DE TIPOS DE EFECTO
# ═══════════════════════════════════════════════════════════════════

class TestEffectTypes:
    def test_immediate_classification(self, analyzer_custom):
        """onset ≤ 1 → immediate."""
        r = analyzer_custom.compute_effect_latency("evt_test_immediate", "test_metric")
        assert r.effect_type == "immediate"

    def test_gradual_classification(self, analyzer_custom):
        """onset 2-4 → gradual."""
        r = analyzer_custom.compute_effect_latency("evt_test_gradual", "test_metric")
        assert r.effect_type == "gradual"

    def test_delayed_classification(self, analyzer_custom):
        """onset ≥ 5 → delayed."""
        r = analyzer_custom.compute_effect_latency("evt_test_delayed", "test_metric")
        assert r.effect_type == "delayed"

    def test_null_classification(self, analyzer_custom):
        """Sin onset → null."""
        r = analyzer_custom.compute_effect_latency("evt_test_null", "test_metric")
        assert r.effect_type == "null"

    def test_effect_types_exhaustive(self):
        """Los 4 tipos cubren todo el espacio."""
        valid_types = {"immediate", "gradual", "delayed", "null", "unknown"}
        a = EffectLatencyAnalyzer()
        # All classifications should fall in valid set
        for onset in [-1, 0, 1, 2, 3, 4, 5, 10]:
            t = a._classify_effect_type(onset, onset + 2)
            assert t in valid_types


# ═══════════════════════════════════════════════════════════════════
# 6. INFERENCIA DE PROFUNDIDAD COGNITIVA
# ═══════════════════════════════════════════════════════════════════

class TestCognitiveDepth:
    def test_immediate_cites_vygotsky(self, analyzer_custom):
        """Efecto inmediato → cita regulación externa (Vygotsky)."""
        r = analyzer_custom.compute_effect_latency("evt_test_immediate", "test_metric")
        assert "Vygotsky" in r.cognitive_depth_inference or "regulación" in r.cognitive_depth_inference.lower()

    def test_gradual_cites_kapur(self, analyzer_custom):
        """Efecto gradual → cita productive failure (Kapur)."""
        r = analyzer_custom.compute_effect_latency("evt_test_gradual", "test_metric")
        assert "Kapur" in r.cognitive_depth_inference or "gradual" in r.cognitive_depth_inference.lower()

    def test_delayed_cites_perry_or_deep(self, analyzer_custom):
        """Efecto retardado → cita reestructuración profunda (Perry/Bjork)."""
        r = analyzer_custom.compute_effect_latency("evt_test_delayed", "test_metric")
        depth_text = r.cognitive_depth_inference.lower()
        assert any(w in depth_text for w in ["perry", "profund", "reestructur", "deep", "bjork"])

    def test_null_mentions_hypothesis(self, analyzer_custom):
        """Sin efecto → genera hipótesis explicativas."""
        r = analyzer_custom.compute_effect_latency("evt_test_null", "test_metric")
        assert "hipótesis" in r.cognitive_depth_inference.lower() or "sin efecto" in r.cognitive_depth_inference.lower()

    def test_inference_never_empty(self, analyzer_with_data):
        """Toda inferencia debe producir texto no vacío."""
        for eid in analyzer_with_data.get_event_ids():
            r = analyzer_with_data.compute_effect_latency(eid, "bloom_mean")
            assert len(r.cognitive_depth_inference) > 0


# ═══════════════════════════════════════════════════════════════════
# 7. MULTI-MÉTRICA
# ═══════════════════════════════════════════════════════════════════

class TestMultiMetric:
    def test_multi_metric_returns_all_metrics(self, analyzer_with_data):
        """compute_multi_metric_latency computa todas las métricas default."""
        eid = analyzer_with_data.get_event_ids()[0]
        multi = analyzer_with_data.compute_multi_metric_latency(eid)
        assert len(multi.results) >= 3

    def test_dominant_is_fastest(self, analyzer_with_data):
        """dominant_metric tiene el onset_n más bajo."""
        eid = analyzer_with_data.get_event_ids()[0]
        multi = analyzer_with_data.compute_multi_metric_latency(eid)
        if multi.dominant_metric and multi.slowest_metric:
            dom = multi.results[multi.dominant_metric]
            slow = multi.results[multi.slowest_metric]
            if dom.onset_n > 0 and slow.onset_n > 0:
                assert dom.onset_n <= slow.onset_n

    def test_latency_spread_non_negative(self, analyzer_with_data):
        """spread = slowest.onset - dominant.onset ≥ 0."""
        eid = analyzer_with_data.get_event_ids()[0]
        multi = analyzer_with_data.compute_multi_metric_latency(eid)
        assert multi.latency_spread >= 0

    def test_summary_type_valid(self, analyzer_with_data):
        """summary_type ∈ {no_effect, mechanical_restriction, deep_restructuring, gradual_adaptation}."""
        valid = {"no_effect", "mechanical_restriction", "deep_restructuring", "gradual_adaptation"}
        for eid in analyzer_with_data.get_event_ids():
            multi = analyzer_with_data.compute_multi_metric_latency(eid)
            assert multi.summary_type in valid

    def test_socratic_asymmetry(self, analyzer_with_data):
        """
        Invariante teórico central: modo socrático produce asimetría
        entre métricas — efecto inmediato en latencia, gradual en Bloom,
        retardado en autonomía. La asimetría entre métricas es la señal
        de que el efecto es cognitivo, no mecánico.
        """
        multi = analyzer_with_data.compute_multi_metric_latency("evt_socratic")
        if multi.dominant_metric and multi.slowest_metric:
            assert multi.latency_spread > 0, (
                "Modo socrático debe producir asimetría entre métricas"
            )


# ═══════════════════════════════════════════════════════════════════
# 8. PUBLIC ACCESSORS (nueva mejora)
# ═══════════════════════════════════════════════════════════════════

class TestPublicAccessors:
    def test_get_event_ids_empty(self, analyzer):
        assert analyzer.get_event_ids() == []

    def test_get_event_ids_after_demo(self, analyzer_with_data):
        ids = analyzer_with_data.get_event_ids()
        assert len(ids) >= 3
        assert "evt_socratic" in ids

    def test_get_event_metrics(self, analyzer_with_data):
        metrics = analyzer_with_data.get_event_metrics("evt_socratic")
        assert "bloom_mean" in metrics
        assert "autonomy_score" in metrics

    def test_get_event_metrics_nonexistent(self, analyzer):
        assert analyzer.get_event_metrics("fake") == []

    def test_no_internal_keys_exposed(self, analyzer_with_data):
        """get_event_ids() no expone claves internas como _config_key."""
        ids = analyzer_with_data.get_event_ids()
        for eid in ids:
            assert not eid.startswith("_")

    def test_no_internal_metrics_exposed(self, analyzer_with_data):
        """get_event_metrics() no expone claves internas."""
        for eid in analyzer_with_data.get_event_ids():
            metrics = analyzer_with_data.get_event_metrics(eid)
            for m in metrics:
                assert not m.startswith("_")


# ═══════════════════════════════════════════════════════════════════
# 9. INTEGRACIÓN
# ═══════════════════════════════════════════════════════════════════

class TestIntegration:
    def test_generate_demo_data(self, analyzer):
        results = analyzer.generate_demo_data()
        assert len(results) >= 3
        assert all(isinstance(v, MultiMetricLatency) for v in results.values())

    def test_full_pipeline(self, analyzer_with_data):
        """Pipeline: events → single metric → multi metric → inference."""
        for eid in analyzer_with_data.get_event_ids():
            # Single metric
            r = analyzer_with_data.compute_effect_latency(eid, "bloom_mean")
            assert isinstance(r, EffectLatencyResult)

            # Multi metric
            multi = analyzer_with_data.compute_multi_metric_latency(eid)
            assert isinstance(multi, MultiMetricLatency)
            assert len(multi.results) > 0

    def test_three_demo_events_different_profiles(self, analyzer_with_data):
        """Los 3 eventos demo deben tener perfiles de latencia distinguibles."""
        ids = analyzer_with_data.get_event_ids()
        profiles = {}
        for eid in ids:
            multi = analyzer_with_data.compute_multi_metric_latency(eid)
            profiles[eid] = multi.summary_type
        # At least 2 different summary types among the 3 events
        assert len(set(profiles.values())) >= 2, (
            f"Demo debe producir perfiles distinguibles, got: {profiles}"
        )
