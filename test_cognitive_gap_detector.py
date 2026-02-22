"""
Tests para CognitiveGapDetector — Módulo 7
==========================================
Verifica los 5 componentes:
1. Knowledge Map construction
2. Student Map tracking
3. Gap detection (3 conditions)
4. Epistemic probe generation
5. System prompt injection
"""

import pytest
from cognitive_gap_detector import (
    CognitiveGapDetector,
    KnowledgeMap,
    KnowledgeNode,
    StudentKnowledgeMap,
    StudentTopicState,
    CognitiveGap,
    EpistemicProbe,
    generate_demo_data,
    print_demo_report,
    _generate_fake_rag_chunks,
    PROGRAMMING_PREREQUISITES,
    TOPIC_DIFFICULTY,
)


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def detector():
    """Detector con knowledge map construido."""
    d = CognitiveGapDetector()
    chunks = _generate_fake_rag_chunks()
    d.build_knowledge_map(chunks)
    return d


@pytest.fixture
def demo():
    """Demo completa con estudiante simulado."""
    return generate_demo_data()


# ═══════════════════════════════════════════════════════════════════
# COMPONENTE 1 — Knowledge Map
# ═══════════════════════════════════════════════════════════════════

class TestKnowledgeMap:
    
    def test_builds_all_topic_nodes(self, detector):
        km = detector.knowledge_map
        assert km is not None
        for topic in PROGRAMMING_PREREQUISITES:
            assert topic in km.nodes, f"Topic '{topic}' missing from knowledge map"
    
    def test_prerequisite_edges_correct(self, detector):
        km = detector.knowledge_map
        # variables → bucles should exist
        assert ("variables", "bucles") in km.prerequisite_edges
        # funciones → recursión should exist
        assert ("funciones", "recursión") in km.prerequisite_edges
        # No self-loops
        for src, dst in km.prerequisite_edges:
            assert src != dst
    
    def test_dependent_topics_are_inverse_of_prerequisites(self, detector):
        km = detector.knowledge_map
        # If variables is prerequisite of bucles, then bucles should be dependent of variables
        assert "bucles" in km.nodes["variables"].dependent_topics
        assert "recursión" in km.nodes["funciones"].dependent_topics
    
    def test_cooccurrence_edges_have_weights(self, detector):
        km = detector.knowledge_map
        assert len(km.cooccurrence_edges) > 0
        for t1, t2, w in km.cooccurrence_edges:
            assert 0.0 < w <= 1.0, f"Invalid weight {w} for edge {t1}-{t2}"
    
    def test_chunks_assigned_to_topics(self, detector):
        km = detector.knowledge_map
        # Variables should have chunks (they appear in multiple fake docs)
        assert len(km.nodes["variables"].chunk_indices) > 0
        assert len(km.nodes["bucles"].chunk_indices) > 0
    
    def test_difficulty_estimates_set(self, detector):
        km = detector.knowledge_map
        # Recursión should be hardest
        assert km.nodes["recursión"].difficulty_estimate > km.nodes["variables"].difficulty_estimate
    
    def test_total_chunks_recorded(self, detector):
        km = detector.knowledge_map
        assert km.total_chunks == 13  # 13 fake chunks


# ═══════════════════════════════════════════════════════════════════
# COMPONENTE 2 — Student Map
# ═══════════════════════════════════════════════════════════════════

class TestStudentMap:
    
    def test_record_interaction_creates_map(self, detector):
        detector.record_interaction("s1", ["variables"], bloom_level=3)
        smap = detector.get_student_map("s1")
        assert smap.student_id == "s1"
        assert smap.total_interactions == 1
    
    def test_visited_flag_set(self, detector):
        detector.record_interaction("s1", ["variables"], bloom_level=2)
        smap = detector.get_student_map("s1")
        assert smap.topic_states["variables"].visited is True
        assert smap.topic_states["bucles"].visited is False
    
    def test_bloom_tracking(self, detector):
        detector.record_interaction("s1", ["variables"], bloom_level=2)
        detector.record_interaction("s1", ["variables"], bloom_level=4)
        state = detector.get_student_map("s1").topic_states["variables"]
        assert state.max_bloom_level == 4
        assert state.bloom_levels_seen == [2, 4]
        assert state.visit_count == 2
    
    def test_coverage_ratio(self, detector):
        # Visit 2 out of 9 topics
        detector.record_interaction("s1", ["variables"], bloom_level=2)
        detector.record_interaction("s1", ["bucles"], bloom_level=3)
        smap = detector.get_student_map("s1")
        expected = 2 / len(detector.knowledge_map.nodes)
        assert abs(smap.coverage_ratio - expected) < 0.01
    
    def test_chunk_consumption_tracked(self, detector):
        detector.record_interaction("s1", ["variables"], bloom_level=2, chunk_indices=[0, 1])
        detector.record_interaction("s1", ["variables"], bloom_level=3, chunk_indices=[1, 2])
        state = detector.get_student_map("s1").topic_states["variables"]
        assert state.chunks_consumed == {0, 1, 2}
    
    def test_multiple_topics_per_interaction(self, detector):
        detector.record_interaction("s1", ["variables", "bucles"], bloom_level=3)
        smap = detector.get_student_map("s1")
        assert smap.topic_states["variables"].visited
        assert smap.topic_states["bucles"].visited
        assert smap.total_interactions == 1  # One interaction, two topics


# ═══════════════════════════════════════════════════════════════════
# COMPONENTE 3 — Gap Detection
# ═══════════════════════════════════════════════════════════════════

class TestGapDetection:
    
    def test_prerequisite_gap_detected(self, detector):
        """Student uses recursión without visiting funciones → critical gap."""
        detector.record_interaction("s1", ["recursión"], bloom_level=3)
        gaps = detector.detect_critical_gaps("s1")
        
        prereq_gaps = [g for g in gaps if g.gap_type == "prerequisite_gap" 
                       or "prerequisite_gap" in g.gap_type]
        # recursión depends on funciones and bucles
        gap_topics = [g.topic for g in prereq_gaps]
        assert "funciones" in gap_topics or "bucles" in gap_topics
    
    def test_prerequisite_gap_severity_is_critical(self, detector):
        detector.record_interaction("s1", ["recursión"], bloom_level=3)
        gaps = detector.detect_critical_gaps("s1")
        prereq_gaps = [g for g in gaps if "prerequisite_gap" in g.gap_type]
        for gap in prereq_gaps:
            assert gap.severity == "critical"
    
    def test_adjacency_gap_detected(self, detector):
        """Student visits variables + bucles but not arrays → adjacency gap."""
        for _ in range(3):
            detector.record_interaction("s1", ["variables"], bloom_level=3)
            detector.record_interaction("s1", ["bucles"], bloom_level=3)
        
        gaps = detector.detect_critical_gaps("s1")
        adjacency_gaps = [g for g in gaps if "adjacency_gap" in g.gap_type]
        gap_topics = [g.topic for g in adjacency_gaps]
        assert "arrays" in gap_topics, f"Expected 'arrays' in adjacency gaps, got {gap_topics}"
    
    def test_metacognitive_gap_detected(self, detector):
        """Single visit to hard topic at low Bloom → metacognitive gap."""
        # recursión has difficulty 0.85, visit once at Bloom 2
        detector.record_interaction("s1", ["recursión"], bloom_level=2)
        # Also need to visit prerequisites so recursión isn't just a prereq gap
        detector.record_interaction("s1", ["funciones"], bloom_level=3)
        detector.record_interaction("s1", ["bucles"], bloom_level=3)
        
        gaps = detector.detect_critical_gaps("s1")
        meta_gaps = [g for g in gaps if "metacognitive_gap" in g.gap_type]
        meta_topics = [g.topic for g in meta_gaps]
        assert "recursión" in meta_topics, f"Expected 'recursión' in metacognitive gaps, got {meta_topics}"
    
    def test_no_gap_for_well_covered_topic(self, detector):
        """Visiting a topic 4x at high Bloom should not produce a gap."""
        for _ in range(4):
            detector.record_interaction("s1", ["variables"], bloom_level=4)
        
        gaps = detector.detect_critical_gaps("s1")
        var_gaps = [g for g in gaps if g.topic == "variables"]
        assert len(var_gaps) == 0, "variables should not have a gap after 4 visits at Bloom 4"
    
    def test_gap_consolidation(self, detector):
        """A topic that is both prereq gap and adjacency gap gets consolidated."""
        # Make funciones appear as both prerequisite gap AND adjacency gap
        detector.record_interaction("s1", ["recursión"], bloom_level=3)
        detector.record_interaction("s1", ["variables"], bloom_level=3)
        detector.record_interaction("s1", ["bucles"], bloom_level=3)
        # funciones is prereq of recursión AND adjacent to variables+bucles
        
        gaps = detector.detect_critical_gaps("s1")
        funciones_gaps = [g for g in gaps if g.topic == "funciones"]
        # Should be consolidated into one gap
        assert len(funciones_gaps) <= 1, "funciones should be consolidated"
        if funciones_gaps:
            # Combined type should reflect both
            assert "prerequisite_gap" in funciones_gaps[0].gap_type
    
    def test_gaps_sorted_by_severity(self, detector):
        detector.record_interaction("s1", ["recursión"], bloom_level=2)
        detector.record_interaction("s1", ["variables"], bloom_level=4)
        gaps = detector.detect_critical_gaps("s1")
        
        if len(gaps) >= 2:
            severity_order = {"critical": 0, "moderate": 1, "low": 2}
            for i in range(len(gaps) - 1):
                assert severity_order[gaps[i].severity] <= severity_order[gaps[i+1].severity]


# ═══════════════════════════════════════════════════════════════════
# COMPONENTE 4 — Epistemic Probes
# ═══════════════════════════════════════════════════════════════════

class TestEpistemicProbes:
    
    def test_prerequisite_gap_generates_contrast_probe(self, detector):
        gap = CognitiveGap(
            topic="funciones",
            gap_type="prerequisite_gap",
            severity="critical",
            dependent_topics_at_risk=["recursión"],
            confidence=0.85,
        )
        probes = detector.generate_epistemic_probes(gap)
        probe_types = [p.probe_type for p in probes]
        assert "contrast" in probe_types
    
    def test_adjacency_gap_generates_scaffolded_probe(self, detector):
        gap = CognitiveGap(
            topic="arrays",
            gap_type="adjacency_gap",
            severity="moderate",
            dependent_topics_at_risk=["bucles", "variables"],
            confidence=0.70,
        )
        probes = detector.generate_epistemic_probes(gap)
        probe_types = [p.probe_type for p in probes]
        assert "scaffolded" in probe_types
    
    def test_metacognitive_gap_generates_boundary_probe(self, detector):
        gap = CognitiveGap(
            topic="recursión",
            gap_type="metacognitive_gap",
            severity="low",
            confidence=0.45,
        )
        probes = detector.generate_epistemic_probes(gap)
        probe_types = [p.probe_type for p in probes]
        assert "boundary" in probe_types
    
    def test_probes_have_priority(self, detector):
        gap = CognitiveGap(
            topic="funciones",
            gap_type="prerequisite_gap",
            severity="critical",
            dependent_topics_at_risk=["recursión"],
            confidence=0.85,
        )
        probes = detector.generate_epistemic_probes(gap)
        for probe in probes:
            assert 0.0 < probe.priority <= 1.0
    
    def test_probes_have_firlej_technique(self, detector):
        gap = CognitiveGap(
            topic="arrays",
            gap_type="adjacency_gap",
            severity="moderate",
            confidence=0.70,
        )
        probes = detector.generate_epistemic_probes(gap)
        for probe in probes:
            assert probe.firlej_technique != ""
    
    def test_max_3_probes_per_gap(self, detector):
        gap = CognitiveGap(
            topic="funciones",
            gap_type="prerequisite_gap+adjacency_gap+metacognitive_gap",
            severity="critical",
            dependent_topics_at_risk=["recursión"],
            confidence=0.90,
        )
        probes = detector.generate_epistemic_probes(gap)
        assert len(probes) <= 3
    
    def test_probes_use_topic_specific_templates(self, detector):
        gap = CognitiveGap(
            topic="arrays",
            gap_type="adjacency_gap",
            severity="moderate",
            dependent_topics_at_risk=["bucles", "variables"],
            confidence=0.70,
        )
        probes = detector.generate_epistemic_probes(gap)
        # The arrays adjacency template mentions "100 elementos"
        scaffolded = [p for p in probes if p.probe_type == "scaffolded"]
        if scaffolded:
            assert "100" in scaffolded[0].probe_text or "colecciones" in scaffolded[0].probe_text


# ═══════════════════════════════════════════════════════════════════
# COMPONENTE 5 — System Prompt Injection
# ═══════════════════════════════════════════════════════════════════

class TestInjection:
    
    def test_no_injection_before_min_interactions(self, detector):
        detector.record_interaction("s1", ["recursión"], bloom_level=2)
        # Only 1 interaction, minimum is 3
        injection = detector.get_probe_for_injection("s1")
        assert injection is None
    
    def test_injection_after_min_interactions(self, detector):
        # Record enough interactions and force counter
        detector.record_interaction("s1", ["recursión"], bloom_level=2)
        detector.record_interaction("s1", ["variables"], bloom_level=3)
        detector.record_interaction("s1", ["bucles"], bloom_level=3)
        detector.interaction_counters["s1"] = 5  # Force past threshold
        
        injection = detector.get_probe_for_injection("s1")
        assert injection is not None
        assert "SONDA EPISTÉMICA" in injection
    
    def test_injection_contains_probe_text(self, detector):
        detector.record_interaction("s1", ["recursión"], bloom_level=2)
        detector.interaction_counters["s1"] = 5
        
        injection = detector.get_probe_for_injection("s1")
        if injection:
            assert "introduce naturalmente" in injection
    
    def test_injection_resets_counter(self, detector):
        detector.record_interaction("s1", ["recursión"], bloom_level=2)
        detector.interaction_counters["s1"] = 5
        
        detector.get_probe_for_injection("s1")
        assert detector.interaction_counters["s1"] == 0
    
    def test_system_prompt_addon_returns_empty_string(self, detector):
        """get_system_prompt_addon should return '' not None."""
        result = detector.get_system_prompt_addon("nonexistent")
        assert isinstance(result, str)
    
    def test_probe_history_recorded(self, detector):
        detector.record_interaction("s1", ["recursión"], bloom_level=2)
        detector.interaction_counters["s1"] = 5
        
        detector.get_probe_for_injection("s1")
        history = detector.probe_history.get("s1", [])
        assert len(history) > 0
        assert "gap_topic" in history[0]


# ═══════════════════════════════════════════════════════════════════
# ANALYTICS & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

class TestAnalytics:
    
    def test_gap_summary_structure(self, demo):
        detector, student_id = demo
        summary = detector.get_gap_summary(student_id)
        assert "total_gaps" in summary
        assert "critical_gaps" in summary
        assert "coverage_ratio" in summary
        assert "topics_visited" in summary
        assert "topics_not_visited" in summary
    
    def test_knowledge_graph_data_format(self, detector):
        data = detector.get_knowledge_graph_data()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == len(PROGRAMMING_PREREQUISITES)
    
    def test_student_overlay_data(self, demo):
        detector, student_id = demo
        overlay = detector.get_student_overlay_data(student_id)
        assert overlay["student_id"] == student_id
        assert len(overlay["node_states"]) > 0


# ═══════════════════════════════════════════════════════════════════
# DEMO DATA
# ═══════════════════════════════════════════════════════════════════

class TestDemoData:
    
    def test_demo_generates_detector_and_student(self):
        detector, student_id = generate_demo_data()
        assert detector.knowledge_map is not None
        assert student_id in detector.student_maps
    
    def test_demo_student_has_12_interactions(self):
        detector, student_id = generate_demo_data()
        smap = detector.get_student_map(student_id)
        assert smap.total_interactions == 12
    
    def test_demo_detects_gaps(self):
        detector, student_id = generate_demo_data()
        gaps = detector.detect_critical_gaps(student_id)
        assert len(gaps) > 0
    
    def test_demo_has_critical_gap(self):
        detector, student_id = generate_demo_data()
        gaps = detector.detect_critical_gaps(student_id)
        critical = [g for g in gaps if g.severity == "critical"]
        assert len(critical) > 0
    
    def test_demo_report_generates(self):
        detector, student_id = generate_demo_data()
        report = print_demo_report(detector, student_id)
        assert "COGNITIVE GAP DETECTOR" in report
        assert "SONDAS EPISTÉMICAS" in report
        assert "INYECCIÓN EN SYSTEM PROMPT" in report


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION SCENARIO
# ═══════════════════════════════════════════════════════════════════

class TestIntegrationScenario:
    """
    Simula el flujo completo tal como operaría en el chatbot:
    1. Construir knowledge map
    2. Estudiante interactúa 6 veces
    3. Detectar gaps
    4. Generar sondas
    5. Inyectar en system prompt
    """
    
    def test_full_flow(self, detector):
        sid = "integration_test"
        
        # Student asks about variables (4x)
        for _ in range(4):
            detector.record_interaction(sid, ["variables"], bloom_level=3)
        
        # Student asks about recursión (without funciones/bucles!)
        detector.record_interaction(sid, ["recursión"], bloom_level=2)
        detector.record_interaction(sid, ["recursión"], bloom_level=2)
        
        # Force interaction counter for injection
        detector.interaction_counters[sid] = 4
        
        # Detect gaps
        gaps = detector.detect_critical_gaps(sid)
        assert len(gaps) > 0
        
        # Should have prerequisite gaps (funciones, bucles are prereqs of recursión)
        critical = [g for g in gaps if g.severity == "critical"]
        assert len(critical) > 0
        
        # Generate injection
        injection = detector.get_probe_for_injection(sid)
        assert injection is not None
        
        # Summary should work
        summary = detector.get_gap_summary(sid)
        assert summary["total_gaps"] > 0
        assert summary["coverage_ratio"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
