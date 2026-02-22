"""
COGNITIVE ANALYZER — Bridge module
====================================
Puente entre app.py (que importa CognitiveAnalyzer) y cognitive_engine.py
(donde vive la implementación real como CognitiveEngine).

Este módulo existe porque durante el desarrollo iterativo los nombres
divergieron. En producción se unificaría.
"""

from cognitive_engine import (
    CognitiveEngine,
    CognitiveAnalysis,
    TrustSignal,
    BLOOM_LEVELS,
    BLOOM_MARKERS,
    ICAP_LEVELS,
)


class CognitiveAnalyzer:
    """
    Wrapper que adapta CognitiveEngine a la interfaz que app.py espera.
    app.py llama analyzer.analyze(prompt) → CognitiveAnalysis
    """

    def __init__(self):
        self._engine = CognitiveEngine()

    def analyze(self, text: str) -> CognitiveAnalysis:
        """Analiza el nivel cognitivo de un prompt."""
        return self._engine.analyze_prompt(text)

    def analyze_trust(self, student_id: str, prompt: str,
                      last_response_time: float = 0) -> TrustSignal:
        return self._engine.analyze_trust(student_id, prompt, last_response_time)

    def track_student(self, student_id: str, analysis: CognitiveAnalysis):
        self._engine.track_student(student_id, analysis)

    def get_student_profile(self, student_id: str) -> dict:
        return self._engine.get_student_profile(student_id)


class EngagementProfiler:
    """
    Perfil de engagement agregado. app.py referencia esta clase
    aunque no la usa activamente todavía — existe para la vista
    de investigador.
    """

    def __init__(self):
        self.profiles: dict[str, dict] = {}

    def update(self, student_id: str, analysis: CognitiveAnalysis):
        if student_id not in self.profiles:
            self.profiles[student_id] = {
                "interactions": 0,
                "avg_bloom": 0,
                "bloom_sum": 0,
                "engagement_scores": [],
            }
        p = self.profiles[student_id]
        p["interactions"] += 1
        p["bloom_sum"] += analysis.bloom_level
        p["avg_bloom"] = round(p["bloom_sum"] / p["interactions"], 2)
        p["engagement_scores"].append(analysis.engagement_score)

    def get_profile(self, student_id: str) -> dict:
        return self.profiles.get(student_id, {"status": "no_data"})


# Re-export for convenience
__all__ = [
    "CognitiveAnalyzer",
    "EngagementProfiler",
    "BLOOM_LEVELS",
    "CognitiveAnalysis",
    "TrustSignal",
]
