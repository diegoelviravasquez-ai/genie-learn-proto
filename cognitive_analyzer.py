"""
COGNITIVE ANALYZER — Bridge module
====================================
Puente entre app.py (que importa CognitiveAnalyzer) y cognitive_engine.py
(donde vive la implementación real como CognitiveEngine).

Este módulo existe porque durante el desarrollo iterativo los nombres
divergieron. En producción se unificaría.
"""

from typing import List

from cognitive_engine import (
    CognitiveEngine,
    CognitiveAnalysis,
    TrustSignal,
    BLOOM_LEVELS,
    BLOOM_MARKERS,
    ICAP_LEVELS,
)


def _linear_trend_slope(levels: List[int]) -> float:
    """Pendiente de regresión lineal (índice -> nivel Bloom)."""
    if not levels or len(levels) < 2:
        return 0.0
    n = len(levels)
    x = list(range(n))
    sum_x = sum(x)
    sum_y = sum(levels)
    sum_xy = sum(xi * yi for xi, yi in zip(x, levels))
    sum_xx = sum(xi * xi for xi in x)
    denom = n * sum_xx - sum_x * sum_x
    if denom == 0:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


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

    def compute_trajectory(
        self, analyses: List[CognitiveAnalysis]
    ) -> dict:
        """
        Genera métricas de trayectoria cognitiva a partir de una lista de análisis.
        Usado por la vista del investigador para el gráfico de evolución Bloom.
        """
        if not analyses:
            return {
                "levels_sequence": [],
                "mean_level": 0.0,
                "trend": 0.0,
                "ceiling": 0,
                "metacognitive_ratio": 0.0,
                "std": 0.0,
                "interpretation": "Sin datos de interacciones.",
                "distribution": {BLOOM_LEVELS[i]["name"]: 0 for i in range(1, 7)},
            }

        levels = [a.bloom_level for a in analyses]
        n = len(levels)
        mean_level = sum(levels) / n
        ceiling = max(levels)
        variance = sum((x - mean_level) ** 2 for x in levels) / n if n else 0
        std = variance ** 0.5
        metacognitive_count = sum(1 for a in analyses if getattr(a, "is_metacognitive", False))
        metacognitive_ratio = metacognitive_count / n
        trend = _linear_trend_slope(levels)

        if trend > 0.1:
            interp = "Trayectoria ascendente: el estudiante progresa en niveles cognitivos."
        elif trend < -0.1:
            interp = "Trayectoria descendente: conviene revisar dificultades o cambio de tema."
        else:
            interp = "Trayectoria estable. Valorar si el reto es adecuado al nivel."

        distribution = {BLOOM_LEVELS[i]["name"]: 0 for i in range(1, 7)}
        for level in levels:
            if 1 <= level <= 6:
                distribution[BLOOM_LEVELS[level]["name"]] += 1

        return {
            "levels_sequence": levels,
            "mean_level": mean_level,
            "trend": trend,
            "ceiling": ceiling,
            "metacognitive_ratio": metacognitive_ratio,
            "std": round(std, 2),
            "interpretation": interp,
            "distribution": distribution,
        }


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

    def classify(
        self,
        analyses: List[CognitiveAnalysis],
        cp_scores: List[float],
        scaffolding_levels: List[int],
    ) -> dict:
        """
        Clasifica el perfil de engagement a partir de análisis, copy-paste scores
        y niveles de scaffolding. Usado por la vista del investigador.
        """
        if not analyses:
            return {
                "label": "Sin datos",
                "color": "#999",
                "emoji": "?",
                "confidence": 0.0,
                "description": "Insuficientes interacciones para clasificar.",
                "all_scores": {},
                "indicators": {"mean_bloom_level": 0.0},
            }

        mean_bloom = sum(a.bloom_level for a in analyses) / len(analyses)
        mean_cp = sum(cp_scores) / len(cp_scores) if cp_scores else 0.0
        max_scaff = max(scaffolding_levels) if scaffolding_levels else 0
        mean_scaff = sum(scaffolding_levels) / len(scaffolding_levels) if scaffolding_levels else 0.0
        meta_ratio = sum(1 for a in analyses if getattr(a, "is_metacognitive", False)) / len(analyses)
        mean_engagement = sum(a.engagement_score for a in analyses) / len(analyses)
        high_bloom_ratio = sum(1 for a in analyses if a.bloom_level >= 4) / len(analyses)

        all_scores = {
            "deep_explorer": min(1.0, mean_bloom / 5 + mean_engagement * 0.3),
            "strategic_user": min(1.0, (5 - mean_scaff) / 5 * 0.8 + mean_bloom / 6 * 0.2),
            "surface_seeker": min(1.0, (6 - mean_bloom) / 5 * 0.7),
            "copy_paster": min(1.0, mean_cp * 1.5),
            "metacognitive": min(1.0, meta_ratio * 2 + mean_engagement * 0.3),
        }

        dominant = max(all_scores, key=all_scores.get)
        profiles_map = {
            "deep_explorer": {"label": "Explorador profundo", "color": "#667eea", "emoji": "·"},
            "strategic_user": {"label": "Estratégico", "color": "#42A5F5", "emoji": "·"},
            "surface_seeker": {"label": "Superficial", "color": "#FFA726", "emoji": "·"},
            "copy_paster": {"label": "Copy-paster", "color": "#EF5350", "emoji": "·"},
            "metacognitive": {"label": "Metacognitivo", "color": "#AB47BC", "emoji": "·"},
        }
        p = profiles_map.get(dominant, {"label": "Mixto", "color": "#888", "emoji": "·"})
        confidence = all_scores[dominant]

        return {
            "label": p["label"],
            "color": p["color"],
            "emoji": p["emoji"],
            "confidence": round(confidence, 2),
            "description": f"Bloom medio {mean_bloom:.1f}, engagement {mean_engagement:.2f}.",
            "all_scores": all_scores,
            "indicators": {
                "mean_bloom_level": round(mean_bloom, 1),
                "mean_copypaste": round(mean_cp, 2),
                "metacognitive_ratio": round(meta_ratio, 2),
                "high_bloom_ratio": round(high_bloom_ratio, 2),
                "max_scaffolding_reached": max_scaff,
            },
        }


# Re-export for convenience
__all__ = [
    "CognitiveAnalyzer",
    "EngagementProfiler",
    "BLOOM_LEVELS",
    "CognitiveAnalysis",
    "TrustSignal",
]
