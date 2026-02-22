"""
GENIE Learn — Motor de Análisis Cognitivo (O1 + O3)
=====================================================
Analiza la profundidad cognitiva de los prompts del estudiante usando:
  - Taxonomía de Bloom Revisada (Anderson & Krathwohl, 2001)
  - Framework ICAP (Chi & Wylie, 2014)
  - Señales de confianza (Lee & See, 2004)
  - Perfiles de engagement dinámicos

Conexión con objetivos GENIE:
  O1 → Marco teórico: los datos generados aquí alimentan los papers
  O3 → Herramientas para estudiantes: el scaffolding se adapta a estos datos
"""

from dataclasses import dataclass, field
from typing import Optional
import re
import time
from datetime import datetime


# ──────────────────────────────────────────────
# TAXONOMÍA DE BLOOM — Operacionalización
# ──────────────────────────────────────────────

BLOOM_LEVELS = {
    1: {"name": "Recordar", "code": "REMEMBER", "color": "#9E9E9E",
        "description": "Recuperar información factual"},
    2: {"name": "Comprender", "code": "UNDERSTAND", "color": "#42A5F5",
        "description": "Explicar conceptos con palabras propias"},
    3: {"name": "Aplicar", "code": "APPLY", "color": "#66BB6A",
        "description": "Usar conocimiento en situaciones nuevas"},
    4: {"name": "Analizar", "code": "ANALYZE", "color": "#FFA726",
        "description": "Descomponer y establecer relaciones"},
    5: {"name": "Evaluar", "code": "EVALUATE", "color": "#EF5350",
        "description": "Juzgar, comparar, criticar"},
    6: {"name": "Crear", "code": "CREATE", "color": "#AB47BC",
        "description": "Diseñar, combinar, producir algo nuevo"},
}

# Marcadores lingüísticos calibrados para español académico
BLOOM_MARKERS = {
    1: {
        "keywords": [
            "qué es", "qué significa", "definición de", "cuál es", "dime",
            "listar", "enumerar", "menciona", "cómo se llama", "cuándo",
            "quién", "dónde", "nombre de", "qué tipos de", "recordar",
        ],
        "patterns": [
            r"^qué (es|son|significa)",
            r"^cuál(es)? (es|son)",
            r"^define",
            r"^di(me)? (qué|cuál)",
        ]
    },
    2: {
        "keywords": [
            "explica", "describe", "por qué", "cómo funciona",
            "para qué sirve", "qué quiere decir",
            "en tus palabras", "resumen", "significa que",
            "entender", "interpretar", "traducir",
        ],
        "patterns": [
            r"^(explica|describe)",
            r"por qué (se usa|funciona|existe)",
            r"para qué (sirve|se usa)",
        ]
    },
    3: {
        "keywords": [
            "cómo se hace", "ejemplo de", "implementar", "resolver",
            "calcular", "ejecutar", "usar para", "aplicar",
            "escribir un", "hacer un", "programar", "código para",
            "muéstrame cómo", "paso a paso",
        ],
        "patterns": [
            r"cómo (hago|puedo|se puede) (hacer|implementar|resolver)",
            r"escribe (un|el) (código|programa|método)",
            r"(dame|muestra) un ejemplo",
        ]
    },
    4: {
        "keywords": [
            "comparar", "diferencia entre", "relación entre",
            "por qué es mejor", "ventajas y desventajas",
            "cuándo usar uno u otro", "estructura de",
            "descomponer", "analizar", "qué pasa si",
            "eficiente", "complejidad", "rendimiento",
            "diferencia hay", "en qué se diferencian",
        ],
        "patterns": [
            r"(compara|diferencia).+(entre|hay)",
            r"por qué (es mejor|se prefiere|se elige)",
            r"qué (ventajas|desventajas|pros|contras)",
            r"qué pasa(ría)? si",
            r"en qué se diferencia",
            r"cuándo (usar|conviene|es mejor)",
        ]
    },
    5: {
        "keywords": [
            "cuál es mejor", "evalúa", "critica", "juzga",
            "está bien", "es correcto", "tiene sentido",
            "qué opinas", "merece la pena", "justifica",
            "es eficiente", "podría mejorar", "falla en",
        ],
        "patterns": [
            r"(está bien|es correcto|tiene sentido)",
            r"cuál (es mejor|conviene|recomendarías)",
            r"(evalúa|critica|revisa) (mi|este|el)",
            r"(podría|debería) mejorar",
        ]
    },
    6: {
        "keywords": [
            "diseñar", "crear", "inventar", "proponer",
            "combinar", "construir", "desarrollar", "planificar",
            "qué pasaría si combino", "nueva forma de",
            "cómo diseñarías", "alternativa a",
        ],
        "patterns": [
            r"(diseña|crea|inventa|propón)",
            r"cómo (diseñarías|crearías|construirías)",
            r"nueva (forma|manera|estrategia) de",
            r"(combinar|integrar) .+ (con|y) ",
        ]
    },
}

# ICAP Framework mapping
ICAP_LEVELS = {
    "passive": {"label": "Pasivo", "description": "Recibe sin procesar", "bloom_range": [1]},
    "active": {"label": "Activo", "description": "Manipula la información", "bloom_range": [2, 3]},
    "constructive": {"label": "Constructivo", "description": "Genera output más allá del input", "bloom_range": [4, 5]},
    "interactive": {"label": "Interactivo", "description": "Co-construye con diálogo", "bloom_range": [5, 6]},
}


@dataclass
class CognitiveAnalysis:
    """Resultado del análisis cognitivo de un prompt."""
    bloom_level: int = 1
    bloom_name: str = "Recordar"
    bloom_code: str = "REMEMBER"
    bloom_confidence: float = 0.0
    icap_level: str = "passive"
    icap_label: str = "Pasivo"
    detected_markers: list = field(default_factory=list)
    engagement_score: float = 0.5  # 0-1


@dataclass
class TrustSignal:
    """Señal de confianza/desconfianza detectada."""
    signal_type: str = "neutral"  # "over_trust" | "under_trust" | "calibrated" | "neutral"
    trust_direction: float = 0.0  # -1 (desconfianza) a +1 (sobre-confianza)
    indicators: list = field(default_factory=list)


class CognitiveEngine:
    """Motor de análisis cognitivo completo."""

    def __init__(self):
        self.student_histories: dict[str, list] = {}  # student_id -> [CognitiveAnalysis]
        self.prompt_timestamps: dict[str, list] = {}   # student_id -> [timestamps]

    def analyze_prompt(self, text: str) -> CognitiveAnalysis:
        """Analiza el nivel cognitivo de un prompt del estudiante."""
        text_lower = text.lower().strip()
        scores = {level: 0.0 for level in range(1, 7)}
        detected_markers = []

        for level, markers in BLOOM_MARKERS.items():
            # Keyword matching
            for kw in markers["keywords"]:
                if kw in text_lower:
                    scores[level] += 1.0
                    detected_markers.append(f"L{level}:{kw}")

            # Pattern matching
            for pattern in markers["patterns"]:
                if re.search(pattern, text_lower):
                    scores[level] += 1.5
                    detected_markers.append(f"L{level}:pattern")

        # Bonus por longitud (prompts largos tienden a ser más elaborados)
        word_count = len(text.split())
        if word_count > 30:
            for level in [4, 5, 6]:
                scores[level] += 0.5
        if word_count > 50:
            for level in [5, 6]:
                scores[level] += 0.5

        # Determinar nivel ganador
        max_score = max(scores.values())
        if max_score == 0:
            # Default a nivel 2 (comprender) si no hay marcadores claros
            bloom_level = 2
            confidence = 0.3
        else:
            bloom_level = max(scores, key=scores.get)
            total = sum(scores.values())
            confidence = max_score / total if total > 0 else 0

        # ICAP mapping
        icap_level = "passive"
        for icap_key, icap_data in ICAP_LEVELS.items():
            if bloom_level in icap_data["bloom_range"]:
                icap_level = icap_key
                break
        if bloom_level >= 4:
            icap_level = "constructive"
        if bloom_level >= 5 and "?" in text:
            icap_level = "interactive"

        bloom_info = BLOOM_LEVELS[bloom_level]
        return CognitiveAnalysis(
            bloom_level=bloom_level,
            bloom_name=bloom_info["name"],
            bloom_code=bloom_info["code"],
            bloom_confidence=round(confidence, 2),
            icap_level=icap_level,
            icap_label=ICAP_LEVELS[icap_level]["label"],
            detected_markers=detected_markers,
            engagement_score=self._compute_engagement(text, bloom_level),
        )

    def analyze_trust(self, student_id: str, prompt: str,
                      last_response_time: float = 0) -> TrustSignal:
        """Analiza señales de confianza del estudiante."""
        indicators = []
        direction = 0.0

        # Latencia como proxy de procesamiento
        timestamps = self.prompt_timestamps.get(student_id, [])
        if len(timestamps) >= 2:
            gap = timestamps[-1] - timestamps[-2]
            if gap < 3.0:
                indicators.append("rapid_fire")
                direction += 0.3  # sobre-confianza: no lee la respuesta
            elif gap > 60.0:
                indicators.append("extended_reflection")
                direction -= 0.1  # reflexión = calibración

        # Patrones lingüísticos
        prompt_lower = prompt.lower()
        trust_up = ["resuélveme", "dime la respuesta", "hazlo por mí", "solo dime"]
        trust_down = ["no estoy seguro", "está bien esto", "¿es correcto?", "revisa"]
        calibrated = ["he intentado", "creo que", "mi idea es", "pero no entiendo"]

        for phrase in trust_up:
            if phrase in prompt_lower:
                indicators.append(f"over_trust:{phrase}")
                direction += 0.2

        for phrase in trust_down:
            if phrase in prompt_lower:
                indicators.append(f"verification:{phrase}")
                direction -= 0.1

        for phrase in calibrated:
            if phrase in prompt_lower:
                indicators.append(f"calibrated:{phrase}")

        # Clasificar
        if direction > 0.3:
            signal_type = "over_trust"
        elif direction < -0.2:
            signal_type = "under_trust"
        elif any("calibrated" in i for i in indicators):
            signal_type = "calibrated"
        else:
            signal_type = "neutral"

        return TrustSignal(
            signal_type=signal_type,
            trust_direction=round(direction, 2),
            indicators=indicators,
        )

    def track_student(self, student_id: str, analysis: CognitiveAnalysis):
        """Registra análisis para tracking longitudinal."""
        if student_id not in self.student_histories:
            self.student_histories[student_id] = []
        self.student_histories[student_id].append(analysis)

        if student_id not in self.prompt_timestamps:
            self.prompt_timestamps[student_id] = []
        self.prompt_timestamps[student_id].append(time.time())

    def get_student_profile(self, student_id: str) -> dict:
        """Perfil de engagement dinámico para un estudiante."""
        history = self.student_histories.get(student_id, [])
        if not history:
            return {"status": "no_data", "interactions": 0}

        bloom_levels = [h.bloom_level for h in history]
        n = len(bloom_levels)

        # Tendencia Bloom (regresión lineal simple)
        if n >= 3:
            x_mean = (n - 1) / 2
            y_mean = sum(bloom_levels) / n
            num = sum((i - x_mean) * (b - y_mean) for i, b in enumerate(bloom_levels))
            den = sum((i - x_mean) ** 2 for i in range(n))
            trend = num / den if den != 0 else 0
        else:
            trend = 0

        # Distribución ICAP
        icap_counts = {"passive": 0, "active": 0, "constructive": 0, "interactive": 0}
        for h in history:
            icap_counts[h.icap_level] = icap_counts.get(h.icap_level, 0) + 1

        # Autonomía epistémica (¿el estudiante sube o baja de nivel?)
        if n >= 4:
            first_half = bloom_levels[:n // 2]
            second_half = bloom_levels[n // 2:]
            autonomy = (sum(second_half) / len(second_half)) - (sum(first_half) / len(first_half))
        else:
            autonomy = 0

        return {
            "status": "active",
            "interactions": n,
            "avg_bloom": round(sum(bloom_levels) / n, 2),
            "bloom_trend": round(trend, 3),
            "bloom_trend_label": "📈 Ascendente" if trend > 0.1 else ("📉 Descendente" if trend < -0.1 else "➡️ Estable"),
            "icap_distribution": icap_counts,
            "dominant_icap": max(icap_counts, key=icap_counts.get),
            "epistemic_autonomy": round(autonomy, 2),
            "autonomy_label": "🟢 Creciente" if autonomy > 0.3 else ("🔴 Decreciente" if autonomy < -0.3 else "🟡 Estable"),
            "last_bloom": bloom_levels[-1] if bloom_levels else 0,
            "max_bloom_reached": max(bloom_levels),
        }

    def _compute_engagement(self, text: str, bloom_level: int) -> float:
        """Score de engagement basado en indicadores textuales."""
        score = 0.3  # base
        words = len(text.split())

        # Longitud como proxy
        if words > 10:
            score += 0.1
        if words > 25:
            score += 0.1

        # Bloom level bonus
        score += bloom_level * 0.05

        # Señales de engagement activo
        if "?" in text:
            score += 0.1
        if any(w in text.lower() for w in ["he intentado", "creo que", "no entiendo por qué"]):
            score += 0.15

        return min(round(score, 2), 1.0)
