"""
COGNITIVE ENGAGEMENT PROFILER
═══════════════════════════════════════════════════════════════════════
Módulo diferencial #1 — Ataca la limitación EXPLÍCITA del paper LAK 2026:

    "The implemented GenAI Analytics are low-level analytics (i.e., raw data).
     It would be interesting to explore higher-level analytics that combine
     multiple indicators to derive participants' profiles or engagement levels."
     — Ortega-Arranz et al. (LAK 2026, Sección 5: Limitations)

DECISIONES DE DISEÑO QUE SOLO ESTE PERFIL TOMARÍA:
────────────────────────────────────────────────────
1. Operacionalización computacional de la taxonomía de Bloom (1956, rev.
   Anderson & Krathwohl 2001) mediante marcadores lingüísticos del español.
   Un ingeniero haría keyword matching. Un sociólogo del conocimiento
   (Mannheim, 1936) sabe que las estructuras cognitivas se manifiestan en
   patrones discursivos, no en palabras sueltas.

2. Perfiles de engagement como TRAYECTORIAS, no como snapshots. Bourdieu
   (1986) demostró que el capital cultural es dinámico y se acumula o
   erosiona en el tiempo. Aplicamos la misma lógica: el engagement de un
   estudiante se mide como pendiente de regresión sobre una ventana
   temporal, no como media aritmética de un solo instante.

3. Clasificación inspirada en la metodología de Análisis de Hipótesis
   Competitivas (ACH, Heuer 1999): cada prompt del estudiante se evalúa
   simultáneamente contra TODAS las categorías de Bloom, y se asigna a la
   que maximiza la evidencia acumulada, no a la primera que haga match.

FUNDAMENTACIÓN:
  - Bloom, B.S. (1956). Taxonomy of Educational Objectives.
  - Anderson, L.W. & Krathwohl, D.R. (2001). A Taxonomy for Learning.
  - Mannheim, K. (1936). Ideology and Utopia.
  - Bourdieu, P. (1986). "The Forms of Capital".
  - Heuer, R.J. (1999). Psychology of Intelligence Analysis.
  - Damasio, A. (1994). Descartes' Error — marcadores somáticos como
    proxy de engagement emocional (aquí: léxico afectivo).

INTEGRACIÓN: Se conecta al middleware existente como post-análisis de
cada interacción, alimentando el dashboard docente con analytics de
alto nivel (no raw data).

Autor: Diego Elvira Vásquez · Prototipo CP25/152 · Feb 2026
"""

import re
import math
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# TAXONOMÍA DE BLOOM — OPERACIONALIZACIÓN LINGÜÍSTICA
# ═══════════════════════════════════════════════════════════════
# Decisión: NO usamos keyword matching plano. Usamos PATRONES
# DISCURSIVOS que capturan la estructura sintáctica de cada nivel
# cognitivo. Un estudiante que dice "qué es X" opera en nivel
# diferente a uno que dice "por qué X funciona diferente a Y".
# La diferencia es pragmática (Austin, 1962; Searle, 1969),
# no léxica.

BLOOM_LEVELS = {
    "remember": {
        "label": "Recordar",
        "description": "Recuperar información factual",
        "weight": 1,
        "patterns": [
            r"qué\s+es\b",
            r"qué\s+significa\b",
            r"cuál\s+es\s+la\s+definición",
            r"cómo\s+se\s+(llama|dice|escribe)\b",
            r"dime\s+(qué|cómo|cuál)",
            r"qué\s+tipos?\s+de\b",
            r"cuáles?\s+son\s+los?\b",
            r"lista(r|me)?\b",
            r"enumera(r|me)?\b",
            r"define\b",
            r"nombra(r|me)?\b",
        ],
    },
    "understand": {
        "label": "Comprender",
        "description": "Construir significado a partir de información",
        "weight": 2,
        "patterns": [
            r"(explica|explicar|explícame)\b",
            r"por\s+qué\b",
            r"para\s+qué\s+(se\s+usa|sirve|funciona)",
            r"cómo\s+funciona\b",
            r"qué\s+relación\s+(hay|tiene|existe)",
            r"(resume|resumir|resumen)\b",
            r"en\s+qué\s+consiste\b",
            r"en\s+(mis|tus|sus)\s+(propias\s+)?palabras",
            r"interpreta(r)?\b",
            r"describe\s+(cómo|qué|el|la)\b",
        ],
    },
    "apply": {
        "label": "Aplicar",
        "description": "Usar procedimientos en situaciones concretas",
        "weight": 3,
        "patterns": [
            r"cómo\s+(puedo|se\s+puede|podría)\s+(hacer|usar|implementar|resolver)",
            r"(usa|usar|utiliza|utilizar)\s+(esto|eso|el|la|un)\b",
            r"(resuelve|resolver|soluciona|solucionar)\b",
            r"(implementa|implementar|programa|programar)\b",
            r"aplica(r)?\s+(esto|eso|el|la)\b",
            r"escribe\s+(un|una|el|la)\s+(código|programa|función|bucle)",
            r"muestra(me)?\s+(un|una|el|la)\s+ejemplo",
            r"calcula(r)?\b",
            r"ejecuta(r)?\b",
        ],
    },
    "analyze": {
        "label": "Analizar",
        "description": "Descomponer en partes y detectar relaciones",
        "weight": 4,
        "patterns": [
            r"(cuál|qué)\s+es\s+la\s+diferencia\s+entre\b",
            r"compara(r|me)?\b",
            r"(contrasta|contrastar)\b",
            r"por\s+qué\s+(falla|no\s+funciona|da\s+error)",
            r"qué\s+pasa(ría)?\s+si\b",
            r"(analiza|analizar|descompone|descomponer)\b",
            r"(clasifica|clasificar)\b",
            r"(identifica|identificar)\s+(el|la|los|las)\s+(error|fallo|problema|causa)",
            r"cuál\s+es\s+(mejor|peor|más\s+eficiente)\b",
            r"en\s+qué\s+se\s+(diferencian|parecen|distinguen)\b",
            r"ventajas?\s+y\s+desventajas?\b",
        ],
    },
    "evaluate": {
        "label": "Evaluar",
        "description": "Emitir juicios basados en criterios",
        "weight": 5,
        "patterns": [
            r"(es\s+correcto|está\s+bien|es\s+buena\s+práctica)\b",
            r"(debería|convendría|sería\s+mejor)\b",
            r"(justifica|justificar|argumenta|argumentar)\b",
            r"(critica|criticar|evalúa|evaluar)\b",
            r"(recomiendas|recomendarías|aconsejas)\b",
            r"merece\s+la\s+pena\b",
            r"tiene\s+sentido\b",
            r"cuál\s+(es|sería)\s+(la\s+)?mejor\s+(forma|manera|opción|alternativa)",
            r"vale\s+la\s+pena\b",
        ],
    },
    "create": {
        "label": "Crear",
        "description": "Generar productos o soluciones originales",
        "weight": 6,
        "patterns": [
            r"(diseña|diseñar|crea|crear|inventa|inventar)\b",
            r"cómo\s+(diseñaría|construiría|crearía|haría)\b",
            r"propone?(r)?\s+(una?\s+)?(solución|alternativa|mejora|diseño)",
            r"modifica(r)?\s+para\s+que\b",
            r"combina(r)?\b.*con\b",
            r"(adapta|adaptar|extiende|extender|amplía|ampliar)\b",
            r"qué\s+pasaría\s+si\s+(cambio|modifico|añado|quito)",
            r"(genera|generar)\s+(un|una)\s+(nuevo|nueva)",
            r"(plantea|plantear)\s+(un|una)\b",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
# MARCADORES DE ENGAGEMENT AFECTIVO
# ═══════════════════════════════════════════════════════════════
# Decisión: Damasio (1994) demostró que la cognición sin emoción
# produce decisiones deficientes. Un estudiante frustrado que
# dice "no entiendo nada" opera en un estado cognitivo
# cualitativamente distinto al que dice "interesante, pero...".
# Capturamos esto como DIMENSIÓN ORTOGONAL al nivel de Bloom.

AFFECTIVE_MARKERS = {
    "frustration": {
        "label": "Frustración",
        "valence": -2,
        "patterns": [
            r"no\s+(entiendo|comprendo|logro|consigo|puedo)\b",
            r"(estoy\s+)?(perdido|bloqueado|atascado|confundido)\b",
            r"(no\s+me\s+(sale|funciona|queda\s+claro))",
            r"(esto\s+es\s+)?(muy\s+)?(difícil|complicado|imposible)\b",
            r"llevo\s+(rato|tiempo|horas)\b",
            r"me\s+rindo\b",
            r"no\s+sé\s+ni\s+por\s+dónde\s+empezar\b",
        ],
    },
    "curiosity": {
        "label": "Curiosidad",
        "valence": 2,
        "patterns": [
            r"(me\s+pregunto|me\s+gustaría\s+saber)\b",
            r"(interesante|curioso|sorprende)\b",
            r"y\s+(si|qué\s+pasa)\b",
            r"(se\s+puede|es\s+posible)\b.*\?",
            r"existe(n)?\s+otras?\s+(formas?|maneras?|alternativas?)",
            r"hay\s+algo\s+más\s+(avanzado|profundo|complejo)\b",
        ],
    },
    "confidence": {
        "label": "Confianza",
        "valence": 1,
        "patterns": [
            r"creo\s+que\b",
            r"(entiendo|comprendo)\s+que\b",
            r"si\s+no\s+me\s+equivoco\b",
            r"he\s+(intentado|probado|pensado)\b",
            r"mi\s+(idea|solución|propuesta)\s+(es|sería)\b",
            r"lo\s+que\s+he\s+hecho\s+es\b",
        ],
    },
    "disengagement": {
        "label": "Desenganche",
        "valence": -1,
        "patterns": [
            r"(dame|dime)\s+(la|el)\s+(solución|respuesta|código|resultado)\b",
            r"no\s+quiero\s+(pensar|explicación)\b",
            r"(solo|sólo)\s+(dime|dame|resuelve)\b",
            r"(rápido|directo|sin\s+rodeos)\b",
            r"(hazlo|resuélvelo)\s+(tú|por\s+mí)\b",
        ],
    },
}


@dataclass
class CognitiveSnapshot:
    """Instantánea cognitiva de una interacción individual."""
    timestamp: str
    student_id: str
    bloom_level: str              # key de BLOOM_LEVELS
    bloom_confidence: float       # 0.0 - 1.0, fuerza del match
    affective_state: str          # key de AFFECTIVE_MARKERS
    affective_valence: int        # -2 a +2
    prompt_complexity: float      # métrica compuesta 0.0 - 1.0
    metacognitive_signals: int    # conteo de marcadores metacognitivos
    raw_prompt: str


@dataclass
class EngagementProfile:
    """
    Perfil de engagement de alto nivel para un estudiante.
    ESTO es lo que el paper LAK 2026 pide y NO tiene:
    analytics derivados que combinan múltiples indicadores.
    """
    student_id: str
    period_start: str
    period_end: str

    # Distribución cognitiva (% de interacciones por nivel Bloom)
    bloom_distribution: dict = field(default_factory=dict)

    # Nivel cognitivo medio ponderado (1.0-6.0)
    cognitive_depth_mean: float = 0.0

    # Tendencia cognitiva: pendiente de regresión (>0 = mejora, <0 = regresión)
    cognitive_trajectory: float = 0.0

    # Estado afectivo dominante
    dominant_affect: str = ""
    affect_valence_mean: float = 0.0

    # Indicadores de autonomía epistémica
    metacognitive_ratio: float = 0.0  # % prompts con señales metacognitivas
    question_sophistication_trend: float = 0.0  # tendencia de complejidad

    # Clasificación derivada
    engagement_type: str = ""  # "deep_learner", "surface_seeker", "struggling", "disengaged", "exploratory"
    risk_flags: list = field(default_factory=list)

    # Número de interacciones analizadas
    n_interactions: int = 0


class CognitiveProfiler:
    """
    Motor de perfilado cognitivo.

    Flujo operativo:
    1. analyze_prompt()  → CognitiveSnapshot por cada interacción
    2. build_profile()   → EngagementProfile agregado por estudiante
    3. classify()        → Tipología de engagement

    Se integra como capa analítica SOBRE el middleware existente,
    sin modificar su flujo de pre/post procesamiento.
    """

    def __init__(self):
        self.snapshots: dict[str, list[CognitiveSnapshot]] = defaultdict(list)
        # Pre-compilar regex para rendimiento
        self._bloom_compiled = {
            level: [re.compile(p, re.IGNORECASE) for p in data["patterns"]]
            for level, data in BLOOM_LEVELS.items()
        }
        self._affect_compiled = {
            state: [re.compile(p, re.IGNORECASE) for p in data["patterns"]]
            for state, data in AFFECTIVE_MARKERS.items()
        }
        # Marcadores metacognitivos (evidencia de reflexión sobre el propio aprendizaje)
        self._metacognitive_patterns = [
            re.compile(p, re.IGNORECASE) for p in [
                r"(no\s+estoy\s+seguro\s+de|no\s+sé\s+si)\b",
                r"(creo|pienso|supongo)\s+que\b",
                r"he\s+(intentado|probado|pensado)\b",
                r"mi\s+estrategia\s+(es|fue|sería)\b",
                r"lo\s+que\s+(no\s+)?entiendo\s+es\b",
                r"(antes|primero|después)\s+(he|había|intenté)\b",
                r"me\s+doy\s+cuenta\s+de\s+que\b",
                r"(me\s+equivoqué|estaba\s+equivocado)\b",
                r"(ahora\s+)?(ya\s+)?entiendo\s+(que|por\s+qué|cómo)\b",
            ]
        ]

    # ───────────────────────────────────────────
    # 1. ANÁLISIS POR INTERACCIÓN
    # ───────────────────────────────────────────

    def analyze_prompt(self, student_id: str, prompt: str) -> CognitiveSnapshot:
        """
        Analiza un prompt individual.

        Metodología ACH (Heuer, 1999): evaluamos el prompt contra TODAS
        las hipótesis (niveles de Bloom) simultáneamente. El nivel asignado
        es el que acumula mayor evidencia, no el primero que hace match.
        """
        prompt_clean = prompt.strip().lower()

        # ─── Bloom: evaluación competitiva ───
        bloom_scores = {}
        for level, patterns in self._bloom_compiled.items():
            score = sum(1 for p in patterns if p.search(prompt_clean))
            # Ponderar por posición en la frase (marcadores al inicio pesan más)
            if score > 0 and patterns[0].search(prompt_clean[:80]):
                score *= 1.3
            bloom_scores[level] = score

        total_bloom = sum(bloom_scores.values())
        if total_bloom > 0:
            best_bloom = max(bloom_scores, key=bloom_scores.get)
            bloom_confidence = bloom_scores[best_bloom] / total_bloom
        else:
            best_bloom = "understand"  # default razonable
            bloom_confidence = 0.3

        # ─── Estado afectivo ───
        affect_scores = {}
        for state, patterns in self._affect_compiled.items():
            score = sum(1 for p in patterns if p.search(prompt_clean))
            affect_scores[state] = score

        total_affect = sum(affect_scores.values())
        if total_affect > 0:
            best_affect = max(affect_scores, key=affect_scores.get)
            affect_valence = AFFECTIVE_MARKERS[best_affect]["valence"]
        else:
            best_affect = "neutral"
            affect_valence = 0

        # ─── Complejidad del prompt ───
        complexity = self._compute_complexity(prompt)

        # ─── Señales metacognitivas ───
        metacognitive = sum(1 for p in self._metacognitive_patterns if p.search(prompt_clean))

        snapshot = CognitiveSnapshot(
            timestamp=datetime.now().isoformat(),
            student_id=student_id,
            bloom_level=best_bloom,
            bloom_confidence=bloom_confidence,
            affective_state=best_affect,
            affective_valence=affect_valence,
            prompt_complexity=complexity,
            metacognitive_signals=metacognitive,
            raw_prompt=prompt[:200],
        )

        self.snapshots[student_id].append(snapshot)
        return snapshot

    def _compute_complexity(self, text: str) -> float:
        """
        Índice de complejidad del prompt (0.0 - 1.0).

        Indicadores (inspirados en análisis de discurso):
        - Longitud del texto (normalizada)
        - Presencia de conectores causales (indica razonamiento)
        - Presencia de código (indica aplicación práctica)
        - Número de interrogaciones (indica profundidad de inquiry)
        """
        score = 0.0

        # Longitud (log-normalizada, techo en 500 chars)
        length_score = min(math.log(len(text) + 1) / math.log(501), 1.0)
        score += length_score * 0.25

        # Conectores causales (Mannheim: pensamiento relacional)
        causal_connectors = [
            "porque", "ya que", "debido a", "por lo tanto", "entonces",
            "sin embargo", "aunque", "pero", "además", "en cambio",
            "por eso", "así que", "de modo que", "puesto que",
        ]
        causal_count = sum(1 for c in causal_connectors if c in text.lower())
        score += min(causal_count / 3.0, 1.0) * 0.25

        # Presencia de código
        code_markers = ["```", "def ", "for ", "while ", "if (", "int ", "System.", "print("]
        has_code = any(m in text for m in code_markers)
        score += (0.2 if has_code else 0.0)

        # Interrogaciones múltiples (profundidad)
        question_count = text.count("?")
        score += min(question_count / 3.0, 1.0) * 0.15

        # Referencia a intentos previos (metacognición)
        if any(w in text.lower() for w in ["he intentado", "he probado", "mi código"]):
            score += 0.15

        return min(score, 1.0)

    # ───────────────────────────────────────────
    # 2. CONSTRUCCIÓN DE PERFILES
    # ───────────────────────────────────────────

    def build_profile(self, student_id: str, last_n: int = 20) -> EngagementProfile:
        """
        Construye el perfil de engagement de alto nivel.

        Esto es EXACTAMENTE lo que el paper LAK 2026 pide como trabajo futuro:
        "higher-level analytics that combine multiple indicators to derive
        participants' profiles or engagement levels with AI."
        """
        history = self.snapshots.get(student_id, [])
        if not history:
            return EngagementProfile(
                student_id=student_id,
                period_start="",
                period_end="",
                engagement_type="unknown",
            )

        # Tomar las últimas N interacciones
        recent = history[-last_n:]

        # ─── Distribución de Bloom ───
        bloom_counts = defaultdict(int)
        for s in recent:
            bloom_counts[s.bloom_level] += 1
        total = len(recent)
        bloom_dist = {k: round(v / total, 3) for k, v in bloom_counts.items()}

        # ─── Profundidad cognitiva media ponderada ───
        weights = [BLOOM_LEVELS[s.bloom_level]["weight"] for s in recent]
        cognitive_mean = sum(weights) / len(weights) if weights else 0

        # ─── Trayectoria cognitiva (regresión lineal simple) ───
        # Bourdieu: el capital (aquí, cognitivo) tiene trayectoria temporal
        if len(recent) >= 3:
            trajectory = self._linear_trend([
                BLOOM_LEVELS[s.bloom_level]["weight"] for s in recent
            ])
        else:
            trajectory = 0.0

        # ─── Estado afectivo dominante ───
        affect_counts = defaultdict(int)
        valences = []
        for s in recent:
            affect_counts[s.affective_state] += 1
            valences.append(s.affective_valence)
        dominant_affect = max(affect_counts, key=affect_counts.get)
        valence_mean = sum(valences) / len(valences) if valences else 0

        # ─── Metacognición ───
        metacognitive_count = sum(1 for s in recent if s.metacognitive_signals > 0)
        metacognitive_ratio = metacognitive_count / total

        # ─── Tendencia de complejidad ───
        if len(recent) >= 3:
            complexity_trend = self._linear_trend([s.prompt_complexity for s in recent])
        else:
            complexity_trend = 0.0

        profile = EngagementProfile(
            student_id=student_id,
            period_start=recent[0].timestamp,
            period_end=recent[-1].timestamp,
            bloom_distribution=bloom_dist,
            cognitive_depth_mean=round(cognitive_mean, 2),
            cognitive_trajectory=round(trajectory, 4),
            dominant_affect=dominant_affect,
            affect_valence_mean=round(valence_mean, 2),
            metacognitive_ratio=round(metacognitive_ratio, 3),
            question_sophistication_trend=round(complexity_trend, 4),
            n_interactions=total,
        )

        # ─── Clasificación ───
        profile.engagement_type = self._classify_engagement(profile)
        profile.risk_flags = self._detect_risks(profile)

        return profile

    def _classify_engagement(self, profile: EngagementProfile) -> str:
        """
        Tipología de engagement derivada.

        Cinco tipos (inspirados en la taxonomía de enfoques de aprendizaje
        de Biggs, 1987 — deep vs. surface — enriquecida con dimensión
        afectiva de Pekrun, 2006):

        - deep_learner: alto Bloom + trayectoria positiva + metacognición
        - surface_seeker: bajo Bloom + baja metacognición + bajo esfuerzo
        - struggling: esfuerzo presente + frustración + bajo Bloom
        - disengaged: búsqueda de solución directa + desenganche afectivo
        - exploratory: alta curiosidad + Bloom variable + muchas preguntas
        """
        depth = profile.cognitive_depth_mean
        trajectory = profile.cognitive_trajectory
        metacog = profile.metacognitive_ratio
        affect = profile.affect_valence_mean
        dominant = profile.dominant_affect

        if depth >= 3.5 and trajectory >= 0 and metacog >= 0.3:
            return "deep_learner"
        elif dominant == "disengagement" or (depth < 2.0 and metacog < 0.1):
            return "disengaged"
        elif dominant == "frustration" and depth < 3.0 and metacog > 0.1:
            return "struggling"
        elif dominant == "curiosity" and affect > 0.5:
            return "exploratory"
        else:
            return "surface_seeker"

    def _detect_risks(self, profile: EngagementProfile) -> list[str]:
        """Detecta señales de alerta para el docente."""
        risks = []
        if profile.cognitive_trajectory < -0.3:
            risks.append("⬇️ Regresión cognitiva: el nivel de preguntas está bajando")
        if profile.dominant_affect == "frustration" and profile.affect_valence_mean < -1.0:
            risks.append("😰 Frustración sostenida: riesgo de abandono")
        if profile.dominant_affect == "disengagement":
            risks.append("💤 Desenganche: el estudiante busca respuestas directas sin esfuerzo")
        if profile.metacognitive_ratio < 0.05 and profile.n_interactions > 5:
            risks.append("🔇 Sin metacognición: el estudiante no reflexiona sobre su proceso")
        if profile.cognitive_depth_mean < 2.0 and profile.n_interactions > 10:
            risks.append("📉 Estancamiento en Bloom bajo: no hay progresión cognitiva")
        return risks

    # ───────────────────────────────────────────
    # 3. UTILIDADES
    # ───────────────────────────────────────────

    def _linear_trend(self, values: list[float]) -> float:
        """Pendiente de regresión lineal simple (sin numpy)."""
        n = len(values)
        if n < 2:
            return 0.0
        x_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        return numerator / denominator if denominator != 0 else 0.0

    def get_cohort_summary(self) -> dict:
        """
        Resumen de cohorte para el dashboard docente.
        Esto es analytics de ALTO NIVEL, no raw data.
        """
        profiles = {
            sid: self.build_profile(sid)
            for sid in self.snapshots
        }

        type_counts = defaultdict(int)
        for p in profiles.values():
            type_counts[p.engagement_type] += 1

        all_risks = []
        for p in profiles.values():
            for r in p.risk_flags:
                all_risks.append({"student": p.student_id, "flag": r})

        depth_values = [p.cognitive_depth_mean for p in profiles.values() if p.n_interactions > 0]

        return {
            "total_students_profiled": len(profiles),
            "engagement_types": dict(type_counts),
            "cohort_cognitive_depth_mean": round(
                sum(depth_values) / len(depth_values), 2
            ) if depth_values else 0,
            "students_at_risk": len([p for p in profiles.values() if p.risk_flags]),
            "risk_alerts": all_risks[:20],  # top 20 alertas
            "bloom_cohort_distribution": self._cohort_bloom_distribution(profiles),
        }

    def _cohort_bloom_distribution(self, profiles: dict) -> dict:
        """Distribución de Bloom agregada de toda la cohorte."""
        totals = defaultdict(float)
        count = 0
        for p in profiles.values():
            if p.n_interactions > 0:
                for level, pct in p.bloom_distribution.items():
                    totals[level] += pct
                count += 1
        if count == 0:
            return {}
        return {k: round(v / count, 3) for k, v in totals.items()}
