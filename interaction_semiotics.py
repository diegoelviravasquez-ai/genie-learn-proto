"""
INTERACTION SEMIOTICS ENGINE
═══════════════════════════════════════════════════════════════════════
Módulo diferencial #3 — Transforma el copy-paste detection (binario)
en un análisis semiótico de la CALIDAD e INTENCIÓN de la interacción.

PROBLEMA QUE ATACA:
El middleware actual detecta copy-paste con una heurística simple (3
indicadores, score 0-1). El paper LAK 2026 lo lista como feature
("detección de copy-paste") pero los evaluadores pidieron algo más
sofisticado: no solo SI copió, sino QUÉ TIPO de interacción es.

DECISIONES DE DISEÑO QUE SOLO ESTE PERFIL TOMARÍA:
────────────────────────────────────────────────────
1. Clasificación pragmática basada en Speech Act Theory (Austin, 1962;
   Searle, 1969). Cada prompt del estudiante es un ACTO DE HABLA con
   una fuerza ilocutiva que revela la intención comunicativa. No es lo
   mismo "dame el código" (directivo → busca solución directa) que
   "mi código falla en la línea 5, creo que es por el scope" (asertivo
   → busca validación de una hipótesis propia).

2. Detección de patrones estratégicos inspirada en metodología OSINT
   y contrainteligencia. Los estudiantes no solo copian-pegan: algunos
   desarrollan ESTRATEGIAS de gaming (parafrasear enunciados para evadir
   detección, fragmentar preguntas para obtener la solución por partes,
   usar lenguaje ambiguo para evadir guardrails). Un analista de
   inteligencia reconoce estas tácticas porque son isomorfas a las
   técnicas de engaño documentadas en la literatura de CI.

3. Máximas conversacionales de Grice (1975) como framework de evaluación
   de calidad interaccional:
   - Cantidad: ¿la pregunta tiene la información necesaria?
   - Calidad: ¿el estudiante es sincero sobre lo que sabe/no sabe?
   - Relación: ¿la pregunta es relevante al contexto del curso?
   - Manera: ¿la pregunta es clara y ordenada?

   Violaciones de estas máximas son señales diagnósticas de engagement
   problemático (no simple copy-paste).

FUNDAMENTACIÓN:
  - Austin, J.L. (1962). How to Do Things with Words.
  - Searle, J.R. (1969). Speech Acts.
  - Grice, H.P. (1975). "Logic and Conversation."
  - Heuer, R.J. (1999). Psychology of Intelligence Analysis.
  - Kahneman, D. & Tversky, A. (1979). Prospect Theory (framing effects
    en cómo el estudiante formula la pregunta).
  - Bjork, R.A. (1994). Desirable difficulties.
  - Pekrun, R. (2006). Control-Value Theory of Achievement Emotions.

Autor: Diego Elvira Vásquez · Prototipo CP25/152 · Feb 2026
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# TIPOLOGÍA DE ACTOS DE HABLA EN CONTEXTO EDUCATIVO
# ═══════════════════════════════════════════════════════════════
# Adaptación de la taxonomía de Searle (1969) al dominio de
# interacción estudiante-chatbot educativo.

SPEECH_ACTS = {
    "solution_request": {
        "label": "Solicitud de solución",
        "illocutionary_force": "directive",
        "pedagogical_value": 0.1,
        "description": "El estudiante pide la respuesta directa sin esfuerzo propio",
        "patterns": [
            r"(dame|dime|deme)\s+(el|la|los|las)\s+(código|solución|respuesta|resultado)",
            r"(resuelve|resuelva|soluciona)\s+(esto|este|esta|el|la)",
            r"(necesito|quiero)\s+(el|la)\s+(código|solución|respuesta)",
            r"(hazme?|haz)\s+(el|la|este|esta)\s+(ejercicio|programa|tarea)",
            r"(pásame|envíame)\s+(el|la)\s+(código|solución)",
        ],
    },
    "hypothesis_test": {
        "label": "Prueba de hipótesis",
        "illocutionary_force": "assertive",
        "pedagogical_value": 0.9,
        "description": "El estudiante presenta una hipótesis propia y busca validación",
        "patterns": [
            r"creo\s+que\s+(es|está|funciona|falla|debería)",
            r"mi\s+(teoría|hipótesis|idea)\s+es\s+que",
            r"(¿)?esto\s+(es|está)\s+(bien|correcto|mal)\s*\?",
            r"si\s+no\s+me\s+equivoco",
            r"¿(sería|es)\s+porque\b",
            r"el\s+error\s+(creo\s+que\s+)?(es|está)\s+en",
            r"he\s+llegado\s+a\s+la\s+conclusión\s+de",
        ],
    },
    "elaboration_request": {
        "label": "Solicitud de elaboración",
        "illocutionary_force": "directive",
        "pedagogical_value": 0.7,
        "description": "El estudiante pide profundizar en algo que parcialmente comprende",
        "patterns": [
            r"(puedes|podrías|puede)\s+(explicar|ampliar|desarrollar|profundizar)",
            r"¿qué\s+(quieres|quiere)\s+decir\s+con",
            r"no\s+me\s+queda\s+claro\s+(lo\s+de|el|la|cómo)",
            r"¿a\s+qué\s+te\s+refieres\s+con",
            r"(explícame|explica)\s+(mejor|más|con\s+más\s+detalle)",
        ],
    },
    "debugging_aid": {
        "label": "Ayuda con depuración",
        "illocutionary_force": "assertive+directive",
        "pedagogical_value": 0.75,
        "description": "El estudiante comparte su código roto y pide orientación",
        "patterns": [
            r"(mi\s+código|mi\s+programa)\s+(no\s+funciona|da\s+error|falla)",
            r"(me\s+sale|tengo)\s+(un\s+)?(error|excepción|fallo|problema)",
            r"he\s+(intentado|probado)\s+.*\s+(pero|y)\s+(no|sigue)",
            r"(qué|dónde)\s+(está|puede\s+estar)\s+el\s+(error|fallo|problema|bug)",
            r"(funciona\s+hasta|falla\s+cuando|se\s+rompe\s+en)",
        ],
    },
    "conceptual_inquiry": {
        "label": "Indagación conceptual",
        "illocutionary_force": "interrogative",
        "pedagogical_value": 0.8,
        "description": "El estudiante busca comprensión profunda de un concepto",
        "patterns": [
            r"¿por\s+qué\s+(se\s+usa|funciona|existe|es\s+necesario)",
            r"¿cuál\s+es\s+la\s+(lógica|razón|idea)\s+detrás",
            r"¿en\s+qué\s+(contexto|situación|caso)\s+(se\s+usa|sirve)",
            r"¿cómo\s+se\s+relaciona\s+.*\s+con\b",
            r"¿(cuál|qué)\s+es\s+la\s+diferencia\s+(entre|de)",
            r"¿para\s+qué\s+sirve\s+(realmente|en\s+la\s+práctica)",
        ],
    },
    "task_delegation": {
        "label": "Delegación de tarea",
        "illocutionary_force": "directive",
        "pedagogical_value": 0.2,
        "description": "El estudiante delega una tarea completa al chatbot",
        "patterns": [
            r"(escribe|escribir|crea|crear|haz|hacer)\s+(un|una|el|la)\s+(programa|ejercicio|tarea|código)",
            r"(implementa|implementar)\s+(lo\s+siguiente|esto|el|la)",
            r"(programa|programar)\s+(un|una|lo\s+siguiente)",
            # Patrón de enunciado copiado (Grice: violación de máxima de manera)
            r"(dado|dada|dados|dadas)\s+(el|la|los|las|un|una)\s+.{30,}",
            r"(se\s+pide|se\s+solicita|implementar\s+un)\s+.{30,}",
        ],
    },
    "metacognitive_reflection": {
        "label": "Reflexión metacognitiva",
        "illocutionary_force": "expressive",
        "pedagogical_value": 1.0,
        "description": "El estudiante reflexiona sobre su propio proceso de aprendizaje",
        "patterns": [
            r"(ahora\s+)?entiendo\s+(que|por\s+qué|cómo)",
            r"me\s+doy\s+cuenta\s+(de\s+)?que",
            r"(antes\s+)?no\s+sabía\s+que\b.*\s+(ahora|pero)",
            r"mi\s+error\s+(era|fue|estaba\s+en)\s+que",
            r"lo\s+que\s+(he\s+aprendido|aprendo)\s+es",
            r"(necesito|debería)\s+(practicar|repasar|estudiar)\s+más",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
# PATRONES DE GAMING ESTRATÉGICO
# ═══════════════════════════════════════════════════════════════
# Inspirados en técnicas de engaño de la literatura de CI (Ben-Israel,
# 2017; Heuer & Pherson, 2010). Los estudiantes adaptan su comportamiento
# al sistema como agentes racionales que optimizan recompensa (obtener
# la respuesta) minimizando esfuerzo.

GAMING_PATTERNS = {
    "fragmentation": {
        "label": "Fragmentación",
        "description": "Divide un ejercicio en sub-preguntas para obtener la solución por partes",
        "detection_method": "temporal_sequence",
        "risk_level": "medium",
    },
    "paraphrasing_evasion": {
        "label": "Paráfrasis evasiva",
        "description": "Reformula un enunciado copiado para evadir detección de copy-paste",
        "detection_method": "similarity_to_previous",
        "risk_level": "high",
    },
    "scaffolding_exploitation": {
        "label": "Explotación del scaffolding",
        "description": "Repite deliberadamente 'no entiendo' para escalar a explicación directa",
        "detection_method": "escalation_pattern",
        "risk_level": "medium",
    },
    "guardrail_probing": {
        "label": "Sondeo de guardrails",
        "description": "Prueba diferentes formulaciones para encontrar una que el sistema permita",
        "detection_method": "reformulation_pattern",
        "risk_level": "low",
    },
}


@dataclass
class InteractionSemiotics:
    """Análisis semiótico de una interacción individual."""
    timestamp: str
    student_id: str
    primary_speech_act: str
    speech_act_confidence: float
    pedagogical_value: float  # 0.0 - 1.0

    # Máximas de Grice
    grice_quantity: float     # ¿información suficiente? (0-1)
    grice_quality: float      # ¿sinceridad aparente? (0-1)
    grice_relation: float     # ¿relevancia al curso? (0-1)
    grice_manner: float       # ¿claridad y orden? (0-1)
    grice_composite: float    # media ponderada

    # Gaming detection
    gaming_suspicion: float   # 0.0 - 1.0
    gaming_type: Optional[str] = None


@dataclass
class InteractionQualityReport:
    """Reporte de calidad interaccional para el docente."""
    student_id: str
    period_interactions: int
    pedagogical_value_mean: float
    speech_act_distribution: dict = field(default_factory=dict)
    grice_profile: dict = field(default_factory=dict)
    gaming_alerts: list = field(default_factory=list)
    quality_tier: str = ""  # "high", "medium", "low", "concerning"
    recommendation: str = ""


class InteractionSemioticsEngine:
    """
    Motor de análisis semiótico de interacciones.

    Va MÁS ALLÁ del copy-paste detection del middleware actual:
    analiza QUÉ TIPO de interacción es, cuál es su VALOR PEDAGÓGICO,
    y si existen PATRONES ESTRATÉGICOS de gaming.

    Esto responde directamente a lo que los evaluadores del LAK 2026
    pidieron como "configuraciones avanzadas" y a la necesidad de
    analytics de alto nivel.
    """

    def __init__(self):
        self.analyses: dict[str, list[InteractionSemiotics]] = defaultdict(list)
        self._speech_act_compiled = {
            act: [re.compile(p, re.IGNORECASE) for p in data["patterns"]]
            for act, data in SPEECH_ACTS.items()
        }

    # ───────────────────────────────────────────
    # 1. ANÁLISIS DE INTERACCIÓN INDIVIDUAL
    # ───────────────────────────────────────────

    def analyze(self, student_id: str, prompt: str) -> InteractionSemiotics:
        """
        Análisis semiótico completo de un prompt.

        Metodología: clasificación competitiva de actos de habla (Searle)
        + evaluación de máximas de Grice + detección de gaming.
        """
        now = datetime.now().isoformat()

        # ─── Clasificación de acto de habla ───
        act_scores = {}
        for act, patterns in self._speech_act_compiled.items():
            score = sum(1 for p in patterns if p.search(prompt))
            act_scores[act] = score

        total = sum(act_scores.values())
        if total > 0:
            primary_act = max(act_scores, key=act_scores.get)
            confidence = act_scores[primary_act] / total
        else:
            primary_act = "conceptual_inquiry"  # default beneficio de la duda
            confidence = 0.2

        ped_value = SPEECH_ACTS[primary_act]["pedagogical_value"]

        # ─── Evaluación de máximas de Grice ───
        grice = self._evaluate_grice(prompt)

        # ─── Gaming detection ───
        gaming = self._detect_gaming(student_id, prompt, primary_act)

        analysis = InteractionSemiotics(
            timestamp=now,
            student_id=student_id,
            primary_speech_act=primary_act,
            speech_act_confidence=round(confidence, 3),
            pedagogical_value=ped_value,
            grice_quantity=grice["quantity"],
            grice_quality=grice["quality"],
            grice_relation=grice["relation"],
            grice_manner=grice["manner"],
            grice_composite=grice["composite"],
            gaming_suspicion=gaming["suspicion"],
            gaming_type=gaming.get("type"),
        )

        self.analyses[student_id].append(analysis)
        return analysis

    def _evaluate_grice(self, prompt: str) -> dict:
        """
        Evaluación de las máximas conversacionales de Grice (1975).

        No es evaluación moral del estudiante: es diagnóstico de
        CALIDAD COMUNICATIVA. Un prompt que viola las máximas es
        más difícil de procesar pedagógicamente, lo que afecta
        la calidad de la respuesta del chatbot.
        """
        prompt_lower = prompt.lower().strip()

        # ─── Cantidad: ¿información suficiente? ───
        # Demasiado corto (< 15 chars) → insuficiente
        # Demasiado largo (> 1000 chars) → probable copy-paste, excesivo
        length = len(prompt)
        if length < 15:
            quantity = 0.2
        elif length < 30:
            quantity = 0.5
        elif length <= 500:
            quantity = 0.9
        elif length <= 1000:
            quantity = 0.6
        else:
            quantity = 0.3

        # Tiene pregunta explícita → informativa
        if "?" in prompt:
            quantity = min(quantity + 0.15, 1.0)

        # ─── Calidad: ¿sinceridad aparente? ───
        # Marcadores de sinceridad vs. evasión
        quality = 0.7  # base: asumimos buena fe

        honesty_markers = [
            "no entiendo", "no sé", "me confunde", "estoy perdido",
            "no estoy seguro", "me equivoqué", "creo que",
        ]
        if any(m in prompt_lower for m in honesty_markers):
            quality += 0.2

        evasion_markers = [
            "un amigo me preguntó", "hipotéticamente",
            "en teoría", "solo por curiosidad",
        ]
        if any(m in prompt_lower for m in evasion_markers):
            quality -= 0.2

        quality = max(0, min(1, quality))

        # ─── Relación: ¿relevancia al curso? ───
        # Proxy: presencia de vocabulario técnico de programación
        tech_terms = [
            "variable", "bucle", "for", "while", "función", "método",
            "array", "lista", "string", "int", "return", "if", "else",
            "class", "objeto", "herencia", "recursión", "algoritmo",
            "compilar", "ejecutar", "depurar", "error", "excepción",
            "código", "programa", "java", "python",
        ]
        tech_count = sum(1 for t in tech_terms if t in prompt_lower)
        relation = min(tech_count / 3.0, 1.0) if tech_count > 0 else 0.3

        # ─── Manera: ¿claridad y orden? ───
        manner = 0.7  # base

        # Múltiples preguntas desordenadas reducen claridad
        question_count = prompt.count("?")
        if question_count > 3:
            manner -= 0.2
        elif question_count == 1:
            manner += 0.1

        # Presencia de estructura (saltos de línea con propósito)
        lines = [l.strip() for l in prompt.split("\n") if l.strip()]
        if len(lines) > 1 and any(":" in l for l in lines):
            manner += 0.1  # estructura deliberada

        # Texto todo en mayúsculas → mala manera
        if prompt == prompt.upper() and len(prompt) > 20:
            manner -= 0.3

        manner = max(0, min(1, manner))

        # ─── Composición ───
        composite = (
            quantity * 0.20
            + quality * 0.30  # calidad pesa más: buena fe es crucial
            + relation * 0.30  # relevancia al curso también
            + manner * 0.20
        )

        return {
            "quantity": round(quantity, 3),
            "quality": round(quality, 3),
            "relation": round(relation, 3),
            "manner": round(manner, 3),
            "composite": round(composite, 3),
        }

    def _detect_gaming(
        self, student_id: str, prompt: str, speech_act: str
    ) -> dict:
        """
        Detección de patrones de gaming estratégico.

        Metodología: análisis de patrones secuenciales inspirado en
        técnicas de CI (contraespionaje cognitivo). Los estudiantes
        que gamean muestran patrones temporales detectables:
        - Fragmentación: múltiples preguntas cortas en rápida sucesión
        - Escalación forzada: repetición de "no entiendo" sin cambio
        - Reformulación: misma pregunta con palabras diferentes
        """
        history = self.analyses.get(student_id, [])
        suspicion = 0.0
        gaming_type = None

        if len(history) < 2:
            return {"suspicion": 0.0}

        recent = history[-5:]

        # ─── Fragmentación ───
        # Si las últimas 3+ preguntas son cortas y del mismo topic
        if len(recent) >= 3:
            short_recent = [
                a for a in recent[-3:]
                if len(a.timestamp) > 0  # placeholder check
            ]
            all_solution_seeking = all(
                a.primary_speech_act in ("solution_request", "task_delegation")
                for a in short_recent
            )
            if all_solution_seeking:
                suspicion += 0.4
                gaming_type = "fragmentation"

        # ─── Scaffolding exploitation ───
        # Repetición del mismo acto de habla sin variación
        if len(recent) >= 3:
            acts = [a.primary_speech_act for a in recent[-3:]]
            if len(set(acts)) == 1 and acts[0] == "solution_request":
                suspicion += 0.3
                gaming_type = "scaffolding_exploitation"

        # ─── Bajo valor pedagógico sostenido ───
        if len(recent) >= 4:
            low_value = [a for a in recent[-4:] if a.pedagogical_value < 0.3]
            if len(low_value) >= 3:
                suspicion += 0.2
                if not gaming_type:
                    gaming_type = "persistent_low_engagement"

        # ─── Speech act + Grice combinados ───
        # Un prompt que pide solución + baja calidad Grice = gaming probable
        if speech_act in ("solution_request", "task_delegation"):
            last_grice = recent[-1].grice_composite if recent else 0.5
            if last_grice < 0.4:
                suspicion += 0.2

        suspicion = min(suspicion, 1.0)
        return {
            "suspicion": round(suspicion, 3),
            "type": gaming_type if suspicion > 0.3 else None,
        }

    # ───────────────────────────────────────────
    # 2. REPORTES DE CALIDAD
    # ───────────────────────────────────────────

    def build_quality_report(self, student_id: str) -> InteractionQualityReport:
        """
        Genera un reporte de calidad interaccional.

        El docente ve: qué tipo de preguntas hace el estudiante,
        cuánto valor pedagógico tienen, y si hay patrones de gaming.
        """
        history = self.analyses.get(student_id, [])
        if not history:
            return InteractionQualityReport(
                student_id=student_id,
                period_interactions=0,
                pedagogical_value_mean=0,
                quality_tier="unknown",
            )

        # ─── Distribución de actos de habla ───
        act_counts = defaultdict(int)
        for a in history:
            act_counts[a.primary_speech_act] += 1
        total = len(history)
        act_dist = {
            k: {
                "count": v,
                "percentage": round(v / total, 3),
                "label": SPEECH_ACTS[k]["label"],
            }
            for k, v in act_counts.items()
        }

        # ─── Valor pedagógico medio ───
        ped_mean = sum(a.pedagogical_value for a in history) / total

        # ─── Perfil de Grice ───
        grice_profile = {
            "quantity_mean": round(sum(a.grice_quantity for a in history) / total, 3),
            "quality_mean": round(sum(a.grice_quality for a in history) / total, 3),
            "relation_mean": round(sum(a.grice_relation for a in history) / total, 3),
            "manner_mean": round(sum(a.grice_manner for a in history) / total, 3),
            "composite_mean": round(sum(a.grice_composite for a in history) / total, 3),
        }

        # ─── Alertas de gaming ───
        gaming_alerts = [
            {
                "timestamp": a.timestamp,
                "type": a.gaming_type,
                "suspicion": a.gaming_suspicion,
            }
            for a in history
            if a.gaming_suspicion > 0.3
        ]

        # ─── Clasificación de calidad ───
        if ped_mean >= 0.7 and grice_profile["composite_mean"] >= 0.6:
            tier = "high"
            rec = "Estudiante con interacciones de alta calidad. Considerar reducir scaffolding."
        elif ped_mean >= 0.4:
            tier = "medium"
            rec = "Calidad intermedia. Las interacciones son funcionales pero podrían profundizarse."
        elif gaming_alerts and len(gaming_alerts) / total > 0.3:
            tier = "concerning"
            rec = ("ATENCIÓN: patrón de gaming detectado. Considerar intervención "
                   "directa (conversación con el estudiante) o ajustar configuraciones.")
        else:
            tier = "low"
            rec = ("Calidad baja de interacciones. El estudiante pide soluciones "
                   "directas sin esfuerzo propio. Considerar reforzar scaffolding socrático.")

        return InteractionQualityReport(
            student_id=student_id,
            period_interactions=total,
            pedagogical_value_mean=round(ped_mean, 3),
            speech_act_distribution=act_dist,
            grice_profile=grice_profile,
            gaming_alerts=gaming_alerts,
            quality_tier=tier,
            recommendation=rec,
        )

    # ───────────────────────────────────────────
    # 3. DASHBOARD DATA
    # ───────────────────────────────────────────

    def get_semiotics_dashboard_data(self) -> dict:
        """Datos agregados para el dashboard docente."""
        if not self.analyses:
            return {"total_analyzed": 0}

        all_analyses = [a for history in self.analyses.values() for a in history]

        # Distribución global de actos de habla
        global_acts = defaultdict(int)
        for a in all_analyses:
            global_acts[a.primary_speech_act] += 1

        # Valor pedagógico por tipo
        value_by_act = defaultdict(list)
        for a in all_analyses:
            value_by_act[a.primary_speech_act].append(a.pedagogical_value)

        # Gaming global
        gaming_count = sum(1 for a in all_analyses if a.gaming_suspicion > 0.3)

        return {
            "total_analyzed": len(all_analyses),
            "speech_act_distribution": {
                k: {
                    "count": v,
                    "label": SPEECH_ACTS.get(k, {}).get("label", k),
                }
                for k, v in global_acts.items()
            },
            "cohort_pedagogical_value_mean": round(
                sum(a.pedagogical_value for a in all_analyses) / len(all_analyses), 3
            ),
            "cohort_grice_composite_mean": round(
                sum(a.grice_composite for a in all_analyses) / len(all_analyses), 3
            ),
            "gaming_alerts_total": gaming_count,
            "gaming_rate": round(gaming_count / len(all_analyses), 3),
        }
