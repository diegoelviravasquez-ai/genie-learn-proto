"""
GENERADOR DE LEARNING DESIGN ASISTIDO POR GenAI
═══════════════════════════════════════════════════════════════════════
Módulo diferencial #7 — Ataca el OBJETIVO O2 del paper CSEDU 2025:

    "To design and develop GenAI-enhanced solutions for teachers and
     academic managers improving learning design and academic decision
     making in SLEs for HL support."
     — Delgado-Kloos et al. (CSEDU 2025, Sección 3: Objectives)

PROBLEMA QUE RESUELVE — EL ESLABÓN PERDIDO ENTRE WP2 Y WP3:
═══════════════════════════════════════════════════════════════
El proyecto GENIE Learn tiene un chatbot que genera analytics (WP3) y
un objetivo declarado de mejorar el Learning Design (WP2). Pero nadie
ha construido el PUENTE: el módulo que tome los analytics generados
por el chatbot y los transforme en propuestas de diseño de actividad
para el docente.

El ciclo actual es:
    estudiante usa chatbot → analytics se generan → docente mira dashboard

El ciclo completo sería:
    estudiante usa chatbot → analytics se generan → docente mira dashboard
    → SISTEMA PROPONE diseño de actividad basado en analytics
    → docente decide si adopta/adapta/descarta
    → diseño se implementa → estudiante interactúa → loop

Este módulo cierra el segundo arco del bucle. Sin él, los analytics son
descriptivos. Con él, se vuelven PRESCRIPTIVOS — pero con el docente
como decisor final (HCAI, Dimitriadis 2021).

FUNDAMENTACIÓN TEÓRICA:
────────────────────────
1. Learning Design como artefacto de diseño (Hernández-Leo et al., 2019)
   El LD no es una lista de actividades — es un artefacto semiótico que
   codifica intenciones pedagógicas, secuencias temporales, roles, recursos
   y mecanismos de evaluación. El generador respeta esta ontología.

2. Design Analytics (Hernández-Leo et al., 2019)
   Las métricas de decisiones de diseño y patrones que caracterizan los LD
   alimentan el generador. No inventamos actividades en el vacío: las
   proponemos en función de lo que los datos dicen que falta.

3. Constructive Alignment (Biggs & Tang, 2011)
   Cada actividad propuesta debe alinear: learning outcome esperado ↔
   actividad de aprendizaje ↔ método de evaluación. El generador verifica
   este alineamiento explícitamente.

4. Zone of Proximal Development (Vygotsky, 1978)
   Las actividades se calibran al nivel cognitivo actual del cohorte
   (medido por Bloom) + un escalón. Proponer actividades de Bloom 5
   a un grupo estancado en Bloom 2 no es ambición — es negligencia.

5. Productive Failure (Kapur, 2008, 2014)
   Algunas actividades deliberadamente proponen problemas que el estudiante
   NO puede resolver con sus herramientas actuales, como andamio para la
   instrucción posterior. El generador incluye este tipo de actividad
   cuando el perfil del cohorte indica meseta cognitiva.

6. Principio de agencia docente (Alonso-Prieto, LASI 2025)
   El sistema PROPONE, NUNCA prescribe. Cada diseño generado incluye una
   justificación basada en datos para que el docente pueda evaluar si la
   propuesta tiene sentido en su contexto — contexto que el sistema no
   conoce completamente.

INTEGRACIÓN: Consume datos de cognitive_profiler.py, epistemic_autonomy.py
y ecosystem_dashboard.py. Produce ActivitySequence serializable a JSON y
renderizable en Streamlit.

Autor: Diego Elvira Vásquez · Prototipo CP25/152 · Feb 2026
"""

import json
import math
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timedelta
from collections import Counter, defaultdict


# ═══════════════════════════════════════════════════════════════
# ONTOLOGÍA DEL LEARNING DESIGN
# ═══════════════════════════════════════════════════════════════
# Siguiendo la estructura de IMS Learning Design y la taxonomía
# de actividades de Laurillard (2012): Acquisition, Inquiry,
# Practice, Production, Discussion, Collaboration.

ACTIVITY_TYPES = {
    "acquisition": {
        "name": "Adquisición",
        "description": "Lectura, vídeo, explicación del docente",
        "laurillard_type": "Acquisition",
        "bloom_range": (1, 2),
        "student_role": "receptor activo",
        "teacher_role": "presentador / curador de contenido",
    },
    "inquiry": {
        "name": "Investigación",
        "description": "Búsqueda guiada, exploración de fuentes, experimentación",
        "laurillard_type": "Inquiry",
        "bloom_range": (3, 4),
        "student_role": "investigador",
        "teacher_role": "facilitador / diseñador de la pregunta",
    },
    "practice": {
        "name": "Práctica",
        "description": "Ejercicios, problemas, simulaciones con feedback",
        "laurillard_type": "Practice",
        "bloom_range": (2, 3),
        "student_role": "ejecutor con retroalimentación",
        "teacher_role": "diseñador de problemas / evaluador formativo",
    },
    "production": {
        "name": "Producción",
        "description": "Crear artefacto: código, documento, presentación, diseño",
        "laurillard_type": "Production",
        "bloom_range": (4, 6),
        "student_role": "creador",
        "teacher_role": "mentor / crítico constructivo",
    },
    "discussion": {
        "name": "Discusión",
        "description": "Debate, argumentación, peer review, foro",
        "laurillard_type": "Discussion",
        "bloom_range": (4, 5),
        "student_role": "argumentador / evaluador de pares",
        "teacher_role": "moderador / provocador intelectual",
    },
    "collaboration": {
        "name": "Colaboración",
        "description": "Proyecto en equipo, co-creación, resolución conjunta",
        "laurillard_type": "Collaboration",
        "bloom_range": (3, 6),
        "student_role": "co-constructor de conocimiento",
        "teacher_role": "orquestador / facilitador de grupo",
    },
    "productive_failure": {
        "name": "Fallo Productivo",
        "description": "Problema deliberadamente más allá del nivel actual, sin instrucción previa",
        "laurillard_type": "Practice (advanced)",
        "bloom_range": (4, 5),
        "student_role": "explorador en zona de fracaso constructivo",
        "teacher_role": "diseñador del problema + instructor posterior",
    },
    "metacognitive_reflection": {
        "name": "Reflexión Metacognitiva",
        "description": "Diario de aprendizaje, autoevaluación, revisión de estrategias",
        "laurillard_type": "Production (meta-level)",
        "bloom_range": (5, 6),
        "student_role": "observador de su propio proceso",
        "teacher_role": "guía reflexivo",
    },
}


# ═══════════════════════════════════════════════════════════════
# PLANTILLAS DE ACTIVIDAD POR DOMINIO (programación)
# ═══════════════════════════════════════════════════════════════

ACTIVITY_TEMPLATES = {
    # --- VARIABLES Y TIPOS ---
    "variables": {
        "acquisition": "Lectura del material sobre tipos de datos y declaración de variables. Incluir tabla comparativa de tipos primitivos con ejemplos.",
        "practice": "Serie de 5 ejercicios de declaración, asignación y conversión de tipos. Difficulty: incremental. Feedback automático del chatbot habilitado.",
        "production": "Crear un mini-programa que use al menos 4 tipos de datos diferentes para resolver un problema real (ej: calcular IMC, convertir monedas).",
        "productive_failure": "Dado un programa con errores de tipo (int vs float vs string), predecir la salida SIN ejecutar. Discusión posterior sobre por qué el sistema de tipos importa.",
    },
    # --- BUCLES ---
    "bucles": {
        "acquisition": "Vídeo/lectura sobre for, while, do-while. Énfasis en cuándo elegir cada uno.",
        "inquiry": "Investigar: ¿en qué situaciones un bucle infinito es DESEABLE? (ej: servidores, game loops). Reportar hallazgos.",
        "practice": "8 ejercicios de dificultad creciente: recorrer arrays, acumuladores, búsqueda, filtrado.",
        "production": "Implementar un juego simple (adivinanza de número) que use bucles anidados y condiciones de salida.",
        "discussion": "Debate: ¿es 'for' más legible que 'while'? Defender ambas posiciones con ejemplos de código.",
        "productive_failure": "Optimizar un algoritmo O(n²) con bucles anidados a O(n). Sin pistas previas. Instrucción posterior sobre eficiencia algorítmica.",
    },
    # --- FUNCIONES ---
    "funciones": {
        "acquisition": "Material sobre definición, parámetros, retorno, scope. Incluir analogía con recetas de cocina.",
        "practice": "Descomponer un programa monolítico de 50 líneas en 4-5 funciones. Ejercicio guiado.",
        "production": "Diseñar e implementar una librería de funciones para manipulación de strings (split, reverse, capitalize, etc.).",
        "discussion": "Peer review: intercambiar código de funciones entre compañeros. Evaluar: ¿los nombres son claros? ¿los parámetros son suficientes? ¿hay efectos secundarios?",
        "metacognitive_reflection": "Escribir un párrafo: '¿Cuándo decido crear una función nueva vs. escribir el código directamente? ¿Qué criterios uso?'",
    },
    # --- RECURSIÓN ---
    "recursión": {
        "acquisition": "Material con visualización de call stack. Analogía con las muñecas rusas (matrioshka).",
        "inquiry": "Investigar: ¿qué problemas son NATURALMENTE recursivos? (árboles, fractales, sistemas de archivos). Encontrar 3 ejemplos en la vida real.",
        "practice": "Implementar factorial, fibonacci, búsqueda binaria recursiva. Con y sin memoización.",
        "productive_failure": "Resolver el problema de las Torres de Hanoi con recursión. Sin explicación previa del algoritmo. Solo la regla: no puedes poner un disco grande sobre uno pequeño.",
        "collaboration": "En parejas: uno escribe la versión recursiva, otro la iterativa del mismo problema. Comparar rendimiento y legibilidad.",
    },
    # --- ARRAYS / ESTRUCTURAS ---
    "arrays": {
        "acquisition": "Material sobre arrays, listas, indexación, slicing.",
        "practice": "10 ejercicios: búsqueda, inserción, eliminación, ordenamiento básico.",
        "production": "Implementar un sistema de gestión de notas de estudiantes usando arrays (CRUD completo).",
        "discussion": "¿Array vs. lista enlazada? Argumentar con complejidad algorítmica. Preparar 2 slides cada uno.",
    },
    # --- GENERAL / CONCEPTUAL ---
    "conceptual": {
        "metacognitive_reflection": "Autoevaluación: ¿qué temas domino? ¿dónde me atasco? ¿qué estrategia de estudio uso con el chatbot?",
        "discussion": "Mesa redonda: ¿el chatbot me está ayudando a aprender o me está resolviendo los ejercicios? Argumentar con evidencia de tu propio uso.",
    },
}


# ═══════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS
# ═══════════════════════════════════════════════════════════════

@dataclass
class CohortAnalytics:
    """Datos agregados del cohorte — input del generador."""
    total_students: int = 0
    bloom_distribution: dict = field(default_factory=dict)      # {1: 12, 2: 25, 3: 18, ...}
    median_bloom: float = 0.0
    topic_difficulty: dict = field(default_factory=dict)         # {"recursión": 0.82, "bucles": 0.31}
    engagement_profiles: dict = field(default_factory=dict)      # {"explorador": 8, "delegador": 5, ...}
    autonomy_distribution: dict = field(default_factory=dict)    # {"dependent": 3, "scaffolded": 12, ...}
    problematic_topics: list = field(default_factory=list)       # topics con >40% Bloom bajo
    gaming_rate: float = 0.0                                     # % estudiantes detectados como gaming
    avg_prompts_per_session: float = 0.0
    productive_struggle_rate: float = 0.0                        # % en zona de dificultad deseable
    plateau_detected: bool = False                               # meseta cognitiva grupal
    timestamp: str = ""


@dataclass
class LearningActivity:
    """Una actividad individual dentro de una secuencia de LD."""
    activity_id: str
    activity_type: str                  # key de ACTIVITY_TYPES
    title: str
    description: str
    bloom_target: int                   # nivel Bloom objetivo (1-6)
    estimated_minutes: int
    student_role: str
    teacher_role: str
    chatbot_role: str                   # cómo debe configurarse el chatbot durante esta actividad
    suggested_config: dict = field(default_factory=dict)  # PedagogicalConfig sugerida
    alignment: dict = field(default_factory=dict)         # {outcome, activity, assessment}
    data_justification: str = ""        # POR QUÉ esta actividad, basado en los datos
    topic: str = ""
    prerequisites: list = field(default_factory=list)
    sequence_position: int = 0


@dataclass
class ActivitySequence:
    """Secuencia completa de Learning Design generada."""
    sequence_id: str
    title: str
    generated_at: str
    target_topic: str
    target_bloom_range: tuple = (1, 6)
    total_estimated_minutes: int = 0
    activities: list = field(default_factory=list)       # list[LearningActivity]
    cohort_summary: str = ""
    design_rationale: str = ""          # fundamentación completa
    teacher_notes: str = ""             # notas para el docente
    config_recommendations: dict = field(default_factory=dict)
    data_snapshot: dict = field(default_factory=dict)    # analytics que generaron este LD


@dataclass
class DesignDecision:
    """Registro de una decisión de diseño del generador — trazabilidad."""
    decision_id: str
    decision_type: str          # "activity_selection" | "sequence_order" | "config_suggestion"
    rationale: str              # por qué se tomó esta decisión
    data_evidence: str          # qué dato la sustenta
    theoretical_basis: str      # qué marco teórico la justifica
    alternatives_considered: list = field(default_factory=list)
    confidence: float = 0.0


# ═══════════════════════════════════════════════════════════════
# MOTOR DE GENERACIÓN DE LEARNING DESIGN
# ═══════════════════════════════════════════════════════════════

class LearningDesignGenerator:
    """
    Genera secuencias de actividades de aprendizaje basadas en
    analytics del cohorte.

    Principio rector: el sistema informa, no prescribe.
    Cada actividad incluye su justificación basada en datos
    para que el docente pueda evaluar y decidir.

    Flujo:
        1. Recibe CohortAnalytics (del dashboard)
        2. Diagnostica necesidades del cohorte
        3. Selecciona actividades apropiadas
        4. Ordena en secuencia pedagógicamente coherente
        5. Sugiere configuraciones del chatbot para cada actividad
        6. Genera la fundamentación y notas docentes
    """

    def __init__(self):
        self.design_log: list[DesignDecision] = []
        self._decision_counter = 0

    # ──────────────────────────────────────────────
    # DIAGNÓSTICO DEL COHORTE
    # ──────────────────────────────────────────────

    def _diagnose_cohort(self, analytics: CohortAnalytics) -> dict:
        """
        Primer paso: analiza los datos del cohorte para identificar
        necesidades pedagógicas.

        Produce un diccionario de 'señales' que guían la generación.
        Cada señal es una observación + implicación pedagógica.
        """
        signals = {
            "bloom_floor": False,        # >50% en Bloom 1-2
            "bloom_ceiling": False,       # >30% en Bloom 5-6
            "bloom_plateau": False,       # meseta sin progresión
            "topic_bottleneck": [],       # topics problemáticos
            "gaming_alert": False,        # >20% gaming
            "low_autonomy": False,        # >60% en dependent/scaffolded
            "high_autonomy": False,       # >40% en transitional/autonomous
            "engagement_skew": "",        # perfil dominante del cohorte
            "needs_metacognition": False, # baja reflexión detectada
            "productive_struggle_low": False,  # poca dificultad deseable
        }

        # --- Análisis de distribución Bloom ---
        total = sum(analytics.bloom_distribution.values()) or 1
        low_bloom = sum(analytics.bloom_distribution.get(k, 0) for k in [1, 2])
        high_bloom = sum(analytics.bloom_distribution.get(k, 0) for k in [5, 6])

        if low_bloom / total > 0.50:
            signals["bloom_floor"] = True
            self._log_decision("diagnosis", "bloom_floor",
                f"{low_bloom}/{total} estudiantes ({low_bloom/total:.0%}) en Bloom 1-2",
                "Biggs & Tang (2011): actividades deben subir un escalón desde nivel actual")
        if high_bloom / total > 0.30:
            signals["bloom_ceiling"] = True
            self._log_decision("diagnosis", "bloom_ceiling",
                f"{high_bloom}/{total} estudiantes en Bloom 5-6",
                "Renzulli (2005): estudiantes avanzados necesitan enriquecimiento, no repetición")

        signals["bloom_plateau"] = analytics.plateau_detected

        # --- Topics problemáticos ---
        for topic, difficulty in analytics.topic_difficulty.items():
            if difficulty > 0.55:
                signals["topic_bottleneck"].append(topic)

        # --- Gaming ---
        if analytics.gaming_rate > 0.20:
            signals["gaming_alert"] = True
            self._log_decision("diagnosis", "gaming_alert",
                f"Gaming rate: {analytics.gaming_rate:.0%}",
                "Desmoralización o desalineación entre actividades y evaluación (Baker et al., 2010)")

        # --- Autonomía ---
        dep = analytics.autonomy_distribution.get("dependent", 0)
        scaf = analytics.autonomy_distribution.get("scaffolded", 0)
        trans = analytics.autonomy_distribution.get("transitional", 0)
        auto = analytics.autonomy_distribution.get("autonomous", 0)
        total_auto = dep + scaf + trans + auto or 1

        if (dep + scaf) / total_auto > 0.60:
            signals["low_autonomy"] = True
        if (trans + auto) / total_auto > 0.40:
            signals["high_autonomy"] = True

        # --- Perfil dominante ---
        if analytics.engagement_profiles:
            dominant = max(analytics.engagement_profiles, key=analytics.engagement_profiles.get)
            signals["engagement_skew"] = dominant

        # --- Metacognición ---
        bloom_5_6 = sum(analytics.bloom_distribution.get(k, 0) for k in [5, 6])
        if bloom_5_6 / total < 0.10:
            signals["needs_metacognition"] = True

        # --- Productive struggle ---
        if analytics.productive_struggle_rate < 0.15:
            signals["productive_struggle_low"] = True

        return signals

    # ──────────────────────────────────────────────
    # SELECCIÓN DE ACTIVIDADES
    # ──────────────────────────────────────────────

    def _select_activities(self, signals: dict, analytics: CohortAnalytics,
                           target_topic: str) -> list[LearningActivity]:
        """
        Selecciona actividades basadas en las señales diagnósticas.

        Lógica:
        - bloom_floor → más practice + acquisition
        - bloom_ceiling → production + discussion
        - topic_bottleneck → actividades específicas del topic
        - gaming_alert → productive_failure + discussion
        - low_autonomy → práctica guiada → inquiry gradual
        - needs_metacognition → metacognitive_reflection
        - productive_struggle_low → productive_failure
        """
        selected = []
        act_id = 0

        topic_key = self._normalize_topic(target_topic)
        templates = ACTIVITY_TEMPLATES.get(topic_key, ACTIVITY_TEMPLATES.get("conceptual", {}))

        def make_activity(atype: str, bloom: int, minutes: int,
                          title: str, description: str, justification: str,
                          chatbot_role: str = "asistente estándar",
                          config: dict = None) -> LearningActivity:
            nonlocal act_id
            act_id += 1
            at = ACTIVITY_TYPES.get(atype, ACTIVITY_TYPES["practice"])
            return LearningActivity(
                activity_id=f"LD-{act_id:03d}",
                activity_type=atype,
                title=title,
                description=description or templates.get(atype, ""),
                bloom_target=bloom,
                estimated_minutes=minutes,
                student_role=at["student_role"],
                teacher_role=at["teacher_role"],
                chatbot_role=chatbot_role,
                suggested_config=config or {},
                alignment={
                    "outcome": f"Alcanzar Bloom {bloom} en {target_topic}",
                    "activity": at["name"],
                    "assessment": self._suggest_assessment(atype, bloom),
                },
                data_justification=justification,
                topic=target_topic,
                sequence_position=act_id,
            )

        # ── Bloque 1: Responder al nivel actual ──

        if signals["bloom_floor"]:
            selected.append(make_activity(
                "acquisition", 2, 20,
                f"Fundamentos de {target_topic}",
                templates.get("acquisition", f"Lectura/vídeo sobre conceptos base de {target_topic}."),
                f"El {sum(analytics.bloom_distribution.get(k,0) for k in [1,2])}/{sum(analytics.bloom_distribution.values())} "
                f"del cohorte está en Bloom 1-2. Necesitan base conceptual antes de practicar.",
                chatbot_role="explicador — respuestas directas y ejemplos",
                config={"scaffolding_mode": "direct", "block_direct_solutions": False},
            ))
            selected.append(make_activity(
                "practice", 3, 30,
                f"Ejercicios guiados de {target_topic}",
                templates.get("practice", f"Serie de ejercicios incrementales sobre {target_topic}."),
                "Después de la adquisición, la práctica guiada consolida. "
                "Bloom 2→3: de comprender a aplicar (Anderson & Krathwohl, 2001).",
                chatbot_role="tutor socrático — guía sin resolver",
                config={"scaffolding_mode": "socratic", "block_direct_solutions": True},
            ))

        # ── Bloque 2: Topics problemáticos ──

        if target_topic in signals["topic_bottleneck"] or \
           self._normalize_topic(target_topic) in [self._normalize_topic(t) for t in signals["topic_bottleneck"]]:
            diff = analytics.topic_difficulty.get(target_topic,
                   analytics.topic_difficulty.get(topic_key, 0.5))
            selected.append(make_activity(
                "inquiry", 4, 35,
                f"Investigación guiada: ¿por qué {target_topic} es difícil?",
                f"Explorar las dificultades conceptuales de {target_topic}. "
                f"Buscar analogías y contraejemplos. Reportar hallazgos.",
                f"Topic '{target_topic}' tiene dificultad {diff:.0%} en el cohorte. "
                f"La investigación guiada activa procesamiento profundo (Chi & Wylie, 2014).",
                chatbot_role="guía de investigación — no da respuestas, sugiere fuentes",
                config={"scaffolding_mode": "socratic", "use_rag": True},
            ))

        # ── Bloque 3: Gaming → actividades que no admiten atajos ──

        if signals["gaming_alert"]:
            selected.append(make_activity(
                "productive_failure", 4, 40,
                f"Reto sin instrucción previa: {target_topic} avanzado",
                templates.get("productive_failure",
                    f"Problema de {target_topic} deliberadamente más allá del nivel actual. "
                    f"Sin acceso al chatbot. Discusión posterior."),
                f"Gaming rate del {analytics.gaming_rate:.0%}. Las actividades de Productive Failure "
                f"(Kapur, 2008) no admiten copy-paste porque el problema no tiene solución directa disponible.",
                chatbot_role="DESACTIVADO durante la actividad. Activo solo en la discusión posterior.",
                config={"max_daily_prompts": 0},  # bloquea chatbot durante actividad
            ))
            selected.append(make_activity(
                "discussion", 5, 25,
                "Debate: ¿cómo estamos usando el chatbot?",
                templates.get("discussion",
                    "Mesa redonda: ¿el chatbot me ayuda a aprender o a evitar aprender? "
                    "Argumentar con evidencia del propio uso."),
                "El debate sobre metacognición del uso del chatbot confronta directamente "
                "el gaming — no con prohibición sino con conciencia (Winne & Hadwin, 1998).",
                chatbot_role="ausente — la reflexión es humana",
            ))

        # ── Bloque 4: Baja autonomía → escalera gradual ──

        if signals["low_autonomy"] and not signals["bloom_floor"]:
            selected.append(make_activity(
                "practice", 3, 25,
                f"Práctica con scaffolding decreciente: {target_topic}",
                f"3 ejercicios sobre {target_topic}. "
                f"Primero: chatbot da pistas. Segundo: chatbot solo confirma/niega. "
                f"Tercero: chatbot desactivado.",
                "El 60%+ del cohorte está en modos dependent/scaffolded. "
                "El scaffolding fading (Wood, Bruner & Ross, 1976) requiere "
                "retirada GRADUAL, no abrupta.",
                chatbot_role="decreciente — de guía activo a verificador a ausente",
                config={"scaffolding_mode": "hints"},
            ))

        # ── Bloque 5: Alta autonomía → enriquecimiento ──

        if signals["bloom_ceiling"] or signals["high_autonomy"]:
            selected.append(make_activity(
                "production", 6, 45,
                f"Proyecto creativo: {target_topic} aplicado",
                templates.get("production",
                    f"Diseñar e implementar una solución original que use {target_topic} "
                    f"para resolver un problema real elegido por el estudiante."),
                "El 30%+ del cohorte está en Bloom 5-6 o autonomía alta. "
                "Necesitan enriquecimiento (Renzulli, 2005), no más ejercicios.",
                chatbot_role="mentor — discute diseño, no da código",
                config={"scaffolding_mode": "socratic", "block_direct_solutions": True},
            ))

        # ── Bloque 6: Metacognición ──

        if signals["needs_metacognition"]:
            selected.append(make_activity(
                "metacognitive_reflection", 5, 15,
                "Diario de aprendizaje: ¿qué he aprendido y cómo?",
                templates.get("metacognitive_reflection",
                    "Escribir un párrafo reflexivo: ¿qué conceptos domino? "
                    "¿dónde me atasco? ¿qué estrategia de estudio uso? "
                    "¿cómo ha cambiado mi comprensión en las últimas sesiones?"),
                f"Solo el {sum(analytics.bloom_distribution.get(k,0) for k in [5,6])}"
                f"/{sum(analytics.bloom_distribution.values())} del cohorte alcanza Bloom 5-6. "
                "La reflexión metacognitiva es el puente hacia niveles superiores (Flavell, 1979).",
                chatbot_role="espejo — reformula lo que el estudiante dice, no añade contenido",
            ))

        # ── Bloque 7: Productive struggle bajo ──

        if signals["productive_struggle_low"] and not signals["gaming_alert"]:
            selected.append(make_activity(
                "productive_failure", 4, 35,
                f"Desafío calibrado: {target_topic}",
                f"Problema de {target_topic} un nivel por encima del promedio del cohorte. "
                f"Permitir 15 min de exploración antes de cualquier instrucción.",
                f"Productive struggle rate del {analytics.productive_struggle_rate:.0%} "
                f"es demasiado bajo. Los 'desirable difficulties' (Bjork, 1994) son "
                f"necesarios para el aprendizaje profundo.",
                chatbot_role="limitado — solo responde después de 15 min",
                config={"scaffolding_mode": "hints", "max_daily_prompts": 3},
            ))

        # ── Meseta cognitiva → ruptura deliberada ──

        if signals["bloom_plateau"]:
            selected.append(make_activity(
                "collaboration", 5, 50,
                f"Proyecto colaborativo: {target_topic} en equipo",
                f"En equipos de 3: diseñar una solución a un problema de {target_topic} "
                f"que ninguno podría resolver solo. Cada miembro aporta una pieza.",
                "Meseta cognitiva detectada. La colaboración introduce perspectivas "
                "que rompen el estancamiento individual (Dillenbourg, 1999).",
                chatbot_role="consultor de equipo — responde al grupo, no a individuos",
                config={"scaffolding_mode": "direct", "use_rag": True},
            ))

        return selected

    # ──────────────────────────────────────────────
    # SECUENCIACIÓN PEDAGÓGICA
    # ──────────────────────────────────────────────

    def _sequence_activities(self, activities: list[LearningActivity]) -> list[LearningActivity]:
        """
        Ordena las actividades en secuencia pedagógicamente coherente.

        Principio de Constructive Alignment (Biggs & Tang, 2011):
        acquisition → practice → inquiry → production → discussion → reflection

        Con excepción: productive_failure puede ir ANTES de la instrucción
        (esa es su esencia — Kapur, 2008).
        """
        type_order = {
            "acquisition": 1,
            "practice": 2,
            "inquiry": 3,
            "productive_failure": 3.5,  # junto con inquiry
            "production": 4,
            "collaboration": 4.5,
            "discussion": 5,
            "metacognitive_reflection": 6,
        }

        sorted_activities = sorted(activities, key=lambda a: type_order.get(a.activity_type, 3))

        for i, act in enumerate(sorted_activities):
            act.sequence_position = i + 1

        return sorted_activities

    # ──────────────────────────────────────────────
    # GENERACIÓN DE FUNDAMENTACIÓN
    # ──────────────────────────────────────────────

    def _generate_rationale(self, signals: dict, analytics: CohortAnalytics,
                            activities: list[LearningActivity]) -> str:
        """Genera la fundamentación completa del diseño propuesto."""
        parts = []
        parts.append(f"Diseño generado para un cohorte de {analytics.total_students} estudiantes.")
        parts.append(f"Bloom mediano: {analytics.median_bloom:.1f}. "
                     f"Se proponen {len(activities)} actividades.\n")

        if signals["bloom_floor"]:
            parts.append("▸ SEÑAL: Mayoría del cohorte en niveles cognitivos bajos (Bloom 1-2). "
                        "Se incluyen actividades de adquisición y práctica guiada para "
                        "construir base antes de escalar.")

        if signals["topic_bottleneck"]:
            topics = ", ".join(signals["topic_bottleneck"])
            parts.append(f"▸ SEÑAL: Topics problemáticos detectados: {topics}. "
                        "Se incluyen actividades de investigación guiada específicas.")

        if signals["gaming_alert"]:
            parts.append(f"▸ SEÑAL: Gaming rate elevado ({analytics.gaming_rate:.0%}). "
                        "Se incluyen actividades de fallo productivo (sin chatbot) "
                        "y debate metacognitivo sobre uso del chatbot.")

        if signals["low_autonomy"]:
            parts.append("▸ SEÑAL: Baja autonomía epistémica. Se implementa scaffolding "
                        "fading en tres fases dentro de la misma sesión.")

        if signals["bloom_plateau"]:
            parts.append("▸ SEÑAL: Meseta cognitiva grupal. Se introduce actividad "
                        "colaborativa para romper estancamiento por perspectiva social.")

        if signals["needs_metacognition"]:
            parts.append("▸ SEÑAL: Baja reflexión metacognitiva. Se añade diario "
                        "de aprendizaje como cierre de la secuencia.")

        parts.append("\n─── MARCOS TEÓRICOS INVOCADOS ───")
        parts.append("• Constructive Alignment (Biggs & Tang, 2011) → secuencia de actividades")
        parts.append("• Bloom revisado (Anderson & Krathwohl, 2001) → niveles objetivo")
        parts.append("• Scaffolding fading (Wood, Bruner & Ross, 1976) → autonomía gradual")
        parts.append("• Productive Failure (Kapur, 2008) → dificultad deseable")
        parts.append("• ICAP framework (Chi & Wylie, 2014) → tipo de engagement por actividad")
        parts.append("• Conversational Framework (Laurillard, 2012) → tipología de actividades")

        return "\n".join(parts)

    # ──────────────────────────────────────────────
    # INTERFAZ PÚBLICA
    # ──────────────────────────────────────────────

    def generate(self, analytics: CohortAnalytics, target_topic: str) -> ActivitySequence:
        """
        Genera una secuencia de Learning Design basada en analytics del cohorte.

        Args:
            analytics: Datos agregados del cohorte
            target_topic: Tema sobre el que diseñar la secuencia

        Returns:
            ActivitySequence con actividades, fundamentación y recomendaciones
        """
        self.design_log = []

        # 1. Diagnosticar
        signals = self._diagnose_cohort(analytics)

        # 2. Seleccionar actividades
        activities = self._select_activities(signals, analytics, target_topic)

        # 3. Si no se generaron actividades (datos insuficientes), crear mínimo
        if not activities:
            activities = [
                LearningActivity(
                    activity_id="LD-001",
                    activity_type="practice",
                    title=f"Práctica estándar: {target_topic}",
                    description=f"Ejercicios de {target_topic} con dificultad incremental.",
                    bloom_target=3,
                    estimated_minutes=30,
                    student_role="ejecutor",
                    teacher_role="evaluador formativo",
                    chatbot_role="asistente estándar",
                    data_justification="Datos insuficientes para diagnóstico específico. "
                                      "Actividad genérica como punto de partida.",
                    topic=target_topic,
                    sequence_position=1,
                ),
            ]

        # 4. Secuenciar
        sequenced = self._sequence_activities(activities)

        # 5. Calcular tiempo total
        total_minutes = sum(a.estimated_minutes for a in sequenced)

        # 6. Generar fundamentación
        rationale = self._generate_rationale(signals, analytics, sequenced)

        # 7. Construir secuencia
        sequence = ActivitySequence(
            sequence_id=f"LD-SEQ-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            title=f"Secuencia de {target_topic} — {len(sequenced)} actividades",
            generated_at=datetime.now().isoformat(),
            target_topic=target_topic,
            target_bloom_range=(
                min(a.bloom_target for a in sequenced),
                max(a.bloom_target for a in sequenced),
            ),
            total_estimated_minutes=total_minutes,
            activities=sequenced,
            cohort_summary=f"{analytics.total_students} estudiantes, "
                          f"Bloom mediano {analytics.median_bloom:.1f}, "
                          f"{len(signals['topic_bottleneck'])} topics problemáticos.",
            design_rationale=rationale,
            teacher_notes=self._generate_teacher_notes(signals, sequenced),
            config_recommendations=self._generate_config_recommendations(sequenced),
            data_snapshot={
                "bloom_distribution": analytics.bloom_distribution,
                "autonomy_distribution": analytics.autonomy_distribution,
                "gaming_rate": analytics.gaming_rate,
                "problematic_topics": analytics.problematic_topics,
                "signals_detected": {k: v for k, v in signals.items() if v},
            },
        )

        return sequence

    def generate_from_raw_logs(self, interaction_logs: list[dict],
                                target_topic: str) -> ActivitySequence:
        """
        Genera LD directamente desde logs de interacción crudos.
        Convierte logs → CohortAnalytics → genera.
        """
        analytics = self._logs_to_analytics(interaction_logs)
        return self.generate(analytics, target_topic)

    # ──────────────────────────────────────────────
    # MÉTODOS AUXILIARES
    # ──────────────────────────────────────────────

    def _logs_to_analytics(self, logs: list[dict]) -> CohortAnalytics:
        """Convierte logs crudos de interacción a CohortAnalytics."""
        if not logs:
            return CohortAnalytics(timestamp=datetime.now().isoformat())

        students = set()
        bloom_counts = Counter()
        topic_scores = defaultdict(list)
        prompts_per_student = Counter()

        for log in logs:
            sid = log.get("student_id", "unknown")
            students.add(sid)
            bl = log.get("bloom_level", 2)
            bloom_counts[bl] += 1
            prompts_per_student[sid] += 1
            for topic in log.get("topics", []):
                topic_scores[topic].append(bl)

        total_students = len(students)
        bloom_dist = dict(bloom_counts)
        all_blooms = [log.get("bloom_level", 2) for log in logs]
        median_bloom = sorted(all_blooms)[len(all_blooms) // 2] if all_blooms else 2.0

        topic_diff = {}
        problematic = []
        for topic, scores in topic_scores.items():
            low = sum(1 for s in scores if s <= 2)
            difficulty = low / len(scores) if scores else 0
            topic_diff[topic] = difficulty
            if difficulty > 0.40:
                problematic.append(topic)

        gaming_count = sum(1 for log in logs if log.get("copy_paste_score", 0) > 0.7)
        gaming_rate = gaming_count / len(logs) if logs else 0

        avg_prompts = sum(prompts_per_student.values()) / total_students if total_students else 0

        return CohortAnalytics(
            total_students=total_students,
            bloom_distribution=bloom_dist,
            median_bloom=median_bloom,
            topic_difficulty=topic_diff,
            problematic_topics=problematic,
            gaming_rate=gaming_rate,
            avg_prompts_per_session=avg_prompts,
            timestamp=datetime.now().isoformat(),
        )

    def _normalize_topic(self, topic: str) -> str:
        """Normaliza el nombre del topic para matchear templates."""
        mapping = {
            "variable": "variables", "tipo": "variables", "declaración": "variables",
            "bucle": "bucles", "for": "bucles", "while": "bucles", "loop": "bucles",
            "función": "funciones", "function": "funciones", "método": "funciones",
            "recursión": "recursión", "recursividad": "recursión", "recursivo": "recursión",
            "array": "arrays", "lista": "arrays", "arreglo": "arrays",
        }
        t = topic.lower().strip()
        return mapping.get(t, t)

    def _suggest_assessment(self, activity_type: str, bloom: int) -> str:
        """Sugiere método de evaluación alineado con actividad y Bloom."""
        assessments = {
            ("acquisition", 1): "Quiz de comprensión rápida (5 preguntas tipo test)",
            ("acquisition", 2): "Resumen en 3 frases del material leído",
            ("practice", 2): "Corrección automática de ejercicios + feedback",
            ("practice", 3): "Review del código/solución por rúbrica de 4 criterios",
            ("inquiry", 4): "Informe breve con hallazgos y fuentes citadas",
            ("production", 5): "Rúbrica de proyecto: funcionalidad + diseño + documentación",
            ("production", 6): "Presentación de 5 min + evaluación de pares",
            ("discussion", 4): "Participación cualitativa + calidad de argumentos",
            ("discussion", 5): "Autoevaluación + evaluación de pares",
            ("collaboration", 5): "Evaluación grupal + contribución individual",
            ("productive_failure", 4): "Reflexión sobre el proceso (no el resultado)",
            ("metacognitive_reflection", 5): "Evaluación formativa del diario",
        }
        return assessments.get((activity_type, bloom),
               assessments.get((activity_type, min(bloom, 5)),
               "Evaluación formativa por el docente"))

    def _generate_teacher_notes(self, signals: dict, activities: list[LearningActivity]) -> str:
        """Notas prácticas para el docente."""
        notes = []
        notes.append(f"Secuencia de {len(activities)} actividades, "
                     f"~{sum(a.estimated_minutes for a in activities)} minutos total.")
        notes.append("")

        if signals["gaming_alert"]:
            notes.append("⚠ GAMING DETECTADO: Se incluye actividad sin chatbot. "
                        "Considere comunicar al grupo por qué se desactiva "
                        "temporalmente — la transparencia reduce la resistencia.")

        if signals["bloom_plateau"]:
            notes.append("📊 MESETA COGNITIVA: El cohorte no progresa. "
                        "La actividad colaborativa busca romper el estancamiento. "
                        "Si persiste, considere revisar el material de base.")

        notes.append("")
        notes.append("CONFIGURACIÓN DEL CHATBOT POR ACTIVIDAD:")
        for act in activities:
            if act.suggested_config:
                config_str = ", ".join(f"{k}={v}" for k, v in act.suggested_config.items())
                notes.append(f"  {act.activity_id} ({act.title}): {config_str}")
            elif act.chatbot_role:
                notes.append(f"  {act.activity_id} ({act.title}): {act.chatbot_role}")

        notes.append("")
        notes.append("NOTA: Este diseño es una PROPUESTA basada en datos. "
                     "Usted conoce el contexto mejor que el sistema. "
                     "Adapte, reordene o descarte según su criterio profesional.")

        return "\n".join(notes)

    def _generate_config_recommendations(self, activities: list) -> dict:
        """Genera recomendaciones de PedagogicalConfig para la secuencia."""
        configs = {}
        for act in activities:
            if act.suggested_config:
                configs[act.activity_id] = {
                    "activity": act.title,
                    "config": act.suggested_config,
                    "reason": act.data_justification[:120] + "...",
                }
        return configs

    def _log_decision(self, dtype: str, did: str, evidence: str, theory: str):
        """Registra una decisión de diseño para trazabilidad."""
        self._decision_counter += 1
        self.design_log.append(DesignDecision(
            decision_id=f"DD-{self._decision_counter:03d}",
            decision_type=dtype,
            rationale=did,
            data_evidence=evidence,
            theoretical_basis=theory,
        ))

    def to_json(self, sequence: ActivitySequence) -> str:
        """Serializa la secuencia completa a JSON."""
        def serialize(obj):
            if hasattr(obj, '__dict__'):
                return {k: serialize(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list):
                return [serialize(i) for i in obj]
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        return json.dumps(serialize(sequence), indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# DEMO EJECUTABLE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("LEARNING DESIGN GENERATOR — Demo")
    print("=" * 70)

    # Cohorte simulado: 45 estudiantes, dificultad en recursión, algo de gaming
    analytics = CohortAnalytics(
        total_students=45,
        bloom_distribution={1: 5, 2: 15, 3: 12, 4: 8, 5: 4, 6: 1},
        median_bloom=2.6,
        topic_difficulty={"recursión": 0.72, "bucles": 0.25, "funciones": 0.38},
        engagement_profiles={"explorador": 12, "delegador": 8, "verificador": 15, "ausente": 10},
        autonomy_distribution={"dependent": 8, "scaffolded": 20, "transitional": 12, "autonomous": 5},
        problematic_topics=["recursión"],
        gaming_rate=0.18,
        productive_struggle_rate=0.10,
        plateau_detected=False,
    )

    generator = LearningDesignGenerator()
    sequence = generator.generate(analytics, "recursión")

    print(f"\n📋 {sequence.title}")
    print(f"   Generado: {sequence.generated_at}")
    print(f"   Tiempo total: {sequence.total_estimated_minutes} min")
    print(f"   Bloom target: {sequence.target_bloom_range}")
    print()

    for act in sequence.activities:
        print(f"  [{act.sequence_position}] {act.activity_id} — {act.title}")
        print(f"      Tipo: {act.activity_type} | Bloom: {act.bloom_target} | {act.estimated_minutes} min")
        print(f"      Chatbot: {act.chatbot_role}")
        print(f"      Justificación: {act.data_justification[:100]}...")
        if act.suggested_config:
            print(f"      Config: {act.suggested_config}")
        print()

    print("─── FUNDAMENTACIÓN ───")
    print(sequence.design_rationale)
    print()
    print("─── NOTAS DOCENTES ───")
    print(sequence.teacher_notes)
    print()
    print(f"Decisiones de diseño registradas: {len(generator.design_log)}")
    for dd in generator.design_log:
        print(f"  {dd.decision_id}: {dd.rationale} → {dd.theoretical_basis[:60]}...")
