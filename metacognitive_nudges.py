"""
METACOGNITIVE NUDGE GENERATOR
═══════════════════════════════════════════════════════════════════════
Módulo Día 1 — Bloque 8: Analytics PARA el estudiante, no SOBRE el estudiante.

PROBLEMA QUE ATACA:
════════════════════
Todos los módulos analíticos de GENIE Learn (Bloom, autonomía, trust,
semiótica, reflexividad) generan datos visibles para el docente e
investigador. El estudiante es el objeto medido, nunca el sujeto
informado de su propia trayectoria.

Esto contradice tres décadas de investigación en metacognición:
    - Flavell (1979): metacognitive knowledge + metacognitive regulation
    - Zimmerman (2002): self-regulated learning requiere self-observation
    - Winne & Hadwin (1998): SRL como ciclo donde monitoring alimenta control
    - Villa-Torrano et al. (2025, C&E): SSRL requiere awareness compartida

El gap: los learning analytics miden la regulación del aprendizaje,
pero no la PROMUEVEN. Este módulo convierte los analytics en
herramientas de autorregulación inyectando nudges metacognitivos
en el flujo del chat.

DISEÑO:
───────
Un nudge NO es un dashboard. No dice "tu autonomía es 0.45".
Dice algo que provoca reflexión metacognitiva — la diferencia entre
un termómetro (mide la temperatura) y una ventana (te hace sentir
el frío).

Tipos de nudges:
    1. PROGRESIÓN   — cuando el estudiante sube de fase de autonomía
    2. ESFUERZO     — cuando demuestra productive struggle (Kapur, 2008)
    3. SPACING      — cuando vuelve a un topic tras 48+ horas
    4. DESACOPLE    — cuando el scaffolding fading indica independencia
    5. REFLEXIÓN    — periódico, invita a mirar hacia atrás
    6. SANDBOX      — cuando el estudiante está en modo práctica libre

Frecuencia: máximo 1 nudge cada N interacciones (configurable por docente).
El nudge es sutil, no intrusivo — un susurro, no un altavoz.

FUNDAMENTACIÓN:
    Flavell, J. (1979). "Metacognition and cognitive monitoring."
    Zimmerman, B. (2002). "Becoming a self-regulated learner."
    Winne, P. & Hadwin, A. (1998). "Studying as self-regulated learning."
    Kapur, M. (2008). "Productive failure."
    Villa-Torrano et al. (2025). SSRL review, Computers & Education.
    Bjork, R.A. (1994). "Memory and metamemory considerations."
    Wood, Bruner & Ross (1976). "The role of tutoring in problem solving."

Autor: Diego Elvira Vásquez · Prototipo CP25/152 · Feb 2026
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime, timedelta
import random


# ═══════════════════════════════════════════════════════════════════════
# MODELO DE DATOS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MetacognitiveNudge:
    """
    Un nudge metacognitivo destinado al estudiante.

    No es una métrica. Es una intervención comunicativa diseñada para
    provocar reflexión metacognitiva — awareness del propio proceso
    de aprendizaje.
    """
    nudge_type: str          # progression | effort | spacing | decouple | reflection | sandbox
    message: str             # texto visible para el estudiante
    icon: str                # emoji del nudge
    theoretical_basis: str   # referencia teórica (para el investigador, no el estudiante)
    trigger_reason: str      # por qué se disparó (logging para research)
    priority: int = 1        # 1-5, 5 = máxima prioridad
    interaction_number: int = 0  # en qué interacción se generó


@dataclass
class NudgeConfig:
    """
    Configuración de nudges — controlada por el docente.

    Cada parámetro es una decisión pedagógica:
    - enabled: ¿quiero que mis estudiantes reciban nudges?
    - min_interval: ¿con qué frecuencia? (demasiado frecuente = ruido)
    - types_enabled: ¿qué tipos? (el docente puede desactivar selectivamente)
    """
    enabled: bool = True
    min_interval: int = 4              # mínimo N interacciones entre nudges
    types_enabled: list = field(default_factory=lambda: [
        "progression", "effort", "spacing", "decouple", "reflection", "sandbox"
    ])
    tone: str = "warm"                 # warm | neutral | academic
    show_theoretical_basis: bool = False  # si True, muestra la referencia al estudiante


# ═══════════════════════════════════════════════════════════════════════
# PLANTILLAS DE NUDGES
# ═══════════════════════════════════════════════════════════════════════

# Cada plantilla tiene variantes para no repetirse.
# El tono "warm" es el default — natural, no condescendiente.

NUDGE_TEMPLATES = {
    "progression": {
        "icon": "🌱",
        "messages_warm": [
            "Algo ha cambiado en cómo preguntas — tus últimas consultas muestran "
            "que estás empezando a formular hipótesis propias antes de pedir ayuda. "
            "Eso es señal de que el tema está empezando a 'encajar'.",

            "He notado que en tus interacciones recientes estás planteando "
            "conexiones entre conceptos que antes tratabas por separado. "
            "Tu comprensión se está estructurando.",

            "Tu forma de interactuar ha evolucionado — estás haciendo preguntas "
            "más específicas y contextualizadas. Eso indica que ya tienes un "
            "mapa mental del tema más claro que al principio.",
        ],
        "messages_neutral": [
            "Tus consultas recientes muestran un incremento en el nivel de "
            "elaboración de tus preguntas.",

            "Se observa una progresión en la complejidad de tus interacciones "
            "con el asistente.",
        ],
        "messages_academic": [
            "Tu trayectoria de interacción muestra una transición desde "
            "preguntas de recuperación (recall) hacia preguntas de análisis. "
            "En términos de la taxonomía de Bloom, estás ascendiendo.",
        ],
        "theoretical_basis": "Zimmerman (2002): self-observation como componente de SRL",
    },

    "effort": {
        "icon": "💪",
        "messages_warm": [
            "Tu última pregunta fue más elaborada que las anteriores — estás "
            "empezando a analizar en lugar de solo recordar. Ese esfuerzo "
            "extra es lo que consolida el aprendizaje, aunque no lo parezca "
            "en el momento.",

            "Noto que estás intentando resolver antes de preguntar — eso es "
            "lo que la investigación llama 'esfuerzo productivo'. Es más "
            "lento pero se retiene mucho mejor.",

            "Tu pregunta incluye una hipótesis propia ('creo que es por...') "
            "— formular hipótesis, aunque sean incorrectas, es una de las "
            "formas más efectivas de aprender.",
        ],
        "messages_neutral": [
            "Tu última interacción demuestra un nivel de esfuerzo cognitivo "
            "superior al promedio de tus consultas anteriores.",
        ],
        "messages_academic": [
            "Se detecta esfuerzo productivo (Kapur, 2008) en tu última "
            "interacción: el intento de resolución previo a la consulta "
            "activa mecanismos de aprendizaje más profundos.",
        ],
        "theoretical_basis": "Kapur (2008): productive failure + Bjork (1994): desirable difficulties",
    },

    "spacing": {
        "icon": "🔄",
        "messages_warm": [
            "Has vuelto al tema de {topic} después de unos días — excelente. "
            "La investigación muestra que este 'espaciado' entre sesiones "
            "es una de las estrategias más efectivas para la retención. "
            "¿Qué recuerdas sin mirar los apuntes?",

            "Retomas {topic} tras una pausa. Antes de continuar: "
            "¿podrías explicarme en una frase lo que recuerdas del tema? "
            "Ese ejercicio de recuperación fortalece la memoria más que "
            "releer los apuntes.",

            "Interesante — vuelves a {topic} después de trabajar otros temas. "
            "Tu cerebro ha tenido tiempo de procesar la información anterior. "
            "Probablemente notes que ahora entiendes cosas que antes no encajaban.",
        ],
        "messages_neutral": [
            "Retomas el tema de {topic} después de un intervalo. "
            "El repaso espaciado favorece la retención a largo plazo.",
        ],
        "messages_academic": [
            "Retorno al topic {topic} tras intervalo de {hours}h. "
            "El spacing effect (Cepeda et al., 2006) predice mejor retención "
            "cuando el repaso ocurre en la ventana de 24-72h post-exposición.",
        ],
        "theoretical_basis": "Bjork (1994): spacing effect + Ebbinghaus (1885): curva del olvido",
    },

    "decouple": {
        "icon": "🦋",
        "messages_warm": [
            "Llevas varias interacciones resolviendo con mínima ayuda del "
            "asistente. Eso sugiere que estás listo para intentar el próximo "
            "ejercicio completamente solo antes de consultarme. ¿Te atreves?",

            "Tu nivel de independencia en las últimas consultas es notable — "
            "estás usando el asistente para verificar, no para obtener respuestas. "
            "Esa es exactamente la transición que buscamos.",

            "Parece que ya no necesitas tanta guía en este tema. El objetivo "
            "del asistente es que cada vez lo necesites menos — y estás "
            "llegando a ese punto.",
        ],
        "messages_neutral": [
            "Tu trayectoria muestra un incremento sostenido en autonomía. "
            "Considera intentar los próximos ejercicios de forma independiente.",
        ],
        "messages_academic": [
            "El scaffolding fading (Wood, Bruner & Ross, 1976) indica que el "
            "nivel de soporte puede reducirse. Tu ratio de consultas autónomas "
            "vs. asistidas sugiere preparación para la fase independiente.",
        ],
        "theoretical_basis": "Wood, Bruner & Ross (1976): scaffolding fading + Deci & Ryan (1985): SDT",
    },

    "reflection": {
        "icon": "🪞",
        "messages_warm": [
            "Llevas {count} interacciones en esta sesión. Antes de continuar: "
            "¿cuál ha sido la idea más importante que has entendido hoy? "
            "Ponerlo en palabras propias consolida el aprendizaje.",

            "Pausa de reflexión: de todo lo que hemos trabajado hoy, "
            "¿hay algo que creías entender pero ahora ves de forma diferente?",

            "Has trabajado bastante hoy. Un ejercicio rápido: "
            "¿podrías explicarle a un compañero en 2 frases lo que has "
            "aprendido en esta sesión? Si puedes, lo has integrado.",
        ],
        "messages_neutral": [
            "Punto de reflexión: resume brevemente lo aprendido en esta sesión.",
        ],
        "messages_academic": [
            "Meta-reflexión (Schön, 1983): ¿qué estrategias de aprendizaje "
            "has usado en esta sesión? ¿Cuáles han funcionado mejor?",
        ],
        "theoretical_basis": "Schön (1983): reflection-on-action + Flavell (1979): metacognitive regulation",
    },

    "sandbox": {
        "icon": "🏖️",
        "messages_warm": [
            "Estás en modo práctica libre — aquí no hay métricas, no hay "
            "evaluación, solo tú y la materia. Aprovecha para hacer las "
            "preguntas que normalmente no harías.",

            "Modo sandbox activo. Este es tu espacio para experimentar "
            "sin consecuencias — prueba cosas, equivócate, explora. "
            "Así es como se aprende de verdad.",
        ],
        "messages_neutral": [
            "Modo práctica libre activo. Esta sesión no se registra en analytics.",
        ],
        "messages_academic": [
            "Sandbox mode: espacio de exploración libre sin registro analítico. "
            "El derecho a equivocarse sin ser medido (efecto Hawthorne pedagógico).",
        ],
        "theoretical_basis": "Efecto Hawthorne (Mayo, 1933) aplicado a LA: la observación modifica lo observado",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# MOTOR DE NUDGES
# ═══════════════════════════════════════════════════════════════════════

class MetacognitiveNudgeGenerator:
    """
    Genera nudges metacognitivos contextuales para el estudiante.

    Principio rector: el nudge provoca reflexión, no informa métricas.
    La diferencia es la que existe entre un espejo (te ves a ti mismo)
    y un expediente (alguien te describe).

    El generador consume señales de los módulos analíticos existentes
    (Bloom, autonomía, trust, semiótica) pero las TRADUCE a lenguaje
    que promueve metacognición, no a números que alienan.
    """

    def __init__(self, config: Optional[NudgeConfig] = None):
        self.config = config or NudgeConfig()
        # Historial por estudiante: {student_id: [nudge, nudge, ...]}
        self.nudge_history: dict[str, list[MetacognitiveNudge]] = {}
        # Contadores por estudiante
        self.interaction_counts: dict[str, int] = {}
        # Topic history: {student_id: {topic: [{"timestamp": ..., "bloom": ...}]}}
        self.topic_history: dict[str, dict[str, list[dict]]] = {}
        # Último nudge entregado por estudiante
        self.last_nudge_at: dict[str, int] = {}
        # Templates ya usados (evitar repetición)
        self._used_templates: dict[str, set] = {}

    # ───────────────────────────────────────────────
    # API PRINCIPAL
    # ───────────────────────────────────────────────

    def record_interaction(
        self,
        student_id: str,
        prompt: str,
        bloom_level: int,
        bloom_name: str,
        detected_topics: list,
        scaffolding_level: int,
        autonomy_score: float = 0.0,
        autonomy_phase: str = "dependent",
        is_sandbox: bool = False,
        timestamp: Optional[str] = None,
    ):
        """
        Registra una interacción y actualiza el estado interno.
        Se llama DESPUÉS de cada interacción completa en el pipeline.
        """
        if student_id not in self.interaction_counts:
            self.interaction_counts[student_id] = 0
            self.topic_history[student_id] = {}
            self.nudge_history[student_id] = []
            self._used_templates[student_id] = set()

        self.interaction_counts[student_id] += 1
        ts = timestamp or datetime.now().isoformat()

        # Registrar topics con bloom y timestamp
        for topic in detected_topics:
            if topic not in self.topic_history[student_id]:
                self.topic_history[student_id][topic] = []
            self.topic_history[student_id][topic].append({
                "timestamp": ts,
                "bloom_level": bloom_level,
                "bloom_name": bloom_name,
                "scaffolding_level": scaffolding_level,
                "interaction_n": self.interaction_counts[student_id],
                "is_sandbox": is_sandbox,
            })

    def maybe_generate_nudge(
        self,
        student_id: str,
        bloom_level: int,
        bloom_name: str,
        detected_topics: list,
        scaffolding_level: int,
        autonomy_score: float = 0.0,
        autonomy_phase: str = "dependent",
        is_sandbox: bool = False,
    ) -> Optional[MetacognitiveNudge]:
        """
        Evalúa si es momento de un nudge y, de serlo, genera el más pertinente.

        Returns None si no toca nudge (por frecuencia o ausencia de trigger).
        Returns MetacognitiveNudge si hay uno pertinente.

        La lógica de priorización:
            1. Sandbox → siempre (si acaba de activarse)
            2. Desacople → alta prioridad (señal fuerte de autonomía)
            3. Progresión → media-alta (señal de mejora)
            4. Esfuerzo → media (refuerzo positivo de productive struggle)
            5. Spacing → media (si el topic vuelve tras pausa)
            6. Reflexión → baja (periódica, cuando no hay nada más específico)
        """
        if not self.config.enabled:
            return None

        count = self.interaction_counts.get(student_id, 0)
        last = self.last_nudge_at.get(student_id, 0)

        # Check intervalo mínimo
        if count - last < self.config.min_interval and count > 1:
            return None

        # Evaluar candidatos en orden de prioridad
        candidates = []

        # 1. Sandbox (siempre, si está activo y es la primera interacción en sandbox)
        if is_sandbox and "sandbox" in self.config.types_enabled:
            sandbox_interactions = sum(
                1 for topic_entries in self.topic_history.get(student_id, {}).values()
                for e in topic_entries if e.get("is_sandbox")
            )
            if sandbox_interactions <= 1:  # primera o segunda interacción sandbox
                candidates.append(self._build_nudge("sandbox", student_id, count))

        # 2. Desacople (scaffolding bajo + autonomía alta)
        if ("decouple" in self.config.types_enabled
                and scaffolding_level <= 1
                and autonomy_score > 0.55
                and autonomy_phase in ("emergent", "autonomous")):
            candidates.append(self._build_nudge(
                "decouple", student_id, count,
                trigger=f"scaffolding={scaffolding_level}, autonomy={autonomy_score:.2f}, phase={autonomy_phase}"
            ))

        # 3. Progresión (Bloom sube respecto a la media reciente)
        if "progression" in self.config.types_enabled:
            progression = self._detect_progression(student_id, bloom_level)
            if progression:
                candidates.append(self._build_nudge(
                    "progression", student_id, count,
                    trigger=progression
                ))

        # 4. Esfuerzo productivo (Bloom >= 4 y scaffolding bajo)
        if ("effort" in self.config.types_enabled
                and bloom_level >= 4
                and scaffolding_level <= 1):
            candidates.append(self._build_nudge(
                "effort", student_id, count,
                trigger=f"bloom={bloom_level} ({bloom_name}) con scaffolding={scaffolding_level}"
            ))

        # 5. Spacing (topic revisitado tras pausa)
        if "spacing" in self.config.types_enabled:
            spacing = self._detect_spacing(student_id, detected_topics)
            if spacing:
                candidates.append(self._build_nudge(
                    "spacing", student_id, count,
                    trigger=spacing["trigger"],
                    topic=spacing["topic"],
                    hours=spacing["hours"],
                ))

        # 6. Reflexión (periódica, cada ~8-12 interacciones si no hay nada mejor)
        if ("reflection" in self.config.types_enabled
                and count > 0 and count % random.choice([8, 10, 12]) == 0
                and not candidates):
            candidates.append(self._build_nudge(
                "reflection", student_id, count,
                trigger=f"periodic at interaction {count}",
                count=count,
            ))

        if not candidates:
            return None

        # Seleccionar el de mayor prioridad
        best = max(candidates, key=lambda n: n.priority)
        self.last_nudge_at[student_id] = count
        self.nudge_history[student_id].append(best)
        return best

    # ───────────────────────────────────────────────
    # DETECTORES DE SEÑALES
    # ───────────────────────────────────────────────

    def _detect_progression(self, student_id: str, current_bloom: int) -> Optional[str]:
        """
        Detecta si el estudiante está progresando en nivel Bloom.
        Compara las últimas 3 interacciones con las 3 anteriores.
        """
        history = self.topic_history.get(student_id, {})
        all_blooms = []
        for topic_entries in history.values():
            for entry in topic_entries:
                all_blooms.append(entry["bloom_level"])

        if len(all_blooms) < 6:
            return None

        recent = all_blooms[-3:]
        previous = all_blooms[-6:-3]
        recent_avg = sum(recent) / len(recent)
        previous_avg = sum(previous) / len(previous)

        # Umbral: subida de al menos 0.8 puntos Bloom
        if recent_avg - previous_avg >= 0.8:
            return (
                f"Bloom avg subió de {previous_avg:.1f} a {recent_avg:.1f} "
                f"(delta +{recent_avg - previous_avg:.1f})"
            )
        return None

    def _detect_spacing(self, student_id: str, current_topics: list) -> Optional[dict]:
        """
        Detecta si el estudiante vuelve a un topic tras pausa de 24+ horas.
        El spacing effect (Bjork, 1994) opera en la ventana 24-168h.
        """
        history = self.topic_history.get(student_id, {})
        now = datetime.now()

        for topic in current_topics:
            if topic not in history or topic == "otro":
                continue
            entries = history[topic]
            if len(entries) < 1:
                continue

            # Buscar la interacción más reciente sobre este topic (excluyendo la actual)
            # Las entries son las ANTERIORES (record_interaction ya se llamó)
            prev_entries = [e for e in entries[:-1] if not e.get("is_sandbox")]
            if not prev_entries:
                continue

            last_ts_str = prev_entries[-1]["timestamp"]
            try:
                last_ts = datetime.fromisoformat(last_ts_str)
                gap_hours = (now - last_ts).total_seconds() / 3600
                if 24 <= gap_hours <= 168:  # 1-7 días
                    return {
                        "topic": topic,
                        "hours": round(gap_hours),
                        "trigger": f"topic '{topic}' revisitado tras {gap_hours:.0f}h de pausa",
                    }
            except (ValueError, TypeError):
                continue

        return None

    # ───────────────────────────────────────────────
    # CONSTRUCTOR DE NUDGES
    # ───────────────────────────────────────────────

    def _build_nudge(
        self,
        nudge_type: str,
        student_id: str,
        interaction_n: int,
        trigger: str = "",
        topic: str = "",
        hours: int = 0,
        count: int = 0,
    ) -> MetacognitiveNudge:
        """Construye un nudge seleccionando una plantilla no repetida."""
        template = NUDGE_TEMPLATES.get(nudge_type, NUDGE_TEMPLATES["reflection"])

        # Seleccionar tono
        tone_key = f"messages_{self.config.tone}"
        messages = template.get(tone_key, template.get("messages_warm", []))

        # Evitar repetición
        used = self._used_templates.get(student_id, set())
        available = [m for i, m in enumerate(messages) if f"{nudge_type}_{i}" not in used]
        if not available:
            # Reset: todas usadas, empezar de nuevo
            used = {u for u in used if not u.startswith(nudge_type)}
            self._used_templates[student_id] = used
            available = messages

        message = random.choice(available) if available else messages[0]
        idx = messages.index(message) if message in messages else 0
        self._used_templates.setdefault(student_id, set()).add(f"{nudge_type}_{idx}")

        # Formatear con variables contextuales
        message = message.format(
            topic=topic or "el tema actual",
            hours=hours,
            count=count or interaction_n,
        )

        # Prioridad por tipo
        priority_map = {
            "sandbox": 5, "decouple": 4, "progression": 3,
            "effort": 3, "spacing": 3, "reflection": 1,
        }

        # Añadir base teórica si configurado
        theoretical = template.get("theoretical_basis", "")
        if self.config.show_theoretical_basis and theoretical:
            message += f"\n\n📚 *Referencia: {theoretical}*"

        return MetacognitiveNudge(
            nudge_type=nudge_type,
            message=message,
            icon=template.get("icon", "🪞"),
            theoretical_basis=theoretical,
            trigger_reason=trigger,
            priority=priority_map.get(nudge_type, 1),
            interaction_number=interaction_n,
        )

    # ───────────────────────────────────────────────
    # ANALYTICS DE NUDGES (para el investigador)
    # ───────────────────────────────────────────────

    def get_nudge_stats(self, student_id: Optional[str] = None) -> dict:
        """Estadísticas de nudges para research — qué tipos se disparan más."""
        if student_id:
            nudges = self.nudge_history.get(student_id, [])
        else:
            nudges = [n for ns in self.nudge_history.values() for n in ns]

        if not nudges:
            return {"total": 0, "by_type": {}, "avg_interval": 0}

        by_type = {}
        for n in nudges:
            by_type[n.nudge_type] = by_type.get(n.nudge_type, 0) + 1

        intervals = []
        for sid, history in self.nudge_history.items():
            for i in range(1, len(history)):
                intervals.append(
                    history[i].interaction_number - history[i-1].interaction_number
                )

        return {
            "total": len(nudges),
            "by_type": by_type,
            "avg_interval": round(sum(intervals) / len(intervals), 1) if intervals else 0,
            "students_nudged": len(self.nudge_history),
            "triggers": [n.trigger_reason for n in nudges[-10:]],
        }

    def get_student_nudge_timeline(self, student_id: str) -> list[dict]:
        """Timeline de nudges para un estudiante — visualizable en research view."""
        return [
            {
                "interaction": n.interaction_number,
                "type": n.nudge_type,
                "icon": n.icon,
                "message_preview": n.message[:80] + "..." if len(n.message) > 80 else n.message,
                "trigger": n.trigger_reason,
                "theory": n.theoretical_basis,
            }
            for n in self.nudge_history.get(student_id, [])
        ]


# ═══════════════════════════════════════════════════════════════════════
# DEMO: generación de datos sintéticos para demostración
# ═══════════════════════════════════════════════════════════════════════

def generate_demo_nudge_sequence() -> list[dict]:
    """
    Genera una secuencia de 15 interacciones con nudges para demo.
    Muestra los 6 tipos de nudge en contexto realista.
    """
    gen = MetacognitiveNudgeGenerator(NudgeConfig(min_interval=2))
    sid = "demo_student_01"
    sequence = []

    # Simular 15 interacciones con progresión natural
    interactions = [
        # Interacciones 1-4: nivel bajo, dependiente
        {"bloom": 1, "name": "Recordar", "topics": ["variables"], "scaff": 0, "auto": 0.1, "phase": "dependent"},
        {"bloom": 2, "name": "Comprender", "topics": ["variables"], "scaff": 0, "auto": 0.15, "phase": "dependent"},
        {"bloom": 2, "name": "Comprender", "topics": ["bucles"], "scaff": 1, "auto": 0.2, "phase": "dependent"},
        {"bloom": 2, "name": "Comprender", "topics": ["bucles"], "scaff": 1, "auto": 0.25, "phase": "dependent"},
        # Interacciones 5-8: mejora gradual
        {"bloom": 3, "name": "Aplicar", "topics": ["funciones"], "scaff": 1, "auto": 0.35, "phase": "scaffolded"},
        {"bloom": 3, "name": "Aplicar", "topics": ["funciones"], "scaff": 1, "auto": 0.4, "phase": "scaffolded"},
        {"bloom": 4, "name": "Analizar", "topics": ["funciones"], "scaff": 0, "auto": 0.5, "phase": "scaffolded"},
        {"bloom": 4, "name": "Analizar", "topics": ["bucles", "funciones"], "scaff": 0, "auto": 0.55, "phase": "emergent"},
        # Interacciones 9-12: autonomía creciente
        {"bloom": 4, "name": "Analizar", "topics": ["recursión"], "scaff": 0, "auto": 0.6, "phase": "emergent"},
        {"bloom": 5, "name": "Evaluar", "topics": ["recursión"], "scaff": 0, "auto": 0.65, "phase": "emergent"},
        {"bloom": 4, "name": "Analizar", "topics": ["variables"], "scaff": 0, "auto": 0.7, "phase": "emergent"},  # spacing!
        {"bloom": 5, "name": "Evaluar", "topics": ["recursión"], "scaff": 0, "auto": 0.75, "phase": "autonomous"},
        # Interacciones 13-15: autonomía alta
        {"bloom": 5, "name": "Evaluar", "topics": ["funciones", "recursión"], "scaff": 0, "auto": 0.8, "phase": "autonomous"},
        {"bloom": 6, "name": "Crear", "topics": ["arrays", "bucles"], "scaff": 0, "auto": 0.85, "phase": "autonomous"},
        {"bloom": 5, "name": "Evaluar", "topics": ["recursión"], "scaff": 0, "auto": 0.88, "phase": "autonomous"},
    ]

    for i, ix in enumerate(interactions):
        gen.record_interaction(
            student_id=sid,
            prompt=f"demo_prompt_{i+1}",
            bloom_level=ix["bloom"],
            bloom_name=ix["name"],
            detected_topics=ix["topics"],
            scaffolding_level=ix["scaff"],
            autonomy_score=ix["auto"],
            autonomy_phase=ix["phase"],
        )

        nudge = gen.maybe_generate_nudge(
            student_id=sid,
            bloom_level=ix["bloom"],
            bloom_name=ix["name"],
            detected_topics=ix["topics"],
            scaffolding_level=ix["scaff"],
            autonomy_score=ix["auto"],
            autonomy_phase=ix["phase"],
        )

        sequence.append({
            "interaction": i + 1,
            "bloom": f"{ix['name']} (N{ix['bloom']})",
            "topics": ix["topics"],
            "autonomy": f"{ix['auto']:.2f} ({ix['phase']})",
            "nudge": {
                "type": nudge.nudge_type,
                "icon": nudge.icon,
                "message": nudge.message,
                "trigger": nudge.trigger_reason,
            } if nudge else None,
        })

    return sequence


# ═══════════════════════════════════════════════════════════════════════
# TEST RÁPIDO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("METACOGNITIVE NUDGE GENERATOR — Demo")
    print("=" * 70)

    sequence = generate_demo_nudge_sequence()
    nudge_count = 0

    for step in sequence:
        prefix = f"  [{step['interaction']:2d}] Bloom: {step['bloom']:<20s} "
        prefix += f"Auto: {step['autonomy']:<25s} "
        prefix += f"Topics: {', '.join(step['topics'])}"

        if step["nudge"]:
            nudge_count += 1
            print(f"\n{prefix}")
            print(f"       {step['nudge']['icon']} NUDGE ({step['nudge']['type']}): "
                  f"{step['nudge']['message'][:100]}...")
            print(f"       Trigger: {step['nudge']['trigger']}")
        else:
            print(prefix)

    print(f"\n{'=' * 70}")
    print(f"Total: {len(sequence)} interacciones, {nudge_count} nudges generados")
    print(f"Ratio: 1 nudge cada {len(sequence) / max(nudge_count, 1):.1f} interacciones")
