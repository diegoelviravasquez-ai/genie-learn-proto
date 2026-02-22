"""
MIDDLEWARE PEDAGÓGICO — Capa intermedia entre estudiante y LLM
=============================================================
Replica la innovación central del paper LAK 2026 (Ortega-Arranz et al.):
las Pedagogical Configurations que modifican el comportamiento del chatbot
según decisiones del docente.

Arquitectura: prompt estudiante → middleware → LLM → middleware → respuesta filtrada
"""

import random
import time
import json
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, date


@dataclass
class PedagogicalConfig:
    """Configuraciones pedagógicas del docente (paper LAK 2026, Sección 3)."""

    # --- Selección de modelo ---
    model_name: str = "gpt-4o-mini"

    # --- Límites de uso ---
    max_daily_prompts: int = 20
    min_response_length: int = 50  # caracteres
    max_response_length: int = 2000

    # --- Comportamiento pedagógico ---
    scaffolding_mode: str = "socratic"  # "socratic" | "hints" | "direct"
    block_direct_solutions: bool = True
    forced_hallucination_pct: float = 0.0  # 0.0 a 1.0 — % de respuestas con error intencional

    # --- Contextualización ---
    use_rag: bool = True
    no_context_behavior: str = "refuse"  # "refuse" | "general"

    # --- Add-ons invisibles ---
    system_addon: str = ""  # texto inyectado silenciosamente al system prompt
    role_play: str = ""  # e.g., "Eres un tutor paciente de programación"

    # --- Analytics tags ---
    topic_tags: list = field(default_factory=lambda: [
        "variables", "bucles", "funciones", "arrays", "recursión",
        "entrada/salida", "depuración", "conceptual", "ejercicio", "otro"
    ])


@dataclass
class InteractionLog:
    """Registro de una interacción para GenAI Analytics."""
    timestamp: str
    student_id: str
    prompt_raw: str
    prompt_processed: str
    response_raw: str
    response_delivered: str
    detected_topics: list
    scaffolding_level: int
    was_blocked: bool
    block_reason: str
    copy_paste_score: float
    response_time_ms: int
    hallucination_injected: bool


class PedagogicalMiddleware:
    """
    Motor de reglas pedagógicas.
    
    Flujo:
    1. pre_process()  → filtra/transforma el prompt ANTES de enviarlo al LLM
    2. [llamada al LLM externa]
    3. post_process() → filtra/transforma la respuesta ANTES de entregarla al estudiante
    4. log()          → registra la interacción para analytics
    """

    def __init__(self, config: PedagogicalConfig):
        self.config = config
        self.interaction_logs: list[InteractionLog] = []
        self.daily_prompt_counts: dict[str, dict[str, int]] = {}  # {student_id: {date: count}}
        self.conversation_states: dict[str, dict] = {}  # estado de scaffolding por estudiante

    # ──────────────────────────────────────────────
    # 1. PRE-PROCESAMIENTO DEL PROMPT
    # ──────────────────────────────────────────────

    def pre_process(self, student_id: str, raw_prompt: str) -> dict:
        """
        Procesa el prompt del estudiante ANTES de enviarlo al LLM.
        Returns dict con:
          - allowed: bool
          - block_reason: str
          - system_prompt: str (construido con configs pedagógicas)
          - processed_prompt: str
          - copy_paste_score: float
          - detected_topics: list
        """
        result = {
            "allowed": True,
            "block_reason": "",
            "system_prompt": "",
            "processed_prompt": raw_prompt,
            "copy_paste_score": 0.0,
            "detected_topics": [],
            "scaffolding_level": 0,
        }

        # --- Check límite diario ---
        today = date.today().isoformat()
        if student_id not in self.daily_prompt_counts:
            self.daily_prompt_counts[student_id] = {}
        if today not in self.daily_prompt_counts[student_id]:
            self.daily_prompt_counts[student_id][today] = 0

        current_count = self.daily_prompt_counts[student_id][today]
        if current_count >= self.config.max_daily_prompts:
            result["allowed"] = False
            result["block_reason"] = (
                f"Has alcanzado el límite diario de {self.config.max_daily_prompts} consultas. "
                f"Esto es una decisión pedagógica de tu profesor/a para fomentar "
                f"la reflexión antes de preguntar. Vuelve mañana."
            )
            return result

        # --- Detección de copy-paste sospechoso ---
        result["copy_paste_score"] = self._detect_copy_paste(raw_prompt)

        # --- Detección de topics ---
        result["detected_topics"] = self._detect_topics(raw_prompt)

        # --- Scaffolding: determinar nivel ---
        state = self.conversation_states.get(student_id, {"level": 0, "topic": None})
        result["scaffolding_level"] = state["level"]

        # --- Construir system prompt con configuraciones pedagógicas ---
        result["system_prompt"] = self._build_system_prompt(result["scaffolding_level"])

        # --- Incrementar contador ---
        self.daily_prompt_counts[student_id][today] += 1

        return result

    def _build_system_prompt(self, scaffolding_level: int) -> str:
        """Construye el system prompt inyectando las configuraciones pedagógicas."""
        parts = []

        # Base role
        if self.config.role_play:
            parts.append(self.config.role_play)
        else:
            parts.append(
                "Eres un asistente educativo para una asignatura universitaria. "
                "Tu objetivo es ayudar al estudiante a APRENDER, no a obtener respuestas."
            )

        # Scaffolding mode
        if self.config.scaffolding_mode == "socratic":
            scaffolding_instructions = {
                0: (
                    "NIVEL SOCRÁTICO: Responde SIEMPRE con preguntas orientadoras. "
                    "NO des la respuesta directa. Haz que el estudiante piense. "
                    "Ejemplo: '¿Qué crees que pasaría si...?', '¿Has considerado...?'"
                ),
                1: (
                    "NIVEL PISTA: El estudiante ya lo intentó. Dale una pista concreta "
                    "pero NO la solución completa. Señala la dirección correcta."
                ),
                2: (
                    "NIVEL EJEMPLO: Proporciona un ejemplo SIMILAR pero NO idéntico "
                    "al problema del estudiante. Que pueda extrapolarlo."
                ),
                3: (
                    "NIVEL EXPLICACIÓN: Ahora sí, explica el concepto paso a paso. "
                    "El estudiante ha pasado por los niveles previos de reflexión."
                ),
            }
            parts.append(scaffolding_instructions.get(scaffolding_level, scaffolding_instructions[3]))
        elif self.config.scaffolding_mode == "hints":
            parts.append("Proporciona pistas progresivas. Nunca la solución completa de golpe.")
        # "direct" → no se añade restricción

        # Bloqueo de soluciones directas
        if self.config.block_direct_solutions:
            parts.append(
                "IMPORTANTE: Si el estudiante pide directamente la solución a un ejercicio "
                "(e.g., 'resuélveme esto', 'dame el código'), NO la proporciones. "
                "En su lugar, guíale hacia la solución con preguntas o pistas."
            )

        # Longitud de respuesta
        parts.append(
            f"Mantén tus respuestas entre {self.config.min_response_length} "
            f"y {self.config.max_response_length} caracteres."
        )

        # Comportamiento sin contexto RAG
        if self.config.no_context_behavior == "refuse":
            parts.append(
                "Si la pregunta no está relacionada con los materiales del curso "
                "proporcionados en el contexto, responde: 'Esta pregunta no está "
                "relacionada con los materiales del curso. Consulta a tu profesor/a.'"
            )

        # Add-on invisible del profesor
        if self.config.system_addon:
            parts.append(self.config.system_addon)

        return "\n\n".join(parts)

    def _detect_copy_paste(self, text: str) -> float:
        """
        Heurística de detección de copy-paste.
        Score 0.0 (orgánico) a 1.0 (probable copy-paste).
        
        NOTA DE DISEÑO (feb 2026):
        Probé inicialmente un enfoque con embeddings: comparar el embedding del
        prompt contra los embeddings de los enunciados de ejercicios del curso
        (que el profesor sube como PDFs). Si la similitud coseno > 0.85, probable
        copy-paste. Funcionaba bien pero requería tener los enunciados indexados
        por separado del contenido del curso — y eso es un requisito que no puedo
        imponer al docente en el prototipo. Lo dejo como TODO para la Fase B.
        
        La heurística actual es cruda pero tiene un false positive rate aceptable
        para un prototipo. En el piloto real mediremos precision/recall contra
        etiquetado manual de los docentes.
        
        # Enfoque descartado por ahora:
        # if self.exercise_embeddings:
        #     sim = cosine_similarity(embed(text), self.exercise_embeddings)
        #     if max(sim) > 0.85: return max(sim)
        """
        indicators = 0.0

        # Texto muy largo sin signos de interrogación → probable enunciado copiado
        if len(text) > 200 and "?" not in text:
            indicators += 0.4

        # Contiene formato de enunciado académico
        # TODO: estos marcadores están sesgados hacia español peninsular.
        # Para UBUN.IA (contexto ProFuturo/LatAm) habría que añadir variantes:
        # "haga un programa", "realice", "desarrolle" son más comunes en LatAm.
        academic_markers = ["ejercicio", "enunciado", "se pide", "implementar", "resolver",
                          "dado el siguiente", "a partir de", "escribe un programa"]
        for marker in academic_markers:
            if marker.lower() in text.lower():
                indicators += 0.15

        # Múltiples saltos de línea → formato copiado
        if text.count("\n") > 3:
            indicators += 0.2

        # Ratio mayúsculas/total alto → probable título o enunciado formal
        if len(text) > 50:
            upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if upper_ratio > 0.15:
                indicators += 0.15

        return min(indicators, 1.0)

    def _detect_topics(self, text: str) -> list:
        """Detección de topics por keywords (simplificado; en producción usaría embeddings)."""
        text_lower = text.lower()
        detected = []
        topic_keywords = {
            "variables": ["variable", "int ", "float", "string", "tipo de dato", "declarar"],
            "bucles": ["for", "while", "bucle", "iteración", "iterar", "repetir"],
            "funciones": ["función", "función", "def ", "return", "parámetro", "argumento"],
            "arrays": ["array", "lista", "vector", "índice", "posición", "recorrer"],
            "recursión": ["recursión", "recursivo", "caso base", "llamada recursiva"],
            "entrada/salida": ["input", "output", "print", "scanner", "leer", "escribir"],
            "depuración": ["error", "bug", "fallo", "no funciona", "no compila", "excepción"],
            "conceptual": ["qué es", "qué significa", "diferencia entre", "para qué sirve"],
            "ejercicio": ["ejercicio", "problema", "enunciado", "resolver", "implementar"],
        }
        for topic, keywords in topic_keywords.items():
            if any(kw in text_lower for kw in keywords):
                detected.append(topic)
        if not detected:
            detected.append("otro")
        return detected

    # ──────────────────────────────────────────────
    # 2. POST-PROCESAMIENTO DE LA RESPUESTA
    # ──────────────────────────────────────────────

    def post_process(self, student_id: str, raw_response: str) -> dict:
        """
        Procesa la respuesta del LLM ANTES de entregarla al estudiante.
        Returns dict con:
          - response: str (respuesta final)
          - hallucination_injected: bool
          - was_truncated: bool
        """
        result = {
            "response": raw_response,
            "hallucination_injected": False,
            "was_truncated": False,
        }

        # --- Truncar si excede max_response_length ---
        if len(raw_response) > self.config.max_response_length:
            result["response"] = raw_response[:self.config.max_response_length] + "..."
            result["was_truncated"] = True

        # --- Inyección de alucinación pedagógica controlada ---
        if self.config.forced_hallucination_pct > 0:
            if random.random() < self.config.forced_hallucination_pct:
                result["response"] = self._inject_hallucination(result["response"])
                result["hallucination_injected"] = True

        # --- Actualizar estado de scaffolding ---
        # DECISIÓN DOCUMENTADA en ADR-002: escalo por conteo, no por análisis semántico.
        # Intenté escalar cuando el CognitiveAnalyzer detectaba Bloom >= 3 en la respuesta
        # del estudiante (señal de comprensión), pero el problema es que Bloom mide la
        # FORMULACIÓN, no la COMPRENSIÓN. Un copy-paste de un enunciado Bloom-5 no indica
        # que el estudiante entienda. Necesitaría un evaluador de comprensión genuino,
        # que es un paper entero por sí mismo.
        #
        # Alternativa futura: usar la latencia entre respuesta del chatbot y siguiente
        # prompt como proxy de procesamiento (Lee & See, 2004 — trust dynamics).
        # Latencia < 3s = no leyó. Latencia > 30s = procesó. Esto sí sería señal
        # de comprensión indirecta. Implementable en Fase B cuando tenga timestamps reales.
        state = self.conversation_states.get(student_id, {"level": 0, "attempts": 0})
        state["attempts"] += 1
        # Escalar nivel cada 2 intentos sobre el mismo tema
        if state["attempts"] >= 2 and state["level"] < 3:
            state["level"] += 1
            state["attempts"] = 0
        self.conversation_states[student_id] = state

        return result

    def _inject_hallucination(self, response: str) -> str:
        """
        Inyecta un error pedagógico controlado.
        
        OPINIÓN PERSONAL (esto no está en ningún paper, es mío):
        La idea de forced hallucinations del paper LAK 2026 es interesante pero
        la implementación obvia (modificar la respuesta para que sea incorrecta)
        tiene un problema ético que los autores no abordan explícitamente:
        
        Si el estudiante cita la respuesta incorrecta en un examen, ¿quién es
        responsable? ¿El docente que activó la configuración? ¿El sistema?
        
        Mi implementación conservadora: SEÑALAR que puede haber error, no OCULTAR
        el error. Esto reduce el efecto pedagógico pero elimina el problema ético.
        Es un compromiso que defiendo: la confianza del estudiante en el sistema
        es un recurso frágil (Lee & See, 2004). Si el estudiante descubre que el
        chatbot le ha mentido sin avisar, la confianza se destruye de forma
        potencialmente irreversible.
        
        Implementación sofisticada para Fase C: en vez de inyectar errores en
        respuestas correctas, GENERAR una segunda respuesta con errores sutiles
        y presentar AMBAS al estudiante para que compare. Eso es una tarea de
        evaluación legítima, no un engaño.
        """
        hallucination_notice = (
            "\n\n⚠️ **NOTA PEDAGÓGICA**: Esta respuesta puede contener información "
            "intencionalmente incorrecta como ejercicio de lectura crítica. "
            "Verifica cada afirmación con los materiales del curso."
        )
        return response + hallucination_notice

    # ──────────────────────────────────────────────
    # 3. LOGGING PARA ANALYTICS
    # ──────────────────────────────────────────────

    def log_interaction(
        self,
        student_id: str,
        prompt_raw: str,
        pre_result: dict,
        response_raw: str,
        post_result: dict,
        response_time_ms: int,
    ) -> InteractionLog:
        """Registra la interacción completa para GenAI Analytics."""
        log = InteractionLog(
            timestamp=datetime.now().isoformat(),
            student_id=student_id,
            prompt_raw=prompt_raw,
            prompt_processed=pre_result.get("processed_prompt", prompt_raw),
            response_raw=response_raw,
            response_delivered=post_result.get("response", response_raw),
            detected_topics=pre_result.get("detected_topics", []),
            scaffolding_level=pre_result.get("scaffolding_level", 0),
            was_blocked=not pre_result.get("allowed", True),
            block_reason=pre_result.get("block_reason", ""),
            copy_paste_score=pre_result.get("copy_paste_score", 0.0),
            response_time_ms=response_time_ms,
            hallucination_injected=post_result.get("hallucination_injected", False),
        )
        self.interaction_logs.append(log)
        return log

    def reset_student(self, student_id: str):
        """Reset del estado de scaffolding (para demos)."""
        self.conversation_states.pop(student_id, None)
        today = date.today().isoformat()
        if student_id in self.daily_prompt_counts:
            self.daily_prompt_counts[student_id].pop(today, None)

    # ──────────────────────────────────────────────
    # 4. ANALYTICS AGGREGATES
    # ──────────────────────────────────────────────

    def get_analytics_summary(self) -> dict:
        """Resumen de analytics para el dashboard docente."""
        if not self.interaction_logs:
            return {"total": 0}

        logs = self.interaction_logs
        topics_flat = [t for log in logs for t in log.detected_topics]
        topic_counts = {}
        for t in topics_flat:
            topic_counts[t] = topic_counts.get(t, 0) + 1

        copy_paste_flags = [l for l in logs if l.copy_paste_score > 0.5]
        blocked = [l for l in logs if l.was_blocked]
        hallucinations = [l for l in logs if l.hallucination_injected]

        students = set(l.student_id for l in logs)
        avg_response_time = sum(l.response_time_ms for l in logs) / len(logs)

        return {
            "total_interactions": len(logs),
            "unique_students": len(students),
            "topic_distribution": topic_counts,
            "copy_paste_alerts": len(copy_paste_flags),
            "blocked_requests": len(blocked),
            "hallucinations_injected": len(hallucinations),
            "avg_response_time_ms": round(avg_response_time),
            "scaffolding_levels": {
                "level_0_socratic": len([l for l in logs if l.scaffolding_level == 0]),
                "level_1_hints": len([l for l in logs if l.scaffolding_level == 1]),
                "level_2_examples": len([l for l in logs if l.scaffolding_level == 2]),
                "level_3_explanation": len([l for l in logs if l.scaffolding_level == 3]),
            },
        }
