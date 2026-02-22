"""
EXTERNAL LEARNING INGESTER — Sinapsis Cross-Plataforma
═══════════════════════════════════════════════════════════════════════
Módulo #63 — Resuelve la fragmentación del aprendizaje en silos.

PROBLEMA QUE ATACA — EL ELEFANTE DISTRIBUIDO:
══════════════════════════════════════════════
Un estudiante aprende programación en GENIE, practica inglés en Duolingo,
escribe código en VS Code, repasa flashcards en Anki, lee documentación
en el navegador, resuelve katas en Codewars, acepta o rechaza sugerencias
de Copilot, busca errores en Google, toma notas en Obsidian, ve tutoriales
en YouTube, practica en Mimo desde el móvil. Cada plataforma ve un
fragmento del aprendiz. Ninguna ve al aprendiz completo.

Ortega-Arranz et al. (LAK 2026) lo intuyen cuando describen los
"GenAI Analytics" como limitados a interacciones dentro del chatbot.
Topali et al. (2024, Sección 5.2) lo confirman: las herramientas HCAI
operan en silos. La línea de Teacher Agency (Alonso-Prieto, 2025)
necesita datos cross-plataforma para que el docente tome decisiones
informadas sobre el estudiante COMPLETO.

FUNDAMENTO TEÓRICO — CINCO TRADICIONES CONVERGENTES:
════════════════════════════════════════════════════
1. Activity Theory (Engeström, 1987)
   Cada app es un instrumento en el triángulo de actividad del estudiante.
   Este módulo reconstruye el triángulo completo.

2. xAPI / Experience API (ADL, 2013)
   Actor + Verb + Object + Result + Context. Adoptamos la ontología
   pero NO la dependencia de cooperación inter-plataforma.

3. Ecological Systems Theory (Bronfenbrenner, 1979)
   El microsistema del estudiante incluye TODAS sus herramientas.

4. Desirable Difficulties (Bjork, 1994)
   La frustración cross-plataforma no siempre es negativa. Distinguimos
   frustración productiva de destructiva vía triangulación multi-fuente.

5. Somatic Markers Hypothesis (Damasio, 1994)
   Los patrones motores (tecleo, pausas, borrado) son proxy de estados
   cognitivos y emocionales. Los adaptadores de keystroke dynamics y
   terminal history materializan esta hipótesis computacionalmente.

17 ADAPTADORES:
═══════════════
Capa 1 — IDE & Código:
  VSCode, Terminal, GitLog, CopilotAcceptance, Replit

Capa 2 — Plataformas de aprendizaje:
  Duolingo, Mimo, LeetCode, FreeCodeCamp

Capa 3 — Memoria & Notas:
  Anki, Notes (Obsidian/Notion)

Capa 4 — Navegación & Búsqueda:
  Browser, SearchQuery, AIConversation

Capa 5 — Señales conductuales:
  KeystrokeDynamics, VideoLearning, Pomodoro

PATRÓN DE INTEGRACIÓN: OBSERVER (idéntico a integration.py)
═══════════════════════════════════════════════════════════
En api.py:
  from external_learning_ingester import ExternalLearningIngester
  self.external = ExternalLearningIngester()

Endpoints:
  POST /external/ingest → state.external.ingest(source, data)
  GET  /external/context/{id} → state.external.get_middleware_injection(id)
  GET  /external/dashboard/{id} → state.external.get_dashboard_data(id)

NO REQUIERE DEPENDENCIAS ADICIONALES. Solo stdlib de Python.

Autor: Diego Elvira Vásquez · Prototipo CP25/152 · Feb 2026
"""

import re
import math
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════

class LearningSource(Enum):
    """Instrumentos del sistema de actividad (Engeström, 1987)."""
    # Capa 1: IDE & Código
    VSCODE = "vscode"
    TERMINAL = "terminal"
    GIT = "git"
    COPILOT = "copilot"
    REPLIT = "replit"
    # Capa 2: Plataformas de aprendizaje
    DUOLINGO = "duolingo"
    MIMO = "mimo"
    LEETCODE = "leetcode"
    FREECODECAMP = "freecodecamp"
    # Capa 3: Memoria & Notas
    ANKI = "anki"
    NOTES = "notes"
    # Capa 4: Navegación & Búsqueda
    BROWSER = "browser"
    SEARCH = "search"
    AI_CHAT = "ai_chat"
    # Capa 5: Señales conductuales
    KEYSTROKE = "keystroke"
    VIDEO = "video"
    POMODORO = "pomodoro"
    # Meta
    MANUAL = "manual"
    CUSTOM = "custom"


class EventType(Enum):
    """Verbos xAPI reducidos a los que producen señal analítica."""
    # Código
    CODE_WRITTEN = "code_written"
    CODE_ERROR = "code_error"
    CODE_FIXED = "code_fixed"
    CODE_EXECUTED = "code_executed"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"
    CHALLENGE_SOLVED = "challenge_solved"
    CHALLENGE_FAILED = "challenge_failed"
    # Lenguaje
    EXERCISE_COMPLETED = "exercise_completed"
    EXERCISE_FAILED = "exercise_failed"
    TRANSLATION_ATTEMPT = "translation_attempt"
    VOCABULARY_LEARNED = "vocabulary_learned"
    # Memoria
    CARD_REVIEWED = "card_reviewed"
    CARD_LAPSED = "card_lapsed"
    CARD_MATURED = "card_matured"
    # Lectura / Navegación
    PAGE_READ = "page_read"
    NOTE_TAKEN = "note_taken"
    SEARCH_PERFORMED = "search_performed"
    # IA
    AI_QUERY = "ai_query"
    AI_COMPLETION_ACCEPTED = "ai_completion_accepted"
    AI_COMPLETION_REJECTED = "ai_completion_rejected"
    # Git
    COMMIT_MADE = "commit_made"
    DIFF_VIEWED = "diff_viewed"
    BRANCH_CREATED = "branch_created"
    # Terminal
    COMMAND_EXECUTED = "command_executed"
    COMMAND_FAILED = "command_failed"
    HELP_INVOKED = "help_invoked"
    # Conductual
    KEYSTROKE_PATTERN = "keystroke_pattern"
    VIDEO_WATCHED = "video_watched"
    VIDEO_SEGMENT_REPLAYED = "video_segment_replayed"
    FOCUS_SESSION_COMPLETED = "focus_session_completed"
    FOCUS_SESSION_BROKEN = "focus_session_broken"
    # Meta
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    FRUSTRATION_SIGNAL = "frustration_signal"
    HELP_REQUESTED = "help_requested"


# ═══════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ExternalLearningEvent:
    """
    Átomo universal de aprendizaje externo.
    xAPI simplificado: Actor + Verb + Object + Result + Context.
    Cada campo alimenta al menos un módulo existente de GENIE.
    """
    event_id: str = ""
    student_id: str = "est_01"
    timestamp: str = ""
    source: str = "custom"
    source_version: str = ""
    session_id: str = ""
    event_type: str = "exercise_completed"
    cognitive_domain: int = 3           # Bloom 1-6
    content: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    difficulty: float = 0.5
    time_spent_seconds: float = 0.0
    attempts: int = 1
    score: float = 1.0
    topic_tags: List[str] = field(default_factory=list)
    related_genie_topics: List[str] = field(default_factory=list)
    frustration_level: float = 0.0
    metacognitive_note: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.event_id:
            raw = f"{self.student_id}:{self.source}:{self.timestamp}:{self.event_type}"
            self.event_id = hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass
class LearningEpisode:
    """
    Unidad molar de actividad (Leontiev, 1978). Secuencia de eventos
    temporalmente cercanos con arco narrativo. cross_platform=True
    evidencia transferencia inter-instrumento.
    """
    episode_id: str = ""
    student_id: str = ""
    start_time: str = ""
    end_time: str = ""
    events: List[ExternalLearningEvent] = field(default_factory=list)
    sources_involved: List[str] = field(default_factory=list)
    dominant_topic: str = ""
    bloom_trajectory: List[int] = field(default_factory=list)
    frustration_arc: List[float] = field(default_factory=list)
    resolution: str = "ongoing"
    cross_platform: bool = False


@dataclass
class CrossPlatformProfile:
    """
    Gemelo Digital de Aprendizaje. Síntesis holística cross-fuente.
    Alimenta middleware vía get_middleware_injection().
    """
    student_id: str = ""
    last_updated: str = ""
    total_events: int = 0
    events_by_source: Dict[str, int] = field(default_factory=dict)
    active_sources: List[str] = field(default_factory=list)
    total_time_minutes: float = 0.0
    bloom_by_source: Dict[str, float] = field(default_factory=dict)
    bloom_global: float = 3.0
    bloom_trend: float = 0.0
    frustration_by_source: Dict[str, float] = field(default_factory=dict)
    frustration_global: float = 0.0
    frustration_hotspots: List[str] = field(default_factory=list)
    transfer_episodes: int = 0
    resolution_rate: float = 0.0
    preferred_help_source: str = ""
    learning_velocity: float = 0.0
    strength_topics: List[str] = field(default_factory=list)
    gap_topics: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    peak_hours: List[int] = field(default_factory=list)
    # Señales exclusivas de los nuevos adaptadores
    ai_dependency_ratio: float = 0.0    # copilot accept / (accept+reject)
    git_reflection_score: float = 0.0   # diffs viewed / commits
    focus_endurance_minutes: float = 0.0 # media de sesiones pomodoro completadas
    search_sophistication: float = 0.0  # bloom medio de las búsquedas


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#
#  CAPA 1 — IDE & CÓDIGO (5 adaptadores)
#
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

class VSCodeAdapter:
    """
    VS Code / Cursor. Señales: código, errores, ejecuciones, tests,
    Ctrl+Z (incertidumbre motora), borrado masivo (abandono de estrategia).
    """
    ERROR_BLOOM = {
        "SyntaxError": 1, "NameError": 2, "TypeError": 3,
        "IndexError": 3, "ValueError": 3, "AttributeError": 3,
        "ImportError": 2, "KeyError": 3, "RecursionError": 4,
        "LogicError": 4, "AssertionError": 5,
    }
    CODE_TOPICS = {
        r"\bfor\b.*\bin\b|\bwhile\b": "bucles",
        r"\bdef\b\s+\w+\s*\(": "funciones",
        r"\bclass\b\s+\w+": "clases",
        r"\breturn\b.*\w+\(": "recursión",
        r"\bif\b.*\belse\b|\belif\b": "condicionales",
        r"\binput\s*\(|\bprint\s*\(": "entrada/salida",
        r"\blist\b|\btuple\b|\bdict\b|\bset\b|\[": "arrays",
        r"\btry\b.*\bexcept\b": "depuración",
        r"\blambda\b|\bmap\b|\bfilter\b": "funcional",
    }

    def normalize(self, raw: dict) -> ExternalLearningEvent:
        event_map = {
            "file_save": EventType.CODE_WRITTEN.value,
            "error": EventType.CODE_ERROR.value,
            "error_fixed": EventType.CODE_FIXED.value,
            "run_success": EventType.CODE_EXECUTED.value,
            "run_fail": EventType.CODE_ERROR.value,
            "test_pass": EventType.TEST_PASSED.value,
            "test_fail": EventType.TEST_FAILED.value,
        }
        code = raw.get("code", "")
        error_type = raw.get("error_type", "")
        event_raw = raw.get("event", "file_save")
        bloom = self.ERROR_BLOOM.get(error_type, 3)
        if event_raw == "test_pass": bloom = 5
        elif event_raw in ("run_success", "error_fixed"): bloom = 4
        topics = self._topics(code)
        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"),
            source="vscode",
            source_version=raw.get("extension_version", "0.1.0"),
            session_id=raw.get("session_id", ""),
            event_type=event_map.get(event_raw, EventType.CODE_WRITTEN.value),
            cognitive_domain=bloom,
            content={"language": raw.get("language", "python"),
                     "code_snippet": code[:500], "filename": raw.get("filename", ""),
                     "error_type": error_type,
                     "error_message": raw.get("error_message", "")[:300],
                     "line_count": raw.get("line_count", 0),
                     "undo_count": raw.get("undo_count", 0)},
            success=event_raw not in ("error", "run_fail", "test_fail"),
            difficulty=self._diff(raw),
            time_spent_seconds=raw.get("time_since_last_save", 0),
            attempts=raw.get("attempt_number", 1),
            topic_tags=topics, related_genie_topics=topics,
            frustration_level=self._frust(raw),
        )

    def _diff(self, r): return round(min(
        0.3*min(r.get("undo_count",0)/20,1)+0.4*min(r.get("error_count_session",0)/10,1)
        +0.3*min(r.get("time_since_last_save",0)/600,1),1),3)

    def _frust(self, r): return round(min(
        0.25*min(r.get("undo_count",0)/15,1)+0.35*min(r.get("same_error_count",0)/5,1)
        +0.20*min(r.get("time_without_run",0)/300,1)+0.20*min(r.get("chars_deleted",0)/500,1),1),3)

    def _topics(self, code):
        t = [topic for p, topic in self.CODE_TOPICS.items() if re.search(p, code, re.I)]
        return t if t else ["otro"]


class TerminalAdapter:
    """
    Shell history (.bash_history / .zsh_history).

    La radiografía de aprendizaje que nadie examina.
    Señales:
    - man/--help → autoaprendizaje activo (Bloom 4: Analyze)
    - Comandos copiados sin variación → copy-paste (Bloom 1)
    - Pipes complejos (|, >, >>)  → composición (Bloom 5: Evaluate)
    - Repetición del mismo comando con variaciones → experimentación
    - sudo sin comprensión → riesgo operativo

    El adaptador lee patrones, NO el contenido sensible de comandos.
    """
    SELF_LEARN_PATTERNS = [r"\bman\b", r"--help\b", r"-h\b", r"\binfo\b", r"\bwhich\b", r"\btype\b"]
    COMPOSITION_PATTERNS = [r"\|", r">>?", r"\$\(", r"`.*`", r"&&", r"\|\|"]
    DANGER_PATTERNS = [r"\brm\s+-rf\b", r"\bsudo\b.*\brm\b", r"\bchmod\s+777\b"]

    def normalize(self, raw: dict) -> ExternalLearningEvent:
        cmd = raw.get("command", "")
        exit_code = raw.get("exit_code", 0)
        success = exit_code == 0

        bloom = 3  # Apply por defecto
        if any(re.search(p, cmd) for p in self.SELF_LEARN_PATTERNS):
            bloom = 4  # Analyze: busca comprender
            event_type = EventType.HELP_INVOKED.value
        elif any(re.search(p, cmd) for p in self.COMPOSITION_PATTERNS):
            bloom = 5  # Evaluate: compone herramientas
            event_type = EventType.COMMAND_EXECUTED.value if success else EventType.COMMAND_FAILED.value
        elif not success:
            bloom = 2
            event_type = EventType.COMMAND_FAILED.value
        else:
            event_type = EventType.COMMAND_EXECUTED.value

        # Detección de repetición mecánica (proxy de copy-paste)
        repeat_count = raw.get("repeat_count", 0)
        frustration = min(repeat_count / 8.0, 1.0) if not success else 0.0

        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"),
            source="terminal",
            session_id=raw.get("session_id", ""),
            event_type=event_type,
            cognitive_domain=bloom,
            content={"command_hash": hashlib.sha256(cmd.encode()).hexdigest()[:8],
                     "shell": raw.get("shell", "bash"),
                     "exit_code": exit_code,
                     "has_pipe": bool(re.search(r"\|", cmd)),
                     "has_redirect": bool(re.search(r">>?", cmd)),
                     "word_count": len(cmd.split()),
                     "is_dangerous": any(re.search(p, cmd) for p in self.DANGER_PATTERNS)},
            success=success,
            difficulty=min(len(cmd.split()) / 10.0, 1.0),
            time_spent_seconds=raw.get("duration_seconds", 0),
            topic_tags=["depuración" if not success else "otro"],
            frustration_level=frustration,
        )


class GitLogAdapter:
    """
    Git microhistoria: commits, diffs, stash, branching.

    NO los commits finales — la MICRO-HISTORIA:
    - git diff frecuente antes de commit → reflexión (Bloom 5)
    - Tiempo entre commits → velocidad de iteración
    - git stash / git stash pop → incertidumbre estratégica
    - Mensajes de commit genéricos ("fix", "update") → Bloom bajo
    - Mensajes descriptivos ("refactor auth to handle JWT") → Bloom 4-5
    - Ratio diffs/commits = "reflection_score" (métrica original)

    Señal única que ningún otro adaptador captura: el PROCESO de
    construcción del código, no solo el resultado.
    """
    GENERIC_MESSAGES = {"fix", "update", "wip", "changes", "test", ".", "commit", "save"}

    def normalize(self, raw: dict) -> ExternalLearningEvent:
        action = raw.get("action", "commit")
        msg = raw.get("message", "").strip().lower()

        if action == "diff":
            event_type = EventType.DIFF_VIEWED.value
            bloom = 5  # Evaluate: revisando antes de commitear
        elif action == "branch":
            event_type = EventType.BRANCH_CREATED.value
            bloom = 4  # Analyze: planificación
        elif action == "commit":
            event_type = EventType.COMMIT_MADE.value
            # Bloom según calidad del mensaje
            words = msg.split()
            if msg in self.GENERIC_MESSAGES or len(words) <= 1:
                bloom = 2  # Remember: commit mecánico
            elif len(words) >= 5:
                bloom = 4  # Analyze: describe lo que hizo y por qué
            else:
                bloom = 3  # Apply
        else:
            event_type = EventType.CODE_WRITTEN.value
            bloom = 3

        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"),
            source="git",
            session_id=raw.get("session_id", ""),
            event_type=event_type,
            cognitive_domain=bloom,
            content={"action": action,
                     "message_quality": "generic" if msg in self.GENERIC_MESSAGES else "descriptive",
                     "files_changed": raw.get("files_changed", 0),
                     "insertions": raw.get("insertions", 0),
                     "deletions": raw.get("deletions", 0),
                     "branch": raw.get("branch", "main"),
                     "is_merge": raw.get("is_merge", False),
                     "is_stash": action in ("stash", "stash_pop")},
            success=True,
            difficulty=min(raw.get("files_changed", 1) / 10.0, 1.0),
            time_spent_seconds=raw.get("time_since_last_commit", 0),
            topic_tags=["funciones"],
            frustration_level=0.3 if action in ("stash", "revert", "reset") else 0.0,
        )


class CopilotAdapter:
    """
    GitHub Copilot / Cursor tab completions — Accept vs Reject.

    LA MÉTRICA QUE NADIE USA EN LEARNING ANALYTICS:
    Ratio de aceptación de sugerencias de IA como proxy de autonomía.

    - Accept rate > 90% → NO está aprendiendo, está copiando de IA (Bloom 1)
    - Accept rate 40-70% → Evalúa críticamente (Bloom 5)
    - Accept rate < 20% → Trabaja independientemente (autonomía alta)
    - Reject + immediate manual write → comprensión profunda
    - Accept + immediate modification → comprensión parcial

    Esta métrica alimenta directamente al EpistemicAutonomyTracker
    como proxy de dependencia tecnológica. Un estudiante con accept
    rate > 85% que luego falla en GENIE sin Copilot evidencia
    transferencia fallida — comprensión ilusoria.
    """

    def normalize(self, raw: dict) -> ExternalLearningEvent:
        action = raw.get("action", "accept")  # accept | reject | partial_accept
        accepted = action in ("accept", "partial_accept")

        if action == "reject":
            bloom = 5   # Evaluate: juzgó y rechazó
            event_type = EventType.AI_COMPLETION_REJECTED.value
        elif action == "partial_accept":
            bloom = 4   # Analyze: tomó lo útil, descartó el resto
            event_type = EventType.AI_COMPLETION_ACCEPTED.value
        else:
            bloom = 1   # Remember: aceptó sin cuestionar
            event_type = EventType.AI_COMPLETION_ACCEPTED.value

        # Si modificó inmediatamente después de aceptar → comprensión parcial
        if accepted and raw.get("modified_after_accept", False):
            bloom = 3  # Apply: adapta la sugerencia

        suggestion_len = len(raw.get("suggestion", "").split("\n"))
        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"),
            source="copilot",
            session_id=raw.get("session_id", ""),
            event_type=event_type,
            cognitive_domain=bloom,
            content={"action": action,
                     "suggestion_lines": suggestion_len,
                     "language": raw.get("language", "python"),
                     "context_length": raw.get("context_length", 0),
                     "modified_after": raw.get("modified_after_accept", False),
                     "time_to_decide_ms": raw.get("time_to_decide_ms", 0),
                     "completion_type": raw.get("completion_type", "inline")},
            success=True,
            difficulty=min(suggestion_len / 20.0, 1.0),
            time_spent_seconds=raw.get("time_to_decide_ms", 0) / 1000.0,
            topic_tags=["funciones"],
            # Aceptación ciega = baja frustración pero alto riesgo pedagógico
            frustration_level=0.0 if accepted else 0.1,
        )


class ReplitAdapter:
    """
    Replit / CodeSandbox — Cloud coding environments.

    Captura el código que se escribe FUERA del IDE local.
    Si practicas desde el móvil, esta sesión se perdería para
    VSCodeAdapter. ReplitAdapter la recupera.

    Replit expone un multiplayer protocol y tiene URLs públicas
    para repls → la extensión de browser captura la actividad.
    """

    def normalize(self, raw: dict) -> ExternalLearningEvent:
        event_raw = raw.get("event", "run")
        success = raw.get("success", True)

        event_map = {
            "run": EventType.CODE_EXECUTED.value,
            "error": EventType.CODE_ERROR.value,
            "save": EventType.CODE_WRITTEN.value,
            "fork": EventType.CODE_WRITTEN.value,
            "share": EventType.CODE_WRITTEN.value,
        }

        bloom = 3
        if event_raw == "error": bloom = 2
        elif raw.get("is_forked", False): bloom = 4  # Analyze: examina código ajeno
        elif success and raw.get("test_output", ""): bloom = 5

        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"),
            source="replit",
            session_id=raw.get("session_id", ""),
            event_type=event_map.get(event_raw, EventType.CODE_EXECUTED.value),
            cognitive_domain=bloom,
            content={"language": raw.get("language", "python"),
                     "repl_name": raw.get("repl_name", ""),
                     "is_forked": raw.get("is_forked", False),
                     "console_output": raw.get("console_output", "")[:200],
                     "file_count": raw.get("file_count", 1)},
            success=success,
            difficulty=min(raw.get("file_count", 1) / 5.0, 1.0),
            time_spent_seconds=raw.get("session_duration", 0),
            topic_tags=["ejercicio"],
            frustration_level=min(raw.get("error_count", 0) / 8.0, 1.0),
        )


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#
#  CAPA 2 — PLATAFORMAS DE APRENDIZAJE (4 adaptadores)
#
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

class DuolingoAdapter:
    """
    Duolingo. Browser extension intercepta DOM: ejercicio, respuesta,
    resultado, pistas, tiempo. Bloom 1-5 según tipo de ejercicio.
    """
    EXERCISE_BLOOM = {
        "word_bank": 1, "match": 1, "translate_from_target": 2, "listening": 2,
        "translate_to_target": 3, "speaking": 3, "free_write": 4, "story": 5,
    }
    TYPE_DIFFICULTY = {
        "word_bank": 0.2, "match": 0.1, "translate_from_target": 0.4, "listening": 0.5,
        "translate_to_target": 0.6, "speaking": 0.7, "free_write": 0.8, "story": 0.6,
    }

    def normalize(self, raw: dict) -> ExternalLearningEvent:
        ex_type = raw.get("exercise_type", "translate_from_target")
        success = raw.get("correct", True)
        bloom = self.EXERCISE_BLOOM.get(ex_type, 2)
        if not success and raw.get("attempts", 1) > 2: bloom = max(1, bloom - 1)
        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"), source="duolingo",
            source_version=raw.get("extension_version", "0.1.0"),
            session_id=raw.get("session_id", ""),
            event_type=(EventType.EXERCISE_COMPLETED.value if success
                        else EventType.EXERCISE_FAILED.value),
            cognitive_domain=bloom,
            content={"target_language": raw.get("target_language", "en"),
                     "source_language": raw.get("source_language", "es"),
                     "exercise_type": ex_type,
                     "prompt_text": raw.get("prompt_text", ""),
                     "correct_answer": raw.get("correct_answer", ""),
                     "student_answer": raw.get("student_answer", ""),
                     "unit_name": raw.get("unit_name", ""),
                     "skill_name": raw.get("skill_name", ""),
                     "xp_earned": raw.get("xp_earned", 0)},
            success=success,
            difficulty=round(0.3*min(len(raw.get("prompt_text","").split())/15,1)
                             +0.5*self.TYPE_DIFFICULTY.get(ex_type,0.4)
                             +0.2*(1.0 if raw.get("is_retry") else 0.0), 3),
            time_spent_seconds=raw.get("time_spent", 0),
            attempts=raw.get("attempts", 1),
            score=1.0 if success else 0.0,
            topic_tags=[raw.get("skill_name", "vocabulary")],
            related_genie_topics=["conceptual"],
            frustration_level=round(min(
                0.30*min(raw.get("attempts",1)/5,1)+0.20*min(raw.get("hints_used",0)/3,1)
                +0.30*(1.0 if raw.get("lesson_abandoned") else 0.0)
                +0.20*(min((raw.get("time_spent",0)-60)/120,1) if raw.get("time_spent",0)>60 else 0),
                1.0), 3),
        )


class MimoAdapter:
    """
    Mimo: Learn to Code — Micro-lecciones de Python, JS, HTML, CSS, SQL.

    Señal diferencial: el patrón de REPETICIÓN. Mimo permite repetir
    lecciones ilimitadamente. Qué lecciones repite el estudiante y
    cuántas veces es la huella dactilar de sus gaps.

    Captura vía browser extension (DOM de mimo.org):
    - Lección actual (nombre, tema, lenguaje)
    - Resultado de cada ejercicio interactivo
    - Si es repetición de una lección anterior
    - Tiempo por ejercicio
    - Streak (días consecutivos)

    Mimo no tiene API pública, pero su DOM es estable y predecible.
    """

    LANGUAGE_TOPICS = {
        "python": ["funciones", "variables", "bucles"],
        "javascript": ["funciones", "arrays", "depuración"],
        "html": ["conceptual"],
        "css": ["conceptual"],
        "sql": ["arrays", "funciones"],
        "swift": ["funciones", "clases"],
        "kotlin": ["funciones", "clases"],
    }

    def normalize(self, raw: dict) -> ExternalLearningEvent:
        success = raw.get("correct", True)
        is_repeat = raw.get("is_repeat", False)
        language = raw.get("language", "python").lower()

        # Bloom: repetir = Remember; completar nuevo = Apply; reto = Analyze
        if is_repeat:
            bloom = 1
        elif raw.get("is_challenge", False):
            bloom = 4
        elif raw.get("is_project", False):
            bloom = 5
        elif success:
            bloom = 3
        else:
            bloom = 2

        topics = self.LANGUAGE_TOPICS.get(language, ["otro"])

        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"),
            source="mimo",
            session_id=raw.get("session_id", ""),
            event_type=(EventType.EXERCISE_COMPLETED.value if success
                        else EventType.EXERCISE_FAILED.value),
            cognitive_domain=bloom,
            content={"language": language,
                     "lesson_name": raw.get("lesson_name", ""),
                     "section": raw.get("section", ""),
                     "is_repeat": is_repeat,
                     "repeat_count": raw.get("repeat_count", 0),
                     "is_challenge": raw.get("is_challenge", False),
                     "is_project": raw.get("is_project", False),
                     "streak_days": raw.get("streak_days", 0),
                     "xp_earned": raw.get("xp_earned", 0)},
            success=success,
            difficulty=0.3 if is_repeat else (0.7 if raw.get("is_challenge") else 0.5),
            time_spent_seconds=raw.get("time_spent", 0),
            attempts=raw.get("attempts", 1),
            topic_tags=topics, related_genie_topics=topics,
            frustration_level=round(min(
                0.4*min(raw.get("repeat_count",0)/5,1)
                +0.3*min(raw.get("attempts",1)/4,1)
                +0.3*(1.0 if raw.get("lesson_abandoned") else 0.0), 1.0), 3),
        )


class LeetCodeAdapter:
    """
    LeetCode / HackerRank / Codewars — Competitive coding.

    Señal extraordinariamente rica: no solo si resolvió el problema
    sino en qué percentil de eficiencia, categoría de algoritmo,
    cuántos intentos, y si usó hints.

    LeetCode tiene API GraphQL semi-pública.
    Codewars tiene API REST documentada.
    HackerRank requiere scraping.

    Bloom mapping:
    - Easy solved on first try → Apply (3)
    - Medium solved → Analyze (4)
    - Hard solved → Evaluate (5)
    - Easy failed → Understand (2)
    - Used editorial/solution → Remember (1)
    """

    DIFFICULTY_BLOOM = {
        "easy": {True: 3, False: 2},
        "medium": {True: 4, False: 3},
        "hard": {True: 5, False: 4},
    }

    CATEGORY_TOPICS = {
        "array": "arrays", "string": "arrays", "hash_table": "arrays",
        "linked_list": "arrays", "stack": "arrays", "queue": "arrays",
        "tree": "recursión", "graph": "recursión", "dfs": "recursión",
        "bfs": "recursión", "recursion": "recursión",
        "dynamic_programming": "recursión", "backtracking": "recursión",
        "sorting": "funciones", "binary_search": "funciones",
        "greedy": "funciones", "math": "conceptual",
        "bit_manipulation": "conceptual",
    }

    def normalize(self, raw: dict) -> ExternalLearningEvent:
        difficulty = raw.get("difficulty", "medium").lower()
        success = raw.get("accepted", False)
        used_editorial = raw.get("used_editorial", False)

        if used_editorial:
            bloom = 1  # Remember: copió la solución
        else:
            bloom = self.DIFFICULTY_BLOOM.get(difficulty, {True: 3, False: 2}).get(success, 3)

        category = raw.get("category", "array").lower()
        genie_topic = self.CATEGORY_TOPICS.get(category, "otro")

        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"),
            source="leetcode",
            session_id=raw.get("session_id", ""),
            event_type=(EventType.CHALLENGE_SOLVED.value if success
                        else EventType.CHALLENGE_FAILED.value),
            cognitive_domain=bloom,
            content={"platform": raw.get("platform", "leetcode"),
                     "problem_name": raw.get("problem_name", ""),
                     "problem_id": raw.get("problem_id", ""),
                     "difficulty": difficulty,
                     "category": category,
                     "language": raw.get("language", "python"),
                     "runtime_percentile": raw.get("runtime_percentile", 50),
                     "memory_percentile": raw.get("memory_percentile", 50),
                     "used_editorial": used_editorial,
                     "submission_count": raw.get("submission_count", 1)},
            success=success,
            difficulty={"easy": 0.3, "medium": 0.6, "hard": 0.9}.get(difficulty, 0.5),
            time_spent_seconds=raw.get("time_spent", 0),
            attempts=raw.get("submission_count", 1),
            topic_tags=[genie_topic], related_genie_topics=[genie_topic],
            frustration_level=round(min(
                0.4*min(raw.get("submission_count",1)/8,1)
                +0.3*(1.0 if raw.get("time_spent",0)>3600 else raw.get("time_spent",0)/3600)
                +0.3*(1.0 if not success and raw.get("submission_count",1)>5 else 0.0), 1.0), 3),
        )


class FreeCodeCampAdapter:
    """
    freeCodeCamp — Curriculum completo con perfil público.

    Scraping limpio: los perfiles públicos de freeCodeCamp exponen
    los challenges completados con timestamp. La extensión parsea
    la página de perfil y extrae el progreso incremental.

    Señal diferencial: freeCodeCamp es secuencial. El orden en que
    el estudiante completa los módulos revela si sigue el currículo
    o salta erráticamente. Saltar indica confianza o impaciencia;
    seguir en orden indica disciplina o inseguridad.
    """

    CERT_BLOOM = {
        "responsive_web_design": 3,
        "javascript_algorithms": 4,
        "front_end_libraries": 4,
        "data_visualization": 5,
        "apis_and_microservices": 5,
        "quality_assurance": 5,
        "scientific_computing_python": 4,
        "data_analysis_python": 5,
        "machine_learning_python": 6,
    }

    def normalize(self, raw: dict) -> ExternalLearningEvent:
        success = raw.get("completed", True)
        cert = raw.get("certification", "").lower().replace(" ", "_")
        bloom = self.CERT_BLOOM.get(cert, 3)
        is_sequential = raw.get("is_sequential", True)

        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"),
            source="freecodecamp",
            session_id=raw.get("session_id", ""),
            event_type=(EventType.CHALLENGE_SOLVED.value if success
                        else EventType.CHALLENGE_FAILED.value),
            cognitive_domain=bloom,
            content={"challenge_name": raw.get("challenge_name", ""),
                     "certification": cert,
                     "is_sequential": is_sequential,
                     "challenges_completed": raw.get("challenges_completed", 0),
                     "challenges_total": raw.get("challenges_total", 0)},
            success=success,
            difficulty=0.5,
            time_spent_seconds=raw.get("time_spent", 0),
            topic_tags=["ejercicio"], related_genie_topics=["ejercicio"],
            frustration_level=0.0,
        )


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#
#  CAPA 3 — MEMORIA & NOTAS (2 adaptadores)
#
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

class AnkiAdapter:
    """
    Anki vía AnkiConnect (localhost:8765). Curva de olvido (Ebbinghaus, 1885).
    Triangulación: bien en Anki + mal en GENIE → memoriza sin comprender.
    """
    def normalize(self, raw: dict) -> ExternalLearningEvent:
        ease = raw.get("ease", 2)
        event_type = EventType.CARD_REVIEWED.value
        if ease == 1: event_type = EventType.CARD_LAPSED.value
        elif raw.get("interval_days", 0) > 30: event_type = EventType.CARD_MATURED.value
        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"), source="anki",
            session_id=raw.get("session_id", ""),
            event_type=event_type, cognitive_domain=1 if ease <= 2 else 2,
            content={"deck_name": raw.get("deck_name", ""),
                     "card_front": raw.get("card_front", ""),
                     "ease": ease, "interval_days": raw.get("interval_days", 0),
                     "lapses": raw.get("lapses", 0)},
            success=ease >= 3, difficulty=1.0-(ease/4.0),
            time_spent_seconds=raw.get("time_ms", 0)/1000.0,
            score=ease/4.0,
            topic_tags=[raw.get("deck_name", "general")],
            frustration_level=min(raw.get("lapses", 0)/5.0, 1.0),
        )


class NotesAdapter:
    """
    Obsidian / Notion — Notas personales del estudiante.

    Lo que el estudiante escribe en sus notas es la manifestación
    más pura de su comprensión. Señales:

    - Estructura jerárquica (headers anidados) → comprensión organizada
    - Caótica (texto plano sin estructura) → fase exploratoria
    - Enlaces internos ([[wikilinks]]) → pensamiento relacional (Bloom 4+)
    - Cantidad de notas enlazadas → densidad de grafo de conocimiento
    - Preguntas pendientes marcadas → metacognición activa

    Obsidian: API local (plugin comunidad). Notion: API REST oficial.
    El adaptador recibe metadata, NO el contenido completo.
    """

    def normalize(self, raw: dict) -> ExternalLearningEvent:
        action = raw.get("action", "create")  # create | edit | link
        has_links = raw.get("internal_links", 0) > 0
        has_structure = raw.get("heading_count", 0) > 2
        has_questions = raw.get("question_marks", 0) > 0

        # Bloom según complejidad estructural de la nota
        if has_links and has_structure:
            bloom = 5  # Evaluate/Create: organiza y conecta
        elif has_links:
            bloom = 4  # Analyze: conecta conceptos
        elif has_structure:
            bloom = 3  # Apply: organiza
        elif has_questions:
            bloom = 4  # Metacognición: se pregunta cosas
        else:
            bloom = 2  # Understand: toma notas lineales

        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"),
            source="notes",
            session_id=raw.get("session_id", ""),
            event_type=EventType.NOTE_TAKEN.value,
            cognitive_domain=bloom,
            content={"platform": raw.get("platform", "obsidian"),
                     "action": action,
                     "word_count": raw.get("word_count", 0),
                     "heading_count": raw.get("heading_count", 0),
                     "internal_links": raw.get("internal_links", 0),
                     "external_links": raw.get("external_links", 0),
                     "code_blocks": raw.get("code_blocks", 0),
                     "question_marks": raw.get("question_marks", 0),
                     "tags": raw.get("tags", []),
                     "vault_total_notes": raw.get("vault_total_notes", 0)},
            success=True,
            difficulty=min(raw.get("word_count", 0) / 500.0, 1.0),
            time_spent_seconds=raw.get("editing_time", 0),
            topic_tags=raw.get("tags", ["conceptual"]) or ["conceptual"],
            frustration_level=0.0,
        )


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#
#  CAPA 4 — NAVEGACIÓN & BÚSQUEDA (3 adaptadores)
#
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

class BrowserAdapter:
    """Navegación web educativa. Bloom estimado por dominio."""
    DOMAINS = {
        "stackoverflow.com": ("depuración", 4),
        "developer.mozilla.org": ("conceptual", 2),
        "docs.python.org": ("conceptual", 2),
        "w3schools.com": ("ejercicio", 3),
        "geeksforgeeks.org": ("ejercicio", 3),
        "github.com": ("funciones", 4),
        "arxiv.org": ("conceptual", 5),
        "wikipedia.org": ("conceptual", 1),
    }
    def normalize(self, raw: dict) -> ExternalLearningEvent:
        url = raw.get("url", "")
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.replace("www.", "")
        except Exception: domain = "unknown"
        topic, bloom = self.DOMAINS.get(domain, ("otro", 2))
        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"), source="browser",
            event_type=EventType.PAGE_READ.value, cognitive_domain=bloom,
            content={"url": url, "domain": domain, "title": raw.get("title", ""),
                     "time_on_page": raw.get("time_on_page", 0),
                     "scroll_depth": raw.get("scroll_depth", 0)},
            success=True, difficulty=0.5,
            time_spent_seconds=raw.get("time_on_page", 0),
            topic_tags=[topic], related_genie_topics=[topic],
        )


class SearchQueryAdapter:
    """
    Búsquedas de error en Google / DuckDuckGo / Bing.

    La query de búsqueda es ventana directa al modelo mental:
    - "python list not working" → Bloom 1 (no sabe nombrar el problema)
    - "python IndexError list empty" → Bloom 2 (identifica el error)
    - "python list comprehension nested generator" → Bloom 4
    - "python generator vs iterator memory efficiency" → Bloom 5

    El adaptador (browser extension) captura queries en dominios
    técnicos y estima Bloom por complejidad léxica.
    """

    TECHNICAL_TERMS = {
        1: {"not working", "help", "error", "how to", "what is", "fix"},
        2: {"example", "tutorial", "syntax", "meaning", "define"},
        3: {"implementation", "method", "function", "convert", "parse"},
        4: {"difference between", "compare", "vs", "optimize", "algorithm", "complexity"},
        5: {"architecture", "design pattern", "tradeoff", "benchmark", "internals"},
    }

    def normalize(self, raw: dict) -> ExternalLearningEvent:
        query = raw.get("query", "").lower()
        bloom = self._estimate_bloom(query)

        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"),
            source="search",
            session_id=raw.get("session_id", ""),
            event_type=EventType.SEARCH_PERFORMED.value,
            cognitive_domain=bloom,
            content={"query": query,
                     "engine": raw.get("engine", "google"),
                     "results_clicked": raw.get("results_clicked", 0),
                     "first_result_domain": raw.get("first_click_domain", ""),
                     "is_error_search": bool(re.search(
                         r"error|exception|traceback|bug|fix|crash", query))},
            success=True,
            difficulty=bloom / 6.0,
            time_spent_seconds=raw.get("search_to_click_seconds", 0),
            topic_tags=["depuración" if "error" in query else "conceptual"],
            frustration_level=round(min(raw.get("searches_in_sequence", 1) / 6.0, 1.0), 3),
        )

    def _estimate_bloom(self, query: str) -> int:
        words = set(query.split())
        for level in range(5, 0, -1):
            if words & self.TECHNICAL_TERMS.get(level, set()):
                return level
        return 2  # Default: Understand


class AIConversationAdapter:
    """
    ChatGPT / Claude / Gemini / otras IAs — Metadata de uso.

    Lo que le preguntas a la IA es exactamente lo que NO sabes.
    El "mapa de ignorancias" del estudiante, cartografiado en
    tiempo real.

    POR PRIVACIDAD: el adaptador captura solo METADATA, nunca
    el contenido completo de la conversación:
    - Plataforma usada (ChatGPT, Claude, Gemini, Perplexity)
    - Tema inferido (keyword extraction del primer mensaje)
    - Longitud del prompt (proxy de especificidad)
    - Si pegó código (copy-paste al chatbot)
    - Si copió la respuesta (copy-paste desde el chatbot)
    - Frecuencia de uso (proxy de dependencia)
    """

    def normalize(self, raw: dict) -> ExternalLearningEvent:
        prompt_length = raw.get("prompt_word_count", 10)
        pasted_code = raw.get("pasted_code", False)
        copied_response = raw.get("copied_response", False)

        # Bloom según comportamiento
        if pasted_code and copied_response:
            bloom = 1  # Remember: copia ida y vuelta sin procesar
        elif pasted_code and not copied_response:
            bloom = 4  # Analyze: usa IA para entender su error
        elif prompt_length > 30:
            bloom = 3  # Apply: formula pregunta elaborada
        elif prompt_length < 5:
            bloom = 1  # Remember: "fix this"
        else:
            bloom = 2

        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"),
            source="ai_chat",
            session_id=raw.get("session_id", ""),
            event_type=EventType.AI_QUERY.value,
            cognitive_domain=bloom,
            content={"platform": raw.get("platform", "chatgpt"),
                     "prompt_word_count": prompt_length,
                     "pasted_code": pasted_code,
                     "copied_response": copied_response,
                     "topic_keywords": raw.get("topic_keywords", []),
                     "conversation_turn": raw.get("turn_number", 1),
                     "model_used": raw.get("model", "unknown")},
            success=True,
            difficulty=min(prompt_length / 50.0, 1.0),
            time_spent_seconds=raw.get("time_composing", 0),
            topic_tags=raw.get("topic_keywords", ["otro"]) or ["otro"],
            frustration_level=0.2 if pasted_code and raw.get("turn_number",1)>3 else 0.0,
        )


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#
#  CAPA 5 — SEÑALES CONDUCTUALES (3 adaptadores)
#
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

class KeystrokeAdapter:
    """
    Keystroke Dynamics — Patrones de tecleo como proxy cognitivo.

    Territorio experimental, fundamentación sólida:
    La velocidad de tecleo cuando escribes código que ENTIENDES
    vs código que estás COPIANDO es mediblemente diferente.

    Señales:
    - WPM (words per minute) durante coding → fluidez
    - Ratio pause/type (pausas largas = planificación o atasco)
    - Burst patterns (ráfagas seguidas de pausas = prueba-y-error)
    - Backspace rate (borrado / total keystrokes = incertidumbre)
    - Key flight time variability (σ del tiempo entre teclas)

    Materialización computacional de los marcadores somáticos
    de Damasio (1994): los patrones motores REVELAN estados
    cognitivos que el estudiante no reporta conscientemente.
    """

    def normalize(self, raw: dict) -> ExternalLearningEvent:
        wpm = raw.get("wpm", 40)
        pause_ratio = raw.get("pause_ratio", 0.3)  # 0-1
        backspace_rate = raw.get("backspace_rate", 0.1)  # 0-1
        burst_count = raw.get("burst_count", 0)  # ráfagas/min

        # Inferir estado cognitivo desde patrones motores
        if wpm > 60 and backspace_rate < 0.05:
            bloom = 4  # Flujo: sabe lo que escribe
            state = "flow"
        elif pause_ratio > 0.6:
            bloom = 4 if backspace_rate < 0.1 else 2
            state = "planning" if backspace_rate < 0.1 else "stuck"
        elif burst_count > 5 and backspace_rate > 0.2:
            bloom = 2  # Trial-and-error sin comprensión
            state = "trial_error"
        elif backspace_rate > 0.3:
            bloom = 1  # No sabe qué escribir
            state = "confused"
        else:
            bloom = 3
            state = "normal"

        frustration = round(min(
            0.3 * backspace_rate * 3 +
            0.3 * min(burst_count / 10.0, 1.0) +
            0.2 * (1.0 if pause_ratio > 0.5 else 0.0) +
            0.2 * (1.0 - min(wpm / 60.0, 1.0)),
            1.0), 3)

        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"),
            source="keystroke",
            session_id=raw.get("session_id", ""),
            event_type=EventType.KEYSTROKE_PATTERN.value,
            cognitive_domain=bloom,
            content={"wpm": wpm, "pause_ratio": pause_ratio,
                     "backspace_rate": backspace_rate,
                     "burst_count": burst_count,
                     "inferred_state": state,
                     "key_flight_stddev": raw.get("key_flight_stddev", 0),
                     "sample_duration_seconds": raw.get("sample_duration", 60)},
            success=state in ("flow", "planning", "normal"),
            difficulty=0.5,
            time_spent_seconds=raw.get("sample_duration", 60),
            topic_tags=["otro"],
            frustration_level=frustration,
        )


class VideoAdapter:
    """
    YouTube / videos educativos — Patrón de consumo, NO contenido.

    Señales que el browser extension captura:
    - Pausas frecuentes → tomando notas O perdido
    - Retrocede al mismo segmento → no entiende ese concepto
    - Velocidad 2x → ya lo sabe, repasa rápido
    - Abandono < 30s → no era lo que buscaba
    - Ver completo a 1x → engaged, aprendiendo

    YouTube no expone esta telemetría, pero el browser extension
    intercepta eventos del video player (play/pause/seek/rate).
    """

    def normalize(self, raw: dict) -> ExternalLearningEvent:
        watch_pct = raw.get("watch_percentage", 100)
        speed = raw.get("playback_speed", 1.0)
        pauses = raw.get("pause_count", 0)
        replays = raw.get("segment_replay_count", 0)
        duration = raw.get("video_duration_seconds", 600)

        # Bloom según comportamiento de consumo
        if replays > 3:
            bloom = 2  # Understand: repite porque no entiende
            event_type = EventType.VIDEO_SEGMENT_REPLAYED.value
        elif speed >= 1.75:
            bloom = 4  # Analyze: ya domina, repasa eficientemente
            event_type = EventType.VIDEO_WATCHED.value
        elif watch_pct < 10:
            bloom = 1  # Skimming
            event_type = EventType.VIDEO_WATCHED.value
        elif pauses > 5 and watch_pct > 80:
            bloom = 4  # Analyze: pausa para procesar
            event_type = EventType.VIDEO_WATCHED.value
        else:
            bloom = 3
            event_type = EventType.VIDEO_WATCHED.value

        frustration = round(min(
            0.4 * min(replays / 5.0, 1.0) +
            0.3 * (1.0 if watch_pct < 20 and duration > 60 else 0.0) +
            0.3 * min(pauses / 15.0, 1.0),
            1.0), 3)

        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"),
            source="video",
            session_id=raw.get("session_id", ""),
            event_type=event_type,
            cognitive_domain=bloom,
            content={"platform": raw.get("platform", "youtube"),
                     "video_title": raw.get("title", ""),
                     "channel": raw.get("channel", ""),
                     "watch_percentage": watch_pct,
                     "playback_speed": speed,
                     "pause_count": pauses,
                     "segment_replay_count": replays,
                     "video_duration": duration,
                     "is_tutorial": raw.get("is_tutorial", False)},
            success=watch_pct > 50,
            difficulty=0.5,
            time_spent_seconds=duration * (watch_pct / 100.0),
            topic_tags=["conceptual"],
            frustration_level=frustration,
        )


class PomodoroAdapter:
    """
    Forest / Toggl / Focus apps — Sesiones de concentración.

    Señal pura de autorregulación (metacognición operativa):
    - Pomodoros completados → disciplina y resistencia atencional
    - Pomodoros rotos → dificultad para mantener foco
    - Duración media → capacidad de concentración sostenida
    - Hora del día → ritmo circadiano de productividad

    Toggl tiene API abierta. Forest no tiene API pero el
    browser extension puede capturar datos de la web app.

    Esta métrica alimenta directamente la recomendación de
    scaffolding: un estudiante que rompe pomodoros necesita
    explicaciones más cortas y segmentadas.
    """

    def normalize(self, raw: dict) -> ExternalLearningEvent:
        completed = raw.get("completed", True)
        duration_planned = raw.get("planned_minutes", 25)
        duration_actual = raw.get("actual_minutes", 25)
        broken_early = not completed and duration_actual < duration_planned * 0.8

        if completed and duration_planned >= 45:
            bloom = 5  # Deep work sostenido
        elif completed:
            bloom = 3  # Disciplina estándar
        elif broken_early:
            bloom = 2  # Dificultad atencional
        else:
            bloom = 3

        return ExternalLearningEvent(
            student_id=raw.get("student_id", "est_01"),
            source="pomodoro",
            session_id=raw.get("session_id", ""),
            event_type=(EventType.FOCUS_SESSION_COMPLETED.value if completed
                        else EventType.FOCUS_SESSION_BROKEN.value),
            cognitive_domain=bloom,
            content={"app": raw.get("app", "generic"),
                     "planned_minutes": duration_planned,
                     "actual_minutes": duration_actual,
                     "completed": completed,
                     "break_reason": raw.get("break_reason", ""),
                     "label": raw.get("label", "study"),
                     "daily_completed": raw.get("daily_completed", 0),
                     "daily_broken": raw.get("daily_broken", 0)},
            success=completed,
            difficulty=min(duration_planned / 60.0, 1.0),
            time_spent_seconds=duration_actual * 60,
            topic_tags=["otro"],
            frustration_level=round(min(
                0.5 * (1.0 if broken_early else 0.0) +
                0.3 * min(raw.get("daily_broken", 0) / 4.0, 1.0) +
                0.2 * (1.0 - min(duration_actual / max(duration_planned, 1), 1.0)),
                1.0), 3),
        )


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
#
#  INFRAESTRUCTURA — Correlador, Profiler, Orquestador
#
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

class TemporalCorrelator:
    """
    Vincula eventos cross-fuente en episodios de aprendizaje.
    Ventana temporal de 30 min. cross_platform=True → transferencia.
    """
    def __init__(self, window_minutes: int = 30):
        self.window = timedelta(minutes=window_minutes)
        self.episodes: List[LearningEpisode] = []

    def ingest(self, event: ExternalLearningEvent) -> Optional[LearningEpisode]:
        active = self._find(event)
        if active:
            active.events.append(event)
            active.end_time = event.timestamp
            if event.source not in active.sources_involved:
                active.sources_involved.append(event.source)
                active.cross_platform = True
            active.bloom_trajectory.append(event.cognitive_domain)
            active.frustration_arc.append(event.frustration_level)
            active.resolution = self._assess(active)
            return active
        ep = LearningEpisode(
            episode_id=hashlib.sha256(f"{event.student_id}:{event.timestamp}".encode()).hexdigest()[:12],
            student_id=event.student_id, start_time=event.timestamp, end_time=event.timestamp,
            events=[event], sources_involved=[event.source],
            dominant_topic=event.topic_tags[0] if event.topic_tags else "otro",
            bloom_trajectory=[event.cognitive_domain], frustration_arc=[event.frustration_level])
        self.episodes.append(ep)
        if len(self.episodes) > 500: self.episodes = self.episodes[-500:]
        return ep

    def _find(self, event):
        try: et = datetime.fromisoformat(event.timestamp)
        except: return None
        for ep in reversed(self.episodes[-50:]):
            if ep.student_id != event.student_id: continue
            try: ee = datetime.fromisoformat(ep.end_time)
            except: continue
            if et - ee > self.window: continue
            if ep.dominant_topic in event.topic_tags or "otro" in event.topic_tags: return ep
            if ep.frustration_arc and ep.frustration_arc[-1] > 0.5 and event.source != ep.sources_involved[-1]:
                return ep
        return None

    def _assess(self, ep):
        if not ep.events: return "ongoing"
        last = ep.events[-1]
        f = ep.frustration_arc[-1] if ep.frustration_arc else 0
        if last.success and f < 0.3: return "resolved"
        if last.event_type == EventType.SESSION_END.value and not last.success: return "abandoned"
        if f > 0.7 and len(ep.events) > 5: return "abandoned"
        return "ongoing"

    def get_cross_platform(self, sid): return [ep for ep in self.episodes if ep.student_id == sid and ep.cross_platform]


class CrossPlatformProfiler:
    """Síntesis incremental O(1) por evento. Alimenta middleware y dashboard."""

    def __init__(self):
        self.profiles: Dict[str, CrossPlatformProfile] = {}
        self._bloom_win: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        self._topic_ok: Dict[str, Dict[str, List[bool]]] = defaultdict(lambda: defaultdict(list))
        self._hourly: Dict[str, List[int]] = defaultdict(lambda: [0]*24)
        # Nuevas señales
        self._copilot_decisions: Dict[str, List[bool]] = defaultdict(list)  # True=accept
        self._git_actions: Dict[str, Dict[str, int]] = defaultdict(lambda: {"commits": 0, "diffs": 0})
        self._focus_sessions: Dict[str, List[float]] = defaultdict(list)
        self._search_blooms: Dict[str, List[int]] = defaultdict(list)

    def update(self, event: ExternalLearningEvent) -> CrossPlatformProfile:
        sid = event.student_id
        if sid not in self.profiles: self.profiles[sid] = CrossPlatformProfile(student_id=sid)
        p = self.profiles[sid]
        p.last_updated = datetime.now().isoformat()
        p.total_events += 1
        p.events_by_source[event.source] = p.events_by_source.get(event.source, 0) + 1
        if event.source not in p.active_sources: p.active_sources.append(event.source)
        p.total_time_minutes += event.time_spent_seconds / 60.0

        # Bloom
        self._bloom_win[sid].append((event.source, event.cognitive_domain))
        if len(self._bloom_win[sid]) > 100: self._bloom_win[sid] = self._bloom_win[sid][-100:]
        bbs = defaultdict(list)
        for s, b in self._bloom_win[sid]: bbs[s].append(b)
        p.bloom_by_source = {s: round(sum(v)/len(v), 2) for s, v in bbs.items()}
        p.bloom_global = round(sum(p.bloom_by_source.values())/max(len(p.bloom_by_source),1), 2)
        blooms = [b for _, b in self._bloom_win[sid][-50:]]
        n = len(blooms)
        if n >= 5:
            xm = (n-1)/2; ym = sum(blooms)/n
            num = sum((i-xm)*(b-ym) for i, b in enumerate(blooms))
            den = sum((i-xm)**2 for i in range(n))
            p.bloom_trend = round(num/den, 4) if den else 0.0
        else: p.bloom_trend = 0.0

        # Frustración EMA
        p.frustration_by_source[event.source] = round(
            0.7*p.frustration_by_source.get(event.source, 0.0)+0.3*event.frustration_level, 3)
        p.frustration_global = round(sum(p.frustration_by_source.values())/max(len(p.frustration_by_source),1), 3)
        p.frustration_hotspots = [s for s, f in p.frustration_by_source.items() if f > 0.5]

        # Topics
        for tag in event.topic_tags: self._topic_ok[sid][tag].append(event.success)
        p.strength_topics = [t for t, r in self._topic_ok[sid].items() if len(r)>=3 and sum(r)/len(r)>0.75]
        p.gap_topics = [t for t, r in self._topic_ok[sid].items() if len(r)>=3 and sum(r)/len(r)<0.4]

        # Ritmos
        try:
            h = datetime.fromisoformat(event.timestamp).hour
            self._hourly[sid][h] += 1
            mx = max(self._hourly[sid])
            p.peak_hours = [i for i, c in enumerate(self._hourly[sid]) if mx > 0 and c >= mx*0.6]
        except: pass

        # === SEÑALES EXCLUSIVAS DE NUEVOS ADAPTADORES ===

        # Copilot: AI dependency ratio
        if event.source == "copilot":
            self._copilot_decisions[sid].append(
                event.event_type == EventType.AI_COMPLETION_ACCEPTED.value)
            if len(self._copilot_decisions[sid]) > 100:
                self._copilot_decisions[sid] = self._copilot_decisions[sid][-100:]
            accepts = sum(self._copilot_decisions[sid])
            total = len(self._copilot_decisions[sid])
            p.ai_dependency_ratio = round(accepts / max(total, 1), 3)

        # Git: reflection score
        if event.source == "git":
            if event.event_type == EventType.COMMIT_MADE.value:
                self._git_actions[sid]["commits"] += 1
            elif event.event_type == EventType.DIFF_VIEWED.value:
                self._git_actions[sid]["diffs"] += 1
            commits = self._git_actions[sid]["commits"]
            diffs = self._git_actions[sid]["diffs"]
            p.git_reflection_score = round(diffs / max(commits, 1), 2)

        # Pomodoro: focus endurance
        if event.source == "pomodoro" and event.success:
            self._focus_sessions[sid].append(event.time_spent_seconds / 60.0)
            if len(self._focus_sessions[sid]) > 50:
                self._focus_sessions[sid] = self._focus_sessions[sid][-50:]
            p.focus_endurance_minutes = round(
                sum(self._focus_sessions[sid]) / len(self._focus_sessions[sid]), 1)

        # Search: sophistication
        if event.source == "search":
            self._search_blooms[sid].append(event.cognitive_domain)
            if len(self._search_blooms[sid]) > 50:
                self._search_blooms[sid] = self._search_blooms[sid][-50:]
            p.search_sophistication = round(
                sum(self._search_blooms[sid]) / len(self._search_blooms[sid]), 2)

        # Recomendaciones
        p.recommended_actions = self._recs(p)
        return p

    def _recs(self, p):
        recs = []
        for g in p.gap_topics[:3]: recs.append(f"Dificultades en '{g}' (éxito < 40%). Reforzar.")
        for s, f in p.frustration_by_source.items():
            if f > 0.6: recs.append(f"Frustración alta en {s} ({f:.0%}). Dividir en pasos.")
        if p.bloom_trend < -0.1: recs.append("Tendencia cognitiva descendente. Posible fatiga.")
        elif p.bloom_trend > 0.15: recs.append("Progresión positiva. Listo para desafíos mayores.")
        if p.ai_dependency_ratio > 0.85:
            recs.append(f"Dependencia de IA alta ({p.ai_dependency_ratio:.0%} aceptaciones Copilot). "
                        f"Fomentar escritura independiente.")
        if p.git_reflection_score < 0.3 and p.events_by_source.get("git", 0) > 5:
            recs.append("Pocos diffs antes de commit. Fomentar revisión de código propio.")
        if p.focus_endurance_minutes > 0 and p.focus_endurance_minutes < 15:
            recs.append(f"Sesiones de foco cortas ({p.focus_endurance_minutes:.0f} min). "
                        f"Respuestas más breves y segmentadas.")
        if len(p.active_sources) > 3:
            recs.append(f"Activo en {len(p.active_sources)} plataformas. Referenciar conexiones.")
        return recs


# ═══════════════════════════════════════════════════════════════════════
# CLASE PRINCIPAL — ExternalLearningIngester
# ═══════════════════════════════════════════════════════════════════════

class ExternalLearningIngester:
    """
    Sinapsis entre GENIE Learn y el ecosistema de aprendizaje externo.
    17 adaptadores. Se instancia UNA VEZ en AppState.

    Integración:
      from external_learning_ingester import ExternalLearningIngester
      self.external = ExternalLearningIngester()

    Dos flujos:
    1. INGESTA: POST /external/ingest → self.external.ingest(source, data)
    2. CONSULTA: middleware → self.external.get_middleware_injection(student_id)
    """

    def __init__(self):
        self.adapters: Dict[str, Any] = {
            # Capa 1: IDE & Código
            "vscode": VSCodeAdapter(),
            "terminal": TerminalAdapter(),
            "git": GitLogAdapter(),
            "copilot": CopilotAdapter(),
            "replit": ReplitAdapter(),
            # Capa 2: Plataformas de aprendizaje
            "duolingo": DuolingoAdapter(),
            "mimo": MimoAdapter(),
            "leetcode": LeetCodeAdapter(),
            "freecodecamp": FreeCodeCampAdapter(),
            # Capa 3: Memoria & Notas
            "anki": AnkiAdapter(),
            "notes": NotesAdapter(),
            # Capa 4: Navegación & Búsqueda
            "browser": BrowserAdapter(),
            "search": SearchQueryAdapter(),
            "ai_chat": AIConversationAdapter(),
            # Capa 5: Señales conductuales
            "keystroke": KeystrokeAdapter(),
            "video": VideoAdapter(),
            "pomodoro": PomodoroAdapter(),
        }
        self.correlator = TemporalCorrelator(window_minutes=30)
        self.profiler = CrossPlatformProfiler()
        self.event_log: List[ExternalLearningEvent] = []

    # ─── INGESTA ───────────────────────────────────────────────

    def ingest(self, source: str, raw_data: dict) -> dict:
        adapter = self.adapters.get(source, self.adapters["browser"])
        event = adapter.normalize(raw_data)
        self.event_log.append(event)
        if len(self.event_log) > 1000: self.event_log = self.event_log[-1000:]
        episode = self.correlator.ingest(event)
        profile = self.profiler.update(event)
        return {
            "event": self._se(event), "episode": self._sep(episode) if episode else None,
            "profile_summary": self._sp(profile),
            "middleware_context": self._ctx(event, episode, profile),
            "alerts": self._alerts(event, profile),
        }

    def ingest_batch(self, events: List[dict]) -> List[str]:
        return [self.ingest(e.get("source","custom"), e.get("data",e))["event"]["event_id"] for e in events]

    # ─── MIDDLEWARE INJECTION ──────────────────────────────────

    def get_middleware_injection(self, student_id: str) -> str:
        """EL PUENTE CLAVE. Inyecta contexto externo al system_prompt."""
        p = self.profiler.profiles.get(student_id)
        if not p or p.total_events == 0: return ""
        lines = ["\n--- CONTEXTO EXTERNO (datos cross-plataforma) ---"]
        if p.gap_topics: lines.append(f"Dificultades en: {', '.join(p.gap_topics)}.")
        if p.strength_topics: lines.append(f"Fortalezas en: {', '.join(p.strength_topics)}.")
        if p.frustration_global > 0.5:
            lines.append(f"⚠ Frustración: {p.frustration_global:.0%}. Tono paciente.")
        if p.bloom_trend > 0.1: lines.append("Progresando. Puede manejar más desafío.")
        elif p.bloom_trend < -0.1: lines.append("Tendencia descendente. Reforzar bases.")
        if p.ai_dependency_ratio > 0.85:
            lines.append(f"Dependencia IA: {p.ai_dependency_ratio:.0%}. No dar soluciones directas.")
        if 0 < p.focus_endurance_minutes < 15:
            lines.append(f"Foco corto ({p.focus_endurance_minutes:.0f}min). Respuestas breves.")
        recent = self._recent(student_id, 3)
        if recent: lines.append(f"Actividad reciente: {recent}")
        for r in p.recommended_actions[:2]: lines.append(f"• {r}")
        lines.append("--- FIN CONTEXTO EXTERNO ---\n")
        return "\n".join(lines)

    def get_scaffolding_recommendation(self, student_id: str) -> str:
        p = self.profiler.profiles.get(student_id)
        if not p: return "socratic"
        if p.frustration_global > 0.6: return "direct"
        if p.ai_dependency_ratio > 0.85: return "socratic"  # Forzar reflexión
        if p.bloom_trend > 0.1: return "socratic"
        if p.bloom_global < 2.5: return "hints"
        return "socratic"

    # ─── DASHBOARD ─────────────────────────────────────────────

    def get_profile(self, sid): return self.profiler.profiles.get(sid)

    def get_dashboard_data(self, sid: str) -> dict:
        p = self.profiler.profiles.get(sid)
        if not p: return {"status": "no_data"}
        eps = [e for e in self.correlator.episodes if e.student_id == sid]
        return {
            "profile": self._sp(p),
            "episodes": {"total": len(eps),
                         "cross_platform": sum(1 for e in eps if e.cross_platform),
                         "resolved": sum(1 for e in eps if e.resolution == "resolved"),
                         "abandoned": sum(1 for e in eps if e.resolution == "abandoned")},
            "source_breakdown": p.events_by_source,
            "bloom_by_source": p.bloom_by_source,
            "frustration_by_source": p.frustration_by_source,
            "hourly_activity": self.profiler._hourly.get(sid, [0]*24),
            "ai_dependency_ratio": p.ai_dependency_ratio,
            "git_reflection_score": p.git_reflection_score,
            "focus_endurance_minutes": p.focus_endurance_minutes,
            "search_sophistication": p.search_sophistication,
            "recent_events": [self._se(e) for e in self.event_log[-20:] if e.student_id == sid],
        }

    def get_episodes(self, sid, cross_only=False):
        eps = (self.correlator.get_cross_platform(sid) if cross_only
               else [e for e in self.correlator.episodes if e.student_id == sid])
        return [self._sep(e) for e in eps]

    def get_available_sources(self) -> List[dict]:
        return [
            {"id": "vscode", "name": "VS Code / Cursor", "layer": "IDE", "difficulty": "easy",
             "description": "Errores, ejecuciones, patrones de edición, Ctrl+Z"},
            {"id": "terminal", "name": "Terminal / Shell", "layer": "IDE", "difficulty": "easy",
             "description": "Comandos, man/help, pipes, errores de shell"},
            {"id": "git", "name": "Git", "layer": "IDE", "difficulty": "easy",
             "description": "Commits, diffs, branches, calidad de mensajes"},
            {"id": "copilot", "name": "Copilot / AI completions", "layer": "IDE", "difficulty": "easy",
             "description": "Accept/reject ratio de sugerencias de IA"},
            {"id": "replit", "name": "Replit / CodeSandbox", "layer": "IDE", "difficulty": "medium",
             "description": "Código en entornos cloud (móvil/web)"},
            {"id": "duolingo", "name": "Duolingo", "layer": "Learning", "difficulty": "medium",
             "description": "Ejercicios, resultados, patrones de error"},
            {"id": "mimo", "name": "Mimo", "layer": "Learning", "difficulty": "medium",
             "description": "Micro-lecciones de código, repeticiones, streaks"},
            {"id": "leetcode", "name": "LeetCode / Codewars", "layer": "Learning", "difficulty": "medium",
             "description": "Challenges, categoría, percentil, editorial usage"},
            {"id": "freecodecamp", "name": "freeCodeCamp", "layer": "Learning", "difficulty": "easy",
             "description": "Progreso curricular, certificaciones"},
            {"id": "anki", "name": "Anki (SRS)", "layer": "Memory", "difficulty": "medium",
             "description": "Revisiones, intervalos, curva de olvido"},
            {"id": "notes", "name": "Obsidian / Notion", "layer": "Memory", "difficulty": "medium",
             "description": "Estructura de notas, enlaces, preguntas pendientes"},
            {"id": "browser", "name": "Navegación web", "layer": "Search", "difficulty": "easy",
             "description": "Lectura de documentación técnica"},
            {"id": "search", "name": "Búsquedas técnicas", "layer": "Search", "difficulty": "easy",
             "description": "Queries de error, sofisticación léxica"},
            {"id": "ai_chat", "name": "ChatGPT / Claude", "layer": "Search", "difficulty": "medium",
             "description": "Metadata de uso de IAs (mapa de ignorancias)"},
            {"id": "keystroke", "name": "Keystroke dynamics", "layer": "Behavioral", "difficulty": "hard",
             "description": "WPM, pausas, backspace rate, estados cognitivos"},
            {"id": "video", "name": "YouTube / Videos", "layer": "Behavioral", "difficulty": "medium",
             "description": "Pausas, replays, velocidad, patrón de consumo"},
            {"id": "pomodoro", "name": "Pomodoro / Focus", "layer": "Behavioral", "difficulty": "easy",
             "description": "Sesiones completadas/rotas, endurance atencional"},
        ]

    # ─── PRIVADO ───────────────────────────────────────────────

    def _ctx(self, event, episode, profile):
        parts = []
        if event.source == "duolingo" and not event.success:
            parts.append(f"Fallo en Duolingo ({event.content.get('target_language','')}):"
                         f" '{event.content.get('prompt_text','')}'.")
        if event.source == "mimo" and not event.success:
            parts.append(f"Fallo en Mimo ({event.content.get('language','')}):"
                         f" lección '{event.content.get('lesson_name','')}'.")
        if event.source == "vscode" and event.event_type == EventType.CODE_ERROR.value:
            parts.append(f"Error {event.content.get('error_type','')} en "
                         f"{event.content.get('language','')}: {event.content.get('error_message','')[:200]}")
        if event.source == "copilot" and event.event_type == EventType.AI_COMPLETION_ACCEPTED.value:
            if profile.ai_dependency_ratio > 0.85:
                parts.append("ALERTA: Alta dependencia de IA. Fomentar escritura independiente.")
        if event.source == "leetcode" and event.content.get("used_editorial"):
            parts.append("Usó editorial/solución en LeetCode. No aprendió a resolverlo solo.")
        if event.frustration_level > 0.7:
            parts.append(f"Frustración alta ({event.frustration_level:.0%}) en {event.source}.")
        if episode and episode.cross_platform:
            parts.append(f"Episodio cross-plataforma: {' → '.join(episode.sources_involved)}.")
        return " ".join(parts)

    def _alerts(self, event, profile):
        alerts = []
        if event.frustration_level > 0.8:
            alerts.append({"level": "critical", "type": "high_frustration", "source": event.source,
                           "student_id": event.student_id,
                           "message": f"Frustración extrema ({event.frustration_level:.0%}) en {event.source}"})
        if profile.bloom_trend < -0.2:
            alerts.append({"level": "warning", "type": "declining_cognitive",
                           "student_id": event.student_id,
                           "message": f"Bloom en descenso ({profile.bloom_trend:+.2f})"})
        if profile.ai_dependency_ratio > 0.90:
            alerts.append({"level": "warning", "type": "ai_dependency",
                           "student_id": event.student_id,
                           "message": f"Dependencia IA: {profile.ai_dependency_ratio:.0%}"})
        abandoned = sum(1 for ep in self.correlator.episodes
                        if ep.student_id == event.student_id and ep.resolution == "abandoned")
        if abandoned > 2:
            alerts.append({"level": "warning", "type": "frequent_abandonment",
                           "student_id": event.student_id,
                           "message": f"{abandoned} episodios abandonados"})
        return alerts

    def _recent(self, sid, n=3):
        evts = [e for e in reversed(self.event_log) if e.student_id == sid][:n]
        return ", ".join(f"{e.source}:{e.event_type}({'✓' if e.success else '✗'})" for e in evts) if evts else ""

    def _se(self, e):
        return {"event_id": e.event_id, "student_id": e.student_id, "timestamp": e.timestamp,
                "source": e.source, "event_type": e.event_type, "cognitive_domain": e.cognitive_domain,
                "success": e.success, "difficulty": e.difficulty, "frustration_level": e.frustration_level,
                "topic_tags": e.topic_tags, "content": e.content}

    def _sep(self, ep):
        return {"episode_id": ep.episode_id, "sources_involved": ep.sources_involved,
                "dominant_topic": ep.dominant_topic, "bloom_trajectory": ep.bloom_trajectory,
                "frustration_arc": ep.frustration_arc, "resolution": ep.resolution,
                "cross_platform": ep.cross_platform, "event_count": len(ep.events)}

    def _sp(self, p):
        return {"student_id": p.student_id, "total_events": p.total_events,
                "active_sources": p.active_sources, "bloom_global": p.bloom_global,
                "bloom_trend": p.bloom_trend, "bloom_by_source": p.bloom_by_source,
                "frustration_global": p.frustration_global,
                "frustration_by_source": p.frustration_by_source,
                "strength_topics": p.strength_topics, "gap_topics": p.gap_topics,
                "peak_hours": p.peak_hours,
                "total_time_minutes": round(p.total_time_minutes, 1),
                "ai_dependency_ratio": p.ai_dependency_ratio,
                "git_reflection_score": p.git_reflection_score,
                "focus_endurance_minutes": p.focus_endurance_minutes,
                "search_sophistication": p.search_sophistication,
                "recommended_actions": p.recommended_actions}
