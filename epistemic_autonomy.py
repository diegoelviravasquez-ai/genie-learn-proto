"""
EPISTEMIC AUTONOMY TRACKER
═══════════════════════════════════════════════════════════════════════
Módulo diferencial #2 — Ataca el gap EXPLÍCITO del SLR de Topali et al.
(BIT, 2024) y la línea de Teacher Agency de Alonso-Prieto:

    "There is a gap between the potential of LA/AI and its actual
     implementation in real cases [...] further research is necessary
     to better understand how and when stakeholders should be involved."
     — Topali et al. (2024, Sección 5.2)

    "Only 4 out of 47 studies integrated learning theories."
     — Topali et al. (2024, Figure 4-5)

DECISIONES DE DISEÑO QUE SOLO ESTE PERFIL TOMARÍA:
────────────────────────────────────────────────────
1. Operacionalización computacional de la Zona de Desarrollo Próximo
   (Vygotsky, 1978). El scaffolding del middleware actual es ESTÁTICO
   (niveles 0-3 que escalan por intentos). Aquí lo hacemos DINÁMICO:
   el nivel óptimo se calcula en función de la distancia entre lo que
   el estudiante demuestra saber y lo que el material del curso espera.

2. Trayectoria de autonomía epistémica como operacionalización del
   concepto de agencia (Bandura, 2018; Emirbayer & Mische, 1998). No
   medimos "cuántas preguntas hace" sino cómo EVOLUCIONA la relación
   entre lo que pregunta y lo que podría resolver solo. La agencia no
   es una propiedad estática: es una capacidad que se ejerce en
   contexto (Priestley, Biesta & Robinson, 2015) — exactamente la
   línea de Alonso-Prieto en el GSIC.

3. Modelo de 4E cognition aplicado (Newen, De Bruin & Gallagher, 2018):
   - Embodied: la fatiga/frustración como señal corporal mapeada
   - Embedded: el contexto del curso como restricción ecológica
   - Enacted: la autonomía se mide por ACCIÓN, no por declaración
   - Extended: el chatbot como extensión cognitiva cuyo uso óptimo
     es la desaparición gradual (scaffolding fading)

FUNDAMENTACIÓN:
  - Vygotsky, L. (1978). Mind in Society.
  - Bandura, A. (2018). "Toward a Psychology of Human Agency."
  - Priestley, Biesta & Robinson (2015). Teacher Agency.
  - Emirbayer & Mische (1998). "What is agency?"
  - Wood, Bruner & Ross (1976). "The role of tutoring in problem solving."
  - Newen, De Bruin & Gallagher (2018). The Oxford Handbook of 4E Cognition.
  - Bjork, R.A. (1994). "Memory and metamemory considerations in the
    training of human beings" (desirable difficulties).

Autor: Diego Elvira Vásquez · Prototipo CP25/152 · Feb 2026
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from datetime import datetime, timedelta
import math


@dataclass
class AutonomyState:
    """
    Estado de autonomía epistémica de un estudiante.

    Modelo: dependencia → scaffolded → emergente → autónomo
    Inspirado en el continuo de autonomía de Deci & Ryan (SDT, 1985)
    enriquecido con las fases de apropiación de Leontiev (1978).
    """
    student_id: str
    phase: str = "dependent"  # dependent | scaffolded | emergent | autonomous
    autonomy_score: float = 0.0  # 0.0 - 1.0
    scaffolding_need: int = 3  # nivel de scaffolding óptimo (0-3)
    zpd_estimate: float = 0.5  # distancia estimada a ZPD
    self_efficacy_proxy: float = 0.5  # proxy de autoeficacia
    dependency_ratio: float = 1.0  # ratio consultas / tiempo
    productive_struggle_count: int = 0  # interacciones con esfuerzo productivo
    total_interactions: int = 0
    last_updated: str = ""


@dataclass
class ScaffoldingRecommendation:
    """
    Recomendación de scaffolding adaptativo para una interacción.

    A diferencia del scaffolding actual (escala linealmente por intentos),
    este se adapta al ESTADO EPISTÉMICO del estudiante.
    """
    recommended_level: int  # 0-3 (socrático, pista, ejemplo, explicación)
    confidence: float  # 0.0 - 1.0
    rationale: str  # explicación legible para el docente
    should_fade: bool  # True si el estudiante está listo para menos ayuda
    zpd_alignment: str  # "below" | "within" | "above"


class EpistemicAutonomyTracker:
    """
    Motor de seguimiento de autonomía epistémica.

    Principio rector (Wood, Bruner & Ross, 1976):
    El scaffolding óptimo se RETIRA progresivamente a medida que el
    aprendiz demuestra competencia. El middleware actual escala el
    scaffolding HACIA ARRIBA (más ayuda). Este módulo calcula cuándo
    debe escalar HACIA ABAJO (menos ayuda = más autonomía = fading).

    Esto conecta directamente con la preocupación HCAI del proyecto:
    ¿cómo sabe el DOCENTE cuándo el chatbot debería dar más o menos
    ayuda? Este tracker le da esa información.
    """

    def __init__(self):
        self.states: dict[str, AutonomyState] = {}
        self.interaction_history: dict[str, list[dict]] = defaultdict(list)

    # ───────────────────────────────────────────
    # 1. REGISTRO DE INTERACCIONES
    # ───────────────────────────────────────────

    def record_interaction(
        self,
        student_id: str,
        prompt: str,
        bloom_level: str,
        bloom_weight: int,
        affective_state: str,
        scaffolding_level_used: int,
        response_was_helpful: bool = True,  # podría venir de feedback
        copy_paste_score: float = 0.0,
    ) -> AutonomyState:
        """
        Registra una interacción y actualiza el estado de autonomía.

        Cada interacción es evidencia Bayesiana que actualiza nuestra
        estimación de la posición del estudiante en el continuo
        dependencia → autonomía.
        """
        now = datetime.now().isoformat()

        # Almacenar interacción
        self.interaction_history[student_id].append({
            "timestamp": now,
            "bloom_weight": bloom_weight,
            "affective_state": affective_state,
            "scaffolding_used": scaffolding_level_used,
            "copy_paste": copy_paste_score,
            "has_own_attempt": self._has_own_attempt(prompt),
            "shows_reasoning": self._shows_reasoning(prompt),
            "asks_why": self._asks_why(prompt),
        })

        # Obtener o crear estado
        state = self.states.get(student_id, AutonomyState(student_id=student_id))
        state.total_interactions += 1
        state.last_updated = now

        # Actualizar indicadores
        state = self._update_autonomy_indicators(student_id, state)
        state = self._classify_phase(state)

        self.states[student_id] = state
        return state

    # ───────────────────────────────────────────
    # 2. INDICADORES DE AUTONOMÍA
    # ───────────────────────────────────────────

    def _update_autonomy_indicators(
        self, student_id: str, state: AutonomyState
    ) -> AutonomyState:
        """
        Actualiza los indicadores compuestos de autonomía.

        Inspirado en la operacionalización de agency de Emirbayer & Mische
        (1998): la agencia tiene tres dimensiones temporales:
        - Iteracional (habitus): patrones repetidos → dependency_ratio
        - Projective (imaginación): intentos propios → self_efficacy_proxy
        - Practical-evaluative (juicio): razonamiento → productive_struggle
        """
        history = self.interaction_history.get(student_id, [])
        recent = history[-15:]  # ventana de 15 interacciones

        if not recent:
            return state

        # ─── Proxy de autoeficacia (Bandura) ───
        # Se estima por la proporción de interacciones donde el estudiante
        # muestra un intento propio ANTES de pedir ayuda
        own_attempts = sum(1 for h in recent if h["has_own_attempt"])
        state.self_efficacy_proxy = own_attempts / len(recent)

        # ─── Ratio de dependencia ───
        # Inverso de la autoeficacia, ponderado por copy-paste
        cp_penalty = sum(h["copy_paste"] for h in recent) / len(recent)
        state.dependency_ratio = max(0, 1.0 - state.self_efficacy_proxy + cp_penalty * 0.5)
        state.dependency_ratio = min(state.dependency_ratio, 1.0)

        # ─── Productive struggle (Kapur, 2008) ───
        # Interacciones donde hay esfuerzo (intento propio + razonamiento)
        # pero sin frustración extrema
        productive = sum(
            1 for h in recent
            if h["has_own_attempt"]
            and h["shows_reasoning"]
            and h["affective_state"] != "frustration"
        )
        state.productive_struggle_count = productive

        # ─── Estimación de ZPD ───
        # La ZPD se aproxima por la distancia entre el Bloom mostrado
        # y el Bloom esperado para el material. Simplificación: usamos
        # la media del Bloom como proxy del nivel actual, y nivel 4
        # (analizar) como target para un curso de programación.
        bloom_weights = [h["bloom_weight"] for h in recent]
        current_bloom = sum(bloom_weights) / len(bloom_weights)
        target_bloom = 4.0  # parameterizable por el docente
        state.zpd_estimate = max(0, min(1, (target_bloom - current_bloom) / 5.0))

        # ─── Score compuesto de autonomía ───
        # Pesos diseñados para reflejar la estructura de la agency:
        # - 35% autoeficacia (dimensión projective)
        # - 25% productive struggle (dimensión practical-evaluative)
        # - 20% proporción de "asks_why" (profundidad epistémica)
        # - 20% inverso del ratio de dependencia (dimensión iteracional)
        asks_why_ratio = sum(1 for h in recent if h["asks_why"]) / len(recent)
        independence = 1.0 - state.dependency_ratio

        state.autonomy_score = (
            0.35 * state.self_efficacy_proxy
            + 0.25 * (productive / len(recent) if recent else 0)
            + 0.20 * asks_why_ratio
            + 0.20 * independence
        )
        state.autonomy_score = round(min(max(state.autonomy_score, 0), 1.0), 3)

        # ─── Nivel de scaffolding óptimo ───
        state.scaffolding_need = self._compute_optimal_scaffolding(state)

        return state

    def _compute_optimal_scaffolding(self, state: AutonomyState) -> int:
        """
        Calcula el nivel óptimo de scaffolding.

        Lógica de fading (Wood et al., 1976):
        - Autonomía alta → nivel 0 (socrático, poca ayuda)
        - Autonomía media → nivel 1-2 (pistas/ejemplos)
        - Autonomía baja → nivel 3 (explicación completa)

        Pero con una corrección crucial: si el estudiante está en
        "productive struggle" (Kapur, 2008), NO escalamos la ayuda
        aunque parezca que lo necesita. La dificultad deseable
        (Bjork, 1994) es pedagógicamente productiva.
        """
        score = state.autonomy_score
        in_productive_struggle = (
            state.productive_struggle_count > 0
            and state.self_efficacy_proxy > 0.3
        )

        if score >= 0.7:
            return 0  # Socrático: ya es bastante autónomo
        elif score >= 0.5:
            return 1 if not in_productive_struggle else 0
        elif score >= 0.3:
            return 2 if not in_productive_struggle else 1
        else:
            return 3  # Explicación directa: necesita ayuda real

    # ───────────────────────────────────────────
    # 3. CLASIFICACIÓN DE FASE
    # ───────────────────────────────────────────

    def _classify_phase(self, state: AutonomyState) -> AutonomyState:
        """
        Clasifica la fase de autonomía epistémica.

        Continuo inspirado en Dreyfus & Dreyfus (1986) —modelo de
        adquisición de habilidades— adaptado al contexto de interacción
        con IA educativa:

        dependent:   El estudiante depende del chatbot para todo.
                     Copy-paste de enunciados, sin intentos propios.
        scaffolded:  El estudiante usa el chatbot como apoyo pero
                     muestra intentos propios. ZPD activa.
        emergent:    El estudiante formula preguntas de nivel alto
                     (Bloom ≥ 4), tiene autoeficacia, lucha productiva.
        autonomous:  El estudiante usa el chatbot como extensión
                     cognitiva (4E: extended mind), no como muleta.
                     Preguntas de evaluación y creación.
        """
        s = state.autonomy_score
        se = state.self_efficacy_proxy
        dep = state.dependency_ratio

        if s >= 0.7 and se >= 0.5:
            state.phase = "autonomous"
        elif s >= 0.45 and se >= 0.3:
            state.phase = "emergent"
        elif se >= 0.15 or state.productive_struggle_count > 0:
            state.phase = "scaffolded"
        else:
            state.phase = "dependent"

        return state

    # ───────────────────────────────────────────
    # 4. RECOMENDACIONES ADAPTATIVAS
    # ───────────────────────────────────────────

    def get_scaffolding_recommendation(
        self, student_id: str, current_bloom_weight: int
    ) -> ScaffoldingRecommendation:
        """
        Genera una recomendación de scaffolding para la interacción actual.

        Esto es lo que conecta este módulo con el middleware existente:
        en lugar de escalar el scaffolding linealmente por intentos,
        lo adapta al estado epistémico del estudiante.
        """
        state = self.states.get(student_id)

        if state is None or state.total_interactions < 3:
            return ScaffoldingRecommendation(
                recommended_level=1,
                confidence=0.3,
                rationale="Datos insuficientes. Scaffolding moderado como default.",
                should_fade=False,
                zpd_alignment="unknown",
            )

        # ZPD alignment
        target = 4.0
        if current_bloom_weight >= target:
            zpd_align = "above"
        elif current_bloom_weight >= target - 1.5:
            zpd_align = "within"
        else:
            zpd_align = "below"

        # Fading decision
        should_fade = (
            state.phase in ("emergent", "autonomous")
            and state.autonomy_score > 0.5
            and state.self_efficacy_proxy > 0.3
        )

        # Rationale para el docente (transparencia HCAI)
        phase_labels = {
            "dependent": "fase dependiente",
            "scaffolded": "fase de scaffolding activo",
            "emergent": "fase emergente",
            "autonomous": "fase autónoma",
        }
        rationale_parts = [
            f"Estudiante en {phase_labels.get(state.phase, state.phase)}",
            f"(autonomía: {state.autonomy_score:.0%}",
            f"autoeficacia: {state.self_efficacy_proxy:.0%}).",
        ]
        if should_fade:
            rationale_parts.append(
                "Se recomienda REDUCIR scaffolding para fomentar autonomía."
            )
        if state.productive_struggle_count > 0:
            rationale_parts.append(
                "Detectada lucha productiva: no escalar ayuda prematuramente."
            )

        return ScaffoldingRecommendation(
            recommended_level=state.scaffolding_need,
            confidence=min(0.3 + state.total_interactions * 0.05, 0.95),
            rationale=" ".join(rationale_parts),
            should_fade=should_fade,
            zpd_alignment=zpd_align,
        )

    # ───────────────────────────────────────────
    # 5. DETECCIÓN DE SEÑALES
    # ───────────────────────────────────────────

    def _has_own_attempt(self, prompt: str) -> bool:
        """
        ¿El estudiante muestra un intento propio?
        Marcador de autoeficacia (Bandura) y dimensión enacted (4E).
        """
        markers = [
            "he intentado", "he probado", "mi código", "lo que tengo",
            "mi solución", "he hecho", "intenté", "probé", "mi idea",
            "lo que se me ocurre", "he escrito", "mi programa",
            "hice esto", "llevo esto", "tengo esto",
        ]
        prompt_lower = prompt.lower()
        return any(m in prompt_lower for m in markers)

    def _shows_reasoning(self, prompt: str) -> bool:
        """
        ¿El estudiante muestra razonamiento causal?
        Marcador de dimensión practical-evaluative (Emirbayer & Mische).
        """
        markers = [
            "porque", "ya que", "creo que", "mi razonamiento",
            "si hago esto entonces", "el problema es que",
            "lo que no entiendo es", "la lógica sería",
            "debería funcionar porque", "no funciona porque",
            "el error está en", "supongo que",
        ]
        prompt_lower = prompt.lower()
        return any(m in prompt_lower for m in markers)

    def _asks_why(self, prompt: str) -> bool:
        """
        ¿El estudiante pregunta por qué, no solo cómo?
        Marcador de profundidad epistémica (Bloom ≥ analyze).
        """
        markers = [
            "por qué", "para qué", "qué sentido tiene",
            "cuál es la razón", "cómo se relaciona",
            "qué ventaja tiene", "en qué se diferencia",
        ]
        prompt_lower = prompt.lower()
        return any(m in prompt_lower for m in markers)

    # ───────────────────────────────────────────
    # 6. DASHBOARD DATA
    # ───────────────────────────────────────────

    def get_autonomy_dashboard_data(self) -> dict:
        """
        Datos para el dashboard de autonomía del docente.

        Esto conecta con la preocupación central de Alonso-Prieto
        (LASI 2025): ¿cómo se ejerce la agencia docente cuando hay
        un agente IA en el aula?

        Respuesta: dándole al docente VISIBILIDAD sobre la trayectoria
        de autonomía de cada estudiante, para que pueda decidir
        (agencia) si interviene o no.
        """
        if not self.states:
            return {"total_tracked": 0}

        phases = defaultdict(int)
        autonomy_scores = []
        at_risk = []

        for sid, state in self.states.items():
            phases[state.phase] += 1
            autonomy_scores.append(state.autonomy_score)

            # Alertas
            if state.phase == "dependent" and state.total_interactions > 10:
                at_risk.append({
                    "student": sid,
                    "alert": "Dependencia prolongada tras 10+ interacciones",
                    "autonomy": state.autonomy_score,
                })
            if state.self_efficacy_proxy < 0.1 and state.total_interactions > 5:
                at_risk.append({
                    "student": sid,
                    "alert": "Autoeficacia muy baja: nunca muestra intentos propios",
                    "autonomy": state.autonomy_score,
                })

        return {
            "total_tracked": len(self.states),
            "phase_distribution": dict(phases),
            "autonomy_mean": round(
                sum(autonomy_scores) / len(autonomy_scores), 3
            ) if autonomy_scores else 0,
            "students_at_risk": at_risk,
            "scaffolding_fading_candidates": [
                sid for sid, s in self.states.items()
                if s.phase in ("emergent", "autonomous")
                and s.autonomy_score > 0.6
            ],
        }
