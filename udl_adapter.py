"""
ADAPTADOR UDL (UNIVERSAL DESIGN FOR LEARNING)
═══════════════════════════════════════════════════════════════════════
Módulo diferencial #9 — Cierra el bucle entre DETECCIÓN y ACCIÓN.

EL PROBLEMA — DETECTAR SIN ADAPTAR ES VOYEURISMO ANALÍTICO:
═══════════════════════════════════════════════════════════════
nd_patterns.py detecta patrones neurodivergentes.
cognitive_profiler.py identifica perfiles de engagement.
epistemic_autonomy.py clasifica fases de autonomía.
temporal_dynamics.py mide ritmos de aprendizaje.

Pero NINGÚN módulo usa esa información para MODIFICAR la respuesta.

El resultado: un sistema que sabe que el estudiante tiene patrones
de hiperfoco episódico (TDAH) pero le entrega la misma respuesta
que a uno con perfil lineal. Un sistema que detecta saltos cognitivos
(AACC) pero no ofrece enriquecimiento. Un sistema que identifica
rendimiento asimétrico temático (2e) pero no adapta la profundidad
al tema.

Este módulo completa el ciclo Sense → Analyze → REACT del SLE.

FUNDAMENTACIÓN TEÓRICA:
────────────────────────
1. Universal Design for Learning (CAST, 2018; Rose & Meyer, 2002)
   UDL establece tres principios:
   - Múltiples medios de REPRESENTACIÓN (cómo se presenta la información)
   - Múltiples medios de ACCIÓN Y EXPRESIÓN (cómo el estudiante demuestra)
   - Múltiples medios de ENGAGEMENT (cómo se motiva y mantiene atención)

   Este módulo actúa sobre el PRIMER principio: modifica CÓMO el chatbot
   presenta la información según el perfil del estudiante.

2. Objetivo O2 del CSEDU 2025 — explícitamente:
   "supporting a wider diversity of learners (UDL) and ethical design
    aspirations"
   — Delgado-Kloos et al. (CSEDU 2025, O2.2)

3. Cognitive Load Theory (Sweller, 1988, 2011)
   La carga cognitiva tiene tres componentes:
   - Intrínseca: complejidad del material (no modificable)
   - Extrínseca: diseño de la presentación (MODIFICABLE — aquí actuamos)
   - Germane: esfuerzo de aprendizaje (deseable)

   Reducir carga extrínseca sin reducir germane: ese es el arte.
   Para TDAH: segmentar, estructurar, reducir longitud.
   Para AACC: compactar, eliminar redundancia, añadir profundidad.
   Para 2e: adaptar por tema, no globalmente.

4. Dual Coding Theory (Paivio, 1986)
   La información se procesa en dos canales: verbal y visual.
   Algunos perfiles se benefician de representación dual (diagrama +
   explicación textual). Este módulo inyecta MARCADORES de formato
   que sugieren representación visual cuando el perfil lo indica.

5. Scaffolding diferenciado (Vygotsky, 1978; Tomlinson, 2001)
   La diferenciación no es dar menos a unos y más a otros. Es dar
   DIFERENTE según la zona de desarrollo próximo de cada uno.
   Un estudiante con hiperfoco no necesita menos contenido — necesita
   contenido entregado en ráfagas más cortas con hooks de enganche.

6. Neurodiversidad como variación, no como déficit (Singer, 1998;
   Armstrong, 2010)
   El diseño UDL no "compensa" una carencia. Diseña para la variación
   humana como norma. Las adaptaciones benefician a todos los
   estudiantes (principio del bordillo rebajado: diseñado para sillas
   de ruedas, usado por todos).

INTEGRACIÓN:
   - Consume: nd_patterns.NeurodivergentPattern, cognitive_profiler output,
     epistemic_autonomy phase
   - Modifica: el system_prompt y/o la respuesta post-LLM
   - Se inserta entre middleware.post_process() y la entrega

Autor: Diego Elvira Vásquez · Prototipo CP25/152 · Feb 2026
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
from collections import defaultdict, Counter


# ═══════════════════════════════════════════════════════════════
# PERFILES ADAPTATIVOS
# ═══════════════════════════════════════════════════════════════
# Cada perfil define CÓMO modificar la presentación de la respuesta.
# Estos NO son diagnósticos — son patrones funcionales de interacción.

@dataclass
class AdaptiveProfile:
    """Perfil adaptativo de un estudiante — acumulativo y dinámico."""
    student_id: str
    # Patrones funcionales detectados (de nd_patterns.py)
    functional_patterns: list = field(default_factory=list)
    # Preferencias observadas (no declaradas — inferidas del comportamiento)
    prefers_short_responses: bool = False          # prompts cortos + re-preguntas → prefiere brevedad
    prefers_examples_first: bool = False           # busca ejemplos antes que explicación
    prefers_step_by_step: bool = False             # pide "paso a paso" frecuentemente
    prefers_visual_markers: bool = False           # mejor rendimiento con código + esquemas
    prefers_minimal_scaffolding: bool = False      # frustración detectada con scaffolding
    prefers_deep_exploration: bool = False          # preguntas que exceden el currículo
    topic_strengths: list = field(default_factory=list)   # topics con Bloom alto
    topic_weaknesses: list = field(default_factory=list)  # topics con Bloom bajo
    # Autonomía y Bloom (de otros módulos)
    autonomy_phase: str = "unknown"
    avg_bloom: float = 2.0
    # Historial de adaptaciones aplicadas
    adaptations_applied: list = field(default_factory=list)
    last_updated: str = ""


@dataclass
class ResponseAdaptation:
    """Adaptación aplicada a una respuesta."""
    adaptation_id: str
    adaptation_type: str        # "format" | "length" | "depth" | "structure" | "engagement"
    description: str            # qué se modificó
    rationale: str              # por qué (patrón funcional + marco teórico)
    udl_principle: str          # "representation" | "action_expression" | "engagement"
    original_length: int        # chars de la respuesta original
    adapted_length: int         # chars de la respuesta adaptada


# ═══════════════════════════════════════════════════════════════
# ESTRATEGIAS DE ADAPTACIÓN
# ═══════════════════════════════════════════════════════════════

class AdaptationStrategy:
    """Estrategia base de adaptación."""
    name: str = "base"
    udl_principle: str = "representation"

    def should_apply(self, profile: AdaptiveProfile) -> bool:
        return False

    def adapt_system_prompt(self, base_prompt: str, profile: AdaptiveProfile) -> str:
        return base_prompt

    def adapt_response(self, response: str, profile: AdaptiveProfile) -> str:
        return response


class SegmentedDeliveryStrategy(AdaptationStrategy):
    """
    Para patrones episódicos (asociados a TDAH):
    Segmenta la respuesta en bloques cortos con encabezados claros.

    Fundamento: la memoria de trabajo limitada procesa mejor chunks
    discretos que flujo continuo (Baddeley, 2000). No reducimos contenido
    — reducimos la carga extrínseca de procesamiento.
    """
    name = "segmented_delivery"
    udl_principle = "representation"

    def should_apply(self, profile: AdaptiveProfile) -> bool:
        has_episodic = any("episod" in p.lower() or "hiperfoco" in p.lower()
                          for p in profile.functional_patterns)
        return has_episodic or profile.prefers_short_responses

    def adapt_system_prompt(self, base_prompt: str, profile: AdaptiveProfile) -> str:
        injection = (
            "\n\n[ADAPTACIÓN UDL — Entrega segmentada]\n"
            "El estudiante se beneficia de respuestas estructuradas en bloques cortos.\n"
            "Reglas de formato:\n"
            "- Máximo 3-4 oraciones por bloque\n"
            "- Cada bloque con un mini-encabezado descriptivo (ej: '▸ Concepto clave:', '▸ Ejemplo:')\n"
            "- Si la respuesta requiere más de 3 bloques, prioriza lo esencial\n"
            "- Usa un hook de enganche al inicio (pregunta retórica, dato sorprendente)\n"
            "- Cierra con UN paso de acción concreto\n"
        )
        return base_prompt + injection

    def adapt_response(self, response: str, profile: AdaptiveProfile) -> str:
        """Post-procesa: si la respuesta vino sin segmentar, la segmenta."""
        if "▸" in response or "###" in response:
            return response  # ya viene segmentada

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response) if s.strip()]
        if len(sentences) <= 4:
            return response  # ya es corta

        # Segmentar en bloques de 3 oraciones
        blocks = []
        labels = ["▸ Lo esencial:", "▸ Detalle:", "▸ Ejemplo:", "▸ Para recordar:"]
        for i in range(0, len(sentences), 3):
            chunk = " ".join(sentences[i:i+3])
            label_idx = min(i // 3, len(labels) - 1)
            blocks.append(f"{labels[label_idx]}\n{chunk}")

        return "\n\n".join(blocks)


class CompactEnrichmentStrategy(AdaptationStrategy):
    """
    Para patrones de salto cognitivo (asociados a AACC):
    Compacta la explicación básica y añade profundidad.

    Fundamento: estudiantes con alto nivel cognitivo se frustran con
    explicaciones que perciben como redundantes (Renzulli, 2005).
    El exceso de scaffolding produce expertise reversal (Kalyuga, 2003).
    """
    name = "compact_enrichment"
    udl_principle = "representation"

    def should_apply(self, profile: AdaptiveProfile) -> bool:
        has_jumps = any("salto" in p.lower() or "cognitivo" in p.lower()
                        for p in profile.functional_patterns)
        return (has_jumps or profile.prefers_minimal_scaffolding or
                profile.prefers_deep_exploration or profile.avg_bloom >= 4.5)

    def adapt_system_prompt(self, base_prompt: str, profile: AdaptiveProfile) -> str:
        injection = (
            "\n\n[ADAPTACIÓN UDL — Enriquecimiento compacto]\n"
            "El estudiante tiene alto nivel cognitivo y se frustra con redundancia.\n"
            "Reglas:\n"
            "- Respuesta directa y densa, sin repeticiones\n"
            "- Añadir una sección '🔬 Para profundizar:' con contenido avanzado\n"
            "- Incluir conexiones con otros temas (no limitar al currículo)\n"
            "- Si el tema es 'trivial' para el nivel del estudiante, ofrecer "
            "la variante avanzada del problema en lugar de la básica\n"
            "- Usar terminología técnica precisa sin simplificación innecesaria\n"
        )
        return base_prompt + injection

    def adapt_response(self, response: str, profile: AdaptiveProfile) -> str:
        """Post-procesa: elimina redundancia si la detecta."""
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', response) if s.strip()]
        if len(sentences) <= 3:
            return response

        # Detectar y eliminar oraciones muy similares (redundancia)
        unique = [sentences[0]]
        for s in sentences[1:]:
            s_words = set(s.lower().split())
            is_redundant = False
            for u in unique:
                u_words = set(u.lower().split())
                if s_words and u_words:
                    overlap = len(s_words & u_words) / max(len(s_words), 1)
                    if overlap > 0.7:
                        is_redundant = True
                        break
            if not is_redundant:
                unique.append(s)

        result = " ".join(unique)

        # Si se eliminó contenido, añadir nota de enriquecimiento
        if len(unique) < len(sentences) - 1:
            result += "\n\n🔬 *[Contenido compactado — sin redundancias]*"

        return result


class ThematicAdaptationStrategy(AdaptationStrategy):
    """
    Para patrones de rendimiento asimétrico temático (asociados a 2e):
    Adapta la profundidad y el scaffolding SEGÚN EL TEMA, no globalmente.

    Fundamento: un estudiante 2e puede estar en Bloom 5 en recursión y
    Bloom 1 en arrays — la misma configuración global no sirve para ambos.
    Tomlinson (2001): la diferenciación es por contenido, no por persona.
    """
    name = "thematic_adaptation"
    udl_principle = "representation"

    def should_apply(self, profile: AdaptiveProfile) -> bool:
        has_asymmetry = any("asim" in p.lower() or "temátic" in p.lower()
                           for p in profile.functional_patterns)
        return (has_asymmetry or
                (len(profile.topic_strengths) > 0 and len(profile.topic_weaknesses) > 0))

    def adapt_system_prompt(self, base_prompt: str, profile: AdaptiveProfile) -> str:
        strengths = ", ".join(profile.topic_strengths[:3]) if profile.topic_strengths else "ninguno detectado"
        weaknesses = ", ".join(profile.topic_weaknesses[:3]) if profile.topic_weaknesses else "ninguno detectado"

        injection = (
            f"\n\n[ADAPTACIÓN UDL — Diferenciación temática]\n"
            f"Este estudiante tiene rendimiento asimétrico por tema.\n"
            f"Temas fuertes: {strengths}\n"
            f"Temas débiles: {weaknesses}\n"
            f"Reglas:\n"
            f"- En temas fuertes: modo compacto, sin scaffolding básico, ofrecer profundidad\n"
            f"- En temas débiles: más ejemplos, paso a paso, scaffolding socrático suave\n"
            f"- NO asumir que el nivel en un tema predice el nivel en otro\n"
            f"- La variación es NORMAL, no un error del estudiante\n"
        )
        return base_prompt + injection

    def adapt_response(self, response: str, profile: AdaptiveProfile) -> str:
        return response  # la adaptación principal es via system prompt


class ExamplesFirstStrategy(AdaptationStrategy):
    """
    Para estudiantes que aprenden mejor desde lo concreto:
    Reorganiza la respuesta para poner ejemplos antes de la explicación.

    Fundamento: Dual Coding (Paivio, 1986) + Concrete-Representational-
    Abstract sequence (Witzel, 2005). Algunos estudiantes necesitan
    anclar el concepto abstracto en un ejemplo concreto primero.
    """
    name = "examples_first"
    udl_principle = "representation"

    def should_apply(self, profile: AdaptiveProfile) -> bool:
        return profile.prefers_examples_first

    def adapt_system_prompt(self, base_prompt: str, profile: AdaptiveProfile) -> str:
        injection = (
            "\n\n[ADAPTACIÓN UDL — Ejemplos primero]\n"
            "El estudiante aprende mejor de lo concreto a lo abstracto.\n"
            "Reglas:\n"
            "- Empezar SIEMPRE con un ejemplo concreto antes de la explicación\n"
            "- El ejemplo debe ser completo y funcional (no fragmento)\n"
            "- DESPUÉS del ejemplo, explicar el concepto subyacente\n"
            "- Usar analogías del mundo real cuando sea posible\n"
        )
        return base_prompt + injection

    def adapt_response(self, response: str, profile: AdaptiveProfile) -> str:
        """Reorganiza: si hay código/ejemplo después de la explicación, moverlo arriba."""
        code_match = re.search(r'(```[\s\S]*?```)', response)
        if not code_match:
            return response

        code_block = code_match.group(1)
        code_start = code_match.start()

        # Si el código está en la primera mitad, ya está bien
        if code_start < len(response) * 0.4:
            return response

        # Mover código al inicio
        text_before = response[:code_start].strip()
        text_after = response[code_match.end():].strip()

        return f"📝 **Ejemplo primero:**\n\n{code_block}\n\n**Explicación:**\n{text_before}\n{text_after}"


class StepByStepStrategy(AdaptationStrategy):
    """
    Para estudiantes que piden estructura secuencial explícita.

    Fundamento: Cognitive Load Theory (Sweller, 2011). La
    estructuración explícita reduce carga extrínseca. La numeración
    actúa como señalización (signaling principle, Mayer 2009).
    """
    name = "step_by_step"
    udl_principle = "representation"

    def should_apply(self, profile: AdaptiveProfile) -> bool:
        return profile.prefers_step_by_step

    def adapt_system_prompt(self, base_prompt: str, profile: AdaptiveProfile) -> str:
        injection = (
            "\n\n[ADAPTACIÓN UDL — Estructura secuencial]\n"
            "El estudiante se beneficia de respuestas paso a paso.\n"
            "Reglas:\n"
            "- Numerar los pasos explícitamente (Paso 1, Paso 2...)\n"
            "- Cada paso debe ser una acción atómica verificable\n"
            "- Incluir qué debería ver/obtener después de cada paso\n"
            "- Máximo 5-7 pasos (si hay más, agrupar en fases)\n"
        )
        return base_prompt + injection

    def adapt_response(self, response: str, profile: AdaptiveProfile) -> str:
        return response  # la adaptación es via system prompt


class EngagementHookStrategy(AdaptationStrategy):
    """
    Para estudiantes con patrones de desenganche rápido.
    Inyecta hooks motivacionales al inicio de la respuesta.

    Fundamento: UDL Principle III (CAST, 2018) — múltiples medios
    de engagement. La curiosidad epistémica (Berlyne, 1960) se activa
    con gaps de información (Loewenstein, 1994).
    """
    name = "engagement_hook"
    udl_principle = "engagement"

    def should_apply(self, profile: AdaptiveProfile) -> bool:
        has_disengagement = any("desenganche" in p.lower() or "abandon" in p.lower()
                                for p in profile.functional_patterns)
        return has_disengagement or profile.autonomy_phase == "dependent"

    def adapt_system_prompt(self, base_prompt: str, profile: AdaptiveProfile) -> str:
        injection = (
            "\n\n[ADAPTACIÓN UDL — Hook de engagement]\n"
            "El estudiante necesita activación motivacional.\n"
            "Reglas:\n"
            "- Empezar con algo que genere curiosidad (pregunta, dato sorprendente, "
            "conexión con la vida real)\n"
            "- Evitar: 'esto es importante porque...' → en su lugar: '¿sabías que...?'\n"
            "- Incluir una mini-meta alcanzable: 'al final de esto podrás...'\n"
            "- Tono: energético sin ser condescendiente\n"
        )
        return base_prompt + injection

    def adapt_response(self, response: str, profile: AdaptiveProfile) -> str:
        return response


# ═══════════════════════════════════════════════════════════════
# MOTOR ADAPTATIVO UDL
# ═══════════════════════════════════════════════════════════════

class UDLAdapter:
    """
    Motor principal de adaptación UDL.

    Flujo:
        1. Recibe perfil del estudiante (de nd_patterns + otros módulos)
        2. Determina qué estrategias aplican
        3. Modifica el system_prompt ANTES del LLM (pre-adaptación)
        4. Modifica la respuesta DESPUÉS del LLM (post-adaptación)
        5. Registra las adaptaciones para analytics

    Las estrategias se aplican en CASCADA: múltiples estrategias
    pueden actuar simultáneamente (ej: segmentada + hook de engagement
    para un estudiante con TDAH y desenganche).
    """

    def __init__(self):
        self.strategies: list[AdaptationStrategy] = [
            SegmentedDeliveryStrategy(),
            CompactEnrichmentStrategy(),
            ThematicAdaptationStrategy(),
            ExamplesFirstStrategy(),
            StepByStepStrategy(),
            EngagementHookStrategy(),
        ]
        self.profiles: dict[str, AdaptiveProfile] = {}
        self.adaptation_log: list[ResponseAdaptation] = []
        self._adaptation_counter = 0

    # ──────────────────────────────────────────────
    # GESTIÓN DE PERFILES
    # ──────────────────────────────────────────────

    def update_profile(self, student_id: str,
                       functional_patterns: list = None,
                       autonomy_phase: str = None,
                       avg_bloom: float = None,
                       topic_strengths: list = None,
                       topic_weaknesses: list = None,
                       interaction_hints: dict = None) -> AdaptiveProfile:
        """
        Actualiza el perfil adaptativo de un estudiante.

        interaction_hints: señales inferidas del comportamiento
            - "short_prompts": prompts <15 palabras consistentemente
            - "asks_examples": pide ejemplos frecuentemente
            - "asks_steps": pide "paso a paso"
            - "code_preference": pregunta más por código que por teoría
            - "quick_abandon": abandona temas rápidamente
        """
        if student_id not in self.profiles:
            self.profiles[student_id] = AdaptiveProfile(student_id=student_id)

        profile = self.profiles[student_id]

        if functional_patterns is not None:
            profile.functional_patterns = functional_patterns
        if autonomy_phase is not None:
            profile.autonomy_phase = autonomy_phase
        if avg_bloom is not None:
            profile.avg_bloom = avg_bloom
        if topic_strengths is not None:
            profile.topic_strengths = topic_strengths
        if topic_weaknesses is not None:
            profile.topic_weaknesses = topic_weaknesses

        # Inferir preferencias de los hints
        if interaction_hints:
            if interaction_hints.get("short_prompts", False):
                profile.prefers_short_responses = True
            if interaction_hints.get("asks_examples", False):
                profile.prefers_examples_first = True
            if interaction_hints.get("asks_steps", False):
                profile.prefers_step_by_step = True
            if interaction_hints.get("code_preference", False):
                profile.prefers_visual_markers = True
            if interaction_hints.get("frustration_with_scaffolding", False):
                profile.prefers_minimal_scaffolding = True
            if interaction_hints.get("exceeds_curriculum", False):
                profile.prefers_deep_exploration = True

        profile.last_updated = datetime.now().isoformat()
        return profile

    def get_profile(self, student_id: str) -> Optional[AdaptiveProfile]:
        """Obtiene el perfil adaptativo de un estudiante."""
        return self.profiles.get(student_id)

    # ──────────────────────────────────────────────
    # ADAPTACIÓN PRE-LLM (system prompt)
    # ──────────────────────────────────────────────

    def adapt_system_prompt(self, student_id: str, base_system_prompt: str) -> tuple[str, list[str]]:
        """
        Modifica el system prompt antes de llamar al LLM.

        Returns:
            (system_prompt_adaptado, lista_de_estrategias_aplicadas)
        """
        profile = self.profiles.get(student_id)
        if not profile:
            return base_system_prompt, []

        applied = []
        prompt = base_system_prompt

        for strategy in self.strategies:
            if strategy.should_apply(profile):
                prompt = strategy.adapt_system_prompt(prompt, profile)
                applied.append(strategy.name)

        return prompt, applied

    # ──────────────────────────────────────────────
    # ADAPTACIÓN POST-LLM (respuesta)
    # ──────────────────────────────────────────────

    def adapt_response(self, student_id: str, response: str) -> tuple[str, list[ResponseAdaptation]]:
        """
        Modifica la respuesta del LLM después de recibirla.

        Returns:
            (respuesta_adaptada, lista_de_adaptaciones_aplicadas)
        """
        profile = self.profiles.get(student_id)
        if not profile:
            return response, []

        adaptations = []
        current = response

        for strategy in self.strategies:
            if strategy.should_apply(profile):
                before_len = len(current)
                current = strategy.adapt_response(current, profile)
                after_len = len(current)

                if before_len != after_len:
                    self._adaptation_counter += 1
                    adaptation = ResponseAdaptation(
                        adaptation_id=f"UDL-{self._adaptation_counter:04d}",
                        adaptation_type=strategy.name,
                        description=f"Estrategia '{strategy.name}' aplicada",
                        rationale=self._get_strategy_rationale(strategy, profile),
                        udl_principle=strategy.udl_principle,
                        original_length=before_len,
                        adapted_length=after_len,
                    )
                    adaptations.append(adaptation)
                    self.adaptation_log.append(adaptation)
                    profile.adaptations_applied.append(strategy.name)

        return current, adaptations

    # ──────────────────────────────────────────────
    # ADAPTACIÓN COMPLETA (pre + post)
    # ──────────────────────────────────────────────

    def full_adapt(self, student_id: str, base_system_prompt: str,
                   llm_response: str) -> dict:
        """
        Adaptación completa: modifica system prompt Y respuesta.

        Returns:
            {
                "adapted_system_prompt": str,
                "adapted_response": str,
                "strategies_applied": list[str],
                "adaptations": list[ResponseAdaptation],
                "profile_summary": dict,
            }
        """
        adapted_prompt, pre_strategies = self.adapt_system_prompt(student_id, base_system_prompt)
        adapted_response, post_adaptations = self.adapt_response(student_id, llm_response)

        profile = self.profiles.get(student_id)
        profile_summary = {}
        if profile:
            profile_summary = {
                "functional_patterns": profile.functional_patterns,
                "autonomy_phase": profile.autonomy_phase,
                "avg_bloom": profile.avg_bloom,
                "preferences": {
                    "short_responses": profile.prefers_short_responses,
                    "examples_first": profile.prefers_examples_first,
                    "step_by_step": profile.prefers_step_by_step,
                    "minimal_scaffolding": profile.prefers_minimal_scaffolding,
                    "deep_exploration": profile.prefers_deep_exploration,
                },
            }

        return {
            "adapted_system_prompt": adapted_prompt,
            "adapted_response": adapted_response,
            "strategies_applied": pre_strategies,
            "adaptations": post_adaptations,
            "profile_summary": profile_summary,
        }

    # ──────────────────────────────────────────────
    # ANALYTICS
    # ──────────────────────────────────────────────

    def get_adaptation_report(self) -> dict:
        """Informe de adaptaciones aplicadas en la sesión."""
        if not self.adaptation_log:
            return {"total_adaptations": 0}

        strategy_counts = defaultdict(int)
        for a in self.adaptation_log:
            strategy_counts[a.adaptation_type] += 1

        return {
            "total_adaptations": len(self.adaptation_log),
            "strategies_used": dict(strategy_counts),
            "students_with_profiles": len(self.profiles),
            "students_adapted": sum(1 for p in self.profiles.values()
                                    if p.adaptations_applied),
            "avg_length_change": round(
                sum(a.adapted_length - a.original_length for a in self.adaptation_log)
                / max(len(self.adaptation_log), 1), 1
            ),
        }

    def get_student_adaptation_history(self, student_id: str) -> dict:
        """Historial de adaptaciones para un estudiante específico."""
        profile = self.profiles.get(student_id)
        if not profile:
            return {"student_id": student_id, "has_profile": False}

        return {
            "student_id": student_id,
            "has_profile": True,
            "functional_patterns": profile.functional_patterns,
            "active_strategies": [s.name for s in self.strategies if s.should_apply(profile)],
            "total_adaptations": len(profile.adaptations_applied),
            "adaptation_types": dict(Counter(profile.adaptations_applied)),
        }

    # ──────────────────────────────────────────────
    # AUXILIARES
    # ──────────────────────────────────────────────

    def _get_strategy_rationale(self, strategy: AdaptationStrategy,
                                profile: AdaptiveProfile) -> str:
        """Genera justificación pedagógica de la adaptación."""
        rationales = {
            "segmented_delivery": (
                "Patrón episódico detectado → respuesta segmentada en bloques cortos. "
                "Fundamento: Cognitive Load Theory (Sweller, 2011) + memoria de trabajo "
                "limitada (Baddeley, 2000). La segmentación reduce carga extrínseca."
            ),
            "compact_enrichment": (
                "Saltos cognitivos / alto Bloom detectado → respuesta compacta sin redundancia "
                "+ sección de profundización. Fundamento: Expertise Reversal Effect (Kalyuga, 2003). "
                "El exceso de scaffolding para un experto REDUCE el aprendizaje."
            ),
            "thematic_adaptation": (
                "Rendimiento asimétrico por tema → scaffolding diferenciado por topic. "
                f"Temas fuertes: {', '.join(profile.topic_strengths[:2])}. "
                f"Temas débiles: {', '.join(profile.topic_weaknesses[:2])}. "
                "Fundamento: Tomlinson (2001) — diferenciación por contenido, no por persona."
            ),
            "examples_first": (
                "Preferencia por aprendizaje concreto→abstracto detectada → ejemplos antes que "
                "explicación. Fundamento: Dual Coding (Paivio, 1986) + secuencia CRA (Witzel, 2005)."
            ),
            "step_by_step": (
                "Preferencia por estructura secuencial → respuesta numerada paso a paso. "
                "Fundamento: Signaling Principle (Mayer, 2009) — la señalización explícita "
                "reduce carga extrínseca y mejora retención."
            ),
            "engagement_hook": (
                "Patrón de desenganche rápido → hook motivacional al inicio. "
                "Fundamento: Information Gap Theory (Loewenstein, 1994) — la curiosidad "
                "epistémica se activa con gaps de información calibrados."
            ),
        }
        return rationales.get(strategy.name, f"Estrategia {strategy.name} aplicada por perfil funcional.")


# ═══════════════════════════════════════════════════════════════
# DEMO EJECUTABLE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("UDL ADAPTER — Demo")
    print("=" * 70)

    adapter = UDLAdapter()

    # --- Estudiante 1: Perfil TDAH (hiperfoco episódico + desenganche) ---
    adapter.update_profile(
        "est_01",
        functional_patterns=["interacción episódica con hiperfoco", "desenganche rápido entre temas"],
        autonomy_phase="scaffolded",
        avg_bloom=2.5,
        interaction_hints={"short_prompts": True, "quick_abandon": True},
    )

    # --- Estudiante 2: Perfil AACC (saltos cognitivos) ---
    adapter.update_profile(
        "est_02",
        functional_patterns=["saltos cognitivos asimétricos", "preguntas fuera de currículo"],
        autonomy_phase="transitional",
        avg_bloom=4.8,
        interaction_hints={"frustration_with_scaffolding": True, "exceeds_curriculum": True},
    )

    # --- Estudiante 3: Perfil 2e (asimetría temática) ---
    adapter.update_profile(
        "est_03",
        functional_patterns=["rendimiento asimétrico temático"],
        autonomy_phase="scaffolded",
        avg_bloom=3.2,
        topic_strengths=["recursión", "funciones"],
        topic_weaknesses=["arrays", "entrada/salida"],
        interaction_hints={"asks_examples": True},
    )

    # Respuesta de ejemplo del LLM
    sample_response = (
        "Un bucle for en Python se utiliza para iterar sobre una secuencia de elementos. "
        "La sintaxis básica es: for variable in secuencia, seguido de dos puntos y el "
        "bloque de código indentado. Por ejemplo, si quieres imprimir los números del "
        "1 al 5, puedes escribir: for i in range(1, 6): print(i). "
        "Es importante recordar que range no incluye el último número. "
        "Los bucles for son muy versátiles y se pueden usar con listas, tuplas, "
        "diccionarios y cualquier objeto iterable. También puedes usar enumerate "
        "si necesitas el índice además del valor. Los bucles anidados son posibles "
        "pero aumentan la complejidad. Recuerda que la indentación es crucial en Python."
    )

    base_prompt = "Eres un tutor de programación para estudiantes universitarios."

    print("\n" + "─" * 50)
    print("ESTUDIANTE 1 — Perfil episódico (TDAH funcional)")
    result1 = adapter.full_adapt("est_01", base_prompt, sample_response)
    print(f"  Estrategias: {result1['strategies_applied']}")
    print(f"  Respuesta adaptada:\n  {result1['adapted_response'][:300]}...")

    print("\n" + "─" * 50)
    print("ESTUDIANTE 2 — Saltos cognitivos (AACC funcional)")
    result2 = adapter.full_adapt("est_02", base_prompt, sample_response)
    print(f"  Estrategias: {result2['strategies_applied']}")
    print(f"  Respuesta adaptada:\n  {result2['adapted_response'][:300]}...")

    print("\n" + "─" * 50)
    print("ESTUDIANTE 3 — Asimetría temática (2e funcional)")
    result3 = adapter.full_adapt("est_03", base_prompt, sample_response)
    print(f"  Estrategias: {result3['strategies_applied']}")
    print(f"  Perfil: {result3['profile_summary']}")

    print("\n" + "═" * 50)
    print("INFORME DE ADAPTACIÓN")
    report = adapter.get_adaptation_report()
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Historial por estudiante
    for sid in ["est_01", "est_02", "est_03"]:
        hist = adapter.get_student_adaptation_history(sid)
        print(f"\n  {sid}: {hist.get('active_strategies', [])}")
