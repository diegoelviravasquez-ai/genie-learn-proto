"""
DETECTOR DE PATRONES DE INTERACCIÓN NEURODIVERGENTE
=====================================================
Identifica patrones de uso del chatbot asociados a perfiles
neurodivergentes (TDAH, AACC, 2e) para ADAPTAR el scaffolding,
no para diagnosticar ni etiquetar.

Principio de diseño fundamental:
  La neurodivergencia no es un déficit a corregir sino una variación
  cognitiva que requiere DISEÑO ADAPTATIVO. Un chatbot educativo que
  ignora estos patrones penaliza sistemáticamente a un segmento de
  estudiantes — exactamente el problema que UBUN.IA abordaba.

Esto NO es un diagnóstico clínico. Es detección de PATRONES DE
INTERACCIÓN que, si se confirman, sugieren adaptaciones del
scaffolding pedagógico. El docente decide si y cómo actuar.

Patrones documentados en la literatura:

  TDAH (Barkley, 2015; Brown, 2017):
  - Ráfagas de hiperfoco: muchas interacciones concentradas seguidas
    de periodos largos de inactividad (distribución bimodal)
  - Salto entre topics: baja persistencia en un tema antes de cambiar
  - Prompts más cortos y menos elaborados (impulsividad)
  - Re-preguntas: vuelve a preguntar lo mismo días después (memoria
    de trabajo limitada, no falta de comprensión)

  AACC / Gifted (Renzulli, 2005; Silverman, 2013):
  - Saltos cognitivos: pasa de Bloom 1 a Bloom 5 sin transitar 2-3-4
  - Preguntas que exceden el temario (busca más allá del currículo)
  - Baja tolerancia a scaffolding socrático en temas que considera
    "triviales" (frustración selectiva, no general)
  - Alta metacognición desde el inicio

  2e — Twice Exceptional (Reis et al., 2014):
  - COMBINACIÓN de patrones TDAH + AACC: hiperfoco en temas de interés
    con profundidad excepcional, seguido de desenganche en otros
  - Rendimiento asimétrico: Bloom 5-6 en unos temas, Bloom 1 en otros
  - Perfil de engagement errático que NO es aleatorio sino temáticamente
    condicionado (la varianza se explica por el tema, no por el tiempo)

Marco ético:
  - Los patrones se reportan al DOCENTE, nunca al estudiante
  - No se usan etiquetas diagnósticas ("posible perfil TDAH") sino
    descripciones funcionales ("patrón de interacción episódica")
  - El docente puede desactivar esta funcionalidad
  - Fundamentación: Value-Sensitive Design (Friedman et al., 2017)
    — stakeholders indirectos (estudiantes neurodivergentes) como
    informantes de diseño

Autor: Diego Elvira Vásquez
Conexión: UBUN.IA (1er premio HACK4EDU 2024, equidad + neurodivergencia),
          perfil 2e personal (AACC + TDAH diagnosticado)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import math


@dataclass
class InteractionEvent:
    """Evento de interacción mínimo para análisis temporal."""
    timestamp: datetime
    bloom_level: int
    topics: list
    prompt_length: int       # palabras
    copy_paste_score: float
    is_metacognitive: bool
    scaffolding_level: int


@dataclass
class NeurodivergentPattern:
    """Patrón detectado con su evidencia y recomendación."""
    pattern_id: str
    pattern_name: str
    functional_description: str   # descripción funcional, NO etiqueta clínica
    confidence: float             # 0-1
    evidence: list                # lista de observaciones que sustentan la detección
    scaffolding_adaptation: str   # cómo adaptar el sistema
    teacher_note: str             # nota para el docente


class NeurodivergentPatternDetector:
    """
    Detecta patrones de interacción que sugieren la necesidad de
    adaptaciones pedagógicas específicas.
    
    Diseño: el detector produce PATRONES FUNCIONALES, no diagnósticos.
    "Interacción episódica con hiperfoco" en vez de "TDAH".
    "Saltos cognitivos asimétricos" en vez de "gifted".
    "Rendimiento temáticamente condicionado" en vez de "2e".
    """

    def analyze(self, events: list[InteractionEvent]) -> list[NeurodivergentPattern]:
        """
        Analiza una secuencia de interacciones y devuelve patrones detectados.
        Requiere mínimo 8 interacciones para resultados significativos.
        """
        if len(events) < 8:
            return []

        patterns = []

        # ── PATRÓN 1: Interacción episódica (asociado a TDAH) ──
        ep = self._detect_episodic_interaction(events)
        if ep:
            patterns.append(ep)

        # ── PATRÓN 2: Salto entre topics (asociado a TDAH) ──
        ts = self._detect_topic_switching(events)
        if ts:
            patterns.append(ts)

        # ── PATRÓN 3: Saltos cognitivos (asociado a AACC) ──
        cj = self._detect_cognitive_jumps(events)
        if cj:
            patterns.append(cj)

        # ── PATRÓN 4: Frustración selectiva (asociado a AACC) ──
        sf = self._detect_selective_frustration(events)
        if sf:
            patterns.append(sf)

        # ── PATRÓN 5: Rendimiento asimétrico temático (asociado a 2e) ──
        ta = self._detect_thematic_asymmetry(events)
        if ta:
            patterns.append(ta)

        # ── PATRÓN 6: Re-preguntas (memoria de trabajo) ──
        rq = self._detect_re_asking(events)
        if rq:
            patterns.append(rq)

        return patterns

    def _detect_episodic_interaction(self, events: list[InteractionEvent]) -> Optional[NeurodivergentPattern]:
        """
        Detecta distribución bimodal del timing: ráfagas + silencios.
        Señal: coeficiente de variación del inter-event time > 1.5
        (distribución normal ≈ 0.3-0.5; episódica > 1.5)
        """
        if len(events) < 5:
            return None

        # Calcular intervalos entre interacciones
        intervals = []
        for i in range(1, len(events)):
            delta = (events[i].timestamp - events[i-1].timestamp).total_seconds()
            if delta > 0:  # ignorar eventos simultáneos
                intervals.append(delta)

        if len(intervals) < 4:
            return None

        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return None

        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std = math.sqrt(variance)
        cv = std / mean_interval  # coeficiente de variación

        # Detectar ráfagas: clusters de intervalos cortos
        short_threshold = mean_interval * 0.3
        burst_count = sum(1 for x in intervals if x < short_threshold)
        burst_ratio = burst_count / len(intervals)

        if cv > 1.2 and burst_ratio > 0.3:
            evidence = [
                f"Coeficiente de variación temporal: {cv:.2f} (umbral: 1.2)",
                f"Ratio de ráfagas: {burst_ratio:.0%} de intervalos son <30% de la media",
                f"Intervalo medio: {mean_interval/60:.0f} min, σ = {std/60:.0f} min",
            ]
            return NeurodivergentPattern(
                pattern_id="EPISODIC",
                pattern_name="Interacción episódica con ráfagas",
                functional_description=(
                    "El estudiante interactúa en ráfagas concentradas seguidas de "
                    "periodos de inactividad. Este patrón sugiere episodios de "
                    "hiperfoco alternados con desenganche."
                ),
                confidence=min(0.95, (cv - 1.0) / 2),
                evidence=evidence,
                scaffolding_adaptation=(
                    "Adaptar el límite de prompts a ventanas temporales en vez de "
                    "diarias. Permitir ráfagas de uso (15-20 prompts/hora) con "
                    "cooldown. No penalizar la distribución irregular."
                ),
                teacher_note=(
                    "Patrón de interacción episódica detectado. Considerar si el "
                    "límite diario de prompts está penalizando a este estudiante — "
                    "su uso concentrado puede ser igual de productivo que el distribuido."
                ),
            )
        return None

    def _detect_topic_switching(self, events: list[InteractionEvent]) -> Optional[NeurodivergentPattern]:
        """
        Detecta cambios frecuentes de topic sin profundizar en ninguno.
        Señal: topic switches / total interactions > 0.6
        """
        if len(events) < 6:
            return None

        switches = 0
        for i in range(1, len(events)):
            prev_topics = set(events[i-1].topics)
            curr_topics = set(events[i].topics)
            if not prev_topics.intersection(curr_topics):
                switches += 1

        switch_ratio = switches / (len(events) - 1)

        # También medir profundidad por topic
        topic_depths = {}
        for e in events:
            for t in e.topics:
                if t not in topic_depths:
                    topic_depths[t] = []
                topic_depths[t].append(e.bloom_level)

        shallow_topics = sum(1 for levels in topic_depths.values() if max(levels) <= 2)
        shallow_ratio = shallow_topics / max(len(topic_depths), 1)

        if switch_ratio > 0.55 and shallow_ratio > 0.5:
            return NeurodivergentPattern(
                pattern_id="TOPIC_SWITCH",
                pattern_name="Navegación temática dispersa",
                functional_description=(
                    "Alta frecuencia de cambio entre topics con baja profundización "
                    "en cada uno. El estudiante explora amplio pero no profundo."
                ),
                confidence=min(0.9, switch_ratio),
                evidence=[
                    f"Ratio de cambio temático: {switch_ratio:.0%}",
                    f"Topics sin superar Bloom 2: {shallow_ratio:.0%}",
                    f"Topics explorados: {len(topic_depths)}",
                ],
                scaffolding_adaptation=(
                    "Introducir 'anclas temáticas': cuando el estudiante cambie de "
                    "tema, el chatbot puede decir 'Antes estábamos con X, ¿quieres "
                    "profundizar o pasamos a Y?' — dando agencia sin forzar."
                ),
                teacher_note=(
                    "Navegación dispersa detectada. Puede indicar dificultad para "
                    "sostener atención en un tema, o puede ser exploración legítima. "
                    "Verificar con el rendimiento en evaluaciones."
                ),
            )
        return None

    def _detect_cognitive_jumps(self, events: list[InteractionEvent]) -> Optional[NeurodivergentPattern]:
        """
        Detecta saltos de ≥3 niveles de Bloom entre interacciones consecutivas.
        Señal de procesamiento no-lineal (asociado a AACC).
        """
        jumps = []
        for i in range(1, len(events)):
            diff = events[i].bloom_level - events[i-1].bloom_level
            if abs(diff) >= 3:
                jumps.append({
                    "from": events[i-1].bloom_level,
                    "to": events[i].bloom_level,
                    "index": i,
                })

        jump_ratio = len(jumps) / max(len(events) - 1, 1)

        # También verificar si los saltos son ascendentes (no frustración)
        upward_jumps = [j for j in jumps if j["to"] > j["from"]]

        if len(jumps) >= 2 and len(upward_jumps) > len(jumps) * 0.5:
            from cognitive_analyzer import BLOOM_LEVELS
            return NeurodivergentPattern(
                pattern_id="COGNITIVE_JUMP",
                pattern_name="Saltos cognitivos no-lineales",
                functional_description=(
                    "El estudiante transita de niveles básicos a avanzados sin pasar "
                    "por los intermedios. Esto sugiere procesamiento no-lineal: "
                    "comprende el concepto completo 'de golpe' sin necesitar "
                    "la escalera gradual del scaffolding."
                ),
                confidence=min(0.85, jump_ratio * 2),
                evidence=[
                    f"Saltos de ≥3 niveles: {len(jumps)} de {len(events)-1} transiciones",
                    f"Ascendentes: {len(upward_jumps)} ({len(upward_jumps)/max(len(jumps),1):.0%})",
                ] + [
                    f"  Salto {j['index']}: {BLOOM_LEVELS[j['from']]['name']} → {BLOOM_LEVELS[j['to']]['name']}"
                    for j in jumps[:5]
                ],
                scaffolding_adaptation=(
                    "Reducir scaffolding socrático a 2 niveles en vez de 4. "
                    "Este estudiante no necesita la escalera completa — forzarla "
                    "genera frustración sin ganancia de aprendizaje. Ofrecer "
                    "preguntas de evaluación (Bloom 5) y creación (Bloom 6) antes."
                ),
                teacher_note=(
                    "Saltos cognitivos detectados — perfil de procesamiento no-lineal. "
                    "El scaffolding gradual puede estar infraestimulando a este "
                    "estudiante. Considerar permitir acceso a problemas avanzados."
                ),
            )
        return None

    def _detect_selective_frustration(self, events: list[InteractionEvent]) -> Optional[NeurodivergentPattern]:
        """
        Detecta frustración que aparece SOLO en ciertos topics pero no en otros.
        Señal: alta varianza de Bloom condicionada al topic.
        """
        topic_levels = {}
        for e in events:
            for t in e.topics:
                if t not in topic_levels:
                    topic_levels[t] = []
                topic_levels[t].append(e.bloom_level)

        # Necesitamos al menos 2 topics con ≥3 interacciones cada uno
        valid_topics = {t: levels for t, levels in topic_levels.items() if len(levels) >= 3}
        if len(valid_topics) < 2:
            return None

        topic_means = {t: sum(l)/len(l) for t, l in valid_topics.items()}

        # Varianza entre medias de topics
        overall_mean = sum(topic_means.values()) / len(topic_means)
        between_variance = sum((m - overall_mean) ** 2 for m in topic_means.values()) / len(topic_means)

        if between_variance > 2.0:  # diferencia de >1.4 niveles entre topics
            high_topics = {t: m for t, m in topic_means.items() if m > overall_mean + 0.7}
            low_topics = {t: m for t, m in topic_means.items() if m < overall_mean - 0.7}

            if high_topics and low_topics:
                return NeurodivergentPattern(
                    pattern_id="SELECTIVE_FRUSTRATION",
                    pattern_name="Rendimiento temáticamente asimétrico",
                    functional_description=(
                        "El estudiante muestra niveles cognitivos altos en ciertos "
                        "temas y bajos en otros. La asimetría no es aleatoria sino "
                        "temáticamente estructurada — sugiere intereses diferenciados "
                        "o prerrequisitos desiguales entre temas."
                    ),
                    confidence=min(0.9, between_variance / 4),
                    evidence=[
                        f"Varianza inter-topic: {between_variance:.2f}",
                        f"Topics altos (>{overall_mean+0.7:.1f}): {', '.join(f'{t} ({m:.1f})' for t,m in high_topics.items())}",
                        f"Topics bajos (<{overall_mean-0.7:.1f}): {', '.join(f'{t} ({m:.1f})' for t,m in low_topics.items())}",
                    ],
                    scaffolding_adaptation=(
                        "Scaffolding adaptativo por tema: modo directo en topics "
                        "de alto rendimiento (no necesita andamiaje), socrático "
                        "en topics bajos. Usar los topics altos como puentes "
                        "analógicos hacia los bajos."
                    ),
                    teacher_note=(
                        "Rendimiento asimétrico significativo entre temas. "
                        "Verificar si los topics bajos corresponden a prerrequisitos "
                        "o a falta de interés. En perfiles twice-exceptional, este "
                        "patrón es característico y responde bien a scaffolding "
                        "diferenciado por tema."
                    ),
                )
        return None

    def _detect_thematic_asymmetry(self, events: list[InteractionEvent]) -> Optional[NeurodivergentPattern]:
        """
        Detecta el patrón 2e completo: hiperfoco + asimetría + metacognición alta
        en ciertos temas pero no en otros.
        """
        # Verificar co-ocurrencia de patrones
        has_episodic = self._detect_episodic_interaction(events) is not None
        has_jumps = self._detect_cognitive_jumps(events) is not None
        has_asymmetry = self._detect_selective_frustration(events) is not None

        # Meta solo en topics de alto rendimiento
        topic_meta = {}
        for e in events:
            for t in e.topics:
                if t not in topic_meta:
                    topic_meta[t] = {"meta": 0, "total": 0}
                topic_meta[t]["total"] += 1
                if e.is_metacognitive:
                    topic_meta[t]["meta"] += 1

        meta_variance = 0
        if len(topic_meta) >= 2:
            ratios = [d["meta"]/max(d["total"], 1) for d in topic_meta.values()]
            mean_ratio = sum(ratios) / len(ratios)
            meta_variance = sum((r - mean_ratio) ** 2 for r in ratios) / len(ratios)

        combined_score = (
            (0.35 if has_episodic else 0) +
            (0.25 if has_jumps else 0) +
            (0.25 if has_asymmetry else 0) +
            (0.15 if meta_variance > 0.05 else 0)
        )

        if combined_score >= 0.5:
            return NeurodivergentPattern(
                pattern_id="TWICE_EXCEPTIONAL",
                pattern_name="Perfil de interacción twice-exceptional",
                functional_description=(
                    "Combinación de: interacción episódica, saltos cognitivos, "
                    "rendimiento asimétrico entre temas, y metacognición selectiva. "
                    "Este patrón compuesto sugiere un perfil que combina alta "
                    "capacidad en áreas de interés con dificultades atencionales "
                    "en otras. El rendimiento global NO refleja la capacidad real."
                ),
                confidence=round(combined_score, 2),
                evidence=[
                    f"Interacción episódica: {'Sí' if has_episodic else 'No'}",
                    f"Saltos cognitivos: {'Sí' if has_jumps else 'No'}",
                    f"Asimetría temática: {'Sí' if has_asymmetry else 'No'}",
                    f"Metacognición selectiva (varianza): {meta_variance:.3f}",
                    f"Score compuesto: {combined_score:.2f}",
                ],
                scaffolding_adaptation=(
                    "SCAFFOLDING DIFERENCIADO COMPLETO:\n"
                    "1. Topics de alto interés: modo directo o evaluativo. "
                    "   Ofrecer extensiones y retos avanzados.\n"
                    "2. Topics de bajo rendimiento: scaffolding socrático con "
                    "   paciencia extra. Conectar con topics de interés como "
                    "   puentes analógicos.\n"
                    "3. Timing: no limitar ráfagas de uso. Permitir sesiones "
                    "   largas en hiperfoco.\n"
                    "4. Memoria de trabajo: el chatbot debe recordar contexto "
                    "   de sesiones anteriores y ofrecer 'resúmenes de dónde "
                    "   nos quedamos'."
                ),
                teacher_note=(
                    "Perfil de interacción complejo detectado que combina señales "
                    "de alta capacidad con patrones atencionales irregulares. "
                    "El rendimiento medio NO es representativo — las evaluaciones "
                    "globales pueden infraestimar significativamente a este "
                    "estudiante. Considerar evaluación diferenciada."
                ),
            )
        return None

    def _detect_re_asking(self, events: list[InteractionEvent]) -> Optional[NeurodivergentPattern]:
        """
        Detecta re-preguntas: el estudiante vuelve a preguntar sobre
        un topic en el que ya había alcanzado Bloom ≥3.
        Señal de memoria de trabajo limitada (no de incomprensión).
        """
        topic_history = {}
        regressions = []

        for i, e in enumerate(events):
            for t in e.topics:
                if t in topic_history:
                    max_prev = max(topic_history[t])
                    if max_prev >= 3 and e.bloom_level <= 1:
                        regressions.append({
                            "topic": t,
                            "prev_max": max_prev,
                            "current": e.bloom_level,
                            "index": i,
                        })
                    topic_history[t].append(e.bloom_level)
                else:
                    topic_history[t] = [e.bloom_level]

        regression_ratio = len(regressions) / max(len(events), 1)

        if len(regressions) >= 2 and regression_ratio > 0.1:
            return NeurodivergentPattern(
                pattern_id="RE_ASKING",
                pattern_name="Patrón de re-consulta con regresión aparente",
                functional_description=(
                    "El estudiante vuelve a hacer preguntas básicas sobre temas "
                    "en los que previamente mostró comprensión avanzada. Esto NO "
                    "indica falta de comprensión sino probable limitación de "
                    "memoria de trabajo o consolidación insuficiente."
                ),
                confidence=min(0.85, regression_ratio * 3),
                evidence=[
                    f"Regresiones detectadas: {len(regressions)}",
                    f"Ratio: {regression_ratio:.0%} de interacciones",
                ] + [
                    f"  Topic '{r['topic']}': fue Bloom {r['prev_max']} → ahora Bloom {r['current']}"
                    for r in regressions[:4]
                ],
                scaffolding_adaptation=(
                    "Activar 'recordatorios de contexto': cuando el estudiante "
                    "pregunte sobre un topic previo, el chatbot ofrece un breve "
                    "resumen de lo que ya se discutió antes de responder. "
                    "Formato: 'La última vez que hablamos de X, habíamos llegado "
                    "a Y. ¿Quieres repasar o avanzar?'"
                ),
                teacher_note=(
                    "Re-consultas detectadas en topics previamente dominados. "
                    "Posible limitación de memoria de trabajo, no de comprensión. "
                    "Considerar proporcionar materiales de repaso rápido o "
                    "cheat sheets del curso."
                ),
            )
        return None
