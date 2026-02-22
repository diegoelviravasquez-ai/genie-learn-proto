"""
SYSTEM REFLEXIVITY ENGINE — El Sistema que Se Observa a Sí Mismo
═══════════════════════════════════════════════════════════════════════
Ningún chatbot educativo publicado hace esto.

REFLEXIVIDAD (Bourdieu, 1990): capacidad de un sistema de observar
sus propias operaciones y derivar conocimiento de esa observación.

Este módulo lee TRANSVERSALMENTE los datos de todos los demás módulos
y detecta PATRONES SISTÉMICOS que ningún módulo individual puede ver:

    - "El modo socrático funciona en horario de mañana pero no por la
       noche — los estudiantes nocturnos tienen diferente tolerancia
       a la ambigüedad"
    - "Las respuestas del LLM están degradándose en el tema de recursión
       desde hace 3 días — el corpus RAG necesita actualización"
    - "Los cambios de configuración del docente X coinciden sistemáticamente
       con bajadas de engagement — posible miscalibration"
    - "El scaffolding socrático produce mejora Bloom en 60% de estudiantes
       pero REGRESIÓN en 15% — efecto heterogéneo no detectado"

ESTO ES META-ANALYTICS: analytics sobre los analytics.
El dataset que genera alimenta DIRECTAMENTE los papers de O1 (2027):
modelos Human-AI que emergen empíricamente de la interacción.

FUNDAMENTACIÓN:
    Bourdieu (1990): reflexividad como herramienta epistemológica
    Luhmann (1995): autopoiesis y auto-observación en sistemas sociales
    Suchman (2007): plans and situated actions — la brecha entre diseño e uso
    Schön (1983): reflection-in-action vs reflection-on-action

Autor: Diego Elvira Vásquez · CP25/152 GSIC/EMIC · Feb 2026
"""

import re
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from collections import defaultdict


@dataclass
class ReflexiveInsight:
    """Un insight emergente del meta-análisis del sistema."""
    insight_id: str = ""
    timestamp: str = ""
    category: str = ""          # systemic|pedagogical|technical|emergent
    severity: str = "info"      # info|warning|critical|discovery
    title: str = ""
    description: str = ""
    evidence: Dict = field(default_factory=dict)
    affected_modules: List[str] = field(default_factory=list)
    theoretical_frame: str = ""
    actionable: bool = False
    suggested_investigation: str = ""


@dataclass
class InteractionTrace:
    """Traza mínima de una interacción para análisis reflexivo."""
    timestamp: str = ""
    student_id: str = ""
    bloom_level: int = 1
    autonomy_score: float = 0.0
    autonomy_phase: str = "dependent"
    scaffolding_mode: str = "socratic"
    hhh_overall: float = 0.0
    rag_quality: float = 0.0
    copy_paste: float = 0.0
    gaming_suspicion: float = 0.0
    topics: List[str] = field(default_factory=list)
    config_snapshot: Dict = field(default_factory=dict)
    was_blocked: bool = False
    hour_of_day: int = 12
    day_of_week: int = 0


class SystemReflexivityEngine:
    """
    Motor de reflexividad: el sistema se observa a sí mismo.
    
    Consume: trazas de interacción acumuladas de todos los módulos.
    Produce: insights emergentes que ningún módulo individual detecta.
    
    Se ejecuta periódicamente (cada N interacciones o cada hora)
    como un cron job del ecosistema, no en tiempo real.
    """

    def __init__(self):
        self.traces: List[InteractionTrace] = []
        self.insights: List[ReflexiveInsight] = []
        self._insight_counter = 0

    def ingest_trace(self, trace: InteractionTrace):
        """Ingesta una traza de interacción."""
        self.traces.append(trace)

    def ingest_batch(self, traces: List[dict]):
        """Ingesta un batch de trazas desde la base de datos."""
        for t in traces:
            self.traces.append(InteractionTrace(**{
                k: v for k, v in t.items()
                if hasattr(InteractionTrace, k)
            }))

    def reflect(self, min_traces: int = 20) -> List[ReflexiveInsight]:
        """
        Ejecuta el ciclo completo de reflexividad.
        Cada detector busca un tipo específico de patrón sistémico.
        """
        if len(self.traces) < min_traces:
            return []

        new_insights = []

        detectors = [
            self._detect_temporal_patterns,
            self._detect_config_effectiveness_heterogeneity,
            self._detect_rag_degradation_by_topic,
            self._detect_scaffolding_paradox,
            self._detect_gaming_ecology,
            self._detect_autonomy_regression,
            self._detect_bloom_ceiling,
            self._detect_hhh_drift,
        ]

        for detector in detectors:
            try:
                insights = detector()
                new_insights.extend(insights)
            except Exception:
                continue

        self.insights.extend(new_insights)
        return new_insights

    # ─── DETECTOR 1: Patrones temporales ──────────────────────

    def _detect_temporal_patterns(self) -> List[ReflexiveInsight]:
        """
        Detecta si el rendimiento varía por hora/día.
        Hipótesis: estudiantes nocturnos tienen diferente tolerancia
        a scaffolding ambiguo (socrático).
        """
        insights = []

        # Agrupar por franja horaria
        morning = [t for t in self.traces if 8 <= t.hour_of_day < 14]
        afternoon = [t for t in self.traces if 14 <= t.hour_of_day < 20]
        night = [t for t in self.traces if t.hour_of_day >= 20 or t.hour_of_day < 8]

        for label, group in [("mañana", morning), ("tarde", afternoon), ("noche", night)]:
            if len(group) < 5:
                continue

            avg_bloom = statistics.mean(t.bloom_level for t in group)
            avg_hhh = statistics.mean(t.hhh_overall for t in group)
            avg_gaming = statistics.mean(t.gaming_suspicion for t in group)

            # Comparar con media global
            global_bloom = statistics.mean(t.bloom_level for t in self.traces)
            global_gaming = statistics.mean(t.gaming_suspicion for t in self.traces)

            if avg_bloom < global_bloom - 0.5 or avg_gaming > global_gaming + 0.15:
                insights.append(self._create_insight(
                    category="emergent",
                    severity="discovery",
                    title=f"Patrón temporal: rendimiento diferente en horario {label}",
                    description=(
                        f"Estudiantes en horario {label} muestran Bloom medio "
                        f"{avg_bloom:.1f} vs global {global_bloom:.1f}, "
                        f"gaming {avg_gaming:.0%} vs global {global_gaming:.0%}. "
                        f"El scaffolding puede necesitar adaptación temporal."
                    ),
                    evidence={
                        "time_slot": label,
                        "n_traces": len(group),
                        "avg_bloom": round(avg_bloom, 2),
                        "global_bloom": round(global_bloom, 2),
                        "avg_gaming": round(avg_gaming, 3),
                    },
                    affected_modules=["middleware", "temporal_config_advisor"],
                    theoretical_frame=(
                        "Schön (1983): reflection-on-action revela patrones "
                        "invisibles durante la acción. El sistema detecta lo que "
                        "el diseñador no anticipó."
                    ),
                    actionable=True,
                    suggested_investigation=(
                        "¿Los estudiantes nocturnos son los mismos que los diurnos en otro horario, "
                        "o es una subpoblación distinta? Cruzar con perfiles de autonomía."
                    ),
                ))

        return insights

    # ─── DETECTOR 2: Heterogeneidad de efectividad ────────────

    def _detect_config_effectiveness_heterogeneity(self) -> List[ReflexiveInsight]:
        """
        Detecta si una configuración produce efectos OPUESTOS en subgrupos.
        El efecto promedio puede ser positivo pero ocultar regresión en un subgrupo.
        """
        insights = []

        # Agrupar por modo de scaffolding
        by_mode = defaultdict(list)
        for t in self.traces:
            by_mode[t.scaffolding_mode].append(t)

        for mode, traces in by_mode.items():
            if len(traces) < 10:
                continue

            # Dividir por estudiante y calcular tendencia Bloom individual
            by_student = defaultdict(list)
            for t in traces:
                by_student[t.student_id].append(t)

            improving = 0
            regressing = 0
            flat = 0
            regressing_students = []

            for sid, student_traces in by_student.items():
                if len(student_traces) < 3:
                    continue
                blooms = [t.bloom_level for t in sorted(student_traces, key=lambda x: x.timestamp)]
                # Tendencia simple: última mitad vs primera mitad
                mid = len(blooms) // 2
                first_half = statistics.mean(blooms[:mid]) if mid > 0 else 0
                second_half = statistics.mean(blooms[mid:]) if mid > 0 else 0
                delta = second_half - first_half

                if delta > 0.3:
                    improving += 1
                elif delta < -0.3:
                    regressing += 1
                    regressing_students.append(sid)
                else:
                    flat += 1

            total = improving + regressing + flat
            if total < 3:
                continue

            regress_pct = regressing / total
            if regress_pct > 0.12:
                insights.append(self._create_insight(
                    category="pedagogical",
                    severity="warning",
                    title=f"Efecto heterogéneo del modo '{mode}'",
                    description=(
                        f"El modo {mode} produce mejora en {improving}/{total} estudiantes "
                        f"pero REGRESIÓN en {regressing}/{total} ({regress_pct:.0%}). "
                        f"El efecto medio oculta un subgrupo que empeora."
                    ),
                    evidence={
                        "mode": mode,
                        "improving": improving,
                        "regressing": regressing,
                        "flat": flat,
                        "regress_pct": round(regress_pct, 3),
                        "affected_students": regressing_students[:5],
                    },
                    affected_modules=["middleware", "epistemic_autonomy", "cognitive_profiler"],
                    theoretical_frame=(
                        "Kalyuga (2003): Expertise Reversal Effect — scaffolding "
                        "beneficioso para novatos puede perjudicar a expertos. "
                        "Suchman (2007): la brecha entre plan y acción situada."
                    ),
                    actionable=True,
                    suggested_investigation=(
                        "Correlacionar regresión con perfil de autonomía: ¿los que regresan "
                        "son estudiantes más avanzados que encuentran el socrático frustrante?"
                    ),
                ))

        return insights

    # ─── DETECTOR 3: Degradación RAG por topic ────────────────

    def _detect_rag_degradation_by_topic(self) -> List[ReflexiveInsight]:
        """
        Detecta si la calidad RAG ha degradado en un tema específico
        mientras se mantiene en otros.
        """
        insights = []

        by_topic = defaultdict(list)
        for t in self.traces:
            for topic in t.topics:
                by_topic[topic].append(t)

        global_rag = statistics.mean(t.rag_quality for t in self.traces) if self.traces else 0.5

        for topic, traces in by_topic.items():
            if len(traces) < 5:
                continue

            avg_rag = statistics.mean(t.rag_quality for t in traces)

            if avg_rag < global_rag - 0.15 and avg_rag < 0.5:
                insights.append(self._create_insight(
                    category="technical",
                    severity="warning",
                    title=f"Degradación RAG en tema '{topic}'",
                    description=(
                        f"La calidad RAG para '{topic}' es {avg_rag:.0%} vs "
                        f"global {global_rag:.0%}. El corpus puede necesitar "
                        f"actualización o ampliación para este tema."
                    ),
                    evidence={
                        "topic": topic,
                        "avg_rag": round(avg_rag, 3),
                        "global_rag": round(global_rag, 3),
                        "n_interactions": len(traces),
                    },
                    affected_modules=["rag_pipeline", "rag_quality_sensor", "ragas_evaluator"],
                    theoretical_frame="RAGAS (Es et al., 2023): context precision y recall por dominio",
                    actionable=True,
                    suggested_investigation=(
                        f"Verificar si existen chunks sobre '{topic}' en el corpus. "
                        f"Si existen, el embedding model puede no capturar la semántica correctamente."
                    ),
                ))

        return insights

    # ─── DETECTOR 4: Paradoja del scaffolding ─────────────────

    def _detect_scaffolding_paradox(self) -> List[ReflexiveInsight]:
        """
        Detecta la paradoja de scaffolding: más ayuda → menos aprendizaje.
        Si los estudiantes con más scaffolding muestran MENOS autonomía,
        el sistema está creando dependencia en vez de independencia.
        """
        insights = []

        by_student = defaultdict(list)
        for t in self.traces:
            by_student[t.student_id].append(t)

        high_scaffold_autonomy = []
        low_scaffold_autonomy = []

        for sid, traces in by_student.items():
            if len(traces) < 5:
                continue

            # Contar interacciones con scaffolding alto vs bajo
            socratic_count = sum(1 for t in traces if t.scaffolding_mode == "socratic")
            socratic_ratio = socratic_count / len(traces)

            final_autonomy = statistics.mean(
                t.autonomy_score for t in sorted(traces, key=lambda x: x.timestamp)[-3:]
            )

            if socratic_ratio > 0.7:
                high_scaffold_autonomy.append(final_autonomy)
            elif socratic_ratio < 0.3:
                low_scaffold_autonomy.append(final_autonomy)

        if len(high_scaffold_autonomy) >= 3 and len(low_scaffold_autonomy) >= 3:
            avg_high = statistics.mean(high_scaffold_autonomy)
            avg_low = statistics.mean(low_scaffold_autonomy)

            if avg_high < avg_low - 0.1:
                insights.append(self._create_insight(
                    category="pedagogical",
                    severity="critical",
                    title="Paradoja del scaffolding detectada",
                    description=(
                        f"Estudiantes con scaffolding intenso muestran MENOR autonomía "
                        f"({avg_high:.2f}) que los con scaffolding bajo ({avg_low:.2f}). "
                        f"El sistema puede estar creando dependencia cognitiva."
                    ),
                    evidence={
                        "high_scaffold_mean_autonomy": round(avg_high, 3),
                        "low_scaffold_mean_autonomy": round(avg_low, 3),
                        "n_high": len(high_scaffold_autonomy),
                        "n_low": len(low_scaffold_autonomy),
                        "delta": round(avg_low - avg_high, 3),
                    },
                    affected_modules=["middleware", "epistemic_autonomy", "teacher_notification_engine"],
                    theoretical_frame=(
                        "Wood, Bruner & Ross (1976): el scaffolding sin fading "
                        "crea dependencia. Vygotsky (1978): ZPD es dinámica, "
                        "no estática. El sistema debe adaptar, no fijar."
                    ),
                    actionable=True,
                    suggested_investigation=(
                        "Implementar fading progresivo: reducir scaffolding automáticamente "
                        "cuando autonomy_score > 0.6 durante 5+ interacciones consecutivas."
                    ),
                ))

        return insights

    # ─── DETECTOR 5: Ecología del gaming ──────────────────────

    def _detect_gaming_ecology(self) -> List[ReflexiveInsight]:
        """
        Detecta si el gaming es un fenómeno individual o social.
        Si aparece en clusters temporales, puede ser comportamiento grupal.
        """
        insights = []

        gaming_traces = [t for t in self.traces if t.gaming_suspicion > 0.5]
        if len(gaming_traces) < 5:
            return insights

        # Agrupar por hora
        by_hour = defaultdict(int)
        for t in gaming_traces:
            by_hour[t.hour_of_day] += 1

        total_gaming = len(gaming_traces)
        peak_hour = max(by_hour, key=by_hour.get) if by_hour else 0
        peak_count = by_hour.get(peak_hour, 0)

        if peak_count > total_gaming * 0.4:
            insights.append(self._create_insight(
                category="emergent",
                severity="discovery",
                title="Gaming como fenómeno social/temporal",
                description=(
                    f"El {peak_count}/{total_gaming} ({peak_count/total_gaming:.0%}) del gaming "
                    f"se concentra a las {peak_hour}:00. Esto sugiere comportamiento grupal "
                    f"(estudiantes trabajando juntos, compartiendo atajos) más que individual."
                ),
                evidence={
                    "peak_hour": peak_hour,
                    "peak_count": peak_count,
                    "total_gaming": total_gaming,
                    "distribution": dict(by_hour),
                },
                affected_modules=["interaction_semiotics", "teacher_notification_engine"],
                theoretical_frame=(
                    "Baker et al. (2008): gaming the system. Hutchins (1995): "
                    "cognición distribuida — el gaming puede ser una estrategia "
                    "colectiva racional, no individual irracional."
                ),
                actionable=True,
                suggested_investigation=(
                    "Cruzar con datos de entregas: ¿el pico de gaming coincide "
                    "con fechas de entrega de prácticas?"
                ),
            ))

        return insights

    # ─── DETECTOR 6: Regresión de autonomía ───────────────────

    def _detect_autonomy_regression(self) -> List[ReflexiveInsight]:
        """
        Detecta estudiantes que estaban en fase 'emergente' o 'autonomous'
        y han regresado a 'dependent'. Señal de alarma pedagógica.
        """
        insights = []

        by_student = defaultdict(list)
        for t in self.traces:
            by_student[t.student_id].append(t)

        regressed = []
        for sid, traces in by_student.items():
            sorted_t = sorted(traces, key=lambda x: x.timestamp)
            if len(sorted_t) < 8:
                continue

            phases = [t.autonomy_phase for t in sorted_t]
            phase_order = {"dependent": 0, "scaffolded": 1, "emergent": 2, "autonomous": 3}

            max_phase = max(phase_order.get(p, 0) for p in phases[:len(phases)//2])
            current_phase = phase_order.get(phases[-1], 0)

            if max_phase >= 2 and current_phase <= 1:
                regressed.append(sid)

        if regressed:
            insights.append(self._create_insight(
                category="pedagogical",
                severity="critical" if len(regressed) > 3 else "warning",
                title=f"Regresión de autonomía en {len(regressed)} estudiante(s)",
                description=(
                    f"{len(regressed)} estudiante(s) que habían alcanzado fase emergente/autónoma "
                    f"han regresado a dependiente/scaffolded. Posibles causas: cambio de tema, "
                    f"aumento de dificultad, o configuración pedagógica inadecuada."
                ),
                evidence={
                    "regressed_students": regressed[:10],
                    "total_regressed": len(regressed),
                },
                affected_modules=["epistemic_autonomy", "teacher_notification_engine"],
                theoretical_frame=(
                    "Bandura (1997): self-efficacy no es lineal; experiencias de fracaso "
                    "pueden producir regresión abrupta. van Geert (1998): dinámicas no "
                    "lineales en desarrollo cognitivo."
                ),
                actionable=True,
                suggested_investigation=(
                    "Correlacionar regresión con cambios de tema o de configuración. "
                    "¿Coincide con un tema nuevo o con un cambio de scaffolding?"
                ),
            ))

        return insights

    # ─── DETECTOR 7: Techo de Bloom ───────────────────────────

    def _detect_bloom_ceiling(self) -> List[ReflexiveInsight]:
        """Detecta si el cohorte ha alcanzado un techo y no progresa."""
        insights = []

        if len(self.traces) < 30:
            return insights

        sorted_traces = sorted(self.traces, key=lambda t: t.timestamp)
        mid = len(sorted_traces) // 2
        first_half = statistics.mean(t.bloom_level for t in sorted_traces[:mid])
        second_half = statistics.mean(t.bloom_level for t in sorted_traces[mid:])

        if abs(second_half - first_half) < 0.15 and second_half < 4.0:
            insights.append(self._create_insight(
                category="pedagogical",
                severity="warning",
                title=f"Techo de Bloom en nivel {second_half:.1f}",
                description=(
                    f"El cohorte se ha estabilizado en Bloom {second_half:.1f} "
                    f"sin progresión significativa ({first_half:.1f} → {second_half:.1f}). "
                    f"El sistema no está empujando hacia niveles superiores."
                ),
                evidence={
                    "first_half_bloom": round(first_half, 2),
                    "second_half_bloom": round(second_half, 2),
                    "total_traces": len(sorted_traces),
                },
                affected_modules=["cognitive_profiler", "middleware", "learning_design_generator"],
                theoretical_frame=(
                    "Anderson & Krathwohl (2001): la taxonomía es jerárquica. "
                    "Estancamiento en niveles bajos indica que las tareas/prompts "
                    "no demandan operaciones cognitivas superiores."
                ),
                actionable=True,
                suggested_investigation=(
                    "¿Las preguntas del sistema invitan a analizar/evaluar/crear? "
                    "El scaffolding socrático puede necesitar preguntas de nivel superior."
                ),
            ))

        return insights

    # ─── DETECTOR 8: Drift de alineación HHH ─────────────────

    def _detect_hhh_drift(self) -> List[ReflexiveInsight]:
        """
        Detecta degradación gradual de la alineación HHH.
        Si honest/harmless bajan lentamente, el sistema está derivando.
        """
        insights = []

        if len(self.traces) < 20:
            return insights

        sorted_traces = sorted(self.traces, key=lambda t: t.timestamp)
        window = max(5, len(sorted_traces) // 4)

        early = sorted_traces[:window]
        recent = sorted_traces[-window:]

        early_hhh = statistics.mean(t.hhh_overall for t in early)
        recent_hhh = statistics.mean(t.hhh_overall for t in recent)

        drift = recent_hhh - early_hhh

        if drift < -0.1:
            insights.append(self._create_insight(
                category="technical",
                severity="warning",
                title=f"Drift de alineación HHH ({drift:+.2f})",
                description=(
                    f"La alineación HHH ha bajado de {early_hhh:.2f} a {recent_hhh:.2f}. "
                    f"Posible causa: cambio en la distribución de preguntas, degradación "
                    f"del corpus RAG, o drift del modelo LLM."
                ),
                evidence={
                    "early_hhh": round(early_hhh, 3),
                    "recent_hhh": round(recent_hhh, 3),
                    "drift": round(drift, 3),
                },
                affected_modules=["hhh_alignment_detector", "llm_client"],
                theoretical_frame="Askell et al. (2021): HHH framework. Monitoring alignment over time.",
                actionable=True,
                suggested_investigation="Revisar logs del LLM y verificar si el modelo ha cambiado de versión.",
            ))

        return insights

    # ─── UTILIDADES ───────────────────────────────────────────

    def _create_insight(self, **kwargs) -> ReflexiveInsight:
        self._insight_counter += 1
        return ReflexiveInsight(
            insight_id=f"RI_{self._insight_counter:04d}",
            timestamp=datetime.now().isoformat(),
            **kwargs,
        )

    def get_insights_report(self) -> dict:
        """Informe completo de insights reflexivos."""
        if not self.insights:
            return {"total_insights": 0, "traces_analyzed": len(self.traces)}

        by_category = defaultdict(list)
        by_severity = defaultdict(int)
        for i in self.insights:
            by_category[i.category].append(i.title)
            by_severity[i.severity] += 1

        return {
            "total_insights": len(self.insights),
            "traces_analyzed": len(self.traces),
            "by_category": {k: len(v) for k, v in by_category.items()},
            "by_severity": dict(by_severity),
            "critical_insights": [
                {"title": i.title, "description": i.description[:200]}
                for i in self.insights if i.severity in ("critical", "discovery")
            ],
            "actionable_count": sum(1 for i in self.insights if i.actionable),
        }

    def export_for_paper(self) -> List[dict]:
        """Exporta insights en formato para incluir en el paper de O1."""
        return [
            {
                "id": i.insight_id,
                "category": i.category,
                "severity": i.severity,
                "title": i.title,
                "description": i.description,
                "evidence": i.evidence,
                "theoretical_frame": i.theoretical_frame,
                "suggested_investigation": i.suggested_investigation,
                "affected_modules": i.affected_modules,
            }
            for i in self.insights
        ]


# ═══════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import random
    random.seed(42)

    engine = SystemReflexivityEngine()

    # Generar trazas sintéticas con patrones ocultos
    students = [f"est_{i:02d}" for i in range(15)]
    topics = ["bucles", "funciones", "recursión", "arrays", "POO"]

    for i in range(80):
        hour = random.choice([9, 10, 11, 14, 15, 16, 21, 22, 23])
        sid = random.choice(students)

        # Patrón oculto: estudiantes nocturnos tienen más gaming
        is_night = hour >= 21
        gaming = random.uniform(0.4, 0.9) if is_night else random.uniform(0, 0.3)

        # Patrón oculto: recursión tiene peor RAG
        topic = random.choice(topics)
        rag = random.uniform(0.2, 0.5) if topic == "recursión" else random.uniform(0.6, 0.95)

        # Patrón oculto: modo socrático crea dependencia en algunos
        mode = random.choice(["socratic", "socratic", "hints", "direct"])
        if mode == "socratic" and sid in students[:3]:
            autonomy = random.uniform(0.1, 0.3)  # estos no mejoran
            phase = "dependent"
        else:
            autonomy = random.uniform(0.3, 0.8)
            phase = "scaffolded" if autonomy < 0.55 else "emergent"

        trace = InteractionTrace(
            timestamp=f"2026-02-{10 + i // 10}T{hour:02d}:{random.randint(0,59):02d}:00",
            student_id=sid,
            bloom_level=random.randint(1, 3),  # techo bajo deliberado
            autonomy_score=autonomy,
            autonomy_phase=phase,
            scaffolding_mode=mode,
            hhh_overall=random.uniform(0.55, 0.85),
            rag_quality=rag,
            copy_paste=random.uniform(0, 0.4),
            gaming_suspicion=gaming,
            topics=[topic],
            hour_of_day=hour,
            day_of_week=i % 7,
        )
        engine.ingest_trace(trace)

    # Reflexionar
    insights = engine.reflect()
    print(f"Insights generados: {len(insights)}\n")
    for insight in insights:
        icon = {"critical": "🔴", "warning": "🟡", "discovery": "🔮", "info": "🔵"}
        print(f"  {icon.get(insight.severity, '⚪')} [{insight.category}] {insight.title}")
        print(f"     {insight.description[:120]}...")
        if insight.suggested_investigation:
            print(f"     → {insight.suggested_investigation[:100]}")
        print()

    report = engine.get_insights_report()
    print(f"Resumen: {report['total_insights']} insights, "
          f"{report['actionable_count']} accionables, "
          f"{report['traces_analyzed']} trazas analizadas")

    print("\n✓ System Reflexivity Engine operativo")
