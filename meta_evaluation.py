"""
META-EVALUACIÓN — Evaluación de Efectividad del Sistema Pedagógico
═══════════════════════════════════════════════════════════════
PROBLEMA QUE ATACA:
Evaluar si las estrategias pedagógicas del chatbot están funcionando
para cada estudiante (subida Bloom, no repetición, engagement, frustración).

FUNDAMENTACIÓN TEÓRICA:
- Formative assessment (Black & Wiliam, 1998)
- Learning analytics feedback loops (Siemens & Long, 2011)
- Double-loop learning (Argyris & Schön, 1978)

POSICIÓN EN EL ECOSISTEMA:
cognitive_engine, trust_dynamics → meta_evaluation
meta_evaluation → researcher_view, app.py (Investigador)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class StrategyEffectiveness:
    """Efectividad de una estrategia pedagógica por estudiante."""
    strategy: str
    student_id: str
    success_count: int
    total_count: int
    success_rate: float


@dataclass
class MetaEvaluationResult:
    """Resultado de meta-evaluación."""
    effectiveness_by_strategy: Dict[str, float]  # strategy -> success_rate
    strategy_recommendations: List[str]
    auto_adjust_suggestion: Optional[str] = None
    per_student_strategy: List[StrategyEffectiveness] = field(default_factory=list)


class MetaEvaluator:
    """
    Meta-evaluador: valora si las estrategias pedagógicas funcionan.
    
    Criterios:
    - Bloom subió en siguiente mensaje → efectivo
    - Misma pregunta repetida → fallido
    - Tiempo respuesta estudiante razonable → engagement
    - Frustración post-respuesta → posible daño
    
    Uso:
        evaluator = MetaEvaluator()
        result = evaluator.evaluate(interactions, student_id="est_01")
    """

    def __init__(self):
        self._consecutive_failures: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def _normalize_prompt(self, p: str) -> str:
        """Normaliza prompt para detectar repeticiones."""
        if not p:
            return ""
        return " ".join((p or "").lower().split())[:100]

    def evaluate(
        self,
        interactions: List[Dict],
        student_id: Optional[str] = None,
    ) -> MetaEvaluationResult:
        """
        Evalúa efectividad de scaffolding_mode por interacción.
        """
        filtered = [i for i in interactions if student_id is None or i.get("student_id") == student_id]
        if not filtered:
            return MetaEvaluationResult(
                effectiveness_by_strategy={},
                strategy_recommendations=["Sin datos suficientes para evaluar"],
            )

        # Agrupar por estrategia (scaffolding_mode)
        by_strategy: Dict[str, List[tuple]] = defaultdict(list)  # strategy -> [(success, interaction), ...]
        consecutive_failures_by_strategy: Dict[str, int] = defaultdict(int)

        sorted_ix = sorted(range(len(filtered)), key=lambda i: filtered[i].get("timestamp", ""))
        prev_bloom = None
        prev_normalized = None

        for idx in sorted_ix:
            i = filtered[idx]
            strategy = i.get("scaffolding_mode") or "socratic"
            bloom = i.get("bloom_level") or i.get("bloom")
            prompt = i.get("prompt_raw") or i.get("prompt", "")
            norm = self._normalize_prompt(prompt)
            trust = i.get("trust_direction")
            response_time = i.get("response_time_ms") or 0

            success = False
            if prev_bloom is not None and bloom is not None:
                if int(bloom) > int(prev_bloom):
                    success = True  # Bloom subió → efectivo
            if norm and prev_normalized and norm == prev_normalized:
                success = False  # Misma pregunta → fallido
            if trust is not None and float(trust) < -0.3:
                success = False  # Frustración post-respuesta
            if response_time > 0 and response_time < 2000:
                pass  # Respuesta rápida = engagement mantenido
            elif prev_bloom is None:
                success = True  # Primera interacción

            by_strategy[strategy].append((success, i))

            if not success and prev_bloom is not None:
                consecutive_failures_by_strategy[strategy] += 1
            else:
                consecutive_failures_by_strategy[strategy] = 0

            prev_bloom = bloom
            prev_normalized = norm

        # Calcular success_rate por estrategia
        effectiveness: Dict[str, float] = {}
        for strat, pairs in by_strategy.items():
            total = len(pairs)
            success_count = sum(1 for s, _ in pairs if s)
            effectiveness[strat] = round(success_count / total, 2) if total > 0 else 0.0

        # Recomendaciones
        recommendations: List[str] = []
        best = max(effectiveness.items(), key=lambda x: x[1]) if effectiveness else None
        worst = min(effectiveness.items(), key=lambda x: x[1]) if effectiveness else None

        if best:
            recommendations.append(f"{best[0].title()} funciona al {best[1]:.0%} con este estudiante")
        if worst and worst[0] != best[0] and worst[1] < 0.5:
            recommendations.append(f"{worst[0].title()} solo alcanza {worst[1]:.0%} — considerar cambiar")

        # Auto-adjust: 3 fallos consecutivos
        auto_adjust = None
        for strat, fails in consecutive_failures_by_strategy.items():
            if fails >= 3:
                auto_adjust = f"Estrategia {strat} ha fallado 3 veces seguidas. Sugerencia: cambiar a hints o direct."
                break

        # Por estudiante
        per_student: List[StrategyEffectiveness] = []
        sid = student_id or "global"
        for strat, pairs in by_strategy.items():
            total = len(pairs)
            success_count = sum(1 for s, _ in pairs if s)
            per_student.append(StrategyEffectiveness(
                strategy=strat,
                student_id=sid,
                success_count=success_count,
                total_count=total,
                success_rate=success_count / total if total > 0 else 0.0,
            ))

        return MetaEvaluationResult(
            effectiveness_by_strategy=effectiveness,
            strategy_recommendations=recommendations or ["Seguir monitorizando"],
            auto_adjust_suggestion=auto_adjust,
            per_student_strategy=per_student,
        )
