"""
DROPOUT PREDICTOR — Predicción de Riesgo de Abandono
═══════════════════════════════════════════════════════════════
PROBLEMA QUE ATACA:
Detectar señales tempranas de desenganche para intervención del docente
antes de que el estudiante abandone.

FUNDAMENTACIÓN TEÓRICA:
- Early warning systems in education (Macfadyen & Dawson, 2010)
- Engagement analytics (Henrie et al., 2015)
- At-risk student indicators (Tinto, 1975)

POSICIÓN EN EL ECOSISTEMA:
trust_dynamics, cognitive_engine → dropout_predictor
dropout_predictor → app.py (Docente-Analytics)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

MONOSYLLABIC_PATTERNS = [
    "ok", "vale", "no entiendo", "da igual", "no", "si", "no se",
    "no funciona", "ayuda", "?", "??", "..."
]


@dataclass
class DropoutPrediction:
    """Resultado del predictor de abandono."""
    dropout_risk: float  # 0-1
    risk_factors: List[str]
    recommended_intervention: str
    time_horizon: str  # "24h", "48h", "1 semana"


class DropoutPredictor:
    """
    Predictor de riesgo de abandono basado en señales conductuales.
    
    Uso:
        predictor = DropoutPredictor()
        pred = predictor.analyze(interactions, student_id="est_01")
    """

    def __init__(self):
        self.monosyllabic = [p.lower() for p in MONOSYLLABIC_PATTERNS]

    def _time_between_messages(self, interactions: List[Dict], student_id: Optional[str] = None) -> float:
        """
        Ratio tiempo entre mensajes: si aumenta → riesgo.
        Devuelve 0-1: 1 = tiempos muy crecientes (riesgo alto).
        """
        filtered = [i for i in interactions if student_id is None or i.get("student_id") == student_id]
        if len(filtered) < 3:
            return 0.0
        filtered = sorted(filtered, key=lambda x: x.get("timestamp", ""))
        deltas = []
        for i in range(1, len(filtered)):
            try:
                t1 = filtered[i - 1].get("timestamp", "")
                t2 = filtered[i].get("timestamp", "")
                if isinstance(t1, str) and isinstance(t2, str):
                    dt1 = datetime.fromisoformat(t1.replace("Z", "+00:00"))
                    dt2 = datetime.fromisoformat(t2.replace("Z", "+00:00"))
                    deltas.append((dt2 - dt1).total_seconds())
            except Exception:
                pass
        if len(deltas) < 2:
            return 0.0
        # Si los últimos deltas son mayores que los primeros → riesgo
        mid = len(deltas) // 2
        avg_first = sum(deltas[:mid]) / max(mid, 1)
        avg_last = sum(deltas[mid:]) / max(len(deltas) - mid, 1)
        if avg_first <= 0:
            return 0.0
        ratio = avg_last / avg_first
        return min(1.0, (ratio - 1) / 3) if ratio > 1 else 0.0  # ratio 4+ → 1

    def _frustration_score(self, interactions: List[Dict], student_id: Optional[str] = None, last_n: int = 5) -> float:
        """
        Frustración acumulada en últimas N interacciones.
        trust_direction < 0, copy_paste alto, was_blocked.
        """
        filtered = [i for i in interactions if student_id is None or i.get("student_id") == student_id]
        last = filtered[-last_n:] if len(filtered) >= last_n else filtered
        if not last:
            return 0.0
        score = 0.0
        for i in last:
            if i.get("was_blocked"):
                score += 0.3
            if i.get("copy_paste_score", 0) > 0.5:
                score += 0.2
            td = i.get("trust_direction")
            if td is not None and float(td) < 0:
                score += 0.25
        return min(1.0, score)

    def _bloom_decreasing(self, interactions: List[Dict], student_id: Optional[str] = None, last_n: int = 5) -> float:
        """
        Bloom decreciente en últimas preguntas → desmotivación.
        Devuelve 0-1: 1 = Bloom claramente bajando.
        """
        filtered = [i for i in interactions if student_id is None or i.get("student_id") == student_id]
        last = filtered[-last_n:] if len(filtered) >= last_n else filtered
        blooms = [i.get("bloom_level") or i.get("bloom") for i in last]
        blooms = [int(b) for b in blooms if b is not None]
        if len(blooms) < 3:
            return 0.0
        first_half = blooms[: len(blooms) // 2]
        second_half = blooms[len(blooms) // 2:]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        if avg_first <= avg_second:
            return 0.0
        drop = avg_first - avg_second
        return min(1.0, drop / 2)  # caída de 2 niveles → 1

    def _monosyllabic_ratio(self, interactions: List[Dict], student_id: Optional[str] = None) -> float:
        """Ratio de respuestas monosilábicas."""
        filtered = [i for i in interactions if student_id is None or i.get("student_id") == student_id]
        prompts = [i.get("prompt_raw") or i.get("prompt", "") for i in filtered]
        if not prompts:
            return 0.0
        mono_count = 0
        for p in prompts:
            p_lower = (p or "").strip().lower()
            if len(p_lower) < 20 and any(m in p_lower for m in self.monosyllabic):
                mono_count += 1
        return mono_count / len(prompts)

    def _irregular_hours(self, interactions: List[Dict], student_id: Optional[str] = None) -> float:
        """
        Horarios irregulares → desenganche.
        Mucha varianza en horas del día de interacción.
        """
        filtered = [i for i in interactions if student_id is None or i.get("student_id") == student_id]
        if len(filtered) < 5:
            return 0.0
        hours = []
        for i in filtered:
            ts = i.get("timestamp", "")
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                else:
                    dt = ts
                hours.append(dt.hour)
            except Exception:
                pass
        if len(hours) < 5:
            return 0.0
        mean_h = sum(hours) / len(hours)
        var = sum((h - mean_h) ** 2 for h in hours) / len(hours)
        # Varianza alta → horarios muy dispersos → riesgo
        return min(1.0, var / 100)  # var 100+ → 1

    def analyze(
        self,
        interactions: List[Dict],
        student_id: Optional[str] = None,
    ) -> DropoutPrediction:
        """
        Analiza interacciones y devuelve predicción de abandono.
        """
        factors: List[tuple] = []  # (score, label)

        tbm = self._time_between_messages(interactions, student_id)
        if tbm > 0.3:
            factors.append((tbm, "Tiempo entre mensajes aumentando"))

        frust = self._frustration_score(interactions, student_id)
        if frust > 0.3:
            factors.append((frust, "Frustración acumulada en últimas interacciones"))

        bloom_d = self._bloom_decreasing(interactions, student_id)
        if bloom_d > 0.2:
            factors.append((bloom_d, "Complejidad decreciente de preguntas (Bloom bajando)"))

        mono = self._monosyllabic_ratio(interactions, student_id)
        if mono > 0.2:
            factors.append((mono, "Respuestas monosilábicas frecuentes"))

        irreg = self._irregular_hours(interactions, student_id)
        if irreg > 0.4:
            factors.append((irreg, "Horarios irregulares de uso"))

        factors.sort(key=lambda x: x[0], reverse=True)
        top_factors = [f[1] for f in factors[:3]]

        # Riesgo ponderado
        weights = [0.25, 0.25, 0.2, 0.15, 0.15]
        scores = [tbm, frust, bloom_d, mono, irreg]
        risk = sum(w * s for w, s in zip(weights, scores))

        if risk > 0.6:
            horizon = "24h"
            intervention = (
                "Contactar al estudiante en las próximas 24h. "
                "Ofrecer sesión de tutoría o revisar si hay obstáculos técnicos o personales. "
                "Considerar ajustar scaffolding a modo más directo temporalmente."
            )
        elif risk > 0.4:
            horizon = "48h"
            intervention = (
                "Monitorizar en las próximas 48h. "
                "Enviar mensaje de apoyo recordando recursos. "
                "Verificar si el modo socrático está generando frustración."
            )
        else:
            horizon = "1 semana"
            intervention = "Sin intervención urgente. Mantener seguimiento habitual."

        return DropoutPrediction(
            dropout_risk=round(min(1.0, risk), 2),
            risk_factors=top_factors or ["Sin señales de riesgo detectadas"],
            recommended_intervention=intervention,
            time_horizon=horizon,
        )
