"""
EFFECT LATENCY ANALYZER — Cuánto Tarda en Funcionar
═══════════════════════════════════════════════════════════════════════
Extensión de config_impact_panel.py — BLOQUE 1

"No solo medimos si una configuración funciona.
 Medimos cuánto tarda en funcionar — y esa latencia contiene
 información sobre la profundidad del cambio cognitivo que produce."

PROBLEMA:
─────────
El config_impact_panel.py mide el DELTA (magnitud). No mide la
LATENCIA: cuántas interacciones pasan entre cambio y estabilización.

ALGORITMO:
──────────
En lugar de dos snapshots fijos (pre/post), calcula delta en ventanas
deslizantes 1, 2, 3, ... N interacciones post-cambio.
  onset_n = N donde delta cruza umbral de significancia
  stabilization_n = N donde delta deja de crecer (plateau)

TIPOS DE EFECTO:
────────────────
  Inmediato (onset≤1): restricción mecánica, no cambio cognitivo
  Gradual (onset 2-4): adaptación estratégica del estudiante
  Retardado (onset≥5): reestructuración epistémica genuina

FUNDAMENTO:
───────────
  Bjork (1994): desirable difficulties → latencia mayor = cambio profundo
  Kapur (2008): productive failure → latencia socrática es esperada
  Vygotsky (1978): la ZDP no se activa instantáneamente
  Perry (1970): transiciones epistémicas tienen latencia inherente

Autor: Diego Elvira Vásquez · CP25/152 GSIC/EMIC · Feb 2026
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import math

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# MODELO DE DATOS
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EffectLatencyResult:
    """Resultado de latencia para un evento × una métrica."""
    event_id: str = ""
    config_key: str = ""
    onset_n: int = -1
    stabilization_n: int = -1
    effect_curve: List[float] = field(default_factory=list)
    metric_name: str = "bloom_mean"
    effect_type: str = "unknown"
    cognitive_depth_inference: str = ""
    n_interactions_available: int = 0
    sufficient_data: bool = False
    baseline_value: float = 0.0
    final_value: float = 0.0
    max_delta: float = 0.0
    onset_threshold: float = 0.0
    stability_threshold: float = 0.0


@dataclass
class MultiMetricLatency:
    """Latencia por múltiples métricas para un mismo evento."""
    event_id: str = ""
    config_key: str = ""
    human_readable: str = ""
    results: Dict[str, EffectLatencyResult] = field(default_factory=dict)
    dominant_metric: str = ""
    slowest_metric: str = ""
    summary_type: str = ""
    latency_spread: int = 0


# ═══════════════════════════════════════════════════════════════════════
# MOTOR DE LATENCIA
# ═══════════════════════════════════════════════════════════════════════

class EffectLatencyAnalyzer:
    """
    Analiza la latencia de efecto de cambios de configuración pedagógica.
    Se conecta al ConfigImpactTracker existente o usa datos sintéticos.
    """

    DEFAULT_ONSET_THRESHOLD = 0.15
    DEFAULT_STABILITY_THRESHOLD = 0.05
    MAX_WINDOW = 20
    PLATEAU_CONFIRMATION = 2

    DEFAULT_METRICS = [
        "bloom_mean", "autonomy_score", "pedagogical_value",
        "latency", "copy_paste",
    ]

    METRIC_LABELS = {
        "bloom_mean": "Nivel Bloom medio",
        "autonomy_score": "Autonomía epistémica",
        "pedagogical_value": "Valor pedagógico",
        "latency": "Latencia entre prompts",
        "copy_paste": "Tasa copy-paste",
        "gaming": "Sospecha gaming",
        "grice_composite": "Calidad Grice",
    }

    def __init__(self, impact_tracker=None):
        self.tracker = impact_tracker
        self._synthetic_data: Dict[str, Dict[str, List[float]]] = {}

    # ──────────────────────────────────────────────────────────────────
    # ANÁLISIS PRINCIPAL
    # ──────────────────────────────────────────────────────────────────

    def compute_effect_latency(
        self,
        event_id: str,
        metric: str = "bloom_mean",
        onset_threshold: float = None,
        stability_threshold: float = None,
    ) -> EffectLatencyResult:
        """
        Computa latencia: ventanas deslizantes → onset + estabilización.
        """
        onset_t = onset_threshold or self.DEFAULT_ONSET_THRESHOLD
        stab_t = stability_threshold or self.DEFAULT_STABILITY_THRESHOLD

        curve, baseline = self._extract_effect_curve(event_id, metric)

        if not curve or len(curve) < 3:
            return EffectLatencyResult(
                event_id=event_id,
                config_key=self._get_config_key(event_id),
                metric_name=metric,
                effect_type="null",
                sufficient_data=False,
                n_interactions_available=len(curve),
                cognitive_depth_inference="Datos insuficientes (< 3 interacciones post-cambio).",
            )

        # --- Onset: primer W donde |delta| >= umbral ---
        onset_n = -1
        for i, delta in enumerate(curve):
            if abs(delta) >= onset_t:
                onset_n = i + 1
                break

        # --- Estabilización: plateau confirmado ---
        stabilization_n = -1
        if onset_n > 0 and onset_n < len(curve):
            consecutive_stable = 0
            for i in range(onset_n, len(curve)):
                if i > 0 and abs(curve[i] - curve[i - 1]) < stab_t:
                    consecutive_stable += 1
                    if consecutive_stable >= self.PLATEAU_CONFIRMATION:
                        stabilization_n = i + 1 - self.PLATEAU_CONFIRMATION + 1
                        break
                else:
                    consecutive_stable = 0

            # Fallback: cola estable
            if stabilization_n < 0 and len(curve) >= onset_n + 3:
                tail = curve[-3:]
                if all(abs(tail[j] - tail[j - 1]) < stab_t for j in range(1, len(tail))):
                    stabilization_n = len(curve) - 2

        effect_type = self._classify_effect_type(onset_n, stabilization_n)
        config_key = self._get_config_key(event_id)
        inference = self._infer_cognitive_depth(
            effect_type, onset_n, stabilization_n, config_key, metric
        )

        max_delta = max(abs(d) for d in curve) if curve else 0.0
        final_value = baseline + curve[-1] if curve else baseline

        return EffectLatencyResult(
            event_id=event_id,
            config_key=config_key,
            onset_n=onset_n,
            stabilization_n=stabilization_n,
            effect_curve=curve,
            metric_name=metric,
            effect_type=effect_type,
            cognitive_depth_inference=inference,
            n_interactions_available=len(curve),
            sufficient_data=True,
            baseline_value=round(baseline, 3),
            final_value=round(final_value, 3),
            max_delta=round(max_delta, 3),
            onset_threshold=onset_t,
            stability_threshold=stab_t,
        )

    def compute_multi_metric_latency(
        self,
        event_id: str,
        metrics: List[str] = None,
    ) -> MultiMetricLatency:
        """Latencia multi-métrica: revela divergencia entre dimensiones."""
        metrics = metrics or self.DEFAULT_METRICS
        results = {}
        for m in metrics:
            results[m] = self.compute_effect_latency(event_id, metric=m)

        valid = {k: v for k, v in results.items()
                 if v.onset_n > 0 and v.sufficient_data}

        dominant = slowest = ""
        spread = 0
        if valid:
            dominant = min(valid, key=lambda k: valid[k].onset_n)
            slowest = max(valid, key=lambda k: valid[k].onset_n)
            spread = valid[slowest].onset_n - valid[dominant].onset_n

        types = [v.effect_type for v in valid.values()] if valid else []
        if not types or all(t == "null" for t in types):
            summary = "no_effect"
        elif all(t == "immediate" for t in types):
            summary = "mechanical_restriction"
        elif "delayed" in types:
            summary = "deep_restructuring"
        else:
            summary = "gradual_adaptation"

        return MultiMetricLatency(
            event_id=event_id,
            config_key=self._get_config_key(event_id),
            human_readable=self._get_human_readable(event_id),
            results=results,
            dominant_metric=dominant,
            slowest_metric=slowest,
            summary_type=summary,
            latency_spread=spread,
        )

    # ──────────────────────────────────────────────────────────────────
    # CLASIFICACIÓN E INFERENCIA
    # ──────────────────────────────────────────────────────────────────

    def _classify_effect_type(self, onset: int, stabilization: int) -> str:
        if onset < 0:
            return "null"
        elif onset <= 1:
            return "immediate"
        elif onset <= 4:
            return "gradual"
        else:
            return "delayed"

    def _infer_cognitive_depth(self, effect_type, onset, stabilization, config_key, metric):
        ml = self.METRIC_LABELS.get(metric, metric)
        stab = f", estab. interacción {stabilization}" if stabilization > 0 else ""

        base = {
            "null": (
                f"Sin efecto detectable en {ml}. Hipótesis: (a) la configuración "
                f"no afecta esta métrica, o (b) requiere más interacciones "
                f"(Bjork 1994: dificultades deseables producen latencia invisible)."
            ),
            "immediate": (
                f"Efecto inmediato en {ml} (onset={onset}{stab}). Restricción "
                f"mecánica, no cambio cognitivo: el sistema IMPIDE un comportamiento. "
                f"Vygotsky (1978): regulación externa precede autorregulación."
            ),
            "gradual": (
                f"Efecto gradual en {ml} (onset={onset}{stab}). El estudiante "
                f"recalibra expectativas sobre qué respuesta recibirá. "
                f"Kapur (2008): productive failure en miniatura. "
                f"Wood et al. (1976): scaffolding itera hasta calibrar ZDP."
            ),
            "delayed": (
                f"Efecto retardado en {ml} (onset={onset}{stab}). "
                f"Reestructuración epistémica: el estudiante modifica su modelo "
                f"de cómo funciona el sistema. Perry (1970): transición entre "
                f"posiciones epistémicas. Latencia alta = profundidad alta."
            ),
        }.get(effect_type, "")

        notes = {
            "socratic_scaffolding": " [Socrático] Latencia esperada 2-6: aceptar frustración de no-respuesta.",
            "max_daily_prompts": " [Límite] Efecto mecánico. Pregunta real: latencia en métricas cognitivas.",
            "hallucination_rate": " [Alucinación] Latencia esperada 5-12: encontrar→dudar→verificar→generalizar.",
            "no_direct_solutions": " [Bloqueo] Dual: inmediato en solicitudes, gradual en Bloom/autonomía.",
            "block_direct_solutions": " [Bloqueo] Dual: inmediato en solicitudes, gradual en cognición.",
        }
        return base + notes.get(config_key, "")

    # ──────────────────────────────────────────────────────────────────
    # EXTRACCIÓN DE DATOS
    # ──────────────────────────────────────────────────────────────────

    def _extract_effect_curve(self, event_id, metric):
        # Datos sintéticos
        if event_id in self._synthetic_data and metric in self._synthetic_data[event_id]:
            values = self._synthetic_data[event_id][metric]
            baseline = values[0]
            curve = []
            for w in range(1, len(values)):
                window_mean = sum(values[1:w + 1]) / w
                curve.append(round(window_mean - baseline, 4))
            return curve, baseline

        # Datos del tracker real
        if self.tracker:
            return self._extract_from_tracker(event_id, metric)

        return [], 0.0

    def _extract_from_tracker(self, event_id, metric):
        if not self.tracker:
            return [], 0.0

        event = None
        for e in self.tracker.events:
            if e.event_id == event_id:
                event = e
                break
        if not event or not event.snapshot_pre:
            return [], 0.0

        baseline = getattr(event.snapshot_pre, metric, 0.0)
        log = self.tracker.interaction_log.get(event.student_id, [])
        if not log:
            return [], baseline

        change_idx = len(log)
        for i, entry in enumerate(log):
            if entry.get("timestamp", "") >= event.timestamp:
                change_idx = i
                break

        post_log = log[change_idx:]
        curve = []
        for w in range(1, min(len(post_log) + 1, self.MAX_WINDOW + 1)):
            vals = [entry.get(metric, 0.0) for entry in post_log[:w]]
            curve.append(round(sum(vals) / len(vals) - baseline, 4))
        return curve, baseline

    def _get_config_key(self, event_id):
        if self.tracker:
            for e in self.tracker.events:
                if e.event_id == event_id:
                    return e.config_key
        return self._synthetic_data.get(event_id, {}).get("_config_key", "unknown")

    def _get_human_readable(self, event_id):
        if self.tracker:
            for e in self.tracker.events:
                if e.event_id == event_id:
                    return e.human_readable
        return self._synthetic_data.get(event_id, {}).get("_human_readable", event_id)

    # ──────────────────────────────────────────────────────────────────
    # REGISTRO SINTÉTICO + DEMO
    # ──────────────────────────────────────────────────────────────────

    def register_synthetic_data(self, event_id, config_key, human_readable, metric, values):
        if event_id not in self._synthetic_data:
            self._synthetic_data[event_id] = {"_config_key": config_key, "_human_readable": human_readable}
        self._synthetic_data[event_id][metric] = values

    def generate_demo_data(self) -> Dict[str, MultiMetricLatency]:
        """
        Tres patrones canónicos:
        1. Socrático: gradual (onset=2, estab=6)
        2. Límite prompts: inmediato (onset=1, estab=1)
        3. Alucinación: retardado (onset=5, estab=12)
        """
        # ── 1. SOCRÁTICO — Gradual ──
        _s = ("evt_socratic", "socratic_scaffolding", "Modo socrático: OFF → ON")
        self.register_synthetic_data(*_s, "bloom_mean",
            [2.1, 2.1, 2.4, 2.8, 3.1, 3.3, 3.5, 3.5, 3.4, 3.5, 3.5, 3.6, 3.5])
        self.register_synthetic_data(*_s, "autonomy_score",
            [0.22, 0.22, 0.24, 0.26, 0.32, 0.38, 0.42, 0.46, 0.48, 0.48, 0.49, 0.48, 0.49])
        self.register_synthetic_data(*_s, "pedagogical_value",
            [0.21, 0.23, 0.35, 0.42, 0.50, 0.56, 0.60, 0.61, 0.60, 0.62, 0.61, 0.60, 0.61])
        self.register_synthetic_data(*_s, "latency",
            [12.0, 18.0, 35.0, 48.0, 55.0, 60.0, 62.0, 61.0, 63.0, 60.0, 62.0, 61.0, 62.0])
        self.register_synthetic_data(*_s, "copy_paste",
            [0.55, 0.50, 0.38, 0.28, 0.22, 0.18, 0.15, 0.14, 0.15, 0.14, 0.15, 0.14, 0.14])

        # ── 2. LÍMITE PROMPTS — Inmediato ──
        _p = ("evt_prompt_limit", "max_daily_prompts", "Límite prompts: 20 → 8")
        self.register_synthetic_data(*_p, "bloom_mean",
            [2.1, 2.5, 2.6, 2.6, 2.7, 2.6, 2.7, 2.6, 2.7, 2.6, 2.7, 2.6, 2.7])
        self.register_synthetic_data(*_p, "autonomy_score",
            [0.29, 0.38, 0.40, 0.39, 0.40, 0.40, 0.39, 0.40, 0.40, 0.39, 0.40, 0.40, 0.39])
        self.register_synthetic_data(*_p, "pedagogical_value",
            [0.30, 0.48, 0.50, 0.49, 0.50, 0.49, 0.50, 0.50, 0.49, 0.50, 0.50, 0.49, 0.50])
        self.register_synthetic_data(*_p, "latency",
            [8.0, 45.0, 48.0, 46.0, 47.0, 46.0, 48.0, 47.0, 46.0, 47.0, 48.0, 46.0, 47.0])
        self.register_synthetic_data(*_p, "copy_paste",
            [0.40, 0.22, 0.20, 0.21, 0.20, 0.20, 0.21, 0.20, 0.20, 0.21, 0.20, 0.20, 0.21])

        # ── 3. ALUCINACIÓN — Retardado ──
        _h = ("evt_hallucination", "hallucination_rate", "Alucinación deliberada: 0% → 15%")
        self.register_synthetic_data(*_h, "bloom_mean",
            [2.0, 2.0, 2.0, 2.1, 2.1, 2.3, 2.5, 2.7, 2.9, 3.1, 3.3, 3.4, 3.5, 3.5, 3.5, 3.6, 3.5])
        self.register_synthetic_data(*_h, "autonomy_score",
            [0.20, 0.20, 0.20, 0.21, 0.21, 0.22, 0.25, 0.30, 0.35, 0.40, 0.44, 0.47, 0.50, 0.51, 0.50, 0.51, 0.50])
        self.register_synthetic_data(*_h, "pedagogical_value",
            [0.20, 0.20, 0.21, 0.22, 0.23, 0.28, 0.34, 0.40, 0.46, 0.52, 0.56, 0.58, 0.60, 0.60, 0.61, 0.60, 0.60])
        self.register_synthetic_data(*_h, "latency",
            [15.0, 16.0, 15.0, 17.0, 18.0, 25.0, 32.0, 40.0, 50.0, 55.0, 58.0, 60.0, 62.0, 61.0, 62.0, 61.0, 62.0])
        self.register_synthetic_data(*_h, "copy_paste",
            [0.45, 0.45, 0.44, 0.42, 0.40, 0.35, 0.28, 0.22, 0.18, 0.15, 0.13, 0.12, 0.11, 0.11, 0.11, 0.11, 0.11])

        results = {}
        for eid in ["evt_socratic", "evt_prompt_limit", "evt_hallucination"]:
            results[eid] = self.compute_multi_metric_latency(eid)
        return results

    # ──────────────────────────────────────────────────────────────────
    # VISUALIZACIÓN PLOTLY
    # ──────────────────────────────────────────────────────────────────

    def render_latency_chart(self, result: EffectLatencyResult, height=300):
        """Gráfico de curva de efecto con onset y estabilización marcados."""
        if not PLOTLY_AVAILABLE:
            return None

        x = list(range(1, len(result.effect_curve) + 1))
        y = result.effect_curve
        ml = self.METRIC_LABELS.get(result.metric_name, result.metric_name)

        colors = {"immediate": "#43a047", "gradual": "#1e88e5", "delayed": "#e65100", "null": "#9e9e9e"}
        c = colors.get(result.effect_type, "#757575")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=f"Δ {ml}",
            line=dict(color=c, width=2.5), marker=dict(size=6, color=c),
            hovertemplate=f"Int %{{x}}<br>Δ {ml}: %{{y:.3f}}<extra></extra>",
        ))

        fig.add_hline(y=0, line_dash="dot", line_color="#bdbdbd", line_width=1)
        fig.add_hline(y=result.onset_threshold, line_dash="dash", line_color="#ff8a65", line_width=1,
                      annotation_text="Umbral onset", annotation_position="top right",
                      annotation_font=dict(size=9, color="#ff8a65"))
        if min(y) < 0:
            fig.add_hline(y=-result.onset_threshold, line_dash="dash", line_color="#ff8a65", line_width=1)

        if result.onset_n > 0:
            fig.add_vline(x=result.onset_n, line_dash="dot", line_color="#e53935", line_width=2,
                          annotation_text=f"Onset (n={result.onset_n})", annotation_position="top left",
                          annotation_font=dict(size=10, color="#e53935"),
                          annotation_bgcolor="rgba(255,255,255,0.8)")
        if result.stabilization_n > 0:
            fig.add_vline(x=result.stabilization_n, line_dash="dot", line_color="#1565c0", line_width=2,
                          annotation_text=f"Plateau (n={result.stabilization_n})", annotation_position="top right",
                          annotation_font=dict(size=10, color="#1565c0"),
                          annotation_bgcolor="rgba(255,255,255,0.8)")
        if result.onset_n > 0 and result.stabilization_n > 0:
            fig.add_vrect(x0=result.onset_n, x1=result.stabilization_n,
                          fillcolor=c, opacity=0.08, layer="below", line_width=0)

        icons = {"immediate": "⚡", "gradual": "📈", "delayed": "🐌", "null": "⊘"}
        fig.update_layout(
            title=dict(text=f"Latencia: {self._get_human_readable(result.event_id)} — "
                            f"{icons.get(result.effect_type, '')} {result.effect_type.title()}", font_size=13),
            xaxis_title="Interacciones post-cambio", yaxis_title=f"Δ {ml}",
            height=height, margin=dict(l=50, r=20, t=45, b=40),
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            xaxis=dict(dtick=1, showgrid=True, gridcolor="#eee"),
            yaxis=dict(showgrid=True, gridcolor="#eee"), showlegend=False,
        )
        return fig

    def render_multi_metric_chart(self, multi: MultiMetricLatency, height=420):
        """Gráfico superpuesto con métricas normalizadas [0-1]."""
        if not PLOTLY_AVAILABLE:
            return None

        fig = go.Figure()
        palette = {"bloom_mean": "#1e88e5", "autonomy_score": "#43a047",
                   "pedagogical_value": "#7b1fa2", "latency": "#ff8f00", "copy_paste": "#e53935"}

        for mn, result in multi.results.items():
            if not result.sufficient_data or not result.effect_curve:
                continue
            curve = result.effect_curve
            mx = max(abs(v) for v in curve) or 1.0
            norm = [v / mx for v in curve]
            x = list(range(1, len(norm) + 1))
            label = self.METRIC_LABELS.get(mn, mn)
            clr = palette.get(mn, "#757575")
            onset_txt = f" (onset={result.onset_n})" if result.onset_n > 0 else ""

            fig.add_trace(go.Scatter(
                x=x, y=norm, mode="lines+markers", name=f"{label}{onset_txt}",
                line=dict(color=clr, width=2), marker=dict(size=4, color=clr),
            ))
            if result.onset_n > 0 and result.onset_n - 1 < len(norm):
                fig.add_trace(go.Scatter(
                    x=[result.onset_n], y=[norm[result.onset_n - 1]],
                    mode="markers", marker=dict(size=12, color=clr, symbol="star",
                                                line=dict(width=1, color="white")),
                    showlegend=False,
                    hovertemplate=f"⭐ Onset {label}: int. {result.onset_n}<extra></extra>",
                ))

        stype = {"mechanical_restriction": "Mecánico", "gradual_adaptation": "Gradual",
                 "deep_restructuring": "Profundo", "no_effect": "Sin efecto"}
        fig.update_layout(
            title=dict(text=f"Multi-métrica: {multi.human_readable} — {stype.get(multi.summary_type, '')}", font_size=13),
            xaxis_title="Interacciones post-cambio", yaxis_title="Delta normalizado [0–1]",
            height=height, margin=dict(l=50, r=20, t=50, b=40),
            paper_bgcolor="white", plot_bgcolor="#fafafa",
            xaxis=dict(dtick=1, showgrid=True, gridcolor="#eee"),
            yaxis=dict(showgrid=True, gridcolor="#eee"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font_size=10),
        )
        return fig

    # ──────────────────────────────────────────────────────────────────
    # PANEL STREAMLIT
    # ──────────────────────────────────────────────────────────────────

    def render_streamlit_panel(self, demo=None):
        """Panel completo para la pestaña 🧠 Análisis Profundo del app.py."""
        if not STREAMLIT_AVAILABLE:
            return
        st.subheader("⏱️ Latencia de Decisión Pedagógica")
        st.caption("Cuántas interacciones tarda cada configuración en producir efecto. "
                   "La latencia revela la profundidad del cambio cognitivo.")

        if demo is None:
            demo = self.generate_demo_data()

        # Tabla resumen
        rows = []
        for eid, multi in demo.items():
            for mn, r in multi.results.items():
                if r.sufficient_data and r.onset_n > 0:
                    rows.append((multi.human_readable, self.METRIC_LABELS.get(mn, mn),
                                 r.onset_n, r.stabilization_n if r.stabilization_n > 0 else "—",
                                 r.effect_type, f"{r.max_delta:+.3f}"))
        if rows:
            st.markdown("##### Resumen")
            icons = {"immediate": "⚡", "gradual": "📈", "delayed": "🐌", "null": "⊘"}
            cols = st.columns([3, 2, 1, 1, 1, 1])
            for col, h in zip(cols, ["Config", "Métrica", "Onset", "Estab", "Tipo", "Δmáx"]):
                col.markdown(f"**{h}**")
            for row in rows:
                cols = st.columns([3, 2, 1, 1, 1, 1])
                cols[0].write(row[0]); cols[1].write(row[1]); cols[2].write(str(row[2]))
                cols[3].write(str(row[3])); cols[4].write(f"{icons.get(row[4], '?')} {row[4]}"); cols[5].write(row[5])
        st.divider()

        for eid, multi in demo.items():
            with st.expander(f"📊 {multi.human_readable} — {multi.summary_type.replace('_', ' ').title()}",
                             expanded=(eid == "evt_socratic")):
                fig_m = self.render_multi_metric_chart(multi)
                if fig_m:
                    st.plotly_chart(fig_m, use_container_width=True)

                st.markdown("##### Detalle por métrica")
                valid_metrics = [(m, r) for m, r in multi.results.items() if r.sufficient_data]
                tabs = st.tabs([self.METRIC_LABELS.get(m, m) for m, _ in valid_metrics])
                for tab, (mn, result) in zip(tabs, valid_metrics):
                    with tab:
                        fig = self.render_latency_chart(result, height=260)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        if result.cognitive_depth_inference:
                            bg = {"immediate": "#e8f5e9", "gradual": "#e3f2fd",
                                  "delayed": "#fff3e0", "null": "#f5f5f5"}.get(result.effect_type, "#f5f5f5")
                            st.markdown(f"<div style='background:{bg};padding:10px;border-radius:6px;"
                                        f"font-size:12px'>🧠 {result.cognitive_depth_inference}</div>",
                                        unsafe_allow_html=True)

                if multi.latency_spread > 0:
                    st.info(f"📐 **Divergencia**: {self.METRIC_LABELS.get(multi.dominant_metric, '')} "
                            f"reacciona {multi.latency_spread} interacciones antes que "
                            f"{self.METRIC_LABELS.get(multi.slowest_metric, '')}. "
                            f"El efecto conductual precede al efecto epistémico.")


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    analyzer = EffectLatencyAnalyzer()
    demo = analyzer.generate_demo_data()

    print("═" * 70)
    print("EFFECT LATENCY ANALYZER — Demo")
    print("═" * 70)
    print('\n"No solo medimos si una configuración funciona.')
    print(' Medimos cuánto tarda en funcionar."\n')

    icons = {"immediate": "⚡", "gradual": "📈", "delayed": "🐌", "null": "⊘"}
    for eid, multi in demo.items():
        print(f"{'─' * 70}")
        print(f"  {multi.human_readable}  [{multi.summary_type}]")
        if multi.latency_spread > 0:
            print(f"  Spread: {multi.latency_spread} int. entre métrica más rápida y más lenta")
        for mn, r in multi.results.items():
            if not r.sufficient_data: continue
            ic = icons.get(r.effect_type, "?")
            lb = analyzer.METRIC_LABELS.get(mn, mn)
            on = f"onset={r.onset_n}" if r.onset_n > 0 else "sin efecto"
            sb = f"estab={r.stabilization_n}" if r.stabilization_n > 0 else ""
            print(f"    {ic} {lb:.<30s} {on:>10s}  {sb:>10s}  Δmax={r.max_delta:+.3f}")
        print()

    print("═" * 70)
    print("  ⚡ Inmediato → restricción conductual")
    print("  📈 Gradual   → adaptación estratégica")
    print("  🐌 Retardado → reestructuración epistémica")
    print("═" * 70)
