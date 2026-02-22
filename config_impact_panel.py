"""
PANEL DE IMPACTO CONFIGURACIONAL
═══════════════════════════════════════════════════════════════════════
Componente Streamlit para interaction_semiotics.py

"CUANDO ACTIVASTE EL MODO SOCRÁTICO, PASÓ ESTO."
═════════════════════════════════════════════════
Eje 4 del contrato CP25/152 convertido en interfaz de usuario.

QUÉ ES Y POR QUÉ IMPORTA:
──────────────────────────
La pieza pedagógicamente más valiosa de GENIE Learn no es el chatbot.
Es el bucle de retroalimentación entre la configuración del docente
y el comportamiento observable del estudiante.

Dimitriadis (2021) lo llama el loop de "human-centered actionability".
"The Teacher in the Loop" (Rodríguez-Triana et al., 2018) lo describe
como el ciclo de decisión docente. Pero ninguno de estos frameworks tiene
una interfaz que muestre ese bucle cerrado en tiempo real.

Este panel lo hace: registra cada cambio de configuración con su contexto
pre-cambio, espera N interacciones, y presenta el delta en las métricas
clave. No es un dashboard genérico — es la evidencia de que la intervención
docente tiene (o no tiene) efecto medible.

MÉTRICAS MONITORIZADAS:
────────────────────────
- Nivel Bloom medio (ventana pre vs. post)
- Score de autonomía epistémica (pre vs. post)
- Valor pedagógico medio de las interacciones (de interaction_semiotics)
- Proporción de actos de habla de exploración/hipótesis
- Latencia entre prompts (proxy de tiempo de reflexión)
- Tasa de copy-paste / gaming
- Distribución por modo epistémico (consumption/verification/exploration/delegation)

DISEÑO VISUAL:
──────────────
Para cada intervención docente:
  [Configuración activada] [Timestamp] [N interacciones analizadas]
  
  ANTES                    FLECHA           DESPUÉS
  Bloom: 2.1               →                Bloom: 3.4   (+1.3 ↑)
  Autonomía: 0.31          →                Autonomía: 0.49 (+0.18 ↑)
  Valor ped.: 0.28          →                Valor ped.: 0.61 (+0.33 ↑)
  
  [Interpretación automática en lenguaje de la teoría]

Autor: Diego Elvira Vásquez · Prototipo CP25/152 · Feb 2026
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from collections import defaultdict

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
class InteractionSnapshot:
    """
    Estado de un conjunto de métricas en un momento dado.
    Representa la 'fotografía' de las métricas antes o después de un cambio.
    """
    timestamp: str
    n_interactions: int

    # Métricas cognitivas
    bloom_mean: float = 0.0
    bloom_trend: str = "stable"         # "ascending" | "descending" | "stable"

    # Métricas de autonomía
    autonomy_score: float = 0.0
    epistemic_mode_distribution: dict = field(default_factory=dict)
    dominant_mode: str = "consumption"

    # Métricas semióticas
    pedagogical_value_mean: float = 0.0
    hypothesis_test_rate: float = 0.0   # fracción de prompts tipo hipótesis
    solution_request_rate: float = 0.0  # fracción de solicitudes directas
    grice_composite_mean: float = 0.0

    # Métricas conductuales
    copy_paste_mean: float = 0.0
    inter_prompt_latency_mean: float = 0.0  # segundos
    gaming_suspicion_mean: float = 0.0


@dataclass
class ConfigImpactEvent:
    """
    Un evento de cambio de configuración con sus métricas antes y después.
    """
    event_id: str
    timestamp: str
    teacher_id: str
    course_id: str
    student_id: str               # "all" si aplica a todo el curso

    # El cambio
    config_key: str
    config_old_value: object
    config_new_value: object
    human_readable: str           # "Modo socrático: OFF → ON"

    # Ventanas de análisis
    window_pre: int = 8           # N interacciones antes del cambio
    window_post: int = 8          # N interacciones después del cambio

    # Snapshots (se completan cuando hay suficientes datos)
    snapshot_pre: Optional[InteractionSnapshot] = None
    snapshot_post: Optional[InteractionSnapshot] = None

    # Estado del análisis
    status: str = "waiting"       # "waiting" | "partial" | "complete"
    interactions_since_change: int = 0

    # Interpretación automática
    interpretation: str = ""
    effect_direction: str = "unknown"   # "positive" | "negative" | "neutral" | "mixed"
    effect_magnitude: str = "unknown"   # "strong" | "moderate" | "weak" | "none"


# ═══════════════════════════════════════════════════════════════════════
# MOTOR DE IMPACTO CONFIGURACIONAL
# ═══════════════════════════════════════════════════════════════════════

class ConfigImpactTracker:
    """
    Registra cambios de configuración y computa el impacto en las métricas.

    Integración con el sistema existente:
    1. En cada cambio de configuración docente (sidebar Streamlit):
       tracker.record_config_change(...)
    2. En cada interacción del estudiante:
       tracker.record_interaction_metrics(...)
    3. Para renderizar el panel:
       render_config_impact_panel(tracker)
    """

    def __init__(self, window_size: int = 8):
        self.window_size = window_size
        self.events: list[ConfigImpactEvent] = []
        self.interaction_log: dict[str, list[dict]] = defaultdict(list)
        # student_id → lista de métricas por interacción

    # ──────────────────────────────────────────────────────────────────
    # REGISTRO
    # ──────────────────────────────────────────────────────────────────

    def record_config_change(
        self,
        teacher_id: str,
        course_id: str,
        student_id: str,
        config_key: str,
        old_value: object,
        new_value: object,
    ) -> ConfigImpactEvent:
        """
        Registra un cambio de configuración y toma el snapshot pre-cambio.
        """
        # Snaphot pre: últimas window_size interacciones antes del cambio
        log = self.interaction_log.get(student_id, [])
        pre_data = log[-self.window_size:] if len(log) >= self.window_size else log

        snapshot_pre = self._build_snapshot(pre_data) if pre_data else None

        # Descripción legible
        human_readable = self._make_human_readable(config_key, old_value, new_value)

        event = ConfigImpactEvent(
            event_id=f"{teacher_id}_{config_key}_{datetime.now().strftime('%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            teacher_id=teacher_id,
            course_id=course_id,
            student_id=student_id,
            config_key=config_key,
            config_old_value=old_value,
            config_new_value=new_value,
            human_readable=human_readable,
            window_pre=self.window_size,
            window_post=self.window_size,
            snapshot_pre=snapshot_pre,
            status="waiting",
        )

        self.events.append(event)
        return event

    def record_interaction_metrics(
        self,
        student_id: str,
        bloom_mean_last5: float,
        autonomy_score: float,
        epistemic_mode: str,
        pedagogical_value: float,
        hypothesis_test: bool,
        solution_request: bool,
        grice_composite: float,
        copy_paste_score: float,
        inter_prompt_latency: float,
        gaming_suspicion: float,
    ) -> None:
        """
        Registra las métricas de una interacción.
        Actualiza automáticamente los eventos pendientes de análisis.
        """
        self.interaction_log[student_id].append({
            "timestamp": datetime.now().isoformat(),
            "bloom_mean": bloom_mean_last5,
            "autonomy_score": autonomy_score,
            "epistemic_mode": epistemic_mode,
            "pedagogical_value": pedagogical_value,
            "hypothesis_test": hypothesis_test,
            "solution_request": solution_request,
            "grice_composite": grice_composite,
            "copy_paste": copy_paste_score,
            "latency": inter_prompt_latency,
            "gaming": gaming_suspicion,
        })

        # Actualizar eventos pendientes de este estudiante
        for event in self.events:
            if event.student_id not in (student_id, "all"):
                continue
            if event.status in ("complete",):
                continue

            event.interactions_since_change += 1

            if event.interactions_since_change >= self.window_size:
                # Suficientes datos post-cambio: completar el análisis
                log = self.interaction_log.get(student_id, [])

                # Localizar posición del evento en el log
                # (los datos registrados después del evento son los post)
                post_data = log[-self.window_size:]
                snapshot_post = self._build_snapshot(post_data)

                event.snapshot_post = snapshot_post
                event.status = "complete"
                event.interpretation = self._interpret(event)
                event.effect_direction, event.effect_magnitude = self._assess_effect(event)

            elif event.interactions_since_change >= 3:
                event.status = "partial"

    # ──────────────────────────────────────────────────────────────────
    # CONSTRUCCIÓN DE SNAPSHOTS
    # ──────────────────────────────────────────────────────────────────

    def _build_snapshot(self, data: list[dict]) -> InteractionSnapshot:
        """Construye un snapshot de métricas a partir de un conjunto de interacciones."""
        if not data:
            return InteractionSnapshot(
                timestamp=datetime.now().isoformat(),
                n_interactions=0,
            )

        n = len(data)

        bloom_values = [d["bloom_mean"] for d in data if "bloom_mean" in d]
        bloom_mean = sum(bloom_values) / n if bloom_values else 0.0

        # Trend de Bloom
        if len(bloom_values) >= 4:
            mid = len(bloom_values) // 2
            first_half = sum(bloom_values[:mid]) / max(mid, 1)
            second_half = sum(bloom_values[mid:]) / max(len(bloom_values) - mid, 1)
            delta = second_half - first_half
            bloom_trend = "ascending" if delta > 0.3 else ("descending" if delta < -0.3 else "stable")
        else:
            bloom_trend = "stable"

        autonomy_values = [d["autonomy_score"] for d in data if "autonomy_score" in d]
        autonomy_mean = sum(autonomy_values) / n if autonomy_values else 0.0

        # Distribución de modos epistémicos
        mode_counts = defaultdict(int)
        for d in data:
            mode_counts[d.get("epistemic_mode", "consumption")] += 1
        mode_dist = {k: v / n for k, v in mode_counts.items()}
        dominant = max(mode_counts, key=mode_counts.get) if mode_counts else "consumption"

        ped_values = [d["pedagogical_value"] for d in data if "pedagogical_value" in d]
        ped_mean = sum(ped_values) / len(ped_values) if ped_values else 0.0

        hyp_rate = sum(1 for d in data if d.get("hypothesis_test")) / n
        sol_rate = sum(1 for d in data if d.get("solution_request")) / n

        grice_values = [d["grice_composite"] for d in data if "grice_composite" in d]
        grice_mean = sum(grice_values) / len(grice_values) if grice_values else 0.0

        cp_values = [d["copy_paste"] for d in data if "copy_paste" in d]
        cp_mean = sum(cp_values) / len(cp_values) if cp_values else 0.0

        latency_values = [d["latency"] for d in data if "latency" in d]
        latency_mean = sum(latency_values) / len(latency_values) if latency_values else 60.0

        gaming_values = [d["gaming"] for d in data if "gaming" in d]
        gaming_mean = sum(gaming_values) / len(gaming_values) if gaming_values else 0.0

        return InteractionSnapshot(
            timestamp=datetime.now().isoformat(),
            n_interactions=n,
            bloom_mean=round(bloom_mean, 2),
            bloom_trend=bloom_trend,
            autonomy_score=round(autonomy_mean, 3),
            epistemic_mode_distribution=dict(mode_dist),
            dominant_mode=dominant,
            pedagogical_value_mean=round(ped_mean, 2),
            hypothesis_test_rate=round(hyp_rate, 2),
            solution_request_rate=round(sol_rate, 2),
            grice_composite_mean=round(grice_mean, 2),
            copy_paste_mean=round(cp_mean, 2),
            inter_prompt_latency_mean=round(latency_mean, 1),
            gaming_suspicion_mean=round(gaming_mean, 2),
        )

    # ──────────────────────────────────────────────────────────────────
    # INTERPRETACIÓN AUTOMÁTICA
    # ──────────────────────────────────────────────────────────────────

    def _interpret(self, event: ConfigImpactEvent) -> str:
        """
        Genera una interpretación pedagógica del impacto en lenguaje accesible.
        Usa el marco teórico apropiado para cada tipo de configuración.
        """
        if not event.snapshot_pre or not event.snapshot_post:
            return "Análisis pendiente."

        pre = event.snapshot_pre
        post = event.snapshot_post
        d_bloom = post.bloom_mean - pre.bloom_mean
        d_autonomy = post.autonomy_score - pre.autonomy_score
        d_ped = post.pedagogical_value_mean - pre.pedagogical_value_mean
        d_latency = post.inter_prompt_latency_mean - pre.inter_prompt_latency_mean

        config = event.config_key
        new_val = event.config_new_value

        # ── Interpretaciones por tipo de configuración ──
        if config == "socratic_scaffolding" and new_val is True:
            parts = []
            if d_bloom > 0.5:
                parts.append(
                    f"El nivel cognitivo medio subió +{d_bloom:.1f} niveles en Bloom "
                    f"({pre.bloom_mean:.1f} → {post.bloom_mean:.1f}). "
                    "El andamiaje socrático activó elaboración más profunda — "
                    "consistente con la hipótesis de esfuerzo deseable (Bjork, 1994)."
                )
            elif d_bloom < -0.3:
                parts.append(
                    f"El nivel cognitivo bajó {d_bloom:.1f} niveles. "
                    "Posible expertise reversal: el andamiaje socrático puede ser "
                    "contraproducente para estudiantes ya en fase emergente-autónoma "
                    "(Kalyuga et al., 2003)."
                )
            if d_latency > 20:
                parts.append(
                    f"La latencia entre prompts aumentó +{d_latency:.0f}s — "
                    "el estudiante reflexionó más antes de consultar."
                )
            if d_autonomy > 0.08:
                parts.append(
                    f"Autonomía epistémica: {pre.autonomy_score:.2f} → {post.autonomy_score:.2f} (+{d_autonomy:.2f}). "
                    "Señal de internalización progresiva (ZDP, Vygotsky 1978)."
                )
            if post.hypothesis_test_rate > pre.hypothesis_test_rate + 0.1:
                parts.append(
                    "Aumentó la proporción de pruebas de hipótesis propias: "
                    f"{pre.hypothesis_test_rate:.0%} → {post.hypothesis_test_rate:.0%}. "
                    "El estudiante está construyendo respuestas propias antes de consultar."
                )
            if post.solution_request_rate < pre.solution_request_rate - 0.1:
                parts.append(
                    f"Las solicitudes de solución directa bajaron: "
                    f"{pre.solution_request_rate:.0%} → {post.solution_request_rate:.0%}."
                )
            return " | ".join(parts) if parts else (
                "Sin cambio significativo detectable en las primeras interacciones. "
                "Recomendado: esperar 5+ interacciones más para evaluar el efecto."
            )

        elif config == "socratic_scaffolding" and new_val is False:
            if d_bloom > 0.3:
                return (
                    f"Efecto paradójico: al retirar el socrático, Bloom subió +{d_bloom:.1f}. "
                    "El andamiaje anterior era posiblemente contraproducente "
                    "(expertise reversal — Kalyuga, 2003). El estudiante rinde mejor "
                    "con acceso directo a la información en su nivel actual."
                )
            elif d_bloom < -0.3:
                return (
                    f"El nivel cognitivo bajó {d_bloom:.1f} niveles al desactivar el socrático. "
                    "El andamiaje era funcionalmente necesario — el estudiante dependía "
                    "de las preguntas guía para estructurar su pensamiento."
                )
            return "Impacto neutral al desactivar el modo socrático."

        elif config == "max_daily_prompts":
            if d_bloom > 0.4:
                return (
                    f"Al ajustar el límite de prompts, el nivel Bloom subió +{d_bloom:.1f}. "
                    "La restricción fomentó una elaboración más cuidadosa antes de consultar "
                    "(consistent con el principio de reducción de disponibilidad, Bjork 1994)."
                )
            elif d_bloom < -0.3 and isinstance(event.config_new_value, int) and \
                 isinstance(event.config_old_value, int) and \
                 event.config_new_value < event.config_old_value:
                return (
                    "La reducción del límite de prompts bajó el nivel cognitivo. "
                    "El estudiante puede estar usando prompts de baja calidad para "
                    "'gastar' su cuota antes de perderla (gaming adaptativo)."
                )
            return f"Ajuste del límite de prompts sin efecto significativo en Bloom o autonomía."

        elif config == "hallucination_rate":
            if d_latency > 30 and post.grice_composite_mean > pre.grice_composite_mean:
                return (
                    f"La tasa de alucinaciones aumentó la reflexión: latencia +{d_latency:.0f}s, "
                    f"Grice composite {pre.grice_composite_mean:.2f} → {post.grice_composite_mean:.2f}. "
                    "El estudiante está verificando activamente — señal de lectura crítica desarrollada."
                )
            return f"Impacto de la tasa de alucinaciones no concluyente todavía."

        else:
            parts = []
            if abs(d_bloom) > 0.3:
                dir_bloom = "subió" if d_bloom > 0 else "bajó"
                parts.append(f"Bloom {dir_bloom} {abs(d_bloom):.1f} niveles.")
            if abs(d_autonomy) > 0.05:
                dir_a = "aumentó" if d_autonomy > 0 else "bajó"
                parts.append(f"Autonomía {dir_a} {abs(d_autonomy):.2f}.")
            return " ".join(parts) if parts else "Sin efecto significativo detectable."

    def _assess_effect(self, event: ConfigImpactEvent) -> tuple[str, str]:
        """Clasifica la dirección y magnitud del efecto."""
        if not event.snapshot_pre or not event.snapshot_post:
            return "unknown", "unknown"

        pre = event.snapshot_pre
        post = event.snapshot_post

        d_bloom = post.bloom_mean - pre.bloom_mean
        d_autonomy = post.autonomy_score - pre.autonomy_score
        d_ped = post.pedagogical_value_mean - pre.pedagogical_value_mean

        positive_signals = sum([d_bloom > 0.3, d_autonomy > 0.05, d_ped > 0.1])
        negative_signals = sum([d_bloom < -0.3, d_autonomy < -0.05, d_ped < -0.1])

        if positive_signals > negative_signals:
            direction = "positive"
        elif negative_signals > positive_signals:
            direction = "negative"
        elif positive_signals > 0 and negative_signals > 0:
            direction = "mixed"
        else:
            direction = "neutral"

        max_change = max(abs(d_bloom) / 3.0, abs(d_autonomy) * 2, abs(d_ped))
        if max_change > 0.5:
            magnitude = "strong"
        elif max_change > 0.25:
            magnitude = "moderate"
        elif max_change > 0.10:
            magnitude = "weak"
        else:
            magnitude = "none"

        return direction, magnitude

    # ──────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _make_human_readable(config_key: str, old_val: object, new_val: object) -> str:
        labels = {
            "socratic_scaffolding": "Modo socrático",
            "no_direct_solutions": "Bloqueo de soluciones",
            "max_daily_prompts": "Límite diario de prompts",
            "hallucination_rate": "Tasa de alucinaciones",
            "response_length": "Longitud de respuesta",
        }
        label = labels.get(config_key, config_key)
        if isinstance(new_val, bool):
            return f"{label}: {'OFF' if not old_val else 'ON'} → {'ON' if new_val else 'OFF'}"
        return f"{label}: {old_val} → {new_val}"


# ═══════════════════════════════════════════════════════════════════════
# COMPONENTE STREAMLIT
# ═══════════════════════════════════════════════════════════════════════

def render_config_impact_panel(
    tracker: ConfigImpactTracker,
    max_events: int = 5,
) -> None:
    """
    Renderiza el panel de impacto configuracional en Streamlit.

    Panel principal del Eje 4 del contrato.
    Cierra el bucle ante Bote-Lorenzo y Asensio-Pérez:
    "No solo configuramos el chatbot — medimos si las configuraciones
    tienen el efecto pedagógico que esperábamos."

    Uso en app.py:
        from config_impact_panel import ConfigImpactTracker, render_config_impact_panel
        render_config_impact_panel(st.session_state.impact_tracker)
    """
    if not STREAMLIT_AVAILABLE:
        raise ImportError("Streamlit requerido")

    complete_events = [e for e in tracker.events if e.status == "complete"]
    waiting_events = [e for e in tracker.events if e.status == "waiting"]
    partial_events = [e for e in tracker.events if e.status == "partial"]

    if not tracker.events:
        st.info(
            "Panel de impacto configuracional activo. "
            "Cuando cambies una configuración en el sidebar, aquí verás "
            "el efecto en las métricas del estudiante."
        )
        return

    # ── Resumen rápido ──
    c1, c2, c3 = st.columns(3)
    c1.metric("Intervenciones analizadas", len(complete_events))
    c2.metric("En proceso (datos insuficientes)", len(partial_events) + len(waiting_events))
    if complete_events:
        positive = sum(1 for e in complete_events if e.effect_direction == "positive")
        c3.metric("Con efecto positivo", f"{positive}/{len(complete_events)}")

    st.divider()

    # ── Eventos completados (los más recientes primero) ──
    shown = 0
    for event in reversed(tracker.events):
        if shown >= max_events:
            break

        with st.container():
            # Cabecera del evento
            status_icon = {"complete": "✅", "partial": "🔄", "waiting": "⏳"}.get(event.status, "?")
            effect_icon = {
                "positive": "🟢", "negative": "🔴",
                "mixed": "🟡", "neutral": "⚪", "unknown": "❓"
            }.get(event.effect_direction, "❓")

            cols = st.columns([3, 1, 1, 2])
            cols[0].markdown(f"**{event.human_readable}**")
            cols[1].markdown(f"{status_icon} {event.status}")
            cols[2].markdown(f"{effect_icon} {event.effect_magnitude}")
            cols[3].markdown(
                f"*{event.timestamp[:16].replace('T', ' ')}*  "
                f"({event.interactions_since_change} int. post-cambio)"
            )

            if event.status == "complete" and event.snapshot_pre and event.snapshot_post:
                _render_impact_comparison(event)
            elif event.status == "partial":
                st.caption(
                    f"Análisis parcial con {event.interactions_since_change}/{event.window_post} "
                    "interacciones post-cambio. El panel se completará automáticamente."
                )
            else:
                st.caption(
                    f"Esperando {event.window_post - event.interactions_since_change} "
                    "interacciones más para análisis completo."
                )

        st.divider()
        shown += 1


def _render_impact_comparison(event: ConfigImpactEvent) -> None:
    """Renderiza la comparación antes/después de un evento."""
    pre = event.snapshot_pre
    post = event.snapshot_post

    if not STREAMLIT_AVAILABLE:
        return

    # Tabla de comparación compacta
    col_before, col_arrow, col_after = st.columns([5, 1, 5])

    with col_before:
        st.markdown("**Antes**")
        _metric_row("Bloom medio", pre.bloom_mean, None, fmt=".1f", max_val=6.0)
        _metric_row("Autonomía", pre.autonomy_score, None, fmt=".2f", max_val=1.0)
        _metric_row("Valor pedagógico", pre.pedagogical_value_mean, None, fmt=".2f", max_val=1.0)
        _metric_row("Latencia (s)", pre.inter_prompt_latency_mean, None, fmt=".0f")
        _metric_row("Solicitudes directas", pre.solution_request_rate, None, fmt=".0%", invert=True)

    with col_arrow:
        st.markdown("<div style='text-align:center; font-size:24px; padding-top:40px'>→</div>",
                    unsafe_allow_html=True)

    with col_after:
        st.markdown("**Después**")
        _metric_row("Bloom medio", post.bloom_mean,
                    post.bloom_mean - pre.bloom_mean, fmt=".1f", max_val=6.0)
        _metric_row("Autonomía", post.autonomy_score,
                    post.autonomy_score - pre.autonomy_score, fmt=".2f", max_val=1.0)
        _metric_row("Valor pedagógico", post.pedagogical_value_mean,
                    post.pedagogical_value_mean - pre.pedagogical_value_mean, fmt=".2f", max_val=1.0)
        _metric_row("Latencia (s)", post.inter_prompt_latency_mean,
                    post.inter_prompt_latency_mean - pre.inter_prompt_latency_mean, fmt=".0f")
        _metric_row("Solicitudes directas", post.solution_request_rate,
                    post.solution_request_rate - pre.solution_request_rate, fmt=".0%", invert=True)

    # Gráfica de barras comparativa si Plotly disponible
    if PLOTLY_AVAILABLE:
        _render_comparison_chart(event)

    # Interpretación automática
    if event.interpretation:
        color = {
            "positive": "#e8f5e9", "negative": "#ffebee",
            "mixed": "#fff8e1", "neutral": "#f5f5f5"
        }.get(event.effect_direction, "#f5f5f5")

        st.markdown(
            f"<div style='background:{color}; padding:10px; border-radius:6px; "
            f"font-size:13px; margin-top:8px;'>"
            f"📊 <b>Interpretación</b>: {event.interpretation}"
            f"</div>",
            unsafe_allow_html=True,
        )


def _metric_row(
    label: str,
    value: float,
    delta: Optional[float],
    fmt: str = ".2f",
    max_val: float = None,
    invert: bool = False,
) -> None:
    """Renderiza una fila de métrica con delta opcional."""
    if not STREAMLIT_AVAILABLE:
        return

    val_str = f"{value:{fmt}}"
    if max_val:
        pct = value / max_val
        bar_width = int(pct * 60)
        bar = "█" * bar_width + "░" * (60 - bar_width)
        val_str += f"  `{bar[:15]}`"

    if delta is not None:
        sign = "+" if delta >= 0 else ""
        delta_str = f"{sign}{delta:{fmt}}"
        is_good = (delta > 0) != invert  # invert para métricas donde menos es mejor
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        color = "#2e7d32" if is_good else "#c62828"
        st.markdown(
            f"**{label}**: {value:{fmt}} "
            f"<span style='color:{color}'>({delta_str} {arrow})</span>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(f"**{label}**: {value:{fmt}}")


def _render_comparison_chart(event: ConfigImpactEvent) -> None:
    """Gráfica radar/bar de comparación antes-después."""
    if not PLOTLY_AVAILABLE or not STREAMLIT_AVAILABLE:
        return

    pre = event.snapshot_pre
    post = event.snapshot_post

    metrics = ["Bloom (norm.)", "Autonomía", "Valor ped.", "Pruebas hipótesis", "Grice"]
    pre_vals = [
        pre.bloom_mean / 6.0,
        pre.autonomy_score,
        pre.pedagogical_value_mean,
        pre.hypothesis_test_rate,
        pre.grice_composite_mean,
    ]
    post_vals = [
        post.bloom_mean / 6.0,
        post.autonomy_score,
        post.pedagogical_value_mean,
        post.hypothesis_test_rate,
        post.grice_composite_mean,
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Antes",
        x=metrics, y=pre_vals,
        marker_color="rgba(158,158,158,0.6)",
        marker_line_color="rgba(158,158,158,0.9)",
        marker_line_width=1,
    ))
    fig.add_trace(go.Bar(
        name="Después",
        x=metrics, y=post_vals,
        marker_color="rgba(66,165,245,0.7)",
        marker_line_color="rgba(66,165,245,1)",
        marker_line_width=1,
    ))

    fig.update_layout(
        height=220,
        barmode="group",
        margin=dict(l=10, r=10, t=20, b=30),
        yaxis=dict(range=[0, 1.1], tickformat=".1f", showgrid=True),
        legend=dict(orientation="h", y=1.1),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
    )

    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# SIMULACIÓN PARA DEMO
# ═══════════════════════════════════════════════════════════════════════

def _generate_demo_tracker() -> ConfigImpactTracker:
    """Genera un tracker con datos de demo para mostrar el panel sin estudiantes reales."""
    import random
    random.seed(7)

    tracker = ConfigImpactTracker(window_size=8)

    # Evento 1: activa modo socrático — efecto positivo claro
    evt1 = tracker.record_config_change(
        "T001", "PROG101", "S001", "socratic_scaffolding", False, True
    )
    # Simular datos pre (bajo engagement)
    evt1.snapshot_pre = InteractionSnapshot(
        timestamp="2026-02-21T09:00:00",
        n_interactions=8,
        bloom_mean=1.8, bloom_trend="stable",
        autonomy_score=0.22,
        epistemic_mode_distribution={"consumption": 0.7, "delegation": 0.3},
        dominant_mode="consumption",
        pedagogical_value_mean=0.21, hypothesis_test_rate=0.05,
        solution_request_rate=0.65, grice_composite_mean=0.30,
        copy_paste_mean=0.55, inter_prompt_latency_mean=12.0,
        gaming_suspicion_mean=0.35,
    )
    # Simular datos post (engagement mejorado)
    evt1.snapshot_post = InteractionSnapshot(
        timestamp="2026-02-21T10:30:00",
        n_interactions=8,
        bloom_mean=3.4, bloom_trend="ascending",
        autonomy_score=0.48,
        epistemic_mode_distribution={"verification": 0.45, "exploration": 0.30, "consumption": 0.25},
        dominant_mode="verification",
        pedagogical_value_mean=0.62, hypothesis_test_rate=0.38,
        solution_request_rate=0.20, grice_composite_mean=0.58,
        copy_paste_mean=0.18, inter_prompt_latency_mean=64.0,
        gaming_suspicion_mean=0.08,
    )
    evt1.interactions_since_change = 8
    evt1.status = "complete"
    evt1.interpretation = tracker._interpret(evt1)
    evt1.effect_direction, evt1.effect_magnitude = tracker._assess_effect(evt1)

    # Evento 2: reduce límite de prompts — efecto mixto
    evt2 = tracker.record_config_change(
        "T001", "PROG101", "S002", "max_daily_prompts", 15, 8
    )
    evt2.snapshot_pre = InteractionSnapshot(
        timestamp="2026-02-21T11:00:00",
        n_interactions=8,
        bloom_mean=2.1, bloom_trend="stable",
        autonomy_score=0.29,
        epistemic_mode_distribution={"consumption": 0.60, "delegation": 0.40},
        dominant_mode="consumption",
        pedagogical_value_mean=0.30, hypothesis_test_rate=0.10,
        solution_request_rate=0.70, grice_composite_mean=0.35,
        copy_paste_mean=0.40, inter_prompt_latency_mean=8.0,
        gaming_suspicion_mean=0.20,
    )
    evt2.snapshot_post = InteractionSnapshot(
        timestamp="2026-02-21T12:00:00",
        n_interactions=8,
        bloom_mean=2.8, bloom_trend="ascending",
        autonomy_score=0.35,
        epistemic_mode_distribution={"verification": 0.40, "consumption": 0.50, "delegation": 0.10},
        dominant_mode="consumption",
        pedagogical_value_mean=0.45, hypothesis_test_rate=0.22,
        solution_request_rate=0.45, grice_composite_mean=0.44,
        copy_paste_mean=0.25, inter_prompt_latency_mean=42.0,
        gaming_suspicion_mean=0.15,
    )
    evt2.interactions_since_change = 8
    evt2.status = "complete"
    evt2.interpretation = tracker._interpret(evt2)
    evt2.effect_direction, evt2.effect_magnitude = tracker._assess_effect(evt2)

    # Evento 3: en espera
    evt3 = tracker.record_config_change(
        "T001", "PROG101", "S001", "hallucination_rate", 0.0, 0.1
    )
    evt3.interactions_since_change = 3
    evt3.status = "partial"

    return tracker


if __name__ == "__main__":
    tracker = _generate_demo_tracker()
    print("Demo tracker generado:")
    for e in tracker.events:
        print(f"  [{e.status}] {e.human_readable} → {e.effect_direction} / {e.effect_magnitude}")
        if e.interpretation:
            print(f"    {e.interpretation[:100]}...")
