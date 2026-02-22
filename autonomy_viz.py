"""
VISUALIZACIÓN DE TRAYECTORIA DE AUTONOMÍA EPISTÉMICA
═══════════════════════════════════════════════════════════════════════
Componente Streamlit para epistemic_autonomy.py

SIN ESTA GRÁFICA, EL MÓDULO ES UN CONTADOR.
CON ELLA, ES UN ARGUMENTO.
══════════════════════════════════════════

La diferencia entre un score de autonomía estático y una curva temporal
de autonomía es la diferencia entre un termómetro y un electrocardiograma.
El primero te dice la temperatura ahora. El segundo te muestra el ritmo
cardíaco — el patrón en el tiempo que revela si el sistema está vivo
y hacia dónde va.

Un docente que ve "autonomía: 0.52" no sabe qué hacer. Un docente que ve
"este estudiante llegó a 0.25, subió a 0.67 en las primeras 8 interacciones,
luego cayó a 0.30 cuando cambió el tema, y lleva 6 interacciones volviendo
a subir" puede actuar con precisión quirúrgica.

DECISIONES DE DISEÑO:
──────────────────────
1. Línea de trayectoria individual con áreas de color por fase:
   dependent (rojo) | scaffolded (naranja) | emergent (azul) | autonomous (verde)
   Las áreas permiten leer la fase sin leer números.

2. Anotaciones automáticas de eventos pedagógicos clave:
   - Cambios de configuración del docente (marcados como puntos rojos en el eje)
   - Picos de productive_struggle (marcados como estrellas)
   - Transiciones de fase (líneas verticales con label)

3. Banda de ZPD como área sombreada: la zona de desarrollo próximo
   estimada en cada punto (state.zpd_estimate) crea un canal dinámico.
   El estudiante dentro del canal = bien calibrado.
   Por encima = demasiado fácil → reducir scaffolding.
   Por debajo = frustración → aumentar scaffolding.

4. Mini-sparklines por estudiante para la vista de cohort docente.

5. Sin dependencias pesadas: Plotly (ya en requirements) + Streamlit.

Autor: Diego Elvira Vásquez · Prototipo CP25/152 · Feb 2026
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from collections import defaultdict

try:
    import plotly.graph_objects as go
    import plotly.express as px
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
# COLORES POR FASE — CONSTANTES DE DISEÑO
# ═══════════════════════════════════════════════════════════════════════

PHASE_CONFIG = {
    "dependent": {
        "color": "#EF5350",        # rojo
        "fill": "rgba(239,83,80,0.15)",
        "label": "Dependiente",
        "y_range": (0.0, 0.30),
    },
    "scaffolded": {
        "color": "#FFA726",        # naranja
        "fill": "rgba(255,167,38,0.15)",
        "label": "Con andamiaje",
        "y_range": (0.25, 0.55),
    },
    "emergent": {
        "color": "#42A5F5",        # azul
        "fill": "rgba(66,165,245,0.15)",
        "label": "Autónomía emergente",
        "y_range": (0.50, 0.80),
    },
    "autonomous": {
        "color": "#66BB6A",        # verde
        "fill": "rgba(102,187,106,0.15)",
        "label": "Autónomo",
        "y_range": (0.75, 1.0),
    },
}

BLOOM_COLOR_MAP = {
    1: "#9E9E9E",
    2: "#42A5F5",
    3: "#66BB6A",
    4: "#FFA726",
    5: "#EF5350",
    6: "#AB47BC",
}


# ═══════════════════════════════════════════════════════════════════════
# DATA MODEL — PUNTOS DE TRAYECTORIA
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TrajectoryPoint:
    """Un punto en la curva temporal de autonomía."""
    interaction_n: int
    timestamp: str
    autonomy_score: float
    phase: str
    bloom_weight: float
    bloom_level_name: str
    self_efficacy: float
    dependency_ratio: float
    zpd_estimate: float
    productive_struggle: bool
    scaffolding_used: int

    # Eventos opcionales en este punto
    config_change: Optional[str] = None       # descripción del cambio si lo hubo
    phase_transition: Optional[str] = None    # "dependent→scaffolded" si hubo transición


@dataclass
class StudentAutonomyTimeline:
    """Timeline completo de un estudiante."""
    student_id: str
    course_id: str
    points: list[TrajectoryPoint] = field(default_factory=list)
    config_events: list[dict] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# MOTOR DE TRAYECTORIA — EXTENSIÓN DE EpistemicAutonomyTracker
# ═══════════════════════════════════════════════════════════════════════

class AutonomyTimelineBuilder:
    """
    Construye y gestiona los timelines de autonomía epistémica.

    Uso: instanciar junto a EpistemicAutonomyTracker y llamar a
    record_point() en cada interacción (después de tracker.record_interaction()).
    """

    def __init__(self):
        self.timelines: dict[str, StudentAutonomyTimeline] = {}
        self.cohort_snapshots: list[dict] = []

    def record_point(
        self,
        student_id: str,
        course_id: str,
        autonomy_state,            # AutonomyState del tracker
        bloom_weight: float,
        bloom_level_name: str,
        productive_struggle: bool = False,
        config_change: Optional[str] = None,
    ) -> TrajectoryPoint:
        """
        Añade un punto al timeline del estudiante.
        Se invoca después de EpistemicAutonomyTracker.record_interaction().
        """
        if student_id not in self.timelines:
            self.timelines[student_id] = StudentAutonomyTimeline(
                student_id=student_id,
                course_id=course_id,
            )

        tl = self.timelines[student_id]
        n = len(tl.points) + 1

        # Detectar transición de fase
        phase_transition = None
        if tl.points:
            prev_phase = tl.points[-1].phase
            if prev_phase != autonomy_state.phase:
                phase_transition = f"{prev_phase}→{autonomy_state.phase}"

        point = TrajectoryPoint(
            interaction_n=n,
            timestamp=datetime.now().isoformat(),
            autonomy_score=autonomy_state.autonomy_score,
            phase=autonomy_state.phase,
            bloom_weight=bloom_weight,
            bloom_level_name=bloom_level_name,
            self_efficacy=autonomy_state.self_efficacy_proxy,
            dependency_ratio=autonomy_state.dependency_ratio,
            zpd_estimate=autonomy_state.zpd_estimate,
            productive_struggle=productive_struggle,
            scaffolding_used=autonomy_state.scaffolding_need,
            config_change=config_change,
            phase_transition=phase_transition,
        )

        tl.points.append(point)
        return point

    def record_config_event(
        self,
        student_id: str,
        interaction_n: int,
        config_key: str,
        old_value: object,
        new_value: object,
    ) -> None:
        """Registra un cambio de configuración docente en el timeline."""
        if student_id not in self.timelines:
            return

        self.timelines[student_id].config_events.append({
            "interaction_n": interaction_n,
            "config_key": config_key,
            "old_value": old_value,
            "new_value": new_value,
            "description": f"{config_key}: {old_value} → {new_value}",
        })

    def get_timeline(self, student_id: str) -> Optional[StudentAutonomyTimeline]:
        return self.timelines.get(student_id)


# ═══════════════════════════════════════════════════════════════════════
# GRÁFICAS
# ═══════════════════════════════════════════════════════════════════════

def build_autonomy_trajectory_figure(
    timeline: StudentAutonomyTimeline,
    show_bloom: bool = True,
    show_zpd_band: bool = True,
    show_events: bool = True,
    height: int = 420,
) -> "go.Figure":
    """
    Construye la figura Plotly de trayectoria de autonomía epistémica.

    GRÁFICA PRINCIPAL del módulo de autonomía. La pieza que transforma
    el tracker de contador en argumento pedagógico.

    Layout:
    - Eje Y principal: autonomy_score (0.0-1.0)
    - Eje Y secundario: bloom_weight (1.0-6.0)
    - Bandas de fondo: fases de autonomía
    - Banda de ZPD: área sombreada alrededor del score actual
    - Anotaciones: cambios de config, transiciones de fase, productive struggle
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly requerido: pip install plotly")

    points = timeline.points
    if not points:
        fig = go.Figure()
        fig.add_annotation(
            text="Sin datos de trayectoria. El estudiante no ha interactuado aún.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="#888"),
        )
        return fig

    xs = [p.interaction_n for p in points]
    ys = [p.autonomy_score for p in points]
    phases = [p.phase for p in points]
    blooms = [p.bloom_weight for p in points]
    bloom_names = [p.bloom_level_name for p in points]
    self_eff = [p.self_efficacy for p in points]
    zpd_est = [p.zpd_estimate for p in points]
    struggles = [p.productive_struggle for p in points]
    scaffolding = [p.scaffolding_used for p in points]

    fig = make_subplots(
        specs=[[{"secondary_y": True}]],
        rows=1, cols=1,
    )

    # ── Bandas de fondo por fase ──
    # Zona verde tenue para autonomía autónoma (0.75-1.0)
    for phase_name, cfg in PHASE_CONFIG.items():
        y0, y1 = cfg["y_range"]
        fig.add_hrect(
            y0=y0, y1=y1,
            fillcolor=cfg["fill"],
            layer="below",
            line_width=0,
        )
        # Label de la zona (a la derecha)
        fig.add_annotation(
            x=1.01, y=(y0 + y1) / 2,
            xref="paper", yref="y",
            text=cfg["label"],
            showarrow=False,
            font=dict(size=9, color=cfg["color"]),
            xanchor="left",
        )

    # ── Banda de ZPD ──
    if show_zpd_band and len(points) > 1:
        zpd_upper = [min(1.0, y + 0.12 * (1 - z)) for y, z in zip(ys, zpd_est)]
        zpd_lower = [max(0.0, y - 0.06 * z) for y, z in zip(ys, zpd_est)]
        fig.add_trace(
            go.Scatter(
                x=xs + xs[::-1],
                y=zpd_upper + zpd_lower[::-1],
                fill="toself",
                fillcolor="rgba(150,150,255,0.10)",
                line=dict(width=0),
                name="Zona de Desarrollo Próximo",
                hoverinfo="skip",
                showlegend=True,
            ),
            secondary_y=False,
        )

    # ── Línea de autonomía principal ──
    # Color por fase en cada segmento
    for i in range(len(points) - 1):
        seg_phase = phases[i]
        seg_color = PHASE_CONFIG.get(seg_phase, {}).get("color", "#555")
        fig.add_trace(
            go.Scatter(
                x=[xs[i], xs[i+1]],
                y=[ys[i], ys[i+1]],
                mode="lines",
                line=dict(color=seg_color, width=2.5),
                showlegend=False,
                hoverinfo="skip",
            ),
            secondary_y=False,
        )

    # ── Puntos de la trayectoria con tooltip rico ──
    hover_texts = [
        f"<b>Interacción #{x}</b><br>"
        f"Autonomía: {y:.2f}<br>"
        f"Fase: {ph}<br>"
        f"Bloom: {bn} ({bw:.0f}/6)<br>"
        f"Autoeficacia: {se:.0%}<br>"
        f"Scaffolding: nivel {sc}"
        for x, y, ph, bn, bw, se, sc in zip(
            xs, ys, phases, bloom_names, blooms, self_eff, scaffolding
        )
    ]

    point_colors = [PHASE_CONFIG.get(ph, {}).get("color", "#555") for ph in phases]

    fig.add_trace(
        go.Scatter(
            x=xs, y=ys,
            mode="markers",
            marker=dict(
                size=[10 if s else 7 for s in struggles],
                color=point_colors,
                symbol=["star" if s else "circle" for s in struggles],
                line=dict(width=1, color="white"),
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
            name="Autonomía epistémica",
            showlegend=True,
        ),
        secondary_y=False,
    )

    # ── Bloom en eje secundario (línea punteada gris) ──
    if show_bloom:
        bloom_colors = [BLOOM_COLOR_MAP.get(int(b), "#888") for b in blooms]
        fig.add_trace(
            go.Bar(
                x=xs, y=blooms,
                marker_color=bloom_colors,
                opacity=0.25,
                name="Nivel Bloom",
                hovertemplate="Bloom: %{y:.0f}/6<extra></extra>",
            ),
            secondary_y=True,
        )

    # ── Eventos de configuración docente ──
    if show_events and timeline.config_events:
        for event in timeline.config_events:
            xi = event["interaction_n"]
            fig.add_vline(
                x=xi,
                line=dict(color="rgba(200,0,0,0.4)", width=1.5, dash="dot"),
            )
            fig.add_annotation(
                x=xi, y=1.0,
                yref="y",
                text=f"⚙ {event['config_key']}",
                showarrow=True,
                arrowhead=2, arrowcolor="rgba(200,0,0,0.5)",
                font=dict(size=8, color="#c00"),
                bgcolor="rgba(255,255,255,0.85)",
                bordercolor="rgba(200,0,0,0.3)",
                borderwidth=1,
                ay=-20,
            )

    # ── Anotaciones de transición de fase ──
    if show_events:
        for p in points:
            if p.phase_transition:
                fig.add_annotation(
                    x=p.interaction_n,
                    y=p.autonomy_score + 0.07,
                    yref="y",
                    text=f"↑ {p.phase_transition.split('→')[1]}",
                    showarrow=False,
                    font=dict(size=8, color="#333"),
                    bgcolor="rgba(255,255,255,0.8)",
                )

    # ── Layout ──
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=100, t=40, b=40),
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=10),
        ),
        hovermode="x unified",
        title=dict(
            text=f"Trayectoria de autonomía epistémica — Estudiante {timeline.student_id}",
            font=dict(size=14),
            x=0,
        ),
    )
    fig.update_yaxes(
        title_text="Score de autonomía (0–1)",
        range=[-0.05, 1.1],
        secondary_y=False,
        gridcolor="#eee",
        tickformat=".1f",
    )
    fig.update_yaxes(
        title_text="Bloom (1–6)",
        range=[0, 8],
        secondary_y=True,
        showgrid=False,
    )
    fig.update_xaxes(
        title_text="Número de interacción",
        dtick=max(1, len(points) // 10),
        gridcolor="#eee",
    )

    return fig


def build_cohort_autonomy_figure(
    timelines: dict[str, StudentAutonomyTimeline],
    height: int = 300,
) -> "go.Figure":
    """
    Vista de cohort: trayectorias de todos los estudiantes superpuestas.
    Permite al docente identificar patrones colectivos y outliers.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly requerido")

    fig = go.Figure()

    for sid, tl in timelines.items():
        if not tl.points:
            continue
        xs = [p.interaction_n for p in tl.points]
        ys = [p.autonomy_score for p in tl.points]
        phase = tl.points[-1].phase
        color = PHASE_CONFIG.get(phase, {}).get("color", "#888")

        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="lines",
            line=dict(color=color, width=1.5),
            opacity=0.6,
            name=sid,
            hovertemplate=f"<b>{sid}</b><br>Int. %{{x}}<br>Autonomía: %{{y:.2f}}<extra></extra>",
        ))

    # Línea media del cohort
    if timelines:
        max_n = max(
            (len(tl.points) for tl in timelines.values()),
            default=0
        )
        if max_n > 0:
            mean_by_n = defaultdict(list)
            for tl in timelines.values():
                for p in tl.points:
                    mean_by_n[p.interaction_n].append(p.autonomy_score)

            mean_xs = sorted(mean_by_n.keys())
            mean_ys = [sum(mean_by_n[x]) / len(mean_by_n[x]) for x in mean_xs]

            fig.add_trace(go.Scatter(
                x=mean_xs, y=mean_ys,
                mode="lines",
                line=dict(color="#333", width=3, dash="dash"),
                name="Media del grupo",
                hovertemplate="Media: %{y:.2f}<extra></extra>",
            ))

    fig.update_layout(
        height=height,
        title="Trayectorias de autonomía — Vista de cohort",
        yaxis=dict(title="Autonomía", range=[-0.05, 1.1], tickformat=".1f"),
        xaxis=dict(title="Número de interacción"),
        showlegend=False,
        hovermode="x",
        margin=dict(l=20, r=20, t=40, b=40),
    )

    return fig


def build_phase_distribution_figure(
    timelines: dict[str, StudentAutonomyTimeline],
    height: int = 250,
) -> "go.Figure":
    """
    Distribución de fases en el cohort. Gráfico de barras apiladas.
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly requerido")

    phase_counts = defaultdict(int)
    for tl in timelines.values():
        if tl.points:
            current_phase = tl.points[-1].phase
            phase_counts[current_phase] += 1

    total = sum(phase_counts.values())
    if total == 0:
        return go.Figure()

    phases = list(PHASE_CONFIG.keys())
    values = [phase_counts.get(p, 0) for p in phases]
    colors = [PHASE_CONFIG[p]["color"] for p in phases]
    labels = [PHASE_CONFIG[p]["label"] for p in phases]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v} ({v/total:.0%})" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        height=height,
        title="Distribución de fases de autonomía",
        yaxis=dict(title="N estudiantes"),
        margin=dict(l=20, r=20, t=40, b=40),
        showlegend=False,
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# COMPONENTE STREAMLIT
# ═══════════════════════════════════════════════════════════════════════

def render_autonomy_trajectory_panel(
    timeline: StudentAutonomyTimeline,
    show_controls: bool = True,
) -> None:
    """
    Renderiza el panel completo de trayectoria de autonomía en Streamlit.

    Uso en app.py:
        from autonomy_viz import render_autonomy_trajectory_panel
        render_autonomy_trajectory_panel(builder.get_timeline(student_id))
    """
    if not STREAMLIT_AVAILABLE:
        raise ImportError("Streamlit requerido")

    if not timeline or not timeline.points:
        st.info("Sin datos de trayectoria todavía. Las interacciones del estudiante irán aquí.")
        return

    points = timeline.points
    latest = points[-1]

    # ── Métricas rápidas en columnas ──
    col1, col2, col3, col4 = st.columns(4)
    phase_color = PHASE_CONFIG.get(latest.phase, {}).get("color", "#888")

    with col1:
        st.metric(
            "Autonomía actual",
            f"{latest.autonomy_score:.2f}",
            delta=f"{latest.autonomy_score - points[-2].autonomy_score:+.2f}"
            if len(points) > 1 else None,
        )
    with col2:
        st.metric("Fase", PHASE_CONFIG.get(latest.phase, {}).get("label", latest.phase))
    with col3:
        st.metric("Bloom actual", f"{latest.bloom_level_name} ({latest.bloom_weight:.0f}/6)")
    with col4:
        st.metric("Interacciones", len(points))

    # ── Controles opcionales ──
    show_bloom = True
    show_zpd = True
    show_events = True
    if show_controls:
        c1, c2, c3 = st.columns(3)
        show_bloom = c1.checkbox("Mostrar Bloom", value=True, key=f"bloom_{timeline.student_id}")
        show_zpd = c2.checkbox("Mostrar banda ZPD", value=True, key=f"zpd_{timeline.student_id}")
        show_events = c3.checkbox("Mostrar eventos", value=True, key=f"events_{timeline.student_id}")

    # ── Gráfica principal ──
    fig = build_autonomy_trajectory_figure(
        timeline,
        show_bloom=show_bloom,
        show_zpd_band=show_zpd,
        show_events=show_events,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Leyenda de símbolos ──
    with st.expander("Leyenda", expanded=False):
        st.markdown(
            "⭐ **Estrella**: interacción con productive struggle (Kapur, 2008) — "
            "el estudiante luchó productivamente antes de pedir ayuda.  \n"
            "⚙ **Línea punteada roja**: cambio de configuración del docente.  \n"
            "🔵 **Banda azul**: Zona de Desarrollo Próximo estimada (Vygotsky, 1978).  \n"
            "**Colores de los puntos**: fase de autonomía en ese momento."
        )


def render_cohort_autonomy_panel(
    timelines: dict[str, StudentAutonomyTimeline],
) -> None:
    """
    Panel de cohort con vista agregada. Para el sidebar del docente.
    """
    if not STREAMLIT_AVAILABLE:
        raise ImportError("Streamlit requerido")

    if not timelines:
        st.info("Sin datos de cohort todavía.")
        return

    tab1, tab2 = st.tabs(["Trayectorias", "Distribución de fases"])

    with tab1:
        fig_cohort = build_cohort_autonomy_figure(timelines)
        st.plotly_chart(fig_cohort, use_container_width=True)
        st.caption(
            "Cada línea es un estudiante. La línea gruesa es la media del cohort. "
            "Color = fase actual."
        )

    with tab2:
        fig_dist = build_phase_distribution_figure(timelines)
        st.plotly_chart(fig_dist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════
# SIMULACIÓN DE DATOS — para demo sin estudiantes reales
# ═══════════════════════════════════════════════════════════════════════

def _generate_demo_timeline(
    student_id: str = "DEMO_S001",
    n_interactions: int = 25,
    pattern: str = "improving",
) -> StudentAutonomyTimeline:
    """
    Genera un timeline sintético para demo.
    pattern: "improving" | "plateau" | "volatile" | "declining"
    """
    import math
    import random
    random.seed(42)

    tl = StudentAutonomyTimeline(student_id=student_id, course_id="PROG101")
    phases = ["dependent", "scaffolded", "emergent", "autonomous"]
    bloom_names = ["Recordar", "Comprender", "Aplicar", "Analizar", "Evaluar", "Crear"]

    for i in range(1, n_interactions + 1):
        t = i / n_interactions

        if pattern == "improving":
            base = 0.2 + 0.6 * t + 0.05 * math.sin(i * 0.8)
        elif pattern == "plateau":
            base = 0.3 + 0.2 * (1 - math.exp(-t * 5)) + 0.04 * math.sin(i)
        elif pattern == "volatile":
            base = 0.45 + 0.3 * math.sin(i * 1.5) * math.cos(i * 0.3)
        else:
            base = 0.7 - 0.4 * t + 0.03 * math.sin(i)

        autonomy = max(0.05, min(0.98, base + random.uniform(-0.05, 0.05)))

        if autonomy < 0.30:
            phase = "dependent"
        elif autonomy < 0.55:
            phase = "scaffolded"
        elif autonomy < 0.78:
            phase = "emergent"
        else:
            phase = "autonomous"

        bloom_w = max(1.0, min(6.0, 1.5 + 3.5 * t + random.uniform(-0.5, 0.5)))
        bloom_name = bloom_names[min(int(bloom_w) - 1, 5)]

        phase_transition = None
        if tl.points and tl.points[-1].phase != phase:
            phase_transition = f"{tl.points[-1].phase}→{phase}"

        # Config change at interaction 10
        config_change = "socratic_scaffolding: False→True" if i == 10 else None

        point = TrajectoryPoint(
            interaction_n=i,
            timestamp=datetime.now().isoformat(),
            autonomy_score=round(autonomy, 3),
            phase=phase,
            bloom_weight=round(bloom_w, 1),
            bloom_level_name=bloom_name,
            self_efficacy=round(autonomy * 0.9, 2),
            dependency_ratio=round(1 - autonomy * 0.85, 2),
            zpd_estimate=round(max(0, 0.6 - autonomy * 0.5), 2),
            productive_struggle=i in (5, 12, 19),
            scaffolding_used=max(0, 3 - int(autonomy * 3)),
            config_change=config_change,
            phase_transition=phase_transition,
        )
        tl.points.append(point)

    if len(tl.points) >= 10:
        tl.config_events.append({
            "interaction_n": 10,
            "config_key": "socratic_scaffolding",
            "old_value": False,
            "new_value": True,
            "description": "socratic_scaffolding: False→True",
        })

    return tl


def _demo_no_streamlit():
    """Demo en modo texto cuando Streamlit no está disponible."""
    tl = _generate_demo_timeline(pattern="improving")
    print(f"Timeline generado: {len(tl.points)} puntos")
    print(f"Rango de autonomía: {min(p.autonomy_score for p in tl.points):.2f} "
          f"→ {max(p.autonomy_score for p in tl.points):.2f}")
    print(f"Fase final: {tl.points[-1].phase}")
    print(f"Eventos de config: {len(tl.config_events)}")

    if PLOTLY_AVAILABLE:
        fig = build_autonomy_trajectory_figure(tl)
        print(f"Figura generada: {len(fig.data)} trazas")
        print("Para visualizar: fig.show() o guardar con fig.write_html('out.html')")
    else:
        print("Plotly no disponible. Instalar: pip install plotly")


if __name__ == "__main__":
    _demo_no_streamlit()
