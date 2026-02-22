"""
VISTA DEL INVESTIGADOR — Análisis de Profundidad
==================================================
Esta vista es lo que transforma el prototipo de "demo técnica"
a "instrumento de investigación". Integra:
  1. Análisis cognitivo (Bloom) de cada interacción
  2. Trayectorias de aprendizaje por estudiante
  3. Perfiles de engagement (deep/surface/strategic)
  4. Dinámicas de confianza (calibración, sobre/infra-dependencia)

Destinatario: el investigador que analiza los datos del piloto.
No es para el docente (esa es la vista Analytics) ni para el estudiante.

Uso: importar como módulo desde app.py
"""

import streamlit as st
import random
from datetime import datetime, timedelta

from cognitive_analyzer import (
    CognitiveAnalyzer, EngagementProfiler, CognitiveAnalysis, BLOOM_LEVELS,
)
from cognitive_engine import ICAP_LEVELS
from trust_dynamics import TrustDynamicsAnalyzer
from ach_diagnostic import ACHDiagnosticEngine, DIAGNOSTIC_HYPOTHESES
from nd_patterns import NeurodivergentPatternDetector, InteractionEvent


def generate_research_demo_data():
    """
    Genera datos de demo RICOS para la vista de investigador.
    Simula 8 estudiantes con perfiles diferenciados a lo largo de 30 interacciones.
    """
    analyzer = CognitiveAnalyzer()
    trust_analyzer = TrustDynamicsAnalyzer()

    # Perfiles de estudiante con patrones predefinidos
    student_profiles = {
        "est_01_deep": {
            "name": "Ana (exploradora profunda)",
            "bloom_pattern": [1, 2, 2, 3, 3, 4, 3, 4, 4, 5, 4, 5, 5, 6, 5, 6],
            "copypaste_base": 0.1,
            "meta_probability": 0.3,
            "trust_pattern": "calibrated",
        },
        "est_02_surface": {
            "name": "Carlos (superficial)",
            "bloom_pattern": [1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1],
            "copypaste_base": 0.6,
            "meta_probability": 0.05,
            "trust_pattern": "over_reliant",
        },
        "est_03_strategic": {
            "name": "María (estratégica)",
            "bloom_pattern": [2, 3, 3, 3, 4, 3, 3, 2, 3, 4, 3, 3, 4, 3, 3, 4],
            "copypaste_base": 0.15,
            "meta_probability": 0.15,
            "trust_pattern": "calibrated",
        },
        "est_04_frustrated": {
            "name": "Pedro (frustrado)",
            "bloom_pattern": [3, 3, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "copypaste_base": 0.3,
            "meta_probability": 0.02,
            "trust_pattern": "under_reliant",
        },
        "est_05_metacognitive": {
            "name": "Laura (metacognitiva)",
            "bloom_pattern": [2, 3, 3, 4, 4, 5, 4, 5, 5, 4, 5, 6, 5, 5, 6, 5],
            "copypaste_base": 0.05,
            "meta_probability": 0.45,
            "trust_pattern": "calibrated",
        },
        "est_06_copypaster": {
            "name": "Diego (copy-paster)",
            "bloom_pattern": [3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1],
            "copypaste_base": 0.8,
            "meta_probability": 0.0,
            "trust_pattern": "over_reliant",
        },
    }

    all_data = {}
    for sid, profile in student_profiles.items():
        analyses = []
        cp_scores = []
        scaffolding_levels = []
        trust_signals = []

        for i, bloom_target in enumerate(profile["bloom_pattern"]):
            # Simular análisis cognitivo con varianza
            actual_bloom = max(1, min(6, bloom_target + random.choice([-1, 0, 0, 0, 1])))
            is_meta = random.random() < profile["meta_probability"]

            icap_key = BLOOM_LEVELS[actual_bloom]["icap_equivalent"]
            icap_label = ICAP_LEVELS.get(icap_key, {}).get("label", "Pasivo")
            analysis = CognitiveAnalysis(
                bloom_level=actual_bloom,
                bloom_name=BLOOM_LEVELS[actual_bloom]["name"],
                bloom_code=BLOOM_LEVELS[actual_bloom]["code"],
                bloom_confidence=round(random.uniform(0.4, 0.95), 2),
                icap_level=icap_key,
                icap_label=icap_label,
                detected_markers=[f"simulated_{i}"],
                engagement_score=0.5 + (0.3 if is_meta else 0),
                is_metacognitive=is_meta,
            )
            analyses.append(analysis)

            cp = min(1.0, max(0.0, profile["copypaste_base"] + random.gauss(0, 0.15)))
            cp_scores.append(round(cp, 2))

            scaffolding_levels.append(min(3, max(0, i // 4)))

            # Trust signal
            trust_dir = {
                "calibrated": random.gauss(0, 0.15),
                "over_reliant": random.gauss(0.5, 0.2),
                "under_reliant": random.gauss(-0.5, 0.2),
            }.get(profile["trust_pattern"], 0)

            trust_signals.append(round(max(-1, min(1, trust_dir)), 2))

        all_data[sid] = {
            "name": profile["name"],
            "analyses": analyses,
            "cp_scores": cp_scores,
            "scaffolding_levels": scaffolding_levels,
            "trust_signals": trust_signals,
        }

    return all_data


def render_researcher_view():
    """Renderiza la vista completa del investigador."""

    st.markdown(
        '<div class="main-header">🔬 Vista del Investigador — Análisis de Profundidad</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "*Instrumentos de análisis para publicación. "
        "Integra taxonomía de Bloom, perfiles ICAP, dinámicas de confianza y trayectorias cognitivas.*"
    )

    # Generar o recuperar datos de demo
    if "research_data" not in st.session_state:
        st.session_state.research_data = generate_research_demo_data()

    data = st.session_state.research_data
    analyzer = CognitiveAnalyzer()
    profiler = EngagementProfiler()

    # ── Selector de estudiante ──
    student_options = {sid: d["name"] for sid, d in data.items()}
    selected_student = st.selectbox(
        "Seleccionar estudiante",
        options=list(student_options.keys()),
        format_func=lambda x: student_options[x],
    )

    student_data = data[selected_student]
    analyses = student_data["analyses"]
    cp_scores = student_data["cp_scores"]
    scaffolding_levels = student_data["scaffolding_levels"]
    trust_signals = student_data["trust_signals"]

    st.markdown("---")

    # ══════════════════════════════════════════
    # SECCIÓN 1: TRAYECTORIA COGNITIVA
    # ══════════════════════════════════════════

    st.markdown("### 📈 Trayectoria Cognitiva (Bloom Revisada)")

    trajectory = analyzer.compute_trajectory(analyses)

    # Métricas de trayectoria
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Nivel medio", f"{trajectory['mean_level']:.1f}/6")
    m2.metric("Tendencia", f"{trajectory['trend']:+.3f}",
              delta="↑ ascendente" if trajectory['trend'] > 0.1 else "↓ descendente" if trajectory['trend'] < -0.1 else "→ estable")
    m3.metric("Techo alcanzado", f"Nivel {trajectory['ceiling']}")
    m4.metric("Metacognición", f"{trajectory['metacognitive_ratio']:.0%}")
    m5.metric("Variabilidad", f"σ = {trajectory['std']:.2f}")

    # Interpretación
    st.info(f"**Interpretación automática:** {trajectory['interpretation']}")

    try:
        import plotly.graph_objects as go
        import plotly.express as px
        import pandas as pd

        # Gráfico de trayectoria con colores de Bloom
        levels_seq = trajectory["levels_sequence"]
        colors = [BLOOM_LEVELS[l]["color"] for l in levels_seq]
        names = [BLOOM_LEVELS[l]["name"] for l in levels_seq]

        fig_trajectory = go.Figure()

        # Línea de tendencia
        n = len(levels_seq)
        trend_line = [trajectory["mean_level"] + trajectory["trend"] * (i - n/2) for i in range(n)]
        fig_trajectory.add_trace(go.Scatter(
            x=list(range(1, n + 1)), y=trend_line,
            mode="lines", name="Tendencia",
            line=dict(color="rgba(0,0,0,0.3)", dash="dash", width=2),
        ))

        # Puntos coloreados por nivel Bloom
        fig_trajectory.add_trace(go.Scatter(
            x=list(range(1, n + 1)), y=levels_seq,
            mode="markers+lines",
            name="Nivel Bloom",
            marker=dict(size=12, color=colors, line=dict(width=1, color="white")),
            line=dict(color="rgba(100,100,100,0.4)", width=1),
            text=[f"{names[i]} (nivel {levels_seq[i]})" for i in range(n)],
            hovertemplate="Interacción %{x}<br>%{text}<extra></extra>",
        ))

        fig_trajectory.update_layout(
            yaxis=dict(
                tickvals=[1, 2, 3, 4, 5, 6],
                ticktext=["1-Recordar", "2-Comprender", "3-Aplicar", "4-Analizar", "5-Evaluar", "6-Crear"],
                range=[0.5, 6.5],
                title="Nivel cognitivo (Bloom)",
            ),
            xaxis=dict(title="Nº de interacción"),
            height=400,
            showlegend=True,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig_trajectory, use_container_width=True)

        # ══════════════════════════════════════════
        # SECCIÓN 2: DISTRIBUCIÓN BLOOM + ICAP
        # ══════════════════════════════════════════

        col_bloom, col_icap = st.columns(2)

        with col_bloom:
            st.markdown("### 🎯 Distribución Bloom")
            dist = trajectory["distribution"]
            fig_bloom = px.bar(
                x=list(dist.keys()), y=list(dist.values()),
                color=list(dist.keys()),
                color_discrete_map={
                    BLOOM_LEVELS[i]["name"]: BLOOM_LEVELS[i]["color"]
                    for i in BLOOM_LEVELS
                },
                labels={"x": "Nivel", "y": "Frecuencia"},
            )
            fig_bloom.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_bloom, use_container_width=True)

        with col_icap:
            st.markdown("### 🧩 Mapeo ICAP (Chi & Wylie, 2014)")
            icap_counts = {}
            for a in analyses:
                key = a.icap_level.title()
                icap_counts[key] = icap_counts.get(key, 0) + 1
            icap_order = ["Passive", "Active", "Constructive", "Interactive"]
            icap_colors = {"Passive": "#9E9E9E", "Active": "#42A5F5", "Constructive": "#FFA726", "Interactive": "#AB47BC"}
            fig_icap = px.pie(
                names=[k for k in icap_order if k in icap_counts],
                values=[icap_counts.get(k, 0) for k in icap_order if k in icap_counts],
                color=[k for k in icap_order if k in icap_counts],
                color_discrete_map=icap_colors,
            )
            fig_icap.update_layout(height=300)
            st.plotly_chart(fig_icap, use_container_width=True)

        st.markdown("---")

        # ══════════════════════════════════════════
        # SECCIÓN 3: PERFIL DE ENGAGEMENT
        # ══════════════════════════════════════════

        st.markdown("### 🎭 Perfil de Engagement")

        profile = profiler.classify(analyses, cp_scores, scaffolding_levels)

        prof_col1, prof_col2 = st.columns([1, 2])

        with prof_col1:
            st.markdown(f"""
            <div style="background: {profile.get('color', '#999')}15; 
                        border: 2px solid {profile.get('color', '#999')}; 
                        border-radius: 12px; padding: 24px; text-align: center;">
                <div style="font-size: 3rem;">{profile.get('emoji', '?')}</div>
                <div style="font-size: 1.4rem; font-weight: 700; color: {profile.get('color', '#333')};">
                    {profile.get('label', '?')}
                </div>
                <div style="font-size: 0.9rem; color: #666; margin-top: 8px;">
                    Confianza: {profile.get('confidence', 0):.0%}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"*{profile.get('description', '')}*")

        with prof_col2:
            if "all_scores" in profile:
                scores = profile["all_scores"]
                profile_names = {
                    "deep_explorer": "🔬 Explorador profundo",
                    "strategic_user": "🎯 Estratégico",
                    "surface_seeker": "📋 Superficial",
                    "copy_paster": "📎 Copy-paster",
                    "metacognitive": "🧠 Metacognitivo",
                }
                fig_radar_data = {
                    "Perfil": [profile_names.get(k, k) for k in scores.keys()],
                    "Score": list(scores.values()),
                }
                fig_profiles = px.bar(
                    fig_radar_data, x="Score", y="Perfil",
                    orientation="h",
                    color="Score",
                    color_continuous_scale="Viridis",
                )
                fig_profiles.update_layout(height=280, showlegend=False)
                st.plotly_chart(fig_profiles, use_container_width=True)

            if "indicators" in profile:
                ind = profile["indicators"]
                st.markdown("**Indicadores subyacentes:**")
                ind_cols = st.columns(5)
                ind_cols[0].metric("Bloom medio", f"{ind['mean_bloom_level']:.1f}")
                ind_cols[1].metric("Copy-paste", f"{ind['mean_copypaste']:.0%}")
                ind_cols[2].metric("Metacognición", f"{ind['metacognitive_ratio']:.0%}")
                ind_cols[3].metric("Bloom alto (≥4)", f"{ind['high_bloom_ratio']:.0%}")
                ind_cols[4].metric("Max scaffolding", f"Nivel {ind['max_scaffolding_reached']}")

        st.markdown("---")

        # ══════════════════════════════════════════
        # SECCIÓN 4: DINÁMICAS DE CONFIANZA
        # ══════════════════════════════════════════

        st.markdown("### 🤝 Dinámicas de Confianza Estudiante-IA")
        st.markdown(
            "*Calibración de confianza: 0 = calibrada (óptimo), "
            ">0.2 = sobre-dependencia, <-0.2 = infra-confianza. "
            "Lee & See (2004), Parasuraman & Riley (1997).*"
        )

        # Gráfico de trust signals a lo largo del tiempo
        fig_trust = go.Figure()

        # Zona de confianza calibrada
        fig_trust.add_hrect(y0=-0.2, y1=0.2,
                           fillcolor="rgba(76,175,80,0.1)",
                           line=dict(width=0),
                           annotation_text="Zona calibrada",
                           annotation_position="top left")

        # Zona de sobre-confianza
        fig_trust.add_hrect(y0=0.2, y1=1.0,
                           fillcolor="rgba(255,152,0,0.08)",
                           line=dict(width=0))

        # Zona de infra-confianza
        fig_trust.add_hrect(y0=-1.0, y1=-0.2,
                           fillcolor="rgba(244,67,54,0.08)",
                           line=dict(width=0))

        # Señales
        x_vals = list(range(1, len(trust_signals) + 1))
        colors_trust = []
        for ts in trust_signals:
            if abs(ts) < 0.2:
                colors_trust.append("#4CAF50")
            elif ts > 0:
                colors_trust.append("#FF9800")
            else:
                colors_trust.append("#F44336")

        fig_trust.add_trace(go.Scatter(
            x=x_vals, y=trust_signals,
            mode="markers+lines",
            marker=dict(size=10, color=colors_trust, line=dict(width=1, color="white")),
            line=dict(color="rgba(100,100,100,0.3)", width=1),
            name="Señal de confianza",
        ))

        # Media móvil
        window = 4
        if len(trust_signals) >= window:
            moving_avg = []
            for i in range(len(trust_signals)):
                start = max(0, i - window + 1)
                moving_avg.append(sum(trust_signals[start:i+1]) / (i - start + 1))
            fig_trust.add_trace(go.Scatter(
                x=x_vals, y=moving_avg,
                mode="lines", name="Media móvil",
                line=dict(color="#1565C0", width=3),
            ))

        fig_trust.update_layout(
            yaxis=dict(title="Score de confianza", range=[-1.1, 1.1]),
            xaxis=dict(title="Interacción"),
            height=350,
            annotations=[
                dict(x=0.02, y=0.85, xref="paper", yref="paper",
                     text="⚠️ Sobre-dependencia", showarrow=False,
                     font=dict(color="#FF9800", size=11)),
                dict(x=0.02, y=0.15, xref="paper", yref="paper",
                     text="⚠️ Infra-confianza", showarrow=False,
                     font=dict(color="#F44336", size=11)),
            ],
        )
        st.plotly_chart(fig_trust, use_container_width=True)

        # Trust profile summary
        mean_trust = sum(trust_signals) / len(trust_signals)
        if abs(mean_trust) < 0.2:
            trust_cat = "✅ Calibrada"
            trust_rec = "Mantener configuración actual."
        elif mean_trust > 0.2:
            trust_cat = "⚠️ Sobre-dependencia"
            trust_rec = "Considerar activar alucinaciones pedagógicas y aumentar scaffolding."
        else:
            trust_cat = "🔴 Infra-confianza"
            trust_rec = "Reducir scaffolding, contactar al estudiante."

        st.markdown(f"**Diagnóstico:** {trust_cat} (score medio: {mean_trust:+.2f})")
        st.markdown(f"**Recomendación:** {trust_rec}")

        st.markdown("---")

        # ══════════════════════════════════════════
        # SECCIÓN 5: DIAGNÓSTICO ACH
        # ══════════════════════════════════════════

        st.markdown("### 🔍 Diagnóstico por Hipótesis Competitivas (ACH)")
        st.markdown(
            "*Adaptación de la metodología de Richards Heuer (CIA, 1999) al diagnóstico "
            "de dificultades de aprendizaje. La hipótesis ganadora es la que tiene "
            "MENOS evidencia en contra, no más a favor — esto contrarresta el sesgo "
            "de confirmación del docente.*"
        )

        ach_engine = ACHDiagnosticEngine()

        # Preparar datos para ACH
        topic_dist = {}
        for a in analyses:
            # Use bloom level as proxy for topics since we don't have real topics in demo
            level_name = BLOOM_LEVELS[a.bloom_level]["name"]
            topic_dist[level_name] = topic_dist.get(level_name, 0) + 1

        diagnosis = ach_engine.diagnose(
            student_id=selected_student,
            bloom_levels=[a.bloom_level for a in analyses],
            bloom_mean=trajectory["mean_level"],
            bloom_trend=trajectory["trend"],
            metacognitive_ratio=trajectory["metacognitive_ratio"],
            copypaste_scores=cp_scores,
            trust_calibration=mean_trust,
            scaffolding_levels_reached=scaffolding_levels,
            topic_distribution=topic_dist,
            n_interactions=len(analyses),
        )

        # Diagnóstico principal
        h_info = DIAGNOSTIC_HYPOTHESES[diagnosis.leading_hypothesis]
        diag_col1, diag_col2 = st.columns([1, 2])

        with diag_col1:
            st.markdown(f"""
            <div style="background: #FFF8E1; border: 2px solid #FFA000; 
                        border-radius: 12px; padding: 20px; text-align: center;">
                <div style="font-size: 1.1rem; font-weight: 700; color: #E65100;">
                    HIPÓTESIS PRINCIPAL
                </div>
                <div style="font-size: 1.3rem; font-weight: 700; margin-top: 8px;">
                    {h_info['name']}
                </div>
                <div style="font-size: 0.9rem; color: #555; margin-top: 6px;">
                    {h_info['short']}
                </div>
                <div style="margin-top: 10px; font-size: 0.85rem; color: #777;">
                    Confianza: {diagnosis.confidence:.0%} · VanLehn: {h_info['van_lehn_type']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with diag_col2:
            # Ranking de hipótesis
            ranked = sorted(diagnosis.hypotheses_scores.items(), key=lambda x: x[1], reverse=True)
            h_names = [DIAGNOSTIC_HYPOTHESES[h]["name"] for h, _ in ranked]
            h_scores = [s for _, s in ranked]
            h_colors = ["#FFA000" if h == diagnosis.leading_hypothesis else "#90A4AE" for h, _ in ranked]

            fig_ach = px.bar(
                x=h_scores, y=h_names,
                orientation="h",
                color=h_names,
                color_discrete_map={DIAGNOSTIC_HYPOTHESES[h]["name"]: c for (h, _), c in zip(ranked, h_colors)},
                labels={"x": "Score ACH (>0 = más probable)", "y": ""},
            )
            fig_ach.update_layout(height=260, showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_ach, use_container_width=True)

        # Cadena de razonamiento
        with st.expander("📝 Cadena de razonamiento diagnóstico"):
            for i, r in enumerate(diagnosis.reasoning_trace, 1):
                st.markdown(f"**{i}.** {r}")

        st.markdown(f"**Intervención recomendada:** {h_info['intervention']}")

        st.markdown("---")

        # ══════════════════════════════════════════
        # SECCIÓN 6: PATRONES NEURODIVERGENTES
        # ══════════════════════════════════════════

        st.markdown("### 🧩 Detección de Patrones de Interacción Neurodivergente")
        st.markdown(
            "*Identificación de patrones que sugieren la necesidad de adaptaciones "
            "pedagógicas específicas. Descripciones FUNCIONALES, no diagnósticas. "
            "Fundamentación: Barkley (2015), Silverman (2013), Reis et al. (2014).*"
        )

        # Generar eventos para el detector ND
        nd_detector = NeurodivergentPatternDetector()
        import random as _rnd
        base_time = datetime.now() - timedelta(days=7)

        events_for_nd = []
        for i, a in enumerate(analyses):
            # Simular timing con varianza
            t = base_time + timedelta(hours=i * _rnd.uniform(0.5, 8))
            events_for_nd.append(InteractionEvent(
                timestamp=t,
                bloom_level=a.bloom_level,
                topics=[BLOOM_LEVELS[a.bloom_level]["name"]],
                prompt_length=_rnd.randint(5, 60),
                copy_paste_score=cp_scores[i] if i < len(cp_scores) else 0,
                is_metacognitive=a.is_metacognitive,
                scaffolding_level=scaffolding_levels[i] if i < len(scaffolding_levels) else 0,
            ))

        nd_patterns = nd_detector.analyze(events_for_nd)

        if nd_patterns:
            for pattern in nd_patterns:
                with st.container():
                    st.markdown(f"""
                    <div style="background: #F3E5F5; border-left: 4px solid #7B1FA2;
                                border-radius: 0 8px 8px 0; padding: 16px; margin-bottom: 12px;">
                        <div style="font-size: 1.1rem; font-weight: 700; color: #4A148C;">
                            {pattern.pattern_name}
                            <span style="font-size: 0.8rem; color: #777; margin-left: 8px;">
                                Confianza: {pattern.confidence:.0%}
                            </span>
                        </div>
                        <div style="font-size: 0.9rem; color: #333; margin-top: 6px;">
                            {pattern.functional_description}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    with st.expander(f"📊 Evidencia y adaptación — {pattern.pattern_id}"):
                        st.markdown("**Evidencia:**")
                        for ev in pattern.evidence:
                            st.markdown(f"- {ev}")
                        st.markdown(f"**Adaptación del scaffolding:** {pattern.scaffolding_adaptation}")
                        st.markdown(f"**Nota para el docente:** {pattern.teacher_note}")
        else:
            st.info(
                "Sin patrones neurodivergentes detectados para este estudiante con "
                "los datos actuales. Se necesitan ≥8 interacciones con varianza temporal suficiente."
            )

        st.markdown("---")

        # ══════════════════════════════════════════
        # SECCIÓN 7: VISTA COMPARATIVA (todos los estudiantes)
        # ══════════════════════════════════════════

        st.markdown("### 👥 Vista Comparativa — Todos los Estudiantes")

        comparison_data = []
        for sid, sdata in data.items():
            traj = analyzer.compute_trajectory(sdata["analyses"])
            prof = profiler.classify(sdata["analyses"], sdata["cp_scores"], sdata["scaffolding_levels"])
            mean_trust_s = sum(sdata["trust_signals"]) / len(sdata["trust_signals"])

            comparison_data.append({
                "Estudiante": sdata["name"],
                "Bloom medio": traj["mean_level"],
                "Tendencia": traj["trend"],
                "Metacognición": f"{traj['metacognitive_ratio']:.0%}",
                "Perfil": f"{prof.get('emoji', '')} {prof.get('label', '?')}",
                "Confianza": mean_trust_s,
                "Copy-paste": f"{sum(sdata['cp_scores'])/len(sdata['cp_scores']):.0%}",
            })

        df_comp = pd.DataFrame(comparison_data)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)

        # Scatterplot: Bloom medio vs Confianza (cada punto = estudiante)
        fig_scatter = px.scatter(
            df_comp,
            x="Bloom medio", y="Confianza",
            text="Estudiante",
            color="Perfil",
            size=[30] * len(df_comp),
            labels={"Bloom medio": "Nivel cognitivo medio (Bloom)", "Confianza": "Score de confianza"},
        )
        fig_scatter.update_traces(textposition="top center", textfont_size=9)
        fig_scatter.add_hline(y=0.2, line_dash="dash", line_color="orange", annotation_text="Umbral sobre-dependencia")
        fig_scatter.add_hline(y=-0.2, line_dash="dash", line_color="red", annotation_text="Umbral infra-confianza")
        fig_scatter.update_layout(height=450)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # ══════════════════════════════════════════
        # SECCIÓN 8: META-EVALUACIÓN DEL SISTEMA
        # ══════════════════════════════════════════

        st.markdown("### 📊 Meta-Evaluación: Efectividad de Estrategias Pedagógicas")
        st.markdown(
            "*¿El scaffolding_mode está funcionando? "
            "Bloom subió, no repitió pregunta, engagement mantenido, sin frustración post-respuesta.*"
        )

        try:
            from meta_evaluation import MetaEvaluator

            evaluator = MetaEvaluator()
            interactions_all = []
            for sid, sdata in data.items():
                for i, a in enumerate(sdata["analyses"]):
                    interactions_all.append({
                        "student_id": sid,
                        "bloom_level": a.bloom_level,
                        "bloom": a.bloom_level,
                        "prompt_raw": f"Pregunta simulada {i}",
                        "prompt": f"Pregunta simulada {i}",
                        "detected_topics": [BLOOM_LEVELS[a.bloom_level]["name"]],
                        "trust_direction": sdata["trust_signals"][i] if i < len(sdata["trust_signals"]) else 0,
                        "scaffolding_mode": ["socratic", "hints", "direct"][sdata["scaffolding_levels"][i] % 3] if i < len(sdata["scaffolding_levels"]) else "socratic",
                        "response_time_ms": 1500,
                        "timestamp": (datetime.now() - timedelta(days=7, hours=i * 2)).isoformat(),
                    })
            meta = evaluator.evaluate(interactions_all, student_id=selected_student)

            m1, m2 = st.columns(2)
            with m1:
                if meta.effectiveness_by_strategy:
                    eff_df = pd.DataFrame([
                        {"Estrategia": k, "Tasa éxito": f"{v:.0%}"}
                        for k, v in meta.effectiveness_by_strategy.items()
                    ])
                    st.dataframe(eff_df, use_container_width=True, hide_index=True)
                for rec in meta.strategy_recommendations:
                    st.info(rec)
            with m2:
                if meta.auto_adjust_suggestion:
                    st.warning("**Sugerencia auto-ajuste:** " + meta.auto_adjust_suggestion)
                st.markdown("**Efectividad por estrategia × estudiante:**")
                for ps in meta.per_student_strategy[:5]:
                    st.caption(f"{ps.strategy}: {ps.success_rate:.0%} ({ps.success_count}/{ps.total_count})")
        except ImportError:
            st.caption("Instala meta_evaluation para ver efectividad de estrategias.")

    except ImportError:
        st.error("Instala plotly y pandas para las visualizaciones: `pip install plotly pandas`")
