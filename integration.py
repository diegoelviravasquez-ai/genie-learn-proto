"""
INTEGRATION LAYER — Differential Modules → Middleware
═══════════════════════════════════════════════════════════════════════
Conecta los tres módulos diferenciales al middleware existente
SIN modificar su flujo de pre/post procesamiento.

Patrón de integración: OBSERVER.
Los módulos observan cada interacción y generan analytics de alto nivel
en paralelo al flujo existente. El middleware sigue funcionando igual;
los módulos ENRIQUECEN los datos, no los reemplazan.

Esto es deliberado: en un proyecto real con código en producción
(GENIE Learn ya tiene un chatbot desplegado), no querrías que un
nuevo contratado rompa lo que funciona. Querrías que AÑADA valor
sobre la infraestructura existente. Esto es exactamente lo que
haría en los meses 1-8 del contrato CP25/152.

Autor: Diego Elvira Vásquez · Prototipo CP25/152 · Feb 2026
"""

from dataclasses import dataclass
from datetime import datetime

from cognitive_profiler import CognitiveProfiler, BLOOM_LEVELS
from epistemic_autonomy import EpistemicAutonomyTracker
from interaction_semiotics import InteractionSemioticsEngine


class EnhancedAnalyticsLayer:
    """
    Capa analítica de alto nivel que integra los tres módulos
    diferenciales sobre el middleware existente.

    Se inicializa UNA VEZ y se invoca en cada interacción.
    Los datos que genera alimentan un dashboard extendido
    (no sustituyen el dashboard actual).
    """

    def __init__(self):
        self.profiler = CognitiveProfiler()
        self.autonomy = EpistemicAutonomyTracker()
        self.semiotics = InteractionSemioticsEngine()

    def analyze_interaction(
        self,
        student_id: str,
        prompt: str,
        scaffolding_level_used: int,
        copy_paste_score: float,
    ) -> dict:
        """
        Análisis completo de una interacción.

        Se invoca DESPUÉS del flujo del middleware existente:
        pre_process → LLM → post_process → log → ESTE MÉTODO.

        Returns dict con los análisis de los tres módulos + síntesis.
        """
        # ─── 1. Cognitive Profiler ───
        snapshot = self.profiler.analyze_prompt(student_id, prompt)

        # ─── 2. Interaction Semiotics ───
        semiotics = self.semiotics.analyze(student_id, prompt)

        # ─── 3. Epistemic Autonomy (necesita datos de los otros dos) ───
        bloom_weight = BLOOM_LEVELS.get(
            snapshot.bloom_level, {}
        ).get("weight", 2)

        autonomy_state = self.autonomy.record_interaction(
            student_id=student_id,
            prompt=prompt,
            bloom_level=snapshot.bloom_level,
            bloom_weight=bloom_weight,
            affective_state=snapshot.affective_state,
            scaffolding_level_used=scaffolding_level_used,
            copy_paste_score=copy_paste_score,
        )

        # ─── 4. Recomendación de scaffolding adaptativo ───
        scaffolding_rec = self.autonomy.get_scaffolding_recommendation(
            student_id, bloom_weight
        )

        return {
            "cognitive": {
                "bloom_level": snapshot.bloom_level,
                "bloom_label": BLOOM_LEVELS[snapshot.bloom_level]["label"],
                "bloom_confidence": snapshot.bloom_confidence,
                "affective_state": snapshot.affective_state,
                "affective_valence": snapshot.affective_valence,
                "complexity": snapshot.prompt_complexity,
                "metacognitive_signals": snapshot.metacognitive_signals,
            },
            "semiotics": {
                "speech_act": semiotics.primary_speech_act,
                "speech_act_label": semiotics.primary_speech_act.replace("_", " ").title(),
                "pedagogical_value": semiotics.pedagogical_value,
                "grice_composite": semiotics.grice_composite,
                "gaming_suspicion": semiotics.gaming_suspicion,
                "gaming_type": semiotics.gaming_type,
            },
            "autonomy": {
                "phase": autonomy_state.phase,
                "autonomy_score": autonomy_state.autonomy_score,
                "self_efficacy": autonomy_state.self_efficacy_proxy,
                "dependency_ratio": autonomy_state.dependency_ratio,
                "productive_struggle": autonomy_state.productive_struggle_count,
            },
            "scaffolding_recommendation": {
                "recommended_level": scaffolding_rec.recommended_level,
                "current_level_used": scaffolding_level_used,
                "should_fade": scaffolding_rec.should_fade,
                "zpd_alignment": scaffolding_rec.zpd_alignment,
                "rationale": scaffolding_rec.rationale,
            },
        }

    def get_enhanced_dashboard_data(self) -> dict:
        """
        Datos para el dashboard extendido.

        Esto es lo que transforma el dashboard de "contadores crudos"
        (limitación LAK 2026) en analytics de ALTO NIVEL:
        - Perfiles de engagement (Bloom + afecto + autonomía)
        - Tipología de interacciones (actos de habla)
        - Alertas de gaming (no solo copy-paste binario)
        - Trayectorias de autonomía
        - Recomendaciones de scaffolding fading
        """
        return {
            "cognitive_cohort": self.profiler.get_cohort_summary(),
            "autonomy_cohort": self.autonomy.get_autonomy_dashboard_data(),
            "semiotics_cohort": self.semiotics.get_semiotics_dashboard_data(),
        }

    def get_student_deep_profile(self, student_id: str) -> dict:
        """
        Perfil profundo de un estudiante individual.

        Combina los tres módulos para dar al docente una vista
        360° del estudiante — exactamente lo que los 16 evaluadores
        del workshop LAK 2026 pidieron como "engagement profiles".
        """
        cognitive = self.profiler.build_profile(student_id)
        quality = self.semiotics.build_quality_report(student_id)
        autonomy_state = self.autonomy.states.get(student_id)
        autonomy_dashboard = self.autonomy.get_autonomy_dashboard_data()

        return {
            "student_id": student_id,
            "engagement_type": cognitive.engagement_type,
            "cognitive_depth": cognitive.cognitive_depth_mean,
            "cognitive_trajectory": cognitive.cognitive_trajectory,
            "bloom_distribution": cognitive.bloom_distribution,
            "dominant_affect": cognitive.dominant_affect,
            "risk_flags": cognitive.risk_flags,
            "autonomy_phase": autonomy_state.phase if autonomy_state else "unknown",
            "autonomy_score": autonomy_state.autonomy_score if autonomy_state else 0,
            "interaction_quality_tier": quality.quality_tier,
            "pedagogical_value_mean": quality.pedagogical_value_mean,
            "gaming_alerts": len(quality.gaming_alerts),
            "grice_profile": quality.grice_profile,
            "recommendation": quality.recommendation,
        }


# ═══════════════════════════════════════════════════════════════
# EJEMPLO DE USO — Cómo encaja en el flujo del middleware actual
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Simular el flujo completo
    analytics = EnhancedAnalyticsLayer()

    # Simular interacciones de un estudiante que progresa
    test_prompts = [
        # Fase 1: dependiente (pide soluciones directas)
        "dame el código del ejercicio 3",
        "resuelve este programa de bucles",
        "necesito la solución del array",

        # Fase 2: scaffolded (empieza a mostrar intentos)
        "he intentado hacer el bucle for pero no me funciona, "
        "creo que el problema está en el contador",
        "mi código da error en la línea 5, he probado con int y double",

        # Fase 3: emergente (preguntas de análisis)
        "¿cuál es la diferencia entre while y do-while? "
        "He probado ambos y mi programa funciona con los dos",
        "¿por qué se usa recursión si los bucles hacen lo mismo? "
        "Creo que es por la legibilidad pero no estoy seguro",

        # Fase 4: metacognitiva
        "Me doy cuenta de que mi error estaba en confundir scope "
        "de variable local con global. Ahora entiendo por qué "
        "el bucle no modificaba el acumulador.",
    ]

    for i, prompt in enumerate(test_prompts):
        result = analytics.analyze_interaction(
            student_id="est_test_01",
            prompt=prompt,
            scaffolding_level_used=min(i // 2, 3),
            copy_paste_score=0.7 if i < 3 else 0.1,
        )

        print(f"\n{'─'*60}")
        print(f"PROMPT {i+1}: {prompt[:60]}...")
        print(f"  Bloom: {result['cognitive']['bloom_label']} "
              f"(confianza: {result['cognitive']['bloom_confidence']:.0%})")
        print(f"  Acto habla: {result['semiotics']['speech_act_label']} "
              f"(valor ped.: {result['semiotics']['pedagogical_value']:.0%})")
        print(f"  Autonomía: {result['autonomy']['phase']} "
              f"({result['autonomy']['autonomy_score']:.0%})")
        print(f"  Scaffolding rec: nivel {result['scaffolding_recommendation']['recommended_level']} "
              f"(fade: {result['scaffolding_recommendation']['should_fade']})")

    # Perfil final
    print(f"\n{'═'*60}")
    print("PERFIL PROFUNDO DEL ESTUDIANTE")
    print(f"{'═'*60}")
    profile = analytics.get_student_deep_profile("est_test_01")
    for k, v in profile.items():
        print(f"  {k}: {v}")
