"""
MOTOR DE DIAGNÓSTICO POR HIPÓTESIS COMPETITIVAS (ACH EDUCATIVO)
=================================================================
Aplica la metodología ACH de Richards Heuer (Psychology of Intelligence
Analysis, CIA, 1999) al diagnóstico de dificultades de aprendizaje.

El problema:
  El docente ve que un estudiante "no entiende bucles". Pero ¿POR QUÉ?
  Las causas posibles son múltiples y compiten entre sí:
    H1: No domina el concepto de variable (prerrequisito ausente)
    H2: Entiende la lógica pero no la sintaxis (brecha procedimental)
    H3: Comprende ambos pero no sabe cuándo aplicar for vs while (brecha transferencia)
    H4: Tiene ansiedad ante la programación (factor afectivo)
    H5: Copia enunciados sin intentar resolver (factor motivacional)

  Un chatbot convencional trata todas estas causas igual: repite la
  explicación. Un sistema con ACH genera HIPÓTESIS COMPETITIVAS y
  busca EVIDENCIA DISCRIMINANTE en los datos de interacción para
  descartar hipótesis progresivamente.

Marco teórico:
  - Heuer, R. (1999). Psychology of Intelligence Analysis. CIA/CSI.
    Cap. 8: "Analysis of Competing Hypotheses" — método de 8 pasos
    para evaluar hipótesis mutuamente excluyentes contra evidencia.
  - Adaptación educativa: VanLehn (2006). "The behavior of tutoring
    systems" — taxonomía de errores de estudiante (slips, bugs,
    misconceptions) que mapea a hipóteses diagnósticas distintas.
  - Webb (1997). Depth of Knowledge — complementa a Bloom indicando
    no solo QUÉ nivel cognitivo muestra el estudiante, sino qué TIPO
    de conocimiento le falta.

Por qué ACH y no un clasificador ML:
  Un clasificador necesita datos de entrenamiento etiquetados. En un
  piloto con 30 estudiantes y 300 interacciones, NO hay suficientes
  datos. ACH opera con RAZONAMIENTO ESTRUCTURADO sobre evidencia
  cualitativa — exactamente lo que un investigador haría manualmente,
  pero sistematizado. Cuando haya datos suficientes (post-piloto 2),
  se puede entrenar un clasificador supervisado usando los diagnósticos
  ACH como etiquetas. Es decir: ACH genera los ground truth labels
  que el ML necesitará después.

Autor: Diego Elvira Vásquez
Conexión: Máster en Análisis de Inteligencia (LISA-UDIMA, 8.9/10)
"""

from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


# ──────────────────────────────────────────────
# HIPÓTESIS DIAGNÓSTICAS
# ──────────────────────────────────────────────
# Cada hipótesis es una posible explicación de POR QUÉ el estudiante
# no progresa. Las hipótesis son mutuamente competitivas pero NO
# mutuamente excluyentes — un estudiante puede tener múltiples
# problemas simultáneos (esto se refleja en el scoring).

DIAGNOSTIC_HYPOTHESES = {
    "H_PREREQ": {
        "id": "H_PREREQ",
        "name": "Prerrequisito ausente",
        "short": "No domina un concepto previo necesario",
        "description": (
            "El estudiante carece de un concepto fundamental que es "
            "prerrequisito del tema actual. Ejemplo: no puede entender "
            "bucles porque no domina variables o condiciones."
        ),
        "van_lehn_type": "bug",  # error sistemático en el modelo mental
        "intervention": (
            "Retroceder al concepto prerrequisito. El scaffolding socrático "
            "debe apuntar al prerrequisito, no al tema actual."
        ),
    },
    "H_SYNTAX": {
        "id": "H_SYNTAX",
        "name": "Brecha sintáctica",
        "short": "Entiende la lógica pero no la sintaxis del lenguaje",
        "description": (
            "El modelo mental es correcto pero la traducción a código falla. "
            "El estudiante sabe QUÉ quiere hacer pero no CÓMO escribirlo. "
            "Frecuente en principiantes y en cambio entre lenguajes."
        ),
        "van_lehn_type": "slip",  # error de ejecución, no de comprensión
        "intervention": (
            "Proporcionar plantillas de código (code templates) y ejercicios "
            "de completar huecos. El scaffolding puede ser más directo."
        ),
    },
    "H_TRANSFER": {
        "id": "H_TRANSFER",
        "name": "Déficit de transferencia",
        "short": "Comprende el concepto aislado pero no sabe cuándo aplicarlo",
        "description": (
            "El estudiante resuelve ejercicios tipo pero no reconoce "
            "el patrón en problemas nuevos. Falla la transferencia cercana "
            "(near transfer, Perkins & Salomon 1992)."
        ),
        "van_lehn_type": "misconception",
        "intervention": (
            "Presentar problemas isomorfos con enunciados distintos. "
            "Preguntar '¿En qué se parece este problema al anterior?' "
            "Scaffolding socrático orientado a reconocimiento de patrones."
        ),
    },
    "H_AFFECTIVE": {
        "id": "H_AFFECTIVE",
        "name": "Barrera afectiva",
        "short": "Ansiedad, frustración o percepción de incompetencia",
        "description": (
            "El estudiante tiene los prerrequisitos cognitivos pero "
            "factores emocionales bloquean el rendimiento. Frecuente "
            "en programación (Connolly et al., 2018: 'programming anxiety')."
        ),
        "van_lehn_type": "none",  # no es error cognitivo
        "intervention": (
            "Reducir scaffolding (no socrático — genera más frustración). "
            "Modo directo con refuerzo positivo. Problemas más fáciles "
            "para reconstruir autoeficacia (Bandura, 1997)."
        ),
    },
    "H_MOTIVATION": {
        "id": "H_MOTIVATION",
        "name": "Déficit motivacional",
        "short": "No invierte esfuerzo genuino — busca atajos",
        "description": (
            "El estudiante podría entender pero elige no esforzarse. "
            "Indicadores: copy-paste de enunciados, aceptación acrítica, "
            "peticiones directas de solución sin intento previo."
        ),
        "van_lehn_type": "none",
        "intervention": (
            "Aumentar scaffolding socrático (forzar reflexión). "
            "Activar alucinaciones pedagógicas. Reducir prompts diarios. "
            "Si persiste, es señal para intervención docente directa."
        ),
    },
    "H_METACOG": {
        "id": "H_METACOG",
        "name": "Déficit metacognitivo",
        "short": "No sabe qué no sabe — no monitoriza su comprensión",
        "description": (
            "El estudiante no distingue entre 'creo que entiendo' y "
            "'realmente entiendo'. No genera preguntas de verificación. "
            "Dunning-Kruger: sobreestima su competencia."
        ),
        "van_lehn_type": "misconception",
        "intervention": (
            "Pedir explícitamente que reformule ('explícame en tus "
            "propias palabras'). Activar preguntas de predicción "
            "('¿qué crees que dará este código?')."
        ),
    },
}


@dataclass
class EvidenceItem:
    """Una pieza de evidencia observacional del comportamiento del estudiante."""
    source: str         # "cognitive", "trust", "middleware", "temporal", "content"
    description: str
    timestamp: str
    # Compatibilidad con cada hipótesis: +1 (consistente), 0 (neutral), -1 (inconsistente)
    compatibility: dict  # {hypothesis_id: score}


@dataclass
class ACHDiagnosis:
    """Resultado del análisis ACH para un estudiante."""
    student_id: str
    timestamp: str
    hypotheses_scores: dict      # {hypothesis_id: weighted_score}
    leading_hypothesis: str      # id de la hipótesis más probable
    confidence: float            # 0-1, basado en la distancia entre H1 y H2
    evidence_items: list         # lista de EvidenceItem
    matrix: dict                 # matriz completa ACH (hipótesis × evidencia)
    recommended_intervention: str
    reasoning_trace: list        # cadena de razonamiento legible


class ACHDiagnosticEngine:
    """
    Motor de diagnóstico ACH.
    
    Pasos del método Heuer adaptado:
    1. Identificar hipótesis (predefinidas + generadas)
    2. Listar evidencia observable
    3. Construir matriz hipótesis × evidencia
    4. Evaluar compatibilidad de cada evidencia con cada hipótesis
    5. Refinar: ¿qué evidencia discrimina entre hipótesis?
    6. Rankear hipótesis por inconsistencia mínima (Heuer: eliminar
       las más inconsistentes, no confirmar las más consistentes)
    7. Evaluar sensibilidad: ¿una pieza de evidencia cambia el ranking?
    8. Generar diagnóstico con nivel de confianza
    """

    def __init__(self):
        self.hypotheses = DIAGNOSTIC_HYPOTHESES
        self.diagnoses_history: list[ACHDiagnosis] = []

    def diagnose(
        self,
        student_id: str,
        bloom_levels: list[int],
        bloom_mean: float,
        bloom_trend: float,
        metacognitive_ratio: float,
        copypaste_scores: list[float],
        trust_calibration: float,
        scaffolding_levels_reached: list[int],
        topic_distribution: dict,
        n_interactions: int,
    ) -> ACHDiagnosis:
        """
        Ejecuta el análisis ACH completo sobre los datos de un estudiante.
        """
        evidence = []
        reasoning = []

        # ── PASO 2: Recoger evidencia observable ──

        # E1: Nivel cognitivo medio
        e1_compat = {}
        if bloom_mean < 2.0:
            e1_compat = {
                "H_PREREQ": 0.7, "H_SYNTAX": 0.3, "H_TRANSFER": -0.3,
                "H_AFFECTIVE": 0.2, "H_MOTIVATION": 0.4, "H_METACOG": 0.3,
            }
            reasoning.append(
                f"Bloom medio bajo ({bloom_mean:.1f}): consistente con prerrequisito "
                f"ausente o motivación baja. Inconsistente con déficit de transferencia "
                f"(requeriría al menos nivel 3 previo)."
            )
        elif bloom_mean < 3.5:
            e1_compat = {
                "H_PREREQ": -0.3, "H_SYNTAX": 0.5, "H_TRANSFER": 0.5,
                "H_AFFECTIVE": 0.2, "H_MOTIVATION": 0.0, "H_METACOG": 0.3,
            }
            reasoning.append(
                f"Bloom medio intermedio ({bloom_mean:.1f}): prerrequisitos probablemente "
                f"cubiertos. Consistente con brechas de sintaxis o transferencia."
            )
        else:
            e1_compat = {
                "H_PREREQ": -0.8, "H_SYNTAX": -0.3, "H_TRANSFER": 0.2,
                "H_AFFECTIVE": -0.5, "H_MOTIVATION": -0.7, "H_METACOG": -0.2,
            }
            reasoning.append(
                f"Bloom medio alto ({bloom_mean:.1f}): descarta prerrequisito ausente "
                f"y déficit motivacional con alta confianza."
            )

        evidence.append(EvidenceItem(
            source="cognitive",
            description=f"Nivel cognitivo medio: {bloom_mean:.1f}/6",
            timestamp=datetime.now().isoformat(),
            compatibility=e1_compat,
        ))

        # E2: Tendencia cognitiva
        e2_compat = {}
        if bloom_trend < -0.15:
            e2_compat = {
                "H_PREREQ": 0.3, "H_SYNTAX": 0.0, "H_TRANSFER": 0.0,
                "H_AFFECTIVE": 0.7, "H_MOTIVATION": 0.5, "H_METACOG": 0.2,
            }
            reasoning.append(
                f"Tendencia descendente ({bloom_trend:+.3f}): fuerte señal de "
                f"frustración/barrera afectiva. El estudiante regresa a estrategias "
                f"más simples — patrón clásico de evitación."
            )
        elif bloom_trend > 0.15:
            e2_compat = {
                "H_PREREQ": -0.5, "H_SYNTAX": -0.2, "H_TRANSFER": -0.2,
                "H_AFFECTIVE": -0.6, "H_MOTIVATION": -0.7, "H_METACOG": -0.3,
            }
            reasoning.append(
                f"Tendencia ascendente ({bloom_trend:+.3f}): progresión cognitiva "
                f"activa. Descarta déficits graves en todas las hipótesis."
            )
        else:
            e2_compat = {
                "H_PREREQ": 0.1, "H_SYNTAX": 0.2, "H_TRANSFER": 0.3,
                "H_AFFECTIVE": 0.0, "H_MOTIVATION": 0.1, "H_METACOG": 0.4,
            }
            reasoning.append(
                f"Tendencia estable ({bloom_trend:+.3f}): meseta cognitiva. "
                f"Consistente con déficit metacognitivo (no percibe que no avanza) "
                f"o con transferencia limitada (domina lo que sabe, no amplía)."
            )

        evidence.append(EvidenceItem(
            source="cognitive",
            description=f"Tendencia cognitiva: {bloom_trend:+.3f}",
            timestamp=datetime.now().isoformat(),
            compatibility=e2_compat,
        ))

        # E3: Tasa de copy-paste
        mean_cp = sum(copypaste_scores) / max(len(copypaste_scores), 1)
        e3_compat = {}
        if mean_cp > 0.5:
            e3_compat = {
                "H_PREREQ": 0.2, "H_SYNTAX": 0.1, "H_TRANSFER": 0.0,
                "H_AFFECTIVE": 0.1, "H_MOTIVATION": 0.8, "H_METACOG": 0.1,
            }
            reasoning.append(
                f"Copy-paste alto ({mean_cp:.0%}): señal fuerte de déficit motivacional. "
                f"El estudiante delega el problema al chatbot sin procesarlo."
            )
        else:
            e3_compat = {
                "H_PREREQ": 0.0, "H_SYNTAX": 0.0, "H_TRANSFER": 0.0,
                "H_AFFECTIVE": 0.0, "H_MOTIVATION": -0.6, "H_METACOG": 0.0,
            }
            reasoning.append(
                f"Copy-paste bajo ({mean_cp:.0%}): descarta motivación como problema "
                f"primario — el estudiante formula sus propias preguntas."
            )

        evidence.append(EvidenceItem(
            source="middleware",
            description=f"Tasa de copy-paste: {mean_cp:.0%}",
            timestamp=datetime.now().isoformat(),
            compatibility=e3_compat,
        ))

        # E4: Metacognición
        e4_compat = {}
        if metacognitive_ratio > 0.2:
            e4_compat = {
                "H_PREREQ": 0.0, "H_SYNTAX": 0.0, "H_TRANSFER": 0.0,
                "H_AFFECTIVE": 0.1, "H_MOTIVATION": -0.4, "H_METACOG": -0.8,
            }
            reasoning.append(
                f"Metacognición alta ({metacognitive_ratio:.0%}): descarta déficit "
                f"metacognitivo con alta confianza. El estudiante monitoriza "
                f"activamente su comprensión."
            )
        elif metacognitive_ratio < 0.05:
            e4_compat = {
                "H_PREREQ": 0.0, "H_SYNTAX": 0.0, "H_TRANSFER": 0.1,
                "H_AFFECTIVE": -0.1, "H_MOTIVATION": 0.2, "H_METACOG": 0.7,
            }
            reasoning.append(
                f"Metacognición ausente ({metacognitive_ratio:.0%}): señal consistente "
                f"con déficit metacognitivo. El estudiante no verifica su comprensión."
            )
        else:
            e4_compat = {h: 0.0 for h in self.hypotheses}

        evidence.append(EvidenceItem(
            source="cognitive",
            description=f"Ratio metacognitivo: {metacognitive_ratio:.0%}",
            timestamp=datetime.now().isoformat(),
            compatibility=e4_compat,
        ))

        # E5: Calibración de confianza
        e5_compat = {}
        if trust_calibration > 0.3:
            e5_compat = {
                "H_PREREQ": 0.1, "H_SYNTAX": 0.0, "H_TRANSFER": 0.0,
                "H_AFFECTIVE": -0.3, "H_MOTIVATION": 0.5, "H_METACOG": 0.5,
            }
            reasoning.append(
                f"Sobre-confianza en IA ({trust_calibration:+.2f}): consistente con "
                f"déficit metacognitivo (no distingue respuesta correcta de incorrecta) "
                f"y motivacional (delega sin verificar)."
            )
        elif trust_calibration < -0.3:
            e5_compat = {
                "H_PREREQ": 0.1, "H_SYNTAX": 0.1, "H_TRANSFER": 0.0,
                "H_AFFECTIVE": 0.6, "H_MOTIVATION": -0.3, "H_METACOG": -0.2,
            }
            reasoning.append(
                f"Infra-confianza en IA ({trust_calibration:+.2f}): señal de barrera "
                f"afectiva. Frustración con el sistema o con el proceso de aprendizaje."
            )
        else:
            e5_compat = {h: 0.0 for h in self.hypotheses}

        evidence.append(EvidenceItem(
            source="trust",
            description=f"Calibración de confianza: {trust_calibration:+.2f}",
            timestamp=datetime.now().isoformat(),
            compatibility=e5_compat,
        ))

        # E6: Concentración temática
        if topic_distribution:
            total_topics = sum(topic_distribution.values())
            max_topic = max(topic_distribution.values()) if topic_distribution else 0
            concentration = max_topic / total_topics if total_topics > 0 else 0

            e6_compat = {}
            if concentration > 0.6:
                e6_compat = {
                    "H_PREREQ": 0.5, "H_SYNTAX": 0.3, "H_TRANSFER": -0.3,
                    "H_AFFECTIVE": 0.0, "H_MOTIVATION": 0.0, "H_METACOG": 0.0,
                }
                dominant = max(topic_distribution, key=topic_distribution.get)
                reasoning.append(
                    f"Concentración temática alta ({concentration:.0%} en '{dominant}'): "
                    f"el estudiante está atascado en un tema — consistente con "
                    f"prerrequisito ausente para ese tema específico."
                )
            else:
                e6_compat = {
                    "H_PREREQ": -0.3, "H_SYNTAX": 0.0, "H_TRANSFER": 0.4,
                    "H_AFFECTIVE": 0.0, "H_MOTIVATION": 0.0, "H_METACOG": 0.0,
                }
                reasoning.append(
                    f"Distribución temática amplia ({concentration:.0%}): el estudiante "
                    f"explora varios temas. Menos probable que sea prerrequisito "
                    f"ausente. Puede ser transferencia (sabe un poco de todo, no conecta)."
                )

            evidence.append(EvidenceItem(
                source="content",
                description=f"Concentración temática: {concentration:.0%}",
                timestamp=datetime.now().isoformat(),
                compatibility=e6_compat,
            ))

        # ── PASO 3-4: Construir matriz y calcular scores ──
        # Método Heuer: la hipótesis ganadora NO es la que tiene más
        # evidencia a favor, sino la que tiene MENOS evidencia en contra.
        # Esto contrarresta el sesgo de confirmación.

        hypothesis_scores = {}
        matrix = {}

        for h_id in self.hypotheses:
            positive = 0  # evidencia consistente
            negative = 0  # evidencia inconsistente
            row = []

            for e in evidence:
                compat = e.compatibility.get(h_id, 0)
                row.append(compat)
                if compat > 0:
                    positive += compat
                elif compat < 0:
                    negative += abs(compat)

            matrix[h_id] = row

            # Score Heuer: penalizar inconsistencia más que premiar consistencia
            # Factor 1.5 en negativo — el método ACH privilegia desconfirmación
            hypothesis_scores[h_id] = round(positive - (negative * 1.5), 3)

        # ── PASO 6: Rankear hipótesis ──
        ranked = sorted(hypothesis_scores.items(), key=lambda x: x[1], reverse=True)
        leading = ranked[0][0]
        leading_score = ranked[0][1]
        second_score = ranked[1][1] if len(ranked) > 1 else 0

        # Confianza: distancia normalizada entre H1 y H2
        score_range = max(abs(leading_score), abs(second_score), 1)
        confidence = min(abs(leading_score - second_score) / score_range, 1.0)

        # ── PASO 8: Generar diagnóstico ──
        diagnosis = ACHDiagnosis(
            student_id=student_id,
            timestamp=datetime.now().isoformat(),
            hypotheses_scores=hypothesis_scores,
            leading_hypothesis=leading,
            confidence=round(confidence, 2),
            evidence_items=evidence,
            matrix=matrix,
            recommended_intervention=self.hypotheses[leading]["intervention"],
            reasoning_trace=reasoning,
        )

        self.diagnoses_history.append(diagnosis)
        return diagnosis

    def get_hypothesis_info(self, hypothesis_id: str) -> dict:
        """Devuelve la información completa de una hipótesis."""
        return self.hypotheses.get(hypothesis_id, {})

    def format_diagnosis_report(self, diagnosis: ACHDiagnosis) -> str:
        """Genera un informe legible del diagnóstico."""
        h_info = self.hypotheses[diagnosis.leading_hypothesis]
        lines = [
            f"══ DIAGNÓSTICO ACH — Estudiante {diagnosis.student_id} ══",
            f"",
            f"HIPÓTESIS PRINCIPAL: {h_info['name']}",
            f"  → {h_info['short']}",
            f"  Confianza: {diagnosis.confidence:.0%}",
            f"  Tipo (VanLehn): {h_info['van_lehn_type']}",
            f"",
            f"RANKING DE HIPÓTESIS:",
        ]

        ranked = sorted(diagnosis.hypotheses_scores.items(), key=lambda x: x[1], reverse=True)
        for h_id, score in ranked:
            h = self.hypotheses[h_id]
            marker = "→" if h_id == diagnosis.leading_hypothesis else " "
            lines.append(f"  {marker} {h['name']}: {score:+.3f}")

        lines.extend([
            f"",
            f"CADENA DE RAZONAMIENTO:",
        ])
        for i, r in enumerate(diagnosis.reasoning_trace, 1):
            lines.append(f"  {i}. {r}")

        lines.extend([
            f"",
            f"INTERVENCIÓN RECOMENDADA:",
            f"  {h_info['intervention']}",
        ])

        return "\n".join(lines)
