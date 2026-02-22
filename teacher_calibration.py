"""
TEACHER CALIBRATION MODULE
═══════════════════════════════════════════════════════════════════════
Módulo diferencial #5 — Analytics de la Interpretación Docente

PROBLEMA QUE ATACA — EL AXIOMA SILENCIOSO:
════════════════════════════════════════════
El paradigma HCAI de Dimitriadis (2021, "Human-Centered Design Principles
for Actionable Learning Analytics") establece el flujo canónico:

    datos de estudiantes → dashboard → docente actúa → mejora

"The Teacher in the Loop" (Rodríguez-Triana et al., 2018) cierra el ciclo:
el docente personaliza, el sistema responde. Correcto. Incompleto.

El axioma silencioso que ningún paper del grupo ha cuestionado:
    QUE EL DOCENTE LEE EL DASHBOARD CORRECTAMENTE.

Este módulo opera la pregunta inversa: ¿cómo de bien calibrada está la
interpretación docente del estado real de sus estudiantes?

FUNDAMENTO TEÓRICO:
────────────────────
1. Cognitive Bias in Teacher Decision-Making (Kahneman, 2011)
   Los docentes están sujetos a los mismos sesgos que cualquier decisor:
   - Availability heuristic: reaccionan al último estudiante visible,
     no al perfil estadístico del grupo.
   - Anchoring: la primera impresión del estudiante contamina lecturas
     posteriores aunque los datos hayan cambiado.
   - Recency bias: sobrevaloran las últimas N interacciones ignorando
     la trayectoria completa.
   - Volume bias: confunden actividad (muchos prompts) con engagement
     de calidad (prompts de nivel alto).

2. Calibration Theory (Lichtenstein & Fischhoff, 1977)
   La calibración es la correspondencia entre la confianza/certeza de un
   decisor y la precisión real de sus juicios. Un docente bien calibrado
   activa scaffolding socrático CUANDO el estudiante está en modo consumo
   (no cuando está en exploración). Un docente mal calibrado aplica la
   misma intervención independientemente del estado real.

3. Actionability y el problema del isomorfismo (Dimitriadis & Wiley, 2021)
   Los principios HCAI de Dimitriadis requieren que los analytics sean
   "actionable" — que generen acciones docentes apropiadas. Pero
   actionability sin calibración es ruido: una acción tomada sobre una
   lectura incorrecta del dashboard puede ser peor que no actuar.
   Este módulo cierra ese loop: mide si las acciones fueron apropiadas
   DADO el estado que los analytics mostraban en ese momento.

4. Expertise Reversal Effect (Kalyuga et al., 2003)
   Docentes expertos tienden a sobre-scaffoldear a estudiantes que ya
   han ganado autonomía. El modelo detecta este patrón específico —
   intervención instructiva cuando los datos señalan autonomía emergente —
   como un bias sistemático de tipo "expertise reversal".

5. Sociología del conocimiento (Mannheim, 1936)
   El docente llega al dashboard con un marco interpretativo previo
   (sus teorías implícitas sobre el aprendizaje). Ese marco determina
   qué datos selecciona y cómo los interpreta. Este módulo cartografía
   ese marco implícito haciéndolo visible — no para criticarlo sino
   para devolvérselo al docente como dato de su propia práctica.

INNOVACIÓN RADICAL:
───────────────────
Ningún sistema de learning analytics existente convierte al docente en
sujeto de análisis además de en destinatario. Este módulo es el primero
en medir la fidelidad interpretativa del docente sobre los propios
analytics que el sistema le entrega. El docente no pierde agencia —
gana un espejo epistémico de sus sesgos interpretativos.

Esto amplía la línea de teacher agency de Alonso-Prieto (LASI 2025):
la agencia no solo debe medirse en decisiones de diseño sino en la
calidad interpretativa durante la orquestación.

NO REQUIERE INSTRUMENTACIÓN ADICIONAL:
Todos los datos necesarios ya existen en los logs del sistema:
- Timestamps de cambios de configuración (ya registrados por middleware)
- Estado del estudiante en ese momento (ya calculado por los módulos 1-4)
- La configuración elegida (ya almacenada en PedagogicalConfig)

Autor: Diego Elvira Vásquez · Prototipo CP25/152 · Feb 2026
"""

from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
from datetime import datetime
import math


# ═══════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StudentStateSnapshot:
    """
    Estado inferido del estudiante en el momento de una intervención docente.
    Captura todos los indicadores disponibles en ese instante.
    """
    timestamp: str
    student_id: str

    # Del cognitive_analyzer / cognitive_profiler
    bloom_level: str = "recordar"           # último nivel Bloom clasificado
    bloom_weight: float = 1.0               # peso numérico 1.0-6.0
    bloom_trend: str = "stable"             # ascending | descending | stable
    bloom_mean_last5: float = 2.0           # media últimas 5 interacciones

    # Del epistemic_autonomy
    epistemic_mode: str = "consumption"     # consumption | verification | exploration | delegation
    autonomy_score: float = 0.3             # 0.0-1.0
    autonomy_trend: str = "stable"          # ascending | descending | stable

    # Del interaction_semiotics
    speech_act_score: float = 30.0          # 0-100 valor pedagógico
    gaming_risk: float = 0.0               # 0.0-1.0

    # Del trust_dynamics
    trust_calibration: float = 0.0         # -1.0 (infra) a +1.0 (sobre)

    # Patrones temporales
    prompt_volume_last_session: int = 0     # prompts en sesión actual
    inter_prompt_latency_mean: float = 60.0 # segundos entre prompts
    session_duration_minutes: float = 15.0

    # Señales de silencio (del epistemic_silence_detector)
    silence_anomaly_score: float = 0.0     # 0.0-1.0 (1.0 = silencio crítico)


@dataclass
class TeacherIntervention:
    """
    Una intervención docente: cambio de configuración pedagógica con contexto.
    """
    timestamp: str
    teacher_id: str
    student_id: str                         # a qué estudiante afecta (o "all")
    course_id: str

    # Qué cambió
    config_key: str                         # p.ej. "socratic_scaffolding"
    config_old_value: object = None
    config_new_value: object = None

    # Estado inferido del estudiante EN ESE MOMENTO
    student_state: Optional[StudentStateSnapshot] = None

    # Supuesto implícito de la intervención (calculado automáticamente)
    assumed_student_state: str = ""         # descripción del estado que justificaría esta acción

    # Resultado de calibración (calculado post-hoc)
    calibration_score: float = 0.0         # -1.0 (inverso) a +1.0 (perfecto)
    calibration_label: str = ""            # perfect | good | neutral | poor | inverted
    mismatch_description: str = ""         # qué parte no coincidió


@dataclass
class TeacherBiasProfile:
    """
    Perfil epistémico acumulado de un docente: sus sesgos interpretativos.
    """
    teacher_id: str
    n_interventions: int = 0
    n_calibrated: int = 0                  # bien calibradas
    n_miscalibrated: int = 0               # mal calibradas
    n_inverted: int = 0                    # invertidas (estado opuesto al supuesto)

    calibration_mean: float = 0.0          # media de calibration_score
    calibration_trend: str = "insufficient_data"

    # Sesgos específicos detectados
    volume_bias: float = 0.0               # tiende a reaccionar al volumen
    recency_bias: float = 0.0             # sobrepondera últimas interacciones
    autonomy_blindspot: float = 0.0       # activa scaffolding aunque haya autonomía
    delegation_miss_rate: float = 0.0     # no detecta delegación cognitiva

    # Fortalezas detectadas
    socratic_calibration: float = 0.0     # precisión al activar modo socrático
    limit_calibration: float = 0.0        # precisión al ajustar límite de prompts

    # Recomendación para el investigador (no se muestra al docente)
    researcher_note: str = ""


# ═══════════════════════════════════════════════════════════════════════
# MAPA CONFIGURACIÓN → SUPUESTO IMPLÍCITO
# ═══════════════════════════════════════════════════════════════════════

# Para cada posible cambio de configuración, qué estado del estudiante
# justificaría RACIONALMENTE esa intervención según la teoría pedagógica.
# Este mapa es la piedra angular del módulo: compara la acción docente
# con lo que la acción debería suponer sobre el estado del estudiante.

CONFIG_ASSUMPTION_MAP = {
    "socratic_scaffolding": {
        True: {
            "description": "Activa socrático: supone que el estudiante está en modo consumo "
                           "o delegación, con Bloom bajo (≤ 2) y autonomía baja (≤ 0.4). "
                           "El docente cree que el estudiante necesita ser guiado para "
                           "construir conocimiento propio en lugar de recibirlo.",
            "expected_epistemic_mode": ["consumption", "delegation"],
            "expected_bloom_max": 2.5,
            "expected_autonomy_max": 0.45,
            "expected_speech_act_max": 40.0,
        },
        False: {
            "description": "Desactiva socrático: supone que el estudiante ha ganado "
                           "suficiente autonomía (exploración/verificación, Bloom ≥ 3) "
                           "y el scaffolding socrático ya no es necesario — puede incluso "
                           "ser contraproducente (expertise reversal effect).",
            "expected_epistemic_mode": ["exploration", "verification"],
            "expected_bloom_min": 3.0,
            "expected_autonomy_min": 0.5,
        },
    },
    "no_direct_solutions": {
        True: {
            "description": "Bloquea soluciones directas: supone que el estudiante está "
                           "externalizando cognitivamente (delegación o consumo) y que "
                           "recibir la solución directa impediría el aprendizaje.",
            "expected_epistemic_mode": ["consumption", "delegation"],
            "expected_bloom_max": 2.0,
            "expected_gaming_risk_min": 0.3,
        },
        False: {
            "description": "Permite soluciones directas: supone que el estudiante tiene "
                           "contexto suficiente para aprovechar una solución (verificación "
                           "o exploración activa) y que el bloqueo generaría frustración "
                           "sin beneficio pedagógico.",
            "expected_epistemic_mode": ["verification", "exploration"],
            "expected_bloom_min": 3.0,
        },
    },
    "max_daily_prompts": {
        "reduce": {
            "description": "Reduce límite de prompts: supone alto volumen de uso superficial "
                           "(muchos prompts, bajo nivel Bloom, modo delegativo). El docente "
                           "cree que la restricción forzará una elaboración más cuidadosa "
                           "antes de consultar.",
            "expected_prompt_volume_min": 8,
            "expected_bloom_max": 2.5,
            "expected_epistemic_mode": ["consumption", "delegation"],
        },
        "increase": {
            "description": "Aumenta límite de prompts: supone que el estudiante está "
                           "en exploración activa y el límite está restringiendo un "
                           "proceso de aprendizaje genuinamente iterativo.",
            "expected_epistemic_mode": ["exploration", "verification"],
            "expected_bloom_min": 3.0,
        },
    },
    "hallucination_rate": {
        "increase": {
            "description": "Sube tasa de alucinaciones: supone que el estudiante acepta "
                           "respuestas sin verificar (sobre-confianza, trust_calibration > 0.4) "
                           "y necesita estímulo para desarrollar lectura crítica.",
            "expected_trust_calibration_min": 0.3,
            "expected_speech_act_max": 50.0,
        },
        "decrease": {
            "description": "Baja tasa de alucinaciones: supone que el estudiante ya "
                           "verifica activamente o que la confusión generada es "
                           "contraproducente para su estado actual.",
            "expected_trust_calibration_max": 0.2,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════
# MOTOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════

class TeacherCalibrationAnalyzer:
    """
    Analiza la calibración interpretativa de los docentes.

    Uso en el flujo del sistema:
    1. Cada vez que un docente cambia una configuración:
       analyzer.record_intervention(teacher_id, config_key, old_val, new_val,
                                    student_state_snapshot)
    2. El módulo calcula automáticamente el score de calibración.
    3. El investigador (no el docente) accede al perfil de sesgos:
       analyzer.get_teacher_bias_profile(teacher_id)
    4. Para el docente, se genera un resumen anónimo de calibración sin
       mencionar "sesgo": solo "precisión de intervención" y sugerencias.
    """

    def __init__(self):
        self.interventions: list[TeacherIntervention] = []
        self.teacher_profiles: dict[str, TeacherBiasProfile] = {}
        self.course_data: dict[str, list[StudentStateSnapshot]] = defaultdict(list)

    # ──────────────────────────────────────────────────────────────────
    # REGISTRO DE INTERVENCIONES
    # ──────────────────────────────────────────────────────────────────

    def record_intervention(
        self,
        teacher_id: str,
        course_id: str,
        student_id: str,
        config_key: str,
        old_value: object,
        new_value: object,
        student_state: StudentStateSnapshot,
    ) -> TeacherIntervention:
        """
        Registra una intervención docente y calcula su calibración.

        Se invoca cada vez que el docente cambia una configuración en el
        sidebar de Streamlit, con el estado actual del estudiante como contexto.
        """
        intervention = TeacherIntervention(
            timestamp=datetime.now().isoformat(),
            teacher_id=teacher_id,
            course_id=course_id,
            student_id=student_id,
            config_key=config_key,
            config_old_value=old_value,
            config_new_value=new_value,
            student_state=student_state,
        )

        # Calcular supuesto implícito de la intervención
        assumed_state, calibration = self._evaluate_calibration(
            config_key, old_value, new_value, student_state
        )
        intervention.assumed_student_state = assumed_state
        intervention.calibration_score = calibration["score"]
        intervention.calibration_label = calibration["label"]
        intervention.mismatch_description = calibration["mismatch"]

        self.interventions.append(intervention)
        self._update_teacher_profile(teacher_id, intervention)

        return intervention

    # ──────────────────────────────────────────────────────────────────
    # EVALUACIÓN DE CALIBRACIÓN
    # ──────────────────────────────────────────────────────────────────

    def _evaluate_calibration(
        self,
        config_key: str,
        old_value: object,
        new_value: object,
        state: StudentStateSnapshot,
    ) -> tuple[str, dict]:
        """
        Evalúa cuán calibrada está la intervención respecto al estado real.

        Lógica: cada intervención implica un supuesto sobre el estado del
        estudiante. Comparamos ese supuesto con el estado real medido.

        Returns:
            (assumed_state_description, calibration_dict)
        """
        if config_key not in CONFIG_ASSUMPTION_MAP:
            return (
                "Intervención sin mapa de supuestos definido.",
                {"score": 0.0, "label": "neutral", "mismatch": "Configuración no modelada."},
            )

        config_map = CONFIG_ASSUMPTION_MAP[config_key]

        # Determinar la dirección del cambio
        if isinstance(new_value, bool):
            direction = new_value
        elif isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
            direction = "reduce" if new_value < old_value else "increase"
        else:
            direction = new_value

        if direction not in config_map:
            return (
                "Dirección de cambio no modelada.",
                {"score": 0.0, "label": "neutral", "mismatch": ""},
            )

        expectation = config_map[direction]
        assumed_desc = expectation["description"]

        # Calcular calibración comparando expectativas con estado real
        score, mismatch = self._compute_calibration_score(expectation, state)

        label = self._score_to_label(score)

        return assumed_desc, {"score": score, "label": label, "mismatch": mismatch}

    def _compute_calibration_score(
        self,
        expectation: dict,
        state: StudentStateSnapshot,
    ) -> tuple[float, str]:
        """
        Compara cada dimensión esperada con el estado real.
        Devuelve score (-1.0 a 1.0) y descripción del desajuste.

        Score:
          +1.0 = estado real coincide perfectamente con el supuesto
           0.0 = estado neutral / no concluyente
          -1.0 = estado real es el opuesto del supuesto (intervención invertida)
        """
        checks = []
        mismatches = []

        # ── Modo epistémico ──
        if "expected_epistemic_mode" in expectation:
            expected_modes = expectation["expected_epistemic_mode"]
            if state.epistemic_mode in expected_modes:
                checks.append(1.0)
            else:
                checks.append(-1.0)
                mismatches.append(
                    f"modo epistémico real='{state.epistemic_mode}' "
                    f"vs esperado={expected_modes}"
                )

        # ── Bloom máximo ──
        if "expected_bloom_max" in expectation:
            if state.bloom_mean_last5 <= expectation["expected_bloom_max"]:
                checks.append(1.0)
            else:
                gap = state.bloom_mean_last5 - expectation["expected_bloom_max"]
                penalty = min(gap / 2.0, 1.0)
                checks.append(-penalty)
                mismatches.append(
                    f"Bloom real={state.bloom_mean_last5:.1f} > "
                    f"máximo esperado={expectation['expected_bloom_max']}"
                )

        # ── Bloom mínimo ──
        if "expected_bloom_min" in expectation:
            if state.bloom_mean_last5 >= expectation["expected_bloom_min"]:
                checks.append(1.0)
            else:
                gap = expectation["expected_bloom_min"] - state.bloom_mean_last5
                penalty = min(gap / 2.0, 1.0)
                checks.append(-penalty)
                mismatches.append(
                    f"Bloom real={state.bloom_mean_last5:.1f} < "
                    f"mínimo esperado={expectation['expected_bloom_min']}"
                )

        # ── Autonomía máxima ──
        if "expected_autonomy_max" in expectation:
            if state.autonomy_score <= expectation["expected_autonomy_max"]:
                checks.append(1.0)
            else:
                gap = state.autonomy_score - expectation["expected_autonomy_max"]
                penalty = min(gap * 2.0, 1.0)
                checks.append(-penalty)
                mismatches.append(
                    f"autonomía real={state.autonomy_score:.2f} > "
                    f"máxima esperada={expectation['expected_autonomy_max']}"
                )

        # ── Autonomía mínima ──
        if "expected_autonomy_min" in expectation:
            if state.autonomy_score >= expectation["expected_autonomy_min"]:
                checks.append(1.0)
            else:
                gap = expectation["expected_autonomy_min"] - state.autonomy_score
                penalty = min(gap * 2.0, 1.0)
                checks.append(-penalty)
                mismatches.append(
                    f"autonomía real={state.autonomy_score:.2f} < "
                    f"mínima esperada={expectation['expected_autonomy_min']}"
                )

        # ── Volumen de prompts ──
        if "expected_prompt_volume_min" in expectation:
            if state.prompt_volume_last_session >= expectation["expected_prompt_volume_min"]:
                checks.append(1.0)
            else:
                checks.append(-0.5)
                mismatches.append(
                    f"volumen real={state.prompt_volume_last_session} < "
                    f"mínimo esperado={expectation['expected_prompt_volume_min']}"
                )

        # ── Trust calibration ──
        if "expected_trust_calibration_min" in expectation:
            if state.trust_calibration >= expectation["expected_trust_calibration_min"]:
                checks.append(1.0)
            else:
                checks.append(-0.5)
                mismatches.append(
                    f"trust real={state.trust_calibration:.2f} < "
                    f"mínimo esperado={expectation['expected_trust_calibration_min']}"
                )

        # ── Speech act score máximo ──
        if "expected_speech_act_max" in expectation:
            if state.speech_act_score <= expectation["expected_speech_act_max"]:
                checks.append(1.0)
            else:
                checks.append(-0.3)
                mismatches.append(
                    f"speech act score real={state.speech_act_score:.0f} > "
                    f"máximo esperado={expectation['expected_speech_act_max']}"
                )

        # ── Risk de gaming ──
        if "expected_gaming_risk_min" in expectation:
            if state.gaming_risk >= expectation["expected_gaming_risk_min"]:
                checks.append(1.0)
            else:
                checks.append(-0.5)
                mismatches.append(
                    f"gaming risk real={state.gaming_risk:.2f} < "
                    f"mínimo esperado={expectation['expected_gaming_risk_min']}"
                )

        if not checks:
            return 0.0, "Sin dimensiones evaluables para esta configuración."

        score = sum(checks) / len(checks)
        mismatch_text = "; ".join(mismatches) if mismatches else "Calibración correcta."

        return score, mismatch_text

    @staticmethod
    def _score_to_label(score: float) -> str:
        if score >= 0.7:
            return "perfect"
        elif score >= 0.3:
            return "good"
        elif score >= -0.2:
            return "neutral"
        elif score >= -0.6:
            return "poor"
        else:
            return "inverted"

    # ──────────────────────────────────────────────────────────────────
    # ACTUALIZACIÓN DEL PERFIL DOCENTE
    # ──────────────────────────────────────────────────────────────────

    def _update_teacher_profile(
        self,
        teacher_id: str,
        intervention: TeacherIntervention,
    ) -> None:
        """Actualiza el perfil de sesgos del docente con la nueva intervención."""
        if teacher_id not in self.teacher_profiles:
            self.teacher_profiles[teacher_id] = TeacherBiasProfile(teacher_id=teacher_id)

        profile = self.teacher_profiles[teacher_id]
        profile.n_interventions += 1

        label = intervention.calibration_label
        if label in ("perfect", "good"):
            profile.n_calibrated += 1
        elif label in ("poor", "inverted"):
            profile.n_miscalibrated += 1
        if label == "inverted":
            profile.n_inverted += 1

        # Actualizar media de calibración (media móvil exponencial)
        alpha = 0.3
        profile.calibration_mean = (
            alpha * intervention.calibration_score +
            (1 - alpha) * profile.calibration_mean
        )

        state = intervention.student_state
        if state is None:
            return

        # ── Detección de volume bias ──
        # El docente actúa sobre volumen aunque Bloom sea alto
        if (intervention.config_key == "max_daily_prompts" and
                state.prompt_volume_last_session > 5 and
                state.bloom_mean_last5 >= 3.5):
            profile.volume_bias = min(1.0, profile.volume_bias + 0.15)

        # ── Detección de recency bias ──
        # Si hay intervenciones cercanas en tiempo con estados muy distintos
        recent = [i for i in self.interventions[-10:]
                  if i.teacher_id == teacher_id and i.student_id == intervention.student_id]
        if len(recent) >= 3:
            recent_scores = [i.calibration_score for i in recent]
            variance = sum((s - sum(recent_scores)/len(recent_scores))**2
                          for s in recent_scores) / len(recent_scores)
            if variance > 0.4:
                profile.recency_bias = min(1.0, profile.recency_bias + 0.1)

        # ── Detección de autonomy blindspot ──
        # Activa socrático cuando el estudiante ya está en exploración
        if (intervention.config_key == "socratic_scaffolding" and
                intervention.config_new_value is True and
                state.epistemic_mode == "exploration"):
            profile.autonomy_blindspot = min(1.0, profile.autonomy_blindspot + 0.2)

        # ── Tasa de miss en delegación ──
        # No activa protecciones cuando el estudiante está delegando
        if (state.epistemic_mode == "delegation" and
                intervention.config_key not in ("no_direct_solutions", "socratic_scaffolding")):
            profile.delegation_miss_rate = min(1.0, profile.delegation_miss_rate + 0.1)

        # ── Tendencia de calibración ──
        if profile.n_interventions >= 5:
            recent_5 = [i.calibration_score
                        for i in self.interventions
                        if i.teacher_id == teacher_id][-5:]
            delta = recent_5[-1] - recent_5[0]
            if delta > 0.15:
                profile.calibration_trend = "improving"
            elif delta < -0.15:
                profile.calibration_trend = "declining"
            else:
                profile.calibration_trend = "stable"
        else:
            profile.calibration_trend = "insufficient_data"

        # ── Nota para el investigador ──
        profile.researcher_note = self._generate_researcher_note(profile)

    def _generate_researcher_note(self, profile: TeacherBiasProfile) -> str:
        """
        Genera una nota de análisis para el investigador (no para el docente).
        Escrita en el lenguaje del análisis de datos educativos.
        """
        notes = []

        calibration_pct = (profile.n_calibrated / profile.n_interventions * 100
                           if profile.n_interventions > 0 else 0)
        notes.append(
            f"Calibración general: {calibration_pct:.0f}% de intervenciones bien calibradas "
            f"({profile.n_interventions} total, {profile.n_inverted} invertidas)."
        )

        if profile.volume_bias > 0.4:
            notes.append(
                "SESGO DETECTADO — Volume bias (Kahneman, 2011): el docente reacciona "
                "al volumen de prompts ignorando la distribución de calidad. "
                "Recomendación: añadir prominencia visual al score de Bloom en el dashboard."
            )

        if profile.autonomy_blindspot > 0.3:
            notes.append(
                "SESGO DETECTADO — Expertise reversal effect (Kalyuga et al., 2003): "
                "el docente activa scaffolding socrático cuando el estudiante ya está "
                "en exploración autónoma. El andamiaje puede ser contraproducente aquí."
            )

        if profile.recency_bias > 0.3:
            notes.append(
                "SESGO DETECTADO — Recency bias: alta varianza en calibración a lo largo "
                "del tiempo, consistente con sobrepondera interacciones recientes. "
                "Considerar añadir visualización de trayectoria (no solo estado actual)."
            )

        if profile.delegation_miss_rate > 0.4:
            notes.append(
                "BRECHA — Delegación cognitiva no detectada: cuando los estudiantes "
                "muestran patrón de delegación, este docente no activa las configuraciones "
                "de protección (socrático o no_direct_solutions). "
                "Puede requerir entrenamiento específico en detección de este patrón."
            )

        if profile.calibration_trend == "improving":
            notes.append("TENDENCIA POSITIVA: calibración mejorando en las últimas 5 intervenciones.")
        elif profile.calibration_trend == "declining":
            notes.append(
                "ATENCIÓN: calibración declinando en las últimas 5 intervenciones. "
                "Posible fatiga docente o cambios en el perfil del grupo."
            )

        return " | ".join(notes)

    # ──────────────────────────────────────────────────────────────────
    # INTERFACES PÚBLICAS
    # ──────────────────────────────────────────────────────────────────

    def get_teacher_bias_profile(self, teacher_id: str) -> Optional[TeacherBiasProfile]:
        """
        Devuelve el perfil de sesgos del docente.
        DESTINADO AL INVESTIGADOR, no al docente.
        """
        return self.teacher_profiles.get(teacher_id)

    def get_teacher_dashboard_summary(self, teacher_id: str) -> dict:
        """
        Resumen de calibración para mostrar AL DOCENTE.
        Sin mencionar "sesgo" — usa lenguaje de "precisión de intervención".
        Compatible con el principio de Value-Sensitive Design (Friedman, 2017).
        """
        profile = self.teacher_profiles.get(teacher_id)
        if not profile or profile.n_interventions < 3:
            return {
                "status": "insufficient_data",
                "message": "Se necesitan al menos 3 intervenciones para análisis.",
                "n_interventions": profile.n_interventions if profile else 0,
            }

        calibration_pct = (profile.n_calibrated / profile.n_interventions * 100)

        # Fortaleza principal
        if profile.socratic_calibration > 0.6:
            strength = "Alta precisión en el uso del modo socrático."
        elif profile.limit_calibration > 0.6:
            strength = "Alta precisión en el ajuste de límites de prompts."
        else:
            strength = "Intervenciones con impacto positivo en engagement."

        # Área de mejora (lenguaje neutro)
        if profile.volume_bias > 0.4:
            improvement = (
                "Considerar la calidad de las preguntas (nivel Bloom) además del "
                "número de prompts al decidir intervenciones."
            )
        elif profile.autonomy_blindspot > 0.3:
            improvement = (
                "Cuando los estudiantes muestran exploración activa, el scaffolding "
                "socrático adicional puede reducir la motivación intrínseca."
            )
        elif profile.delegation_miss_rate > 0.4:
            improvement = (
                "Algunos estudiantes con baja frecuencia de prompts pueden estar "
                "delegando todo el trabajo cognitivo al chatbot. La activación de "
                "'no_direct_solutions' puede ser útil en esos casos."
            )
        else:
            improvement = "Calibración dentro de rangos esperados para este perfil de curso."

        return {
            "status": "available",
            "n_interventions": profile.n_interventions,
            "calibration_pct": round(calibration_pct, 1),
            "trend": profile.calibration_trend,
            "strength": strength,
            "improvement_area": improvement,
            "last_intervention_label": (
                self.interventions[-1].calibration_label
                if self.interventions else "none"
            ),
        }

    def get_course_calibration_report(self, teacher_id: str, course_id: str) -> dict:
        """
        Informe de calibración por curso para el investigador.
        Incluye distribución de etiquetas y análisis de patrones.
        """
        teacher_interventions = [
            i for i in self.interventions
            if i.teacher_id == teacher_id and i.course_id == course_id
        ]

        if not teacher_interventions:
            return {"error": "No hay intervenciones registradas."}

        label_counts = defaultdict(int)
        config_calibration = defaultdict(list)

        for i in teacher_interventions:
            label_counts[i.calibration_label] += 1
            config_calibration[i.config_key].append(i.calibration_score)

        config_summary = {}
        for key, scores in config_calibration.items():
            config_summary[key] = {
                "n": len(scores),
                "mean_calibration": round(sum(scores) / len(scores), 3),
                "best": round(max(scores), 3),
                "worst": round(min(scores), 3),
            }

        return {
            "teacher_id": teacher_id,
            "course_id": course_id,
            "n_total": len(teacher_interventions),
            "label_distribution": dict(label_counts),
            "overall_calibration": round(
                sum(i.calibration_score for i in teacher_interventions) /
                len(teacher_interventions), 3
            ),
            "by_config_key": config_summary,
            "profile": self.get_teacher_bias_profile(teacher_id),
        }


# ═══════════════════════════════════════════════════════════════════════
# DEMO / TEST
# ═══════════════════════════════════════════════════════════════════════

def _demo():
    """
    Demuestra el módulo con una secuencia de intervenciones docentes
    con distintos niveles de calibración.
    """
    analyzer = TeacherCalibrationAnalyzer()

    # Estado 1: estudiante en modo delegación → docente activa socrático → BIEN CALIBRADO
    state_delegation = StudentStateSnapshot(
        timestamp=datetime.now().isoformat(),
        student_id="S001",
        bloom_level="recordar",
        bloom_weight=1.0,
        bloom_mean_last5=1.5,
        epistemic_mode="delegation",
        autonomy_score=0.15,
        speech_act_score=12.0,
        gaming_risk=0.7,
        trust_calibration=0.6,
        prompt_volume_last_session=15,
        inter_prompt_latency_mean=8.0,
    )
    i1 = analyzer.record_intervention(
        teacher_id="T001", course_id="PROG101", student_id="S001",
        config_key="socratic_scaffolding",
        old_value=False, new_value=True,
        student_state=state_delegation,
    )
    print(f"Intervención 1 — {i1.calibration_label} ({i1.calibration_score:.2f})")
    print(f"  Supuesto: {i1.assumed_student_state[:80]}...")
    print(f"  Desajuste: {i1.mismatch_description}")

    # Estado 2: estudiante en exploración → docente activa socrático → MAL CALIBRADO
    state_exploration = StudentStateSnapshot(
        timestamp=datetime.now().isoformat(),
        student_id="S002",
        bloom_level="analizar",
        bloom_weight=4.0,
        bloom_mean_last5=4.2,
        epistemic_mode="exploration",
        autonomy_score=0.78,
        speech_act_score=82.0,
        gaming_risk=0.0,
        trust_calibration=0.1,
        prompt_volume_last_session=4,
        inter_prompt_latency_mean=180.0,
    )
    i2 = analyzer.record_intervention(
        teacher_id="T001", course_id="PROG101", student_id="S002",
        config_key="socratic_scaffolding",
        old_value=False, new_value=True,
        student_state=state_exploration,
    )
    print(f"\nIntervención 2 — {i2.calibration_label} ({i2.calibration_score:.2f})")
    print(f"  Desajuste: {i2.mismatch_description}")

    # Estado 3: mucho volumen pero alta calidad → reduce prompts → VOLUME BIAS
    state_high_volume_quality = StudentStateSnapshot(
        timestamp=datetime.now().isoformat(),
        student_id="S003",
        bloom_level="analizar",
        bloom_weight=4.0,
        bloom_mean_last5=4.5,
        epistemic_mode="exploration",
        autonomy_score=0.71,
        speech_act_score=78.0,
        gaming_risk=0.0,
        trust_calibration=0.05,
        prompt_volume_last_session=20,
        inter_prompt_latency_mean=45.0,
    )
    i3 = analyzer.record_intervention(
        teacher_id="T001", course_id="PROG101", student_id="S003",
        config_key="max_daily_prompts",
        old_value=15, new_value=8,
        student_state=state_high_volume_quality,
    )
    print(f"\nIntervención 3 — {i3.calibration_label} ({i3.calibration_score:.2f})")
    print(f"  Desajuste: {i3.mismatch_description}")

    # Perfil del docente
    profile = analyzer.get_teacher_bias_profile("T001")
    print(f"\n{'='*60}")
    print(f"PERFIL DOCENTE T001:")
    print(f"  Intervenciones: {profile.n_interventions}")
    print(f"  Calibradas: {profile.n_calibrated}")
    print(f"  Mal calibradas: {profile.n_miscalibrated}")
    print(f"  Calibración media: {profile.calibration_mean:.2f}")
    print(f"  Volume bias: {profile.volume_bias:.2f}")
    print(f"  Autonomy blindspot: {profile.autonomy_blindspot:.2f}")
    print(f"  Tendencia: {profile.calibration_trend}")
    print(f"\n  NOTA INVESTIGADOR: {profile.researcher_note}")

    # Dashboard para el docente (lenguaje neutro)
    dashboard = analyzer.get_teacher_dashboard_summary("T001")
    print(f"\n{'='*60}")
    print(f"DASHBOARD PARA DOCENTE:")
    print(f"  Calibración: {dashboard.get('calibration_pct', 'N/A')}%")
    print(f"  Fortaleza: {dashboard.get('strength', '')}")
    print(f"  Área de mejora: {dashboard.get('improvement_area', '')}")


if __name__ == "__main__":
    _demo()
