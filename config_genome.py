"""
CONFIG GENOME — El Genoma de Configuración Docente
═══════════════════════════════════════════════════════════════════════
Convierte las configuraciones pedagógicas de estado estático a evento
temporal con contexto: quién cambió qué, cuándo, ante qué señal analítica,
y con qué efecto medido sobre el comportamiento del estudiante.

HIPÓTESIS CENTRAL:
Los patrones de configuración de los docentes no son aleatorios.
Son fingerprints cognitivos: revelan modelos mentales implícitos sobre
el aprendizaje. Un docente que activa modo socrático + bloqueo soluciones
+ hallucination_rate=0 opera desde una epistemología diferente a uno que
usa modo directo + sin límite de prompts + alta alucinación.
Con N docentes en el piloto GENIE Learn, emergen clusters de estilos
pedagógicos digitalmente observables por primera vez.

PAPERS HABILITADOS:
- WP2: análisis de design analytics (fingerprints → pedagogical intent)
- WP3: análisis atribucional (qué configuración produce qué comportamiento)
- LAK 2027: tipología empírica de estilos pedagógicos mediados por IA

POSICIÓN EN EL ECOSISTEMA:
    system_event_logger.py → log_event(config_change_event)
    config_genome.py       → analiza la colección de config_change_events
    teacher_calibration.py → usa los fingerprints para medir fidelidad

Autor: Diego Elvira Vásquez · Ecosistema GENIE Learn · Feb 2026
"""

import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import math


# ──────────────────────────────────────────────────────────────
# FINGERPRINT DE CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────

@dataclass
class ConfigFingerprint:
    """
    Representación compacta de un estado de configuración pedagógica.
    
    El fingerprint no es el estado en sí — es el patrón de intención
    que el estado revela. Dos configuraciones con valores numéricos
    distintos pueden revelar la misma intención pedagógica.
    """
    # Identidad
    teacher_id: str
    fingerprint_id: str  # hash determinístico del config
    computed_at: str

    # Dimensiones pedagógicas (no los parámetros raw)
    restrictiveness_score: float    # [0-1] qué tanto restringe el acceso a respuestas
    socratic_intensity: float       # [0-1] qué tan socrática es la configuración
    tolerance_for_error: float      # [0-1] permisividad pedagógica ante errores
    trust_in_student: float         # [0-1] autonomía que el docente cede al estudiante

    # Clasificación de estilo
    pedagogical_style: str          # "scaffolded_explorer" | "strict_guardian" |
                                    # "permissive_guide" | "challenge_based" | "mixed"
    
    # Config raw para referencia
    config_snapshot: Dict = field(default_factory=dict)


@dataclass
class ConfigChangeRecord:
    """
    Registro de un cambio específico de configuración con su contexto.
    
    La pieza que falta en el sistema actual: no solo QUÉ cambió,
    sino ANTE QUÉ dato cambió y CON QUÉ EFECTO.
    """
    event_id: str
    timestamp: str
    teacher_id: str
    param_changed: str
    value_before: Any
    value_after: Any
    
    # Contexto del cambio (el dato que lo motivó)
    analytics_trigger: Optional[str]      # qué métrica del dashboard motivó el cambio
    cohort_bloom_at_change: float         # nivel Bloom medio del cohorte en ese momento
    cohort_autonomy_at_change: float      # nivel de autonomía medio del cohorte
    session_pressure_at_change: float     # presión académica contextual
    
    # Efecto medido (se completa retroactivamente)
    outcome_bloom_delta: Optional[float] = None      # cambio en Bloom 7 días después
    outcome_autonomy_delta: Optional[float] = None   # cambio en autonomía 7 días después
    outcome_gaming_delta: Optional[float] = None     # cambio en gaming 7 días después
    outcome_n_students: Optional[int] = None


@dataclass 
class TeacherConfigProfile:
    """
    Perfil agregado del comportamiento configuracional de un docente.
    
    El espejo que Yannis no tiene todavía: cómo cada docente usa
    el espacio de configuración a lo largo del tiempo.
    """
    teacher_id: str
    
    # Parámetros favoritos (los más cambiados)
    most_adjusted_params: List[str]
    
    # Estabilidad configuracional: docentes que cambian poco vs. mucho
    config_volatility: float  # [0-1] 0=nunca cambia, 1=cambia constantemente
    
    # Calibración: ¿el docente actúa sobre datos o por intuición?
    data_driven_ratio: float  # % de cambios con analytics_trigger != None
    
    # Estilo dominante a lo largo del curso
    dominant_style: str
    style_evolution: List[str]  # cómo evoluciona el estilo semana a semana
    
    # Efectividad media de sus intervenciones
    mean_bloom_delta: Optional[float]
    mean_autonomy_delta: Optional[float]
    
    # Sesgos detectados
    detected_biases: List[str]


# ──────────────────────────────────────────────────────────────
# MOTOR DE ANÁLISIS DE CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────

class ConfigGenomeAnalyzer:
    """
    Analiza patrones de configuración pedagógica.
    
    Se alimenta del system_event_logger para obtener los eventos
    de tipo 'config_change' y los analiza para producir:
    1. Fingerprints de docentes individuales
    2. Clusters de estilos pedagógicos del cohorte docente
    3. Análisis atribucional: configuración → efecto en estudiantes
    4. Detección de sesgos interpretativos
    """

    # Parámetros que revelan la intención pedagógica más directamente
    INTENT_PARAMS = {
        "scaffolding_mode": {
            "socratic": {"socratic_intensity": 1.0, "trust_in_student": 0.8},
            "hints":    {"socratic_intensity": 0.5, "trust_in_student": 0.5},
            "direct":   {"socratic_intensity": 0.0, "trust_in_student": 0.3},
        },
        "block_direct_solutions": {
            True:  {"restrictiveness_score": +0.3},
            False: {"restrictiveness_score": -0.1},
        },
        "max_daily_prompts": {
            # normalizado: 5 prompts=alta restricción, 50=sin restricción
            "__continuous__": lambda v: {"restrictiveness_score": max(0, 1 - (v / 50))},
        },
        "forced_hallucination_pct": {
            # alta alucinación = mucha tolerancia al error, confianza en el estudiante
            "__continuous__": lambda v: {
                "tolerance_for_error": v,
                "trust_in_student": min(v * 2, 1.0),
            },
        },
        "use_rag": {
            True:  {"restrictiveness_score": +0.1},  # ciñe respuestas al curso
            False: {"restrictiveness_score": -0.2},
        },
    }

    STYLE_RULES = [
        # (condición, estilo)
        (lambda f: f["socratic_intensity"] > 0.7 and f["restrictiveness_score"] > 0.5,
         "scaffolded_explorer"),
        (lambda f: f["restrictiveness_score"] > 0.7 and f["socratic_intensity"] < 0.4,
         "strict_guardian"),
        (lambda f: f["restrictiveness_score"] < 0.3 and f["tolerance_for_error"] > 0.4,
         "permissive_guide"),
        (lambda f: f["socratic_intensity"] > 0.6 and f["tolerance_for_error"] > 0.3,
         "challenge_based"),
    ]

    def __init__(self):
        self.change_records: List[ConfigChangeRecord] = []
        self.teacher_profiles: Dict[str, TeacherConfigProfile] = {}

    def ingest_config_change(
        self,
        event_id: str,
        timestamp: str,
        teacher_id: str,
        param_changed: str,
        value_before: Any,
        value_after: Any,
        analytics_trigger: Optional[str],
        cohort_bloom: float = 2.0,
        cohort_autonomy: float = 0.3,
        session_pressure: float = 0.3,
    ) -> ConfigChangeRecord:
        """Registra un cambio de configuración para análisis posterior."""
        record = ConfigChangeRecord(
            event_id=event_id,
            timestamp=timestamp,
            teacher_id=teacher_id,
            param_changed=param_changed,
            value_before=value_before,
            value_after=value_after,
            analytics_trigger=analytics_trigger,
            cohort_bloom_at_change=cohort_bloom,
            cohort_autonomy_at_change=cohort_autonomy,
            session_pressure_at_change=session_pressure,
        )
        self.change_records.append(record)
        return record

    def compute_fingerprint(
        self,
        teacher_id: str,
        config_snapshot: Dict,
    ) -> ConfigFingerprint:
        """
        Genera el fingerprint pedagógico de una configuración.
        
        Transforma parámetros técnicos en dimensiones de intención:
        un docente que ve su fingerprint entiende su propio estilo
        en términos pedagógicos, no técnicos.
        """
        dims = {
            "restrictiveness_score": 0.3,  # base neutral
            "socratic_intensity": 0.3,
            "tolerance_for_error": 0.1,
            "trust_in_student": 0.5,
        }

        for param, rules in self.INTENT_PARAMS.items():
            value = config_snapshot.get(param)
            if value is None:
                continue

            if "__continuous__" in rules:
                delta = rules["__continuous__"](value)
                for k, v in delta.items():
                    dims[k] = v
            elif value in rules:
                for k, v in rules[value].items():
                    dims[k] = max(0.0, min(1.0, dims[k] + v))

        # Clasificar estilo
        style = "mixed"
        for condition, style_name in self.STYLE_RULES:
            try:
                if condition(dims):
                    style = style_name
                    break
            except (KeyError, TypeError):
                continue

        # Hash determinístico del snapshot
        config_str = json.dumps(config_snapshot, sort_keys=True)
        fingerprint_id = hashlib.md5(config_str.encode()).hexdigest()[:8]

        return ConfigFingerprint(
            teacher_id=teacher_id,
            fingerprint_id=fingerprint_id,
            computed_at=datetime.now().isoformat(),
            restrictiveness_score=round(dims["restrictiveness_score"], 3),
            socratic_intensity=round(dims["socratic_intensity"], 3),
            tolerance_for_error=round(dims["tolerance_for_error"], 3),
            trust_in_student=round(dims["trust_in_student"], 3),
            pedagogical_style=style,
            config_snapshot=config_snapshot,
        )

    def build_teacher_profile(self, teacher_id: str) -> TeacherConfigProfile:
        """
        Construye el perfil completo de un docente.
        
        Detecta sesgos interpretativos — la pieza que Yannis 
        no ha operacionalizado porque no tenía el dato.
        """
        teacher_records = [r for r in self.change_records if r.teacher_id == teacher_id]
        
        if not teacher_records:
            return TeacherConfigProfile(
                teacher_id=teacher_id,
                most_adjusted_params=[],
                config_volatility=0.0,
                data_driven_ratio=0.0,
                dominant_style="unknown",
                style_evolution=[],
                mean_bloom_delta=None,
                mean_autonomy_delta=None,
                detected_biases=[],
            )

        # Parámetros más cambiados
        param_counts: Dict[str, int] = {}
        for r in teacher_records:
            param_counts[r.param_changed] = param_counts.get(r.param_changed, 0) + 1
        most_adjusted = sorted(param_counts, key=param_counts.get, reverse=True)[:3]

        # Volatilidad: cambios por semana
        if len(teacher_records) > 1:
            t_first = datetime.fromisoformat(teacher_records[0].timestamp)
            t_last = datetime.fromisoformat(teacher_records[-1].timestamp)
            weeks = max((t_last - t_first).days / 7, 0.1)
            volatility = min(len(teacher_records) / (weeks * 5), 1.0)  # 5 cambios/semana = max
        else:
            volatility = 0.1

        # Ratio data-driven
        triggered = sum(1 for r in teacher_records if r.analytics_trigger is not None)
        data_driven_ratio = triggered / len(teacher_records)

        # Efectividad media (si hay outcomes)
        bloom_deltas = [r.outcome_bloom_delta for r in teacher_records 
                        if r.outcome_bloom_delta is not None]
        autonomy_deltas = [r.outcome_autonomy_delta for r in teacher_records 
                           if r.outcome_autonomy_delta is not None]
        
        mean_bloom = sum(bloom_deltas) / len(bloom_deltas) if bloom_deltas else None
        mean_autonomy = sum(autonomy_deltas) / len(autonomy_deltas) if autonomy_deltas else None

        # Detección de sesgos
        biases = self._detect_biases(teacher_records)

        return TeacherConfigProfile(
            teacher_id=teacher_id,
            most_adjusted_params=most_adjusted,
            config_volatility=round(volatility, 3),
            data_driven_ratio=round(data_driven_ratio, 3),
            dominant_style=self._infer_dominant_style(teacher_records),
            style_evolution=[],
            mean_bloom_delta=round(mean_bloom, 3) if mean_bloom else None,
            mean_autonomy_delta=round(mean_autonomy, 3) if mean_autonomy else None,
            detected_biases=biases,
        )

    def get_attributional_analysis(self) -> Dict:
        """
        Análisis atribucional: qué configuraciones producen qué efectos.
        
        Esta es la tabla que los papers WP2 y WP3 necesitan.
        Con suficientes observaciones, responde preguntas como:
        "El modo socrático + límite de 10 prompts produce un
        incremento de +0.8 en Bloom a los 7 días para estudiantes
        que parten de Bloom-2, pero no tiene efecto en Bloom-1."
        """
        records_with_outcome = [
            r for r in self.change_records
            if r.outcome_bloom_delta is not None
        ]

        if not records_with_outcome:
            return {"message": "Sin outcomes registrados aún. Los outcomes se calculan 7 días después de cada cambio."}

        # Agrupar por parámetro cambiado
        by_param: Dict[str, List[ConfigChangeRecord]] = {}
        for r in records_with_outcome:
            by_param.setdefault(r.param_changed, []).append(r)

        analysis = {}
        for param, records in by_param.items():
            bloom_deltas = [r.outcome_bloom_delta for r in records if r.outcome_bloom_delta]
            autonomy_deltas = [r.outcome_autonomy_delta for r in records if r.outcome_autonomy_delta]
            
            analysis[param] = {
                "n_observations": len(records),
                "mean_bloom_delta": round(sum(bloom_deltas) / len(bloom_deltas), 3) if bloom_deltas else None,
                "mean_autonomy_delta": round(sum(autonomy_deltas) / len(autonomy_deltas), 3) if autonomy_deltas else None,
                "data_driven_pct": round(
                    sum(1 for r in records if r.analytics_trigger) / len(records), 2
                ),
                "cohort_bloom_context": round(
                    sum(r.cohort_bloom_at_change for r in records) / len(records), 2
                ),
            }

        return analysis

    def _detect_biases(self, records: List[ConfigChangeRecord]) -> List[str]:
        """
        Detecta sesgos interpretativos del docente.
        
        Los cuatro sesgos más comunes en la interpretación
        de dashboards de learning analytics:
        """
        biases = []

        if not records:
            return biases

        # Sesgo de recencia: cambia aunque la tendencia general sea positiva
        recent = records[-3:] if len(records) >= 3 else records
        if len(recent) >= 2:
            rapid_changes = sum(1 for r in recent if r.analytics_trigger is None)
            if rapid_changes >= 2:
                biases.append("recency_bias: interviene ante eventos atípicos recientes, ignora tendencia")

        # Sesgo de volumen: reacciona al número de prompts más que a su calidad
        volume_changes = [r for r in records if r.param_changed == "max_daily_prompts"]
        bloom_changes = [r for r in records if r.param_changed == "scaffolding_mode"]
        if len(volume_changes) > len(bloom_changes) * 2:
            biases.append("volume_bias: pondera la cantidad de prompts sobre la calidad cognitiva")

        # Sesgo de presión: cambia configuraciones en momentos de alta presión académica
        high_pressure_changes = [r for r in records if r.session_pressure_at_change > 0.7]
        if len(high_pressure_changes) > len(records) * 0.6:
            biases.append("pressure_reactivity: concentra intervenciones en períodos de alta presión")

        # Sesgo de intuición: ratio de cambios sin trigger analítico
        intuition_ratio = sum(1 for r in records if r.analytics_trigger is None) / len(records)
        if intuition_ratio > 0.7:
            biases.append("intuition_over_data: el 70%+ de cambios son por intuición, no por dashboard")

        return biases

    def _infer_dominant_style(self, records: List[ConfigChangeRecord]) -> str:
        """Infiere el estilo pedagógico dominante a partir del historial de cambios."""
        if not records:
            return "unknown"

        # Analizar la dirección de los cambios más frecuentes
        restrictiveness_trajectory = []
        for r in records:
            if r.param_changed == "block_direct_solutions":
                restrictiveness_trajectory.append(1.0 if r.value_after else -1.0)
            elif r.param_changed == "max_daily_prompts":
                # Reducir prompts = más restrictivo
                if isinstance(r.value_before, (int, float)) and isinstance(r.value_after, (int, float)):
                    delta = r.value_after - r.value_before
                    restrictiveness_trajectory.append(-delta / 20)  # normalizado

        if not restrictiveness_trajectory:
            return "stable"

        mean_direction = sum(restrictiveness_trajectory) / len(restrictiveness_trajectory)

        if mean_direction > 0.3:
            return "progressive_restrictive"
        elif mean_direction < -0.3:
            return "progressive_permissive"
        else:
            return "adaptive_balanced"


# ──────────────────────────────────────────────
# DEMO AUTOEJECTABLE
# ──────────────────────────────────────────────

if __name__ == "__main__":
    analyzer = ConfigGenomeAnalyzer()

    print("═" * 60)
    print("CONFIG GENOME — Demo de fingerprinting pedagógico")
    print("═" * 60)

    # Simular dos docentes con estilos distintos
    changes_prof_A = [
        # Prof A: empieza restrictiva, va relajando basado en datos
        ("scaffolding_mode", "direct", "socratic", "bloom_mean_low"),
        ("block_direct_solutions", False, True, "gaming_alerts_spike"),
        ("max_daily_prompts", 20, 12, "high_volume_low_quality"),
        ("forced_hallucination_pct", 0.0, 0.15, "critical_reading_goal"),
    ]

    changes_prof_B = [
        # Prof B: intuitivo, cambia sin datos, oscila
        ("max_daily_prompts", 20, 8, None),
        ("max_daily_prompts", 8, 20, None),
        ("scaffolding_mode", "socratic", "direct", None),
        ("max_daily_prompts", 20, 5, None),
    ]

    for param, before, after, trigger in changes_prof_A:
        analyzer.ingest_config_change(
            event_id=f"e_{param}_A",
            timestamp=datetime.now().isoformat(),
            teacher_id="prof_A",
            param_changed=param,
            value_before=before,
            value_after=after,
            analytics_trigger=trigger,
            cohort_bloom=2.3,
            cohort_autonomy=0.4,
            session_pressure=0.5,
        )

    for param, before, after, trigger in changes_prof_B:
        analyzer.ingest_config_change(
            event_id=f"e_{param}_B",
            timestamp=datetime.now().isoformat(),
            teacher_id="prof_B",
            param_changed=param,
            value_before=before,
            value_after=after,
            analytics_trigger=trigger,
            cohort_bloom=1.8,
            cohort_autonomy=0.25,
            session_pressure=0.7,
        )

    # Fingerprints
    config_A = {"scaffolding_mode": "socratic", "block_direct_solutions": True,
                "max_daily_prompts": 12, "forced_hallucination_pct": 0.15, "use_rag": True}
    config_B = {"scaffolding_mode": "direct", "block_direct_solutions": False,
                "max_daily_prompts": 5, "forced_hallucination_pct": 0.0, "use_rag": True}

    fp_A = analyzer.compute_fingerprint("prof_A", config_A)
    fp_B = analyzer.compute_fingerprint("prof_B", config_B)

    print(f"\n🧬 FINGERPRINT — Prof. A ({fp_A.pedagogical_style})")
    print(f"   Restrictividad: {fp_A.restrictiveness_score:.2f}")
    print(f"   Intensidad socrática: {fp_A.socratic_intensity:.2f}")
    print(f"   Tolerancia al error: {fp_A.tolerance_for_error:.2f}")
    print(f"   Confianza en el estudiante: {fp_A.trust_in_student:.2f}")

    print(f"\n🧬 FINGERPRINT — Prof. B ({fp_B.pedagogical_style})")
    print(f"   Restrictividad: {fp_B.restrictiveness_score:.2f}")
    print(f"   Intensidad socrática: {fp_B.socratic_intensity:.2f}")
    print(f"   Tolerancia al error: {fp_B.tolerance_for_error:.2f}")
    print(f"   Confianza en el estudiante: {fp_B.trust_in_student:.2f}")

    profile_A = analyzer.build_teacher_profile("prof_A")
    profile_B = analyzer.build_teacher_profile("prof_B")

    print(f"\n👤 PERFIL DOCENTE — Prof. A")
    print(f"   Volatilidad: {profile_A.config_volatility:.2f} (estabilidad configuracional)")
    print(f"   Data-driven: {profile_A.data_driven_ratio:.0%} de cambios basados en datos")
    print(f"   Sesgos: {profile_A.detected_biases or ['ninguno detectado']}")

    print(f"\n👤 PERFIL DOCENTE — Prof. B")
    print(f"   Volatilidad: {profile_B.config_volatility:.2f}")
    print(f"   Data-driven: {profile_B.data_driven_ratio:.0%} de cambios basados en datos")
    print(f"   Sesgos: {profile_B.detected_biases or ['ninguno detectado']}")

    print("\n═" * 60)
