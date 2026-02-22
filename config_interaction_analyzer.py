"""
CONFIG INTERACTION ANALYZER — Cuando las Configuraciones se Sabotean
═══════════════════════════════════════════════════════════════════════
Extensión de config_genome.py — BLOQUE 5

"Los docentes configuran 5 parámetros independientes. Pero las
 configuraciones interactúan — el modo socrático con límite bajo
 de prompts produce un efecto antagónico que nadie esperaba. El
 sistema detecta y visualiza estas interacciones para que el docente
 no combine configuraciones que se sabotean mutuamente."

PROBLEMA:
─────────
El config_impact_panel.py registra cambios INDIVIDUALES.
El config_genome.py modela configuraciones como fingerprint.
Nadie analiza qué pasa cuando MÚLTIPLES configuraciones operan
simultáneamente — y los efectos de interacción son no-triviales.

CONCEPTO CENTRAL:
─────────────────
Cada configuración activa = una dimensión de un espacio.
Dos configuraciones activas = un punto en espacio bidimensional.
El efecto observado en ese punto puede ser:

  Aditivo:      efecto_real ≈ efecto_A + efecto_B
  Sinérgico:    efecto_real > efecto_A + efecto_B (amplificación)
  Antagónico:   efecto_real < efecto_A + efecto_B (sabotaje)
  Emergente:    efecto_real es cualitativamente distinto de ambos

FUNDAMENTO TEÓRICO:
───────────────────
- Vygotsky (1978): La ZDP no es un punto fijo; la combinación de
  scaffolding y restricción puede CONTRAER la zona en lugar de expandirla.
- Bjork (1994): Las dificultades deseables se vuelven indeseables
  cuando se acumulan — dos dificultades simultáneas no suman, saturan.
- Kalyuga et al. (2003): Expertise reversal — la intervención que
  funciona sola puede degradarse cuando se combina con otra.
- Sweller (1988): Cognitive Load Theory — dos intervenciones que
  individualmente están dentro de la carga tolerable pueden
  exceder la carga máxima cuando se combinan.

INTERACCIONES MODELADAS EN DEMO:
────────────────────────────────
  Socrático × límite bajo prompts → ANTAGÓNICO
    El socrático genera ida-y-vuelta (cada respuesta guía requiere
    follow-up). El límite bajo corta ese diálogo. El estudiante
    "gasta" prompts en el intercambio socrático sin llegar a resolver.

  Socrático × alucinación deliberada → SINÉRGICO
    El socrático obliga a pensar. La alucinación obliga a verificar.
    Juntos: el estudiante piensa Y verifica — engagement profundo.

  Bloqueo soluciones × RAG activo → ADITIVO
    Cada uno contribuye independientemente sin interferir.

INTEGRACIÓN:
────────────
    from config_genome import ConfigGenomeAnalyzer
    from config_interaction_analyzer import ConfigInteractionAnalyzer

    genome = ConfigGenomeAnalyzer()
    interaction_analyzer = ConfigInteractionAnalyzer(genome)

    # Registrar observaciones bajo combinaciones de configs:
    interaction_analyzer.record_observation(active_configs, metrics)

    # Computar efectos de interacción:
    results = interaction_analyzer.compute_interaction_effects()

    # Heatmap:
    fig = interaction_analyzer.render_interaction_heatmap()

Autor: Diego Elvira Vásquez · CP25/152 GSIC/EMIC · Feb 2026
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, FrozenSet, Any
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
class ConfigCombination:
    """
    Registro de una combinación de configuraciones activas simultáneamente
    con las métricas observadas bajo esa combinación.
    """
    active_configs: FrozenSet[Tuple[str, Any]]  # frozenset de (config_name, value)
    interaction_count: int = 0
    metrics_observed: Dict[str, List[float]] = field(default_factory=dict)
    # metric_name → lista de valores observados bajo esta combinación

    def config_label(self) -> str:
        """Representación legible."""
        parts = []
        for name, val in sorted(self.active_configs):
            if isinstance(val, bool):
                parts.append(f"{name}={'ON' if val else 'OFF'}")
            else:
                parts.append(f"{name}={val}")
        return " + ".join(parts)

    def mean_metric(self, metric: str) -> float:
        vals = self.metrics_observed.get(metric, [])
        return sum(vals) / len(vals) if vals else 0.0


@dataclass
class ConfigInteractionResult:
    """
    Resultado del análisis de interacción entre un par de configuraciones.
    """
    config_a: Tuple[str, Any]       # (param_name, value)
    config_b: Tuple[str, Any]
    metric: str

    # Efectos individuales
    effect_a_alone: float = 0.0     # delta vs baseline cuando solo A activa
    effect_b_alone: float = 0.0     # delta vs baseline cuando solo B activa
    expected_additive: float = 0.0  # effect_a + effect_b

    # Efecto combinado real
    actual_combined: float = 0.0    # delta vs baseline cuando A+B activas

    # Tipo de interacción
    interaction_type: str = "unknown"   # additive|synergistic|antagonistic|emergent
    interaction_ratio: float = 1.0      # actual / expected (>1.2=sinérgico, <0.8=antagónico)
    interaction_magnitude: float = 0.0  # |actual - expected|

    # Interpretación
    interpretation: str = ""
    theoretical_basis: str = ""
    recommendation: str = ""

    # Confianza
    n_observations_a: int = 0
    n_observations_b: int = 0
    n_observations_ab: int = 0
    sufficient_data: bool = False

    def label_a(self) -> str:
        n, v = self.config_a
        return f"{n}={'ON' if isinstance(v, bool) and v else v}"

    def label_b(self) -> str:
        n, v = self.config_b
        return f"{n}={'ON' if isinstance(v, bool) and v else v}"


@dataclass
class InteractionMatrix:
    """
    Matriz NxN de interacciones entre todas las configuraciones.
    """
    config_names: List[str]                     # nombres de configuraciones
    matrix: Dict[Tuple[str, str], ConfigInteractionResult] = field(default_factory=dict)
    primary_metric: str = "bloom_mean"
    timestamp: str = ""

    def get(self, a: str, b: str) -> Optional[ConfigInteractionResult]:
        return self.matrix.get((a, b)) or self.matrix.get((b, a))


# ═══════════════════════════════════════════════════════════════════════
# MOTOR DE INTERACCIÓN
# ═══════════════════════════════════════════════════════════════════════

class ConfigInteractionAnalyzer:
    """
    Analiza efectos de interacción entre configuraciones pedagógicas
    que operan simultáneamente.

    Se conecta al ConfigGenomeAnalyzer para context fingerprint,
    y al ConfigImpactTracker para datos de métricas.
    También opera en modo standalone con datos sintéticos para demo.
    """

    # Umbrales de clasificación
    SYNERGISTIC_THRESHOLD = 1.2     # ratio > 1.2 → sinérgico
    ANTAGONISTIC_THRESHOLD = 0.8    # ratio < 0.8 → antagónico
    ADDITIVE_TOLERANCE = 0.2        # margen para considerar aditivo
    EMERGENT_SIGN_FLIP = True       # si el signo del efecto cambia → emergente
    MIN_OBSERVATIONS = 3            # mínimo para computar efecto

    # Configuraciones a analizar
    CONFIG_PARAMS = [
        ("socratic_scaffolding", True),
        ("max_daily_prompts", 8),       # "bajo" = restrictivo
        ("hallucination_rate", 0.15),
        ("block_direct_solutions", True),
        ("use_rag", True),
    ]

    # Nombres cortos para display
    SHORT_NAMES = {
        ("socratic_scaffolding", True): "Socrático",
        ("max_daily_prompts", 8): "Límite bajo",
        ("hallucination_rate", 0.15): "Alucinación 15%",
        ("block_direct_solutions", True): "Bloqueo soluciones",
        ("use_rag", True): "RAG activo",
    }

    METRICS = ["bloom_mean", "autonomy_score", "pedagogical_value"]

    def __init__(self, genome_analyzer=None):
        self.genome = genome_analyzer
        self.combinations: Dict[FrozenSet, ConfigCombination] = {}
        self._baseline_metrics: Dict[str, float] = {}
        self._individual_effects: Dict[Tuple[str, Any], Dict[str, float]] = {}

    # ──────────────────────────────────────────────────────────────────
    # REGISTRO DE OBSERVACIONES
    # ──────────────────────────────────────────────────────────────────

    def record_observation(
        self,
        active_configs: Dict[str, Any],
        metrics: Dict[str, float],
    ):
        """
        Registra una observación (interacción) bajo un set de configs activas.

        Args:
            active_configs: {"socratic_scaffolding": True, "max_daily_prompts": 8, ...}
            metrics: {"bloom_mean": 3.2, "autonomy_score": 0.45, ...}
        """
        # Convertir a frozenset de configuraciones activas (no-default)
        key = frozenset(active_configs.items())

        if key not in self.combinations:
            self.combinations[key] = ConfigCombination(
                active_configs=key,
                interaction_count=0,
                metrics_observed={m: [] for m in self.METRICS},
            )

        combo = self.combinations[key]
        combo.interaction_count += 1
        for m in self.METRICS:
            if m in metrics:
                combo.metrics_observed.setdefault(m, []).append(metrics[m])

    def set_baseline(self, metrics: Dict[str, float]):
        """Define métricas baseline (sin configuraciones pedagógicas activas)."""
        self._baseline_metrics = dict(metrics)

    def set_individual_effect(
        self,
        config: Tuple[str, Any],
        effects: Dict[str, float],
    ):
        """
        Registra el efecto individual conocido de una configuración sola.
        effects: {"bloom_mean": +0.8, "autonomy_score": +0.15, ...}
        """
        self._individual_effects[config] = dict(effects)

    # ──────────────────────────────────────────────────────────────────
    # CÓMPUTO DE INTERACCIONES
    # ──────────────────────────────────────────────────────────────────

    def compute_interaction_effects(
        self,
        metric: str = "bloom_mean",
    ) -> List[ConfigInteractionResult]:
        """
        Para cada par de configuraciones que han co-ocurrido,
        calcula el efecto real y lo compara con la suma de efectos
        individuales.

        ratio = efecto_real / efecto_esperado_aditivo
          > 1.2 → sinérgico  (las configuraciones se amplifican)
          < 0.8 → antagónico (se sabotean)
          ≈ 1.0 → aditivo   (independientes)
          signo opuesto → emergente (cualitativamente distinto)
        """
        results = []
        configs = list(self._individual_effects.keys())

        for i in range(len(configs)):
            for j in range(i + 1, len(configs)):
                a, b = configs[i], configs[j]
                result = self._compute_pair_interaction(a, b, metric)
                if result:
                    results.append(result)

        return results

    def _compute_pair_interaction(
        self,
        config_a: Tuple[str, Any],
        config_b: Tuple[str, Any],
        metric: str,
    ) -> Optional[ConfigInteractionResult]:
        """Computa interacción entre un par específico."""
        effect_a = self._individual_effects.get(config_a, {}).get(metric, 0.0)
        effect_b = self._individual_effects.get(config_b, {}).get(metric, 0.0)
        expected_additive = effect_a + effect_b

        # Buscar combinación donde ambas están activas
        actual = self._find_combined_effect(config_a, config_b, metric)
        n_ab = self._find_combined_count(config_a, config_b)

        n_a = self._find_individual_count(config_a)
        n_b = self._find_individual_count(config_b)

        if actual is None:
            return None

        # Clasificar
        if expected_additive == 0:
            ratio = 1.0 if actual == 0 else float('inf')
        else:
            ratio = actual / expected_additive

        # Detección de emergencia: cambio de signo
        sign_a = 1 if effect_a >= 0 else -1
        sign_b = 1 if effect_b >= 0 else -1
        sign_combined = 1 if actual >= 0 else -1

        if (sign_a > 0 and sign_b > 0 and sign_combined < 0) or \
           (sign_a < 0 and sign_b < 0 and sign_combined > 0):
            itype = "emergent"
        elif ratio > self.SYNERGISTIC_THRESHOLD:
            itype = "synergistic"
        elif ratio < self.ANTAGONISTIC_THRESHOLD:
            itype = "antagonistic"
        else:
            itype = "additive"

        magnitude = abs(actual - expected_additive)

        # Interpretación
        interpretation, theory, recommendation = self._interpret_interaction(
            config_a, config_b, itype, ratio, effect_a, effect_b, actual, metric
        )

        return ConfigInteractionResult(
            config_a=config_a,
            config_b=config_b,
            metric=metric,
            effect_a_alone=round(effect_a, 3),
            effect_b_alone=round(effect_b, 3),
            expected_additive=round(expected_additive, 3),
            actual_combined=round(actual, 3),
            interaction_type=itype,
            interaction_ratio=round(ratio, 3),
            interaction_magnitude=round(magnitude, 3),
            interpretation=interpretation,
            theoretical_basis=theory,
            recommendation=recommendation,
            n_observations_a=n_a,
            n_observations_b=n_b,
            n_observations_ab=n_ab,
            sufficient_data=(n_ab >= self.MIN_OBSERVATIONS),
        )

    def _find_combined_effect(self, a, b, metric):
        """Busca el efecto observado cuando ambas configs están activas."""
        for key, combo in self.combinations.items():
            configs_dict = dict(combo.active_configs)
            if a[0] in configs_dict and configs_dict[a[0]] == a[1] and \
               b[0] in configs_dict and configs_dict[b[0]] == b[1]:
                vals = combo.metrics_observed.get(metric, [])
                if vals:
                    baseline = self._baseline_metrics.get(metric, 0.0)
                    return sum(vals) / len(vals) - baseline
        return None

    def _find_combined_count(self, a, b):
        for key, combo in self.combinations.items():
            configs_dict = dict(combo.active_configs)
            if a[0] in configs_dict and configs_dict[a[0]] == a[1] and \
               b[0] in configs_dict and configs_dict[b[0]] == b[1]:
                return combo.interaction_count
        return 0

    def _find_individual_count(self, config):
        total = 0
        for key, combo in self.combinations.items():
            configs_dict = dict(combo.active_configs)
            if config[0] in configs_dict and configs_dict[config[0]] == config[1]:
                total += combo.interaction_count
        return total

    # ──────────────────────────────────────────────────────────────────
    # MATRIZ DE INTERACCIÓN
    # ──────────────────────────────────────────────────────────────────

    def generate_interaction_matrix(
        self,
        metric: str = "bloom_mean",
    ) -> InteractionMatrix:
        """
        Matriz NxN: cada celda = tipo de interacción + magnitud.
        Visualizable como heatmap.
        """
        results = self.compute_interaction_effects(metric)

        config_names = sorted(set(
            self.SHORT_NAMES.get(r.config_a, r.label_a()) for r in results
        ) | set(
            self.SHORT_NAMES.get(r.config_b, r.label_b()) for r in results
        ))

        matrix = InteractionMatrix(
            config_names=config_names,
            primary_metric=metric,
            timestamp=datetime.now().isoformat(),
        )

        for r in results:
            la = self.SHORT_NAMES.get(r.config_a, r.label_a())
            lb = self.SHORT_NAMES.get(r.config_b, r.label_b())
            matrix.matrix[(la, lb)] = r

        return matrix

    # ──────────────────────────────────────────────────────────────────
    # INTERPRETACIÓN
    # ──────────────────────────────────────────────────────────────────

    def _interpret_interaction(self, ca, cb, itype, ratio, ea, eb, actual, metric):
        na = self.SHORT_NAMES.get(ca, f"{ca[0]}={ca[1]}")
        nb = self.SHORT_NAMES.get(cb, f"{cb[0]}={cb[1]}")
        ml = {"bloom_mean": "nivel Bloom", "autonomy_score": "autonomía",
              "pedagogical_value": "valor pedagógico"}.get(metric, metric)

        interpretations = {
            "synergistic": (
                f"{na} + {nb} producen efecto sinérgico en {ml}: "
                f"el efecto combinado ({actual:+.2f}) supera la suma de efectos "
                f"individuales ({ea:+.2f} + {eb:+.2f} = {ea+eb:+.2f}). "
                f"Ratio: {ratio:.2f}x. Las configuraciones se amplifican mutuamente.",
                "Bjork (1994): dos dificultades deseables complementarias producen "
                "profundidad mayor que la suma. La verificación (alucinación) y "
                "la elaboración (socrático) activan circuitos cognitivos distintos "
                "que se refuerzan — Sweller (1988): carga cognitiva germánica, "
                "no extrínseca.",
                f"RECOMENDACIÓN: Combinar {na} + {nb} cuando se busca engagement profundo. "
                f"Vigilar que no exceda carga cognitiva en estudiantes iniciales."
            ),
            "antagonistic": (
                f"{na} + {nb} producen efecto antagónico en {ml}: "
                f"el efecto combinado ({actual:+.2f}) es inferior a la suma esperada "
                f"({ea:+.2f} + {eb:+.2f} = {ea+eb:+.2f}). "
                f"Ratio: {ratio:.2f}x. Las configuraciones se sabotean.",
                "Sweller (1988): Cognitive Load Theory — dos intervenciones que "
                "individualmente están dentro de la carga tolerable pueden excederla "
                "cuando se combinan. Kalyuga (2003): expertise reversal — la "
                "combinación produce sobrecarga que anula el beneficio de cada una.",
                f"ALERTA: Evitar combinar {na} + {nb} o compensar con scaffolding "
                f"adicional. Si se combinan, monitorear {ml} activamente."
            ),
            "additive": (
                f"{na} + {nb} tienen efecto aditivo en {ml}: "
                f"el efecto combinado ({actual:+.2f}) ≈ suma de individuales "
                f"({ea:+.2f} + {eb:+.2f} = {ea+eb:+.2f}). "
                f"Ratio: {ratio:.2f}x. Las configuraciones son independientes.",
                "Las configuraciones operan en dimensiones ortogonales del espacio "
                "pedagógico. Cada una afecta un mecanismo distinto sin interferencia. "
                "Resultado predecible por principio de superposición.",
                f"NEUTRAL: {na} + {nb} pueden combinarse libremente. "
                f"El efecto es predecible."
            ),
            "emergent": (
                f"{na} + {nb} producen efecto EMERGENTE en {ml}: "
                f"el efecto combinado ({actual:+.2f}) invierte el signo esperado "
                f"({ea:+.2f} + {eb:+.2f} = {ea+eb:+.2f}). "
                f"La combinación produce un fenómeno cualitativamente distinto.",
                "Suchman (2007): plans and situated actions — la interacción entre "
                "intervenciones produce efectos que no son deducibles de los componentes. "
                "Luhmann (1995): emergencia sistémica — el todo es cualitativamente "
                "distinto de la suma.",
                f"INVESTIGAR: La combinación {na} + {nb} produce un fenómeno no previsto. "
                f"Documentar para publicación: es un hallazgo genuino."
            ),
        }

        return interpretations.get(itype, ("", "", ""))

    # ──────────────────────────────────────────────────────────────────
    # DATOS SINTÉTICOS PARA DEMO
    # ──────────────────────────────────────────────────────────────────

    def generate_demo_data(self) -> InteractionMatrix:
        """
        Genera datos demo con las tres interacciones canónicas:

        1. Socrático × límite bajo → ANTAGÓNICO
           El socrático genera ida-y-vuelta. El límite corta el diálogo.
           Bloom: socrático solo +0.8, límite solo +0.3, combinados -0.3

        2. Socrático × alucinación → SINÉRGICO
           Piensa + verifica = engagement profundo.
           Autonomía: socrático +0.15, alucinación +0.10, combinados +0.40

        3. Bloqueo soluciones × RAG → ADITIVO
           Independientes, sin interferencia.
        """

        # Baseline
        self.set_baseline({
            "bloom_mean": 2.0,
            "autonomy_score": 0.25,
            "pedagogical_value": 0.20,
        })

        # Efectos individuales
        self.set_individual_effect(
            ("socratic_scaffolding", True),
            {"bloom_mean": 0.8, "autonomy_score": 0.15, "pedagogical_value": 0.25}
        )
        self.set_individual_effect(
            ("max_daily_prompts", 8),
            {"bloom_mean": 0.3, "autonomy_score": 0.08, "pedagogical_value": 0.10}
        )
        self.set_individual_effect(
            ("hallucination_rate", 0.15),
            {"bloom_mean": 0.5, "autonomy_score": 0.10, "pedagogical_value": 0.15}
        )
        self.set_individual_effect(
            ("block_direct_solutions", True),
            {"bloom_mean": 0.4, "autonomy_score": 0.12, "pedagogical_value": 0.18}
        )
        self.set_individual_effect(
            ("use_rag", True),
            {"bloom_mean": 0.3, "autonomy_score": 0.05, "pedagogical_value": 0.12}
        )

        # ── 1. Socrático + límite bajo → ANTAGÓNICO ──
        # El socrático solo da +0.8 Bloom, el límite solo +0.3
        # Juntos: -0.3 Bloom (el estudiante gasta prompts en ida-y-vuelta)
        for _ in range(8):
            self.record_observation(
                {"socratic_scaffolding": True, "max_daily_prompts": 8},
                {"bloom_mean": 1.7, "autonomy_score": 0.20, "pedagogical_value": 0.15},
            )

        # ── 2. Socrático + alucinación → SINÉRGICO ──
        # Socrático solo +0.15 autonomía, alucinación sola +0.10
        # Juntos: +0.40 autonomía (piensa + verifica = profundo)
        for _ in range(8):
            self.record_observation(
                {"socratic_scaffolding": True, "hallucination_rate": 0.15},
                {"bloom_mean": 3.6, "autonomy_score": 0.65, "pedagogical_value": 0.62},
            )

        # ── 3. Bloqueo + RAG → ADITIVO ──
        # Bloqueo +0.4 Bloom, RAG +0.3 Bloom → juntos ≈ +0.7
        for _ in range(8):
            self.record_observation(
                {"block_direct_solutions": True, "use_rag": True},
                {"bloom_mean": 2.68, "autonomy_score": 0.42, "pedagogical_value": 0.50},
            )

        # ── 4. Socrático + bloqueo → SINÉRGICO MODERADO ──
        for _ in range(6):
            self.record_observation(
                {"socratic_scaffolding": True, "block_direct_solutions": True},
                {"bloom_mean": 3.8, "autonomy_score": 0.58, "pedagogical_value": 0.60},
            )

        # ── 5. Límite + alucinación → ANTAGÓNICO ──
        # El estudiante necesita muchas interacciones para descubrir alucinaciones.
        # El límite corta esas interacciones. Se anulan.
        for _ in range(6):
            self.record_observation(
                {"max_daily_prompts": 8, "hallucination_rate": 0.15},
                {"bloom_mean": 2.2, "autonomy_score": 0.28, "pedagogical_value": 0.22},
            )

        # ── 6. Alucinación + RAG → ADITIVO/SINÉRGICO LEVE ──
        for _ in range(6):
            self.record_observation(
                {"hallucination_rate": 0.15, "use_rag": True},
                {"bloom_mean": 3.0, "autonomy_score": 0.42, "pedagogical_value": 0.50},
            )

        # Generar matrices para cada métrica principal
        return self.generate_interaction_matrix("bloom_mean")

    # ──────────────────────────────────────────────────────────────────
    # VISUALIZACIÓN
    # ──────────────────────────────────────────────────────────────────

    def render_interaction_heatmap(
        self,
        metric: str = "bloom_mean",
        height: int = 500,
    ) -> "go.Figure":
        """
        Heatmap NxN:
          Filas/columnas = configuraciones
          Color = tipo de interacción (verde=sinérgico, rojo=antagónico,
                  gris=aditivo, azul=emergente)
          Intensidad = magnitud
          Hover = explicación teórica
        """
        if not PLOTLY_AVAILABLE:
            return None

        results = self.compute_interaction_effects(metric)
        if not results:
            return None

        # Recopilar nombres
        all_names = sorted(set(
            self.SHORT_NAMES.get(r.config_a, r.label_a()) for r in results
        ) | set(
            self.SHORT_NAMES.get(r.config_b, r.label_b()) for r in results
        ))

        n = len(all_names)
        z = [[0.0] * n for _ in range(n)]           # valor numérico para color
        text = [[""] * n for _ in range(n)]          # hover text
        annotations = [[""] * n for _ in range(n)]   # cell labels

        type_values = {
            "synergistic": 1.0,
            "additive": 0.0,
            "antagonistic": -1.0,
            "emergent": 0.5,
        }
        type_symbols = {
            "synergistic": "✦",
            "additive": "═",
            "antagonistic": "✗",
            "emergent": "◆",
        }

        for r in results:
            la = self.SHORT_NAMES.get(r.config_a, r.label_a())
            lb = self.SHORT_NAMES.get(r.config_b, r.label_b())
            if la in all_names and lb in all_names:
                i = all_names.index(la)
                j = all_names.index(lb)

                val = type_values.get(r.interaction_type, 0) * min(r.interaction_magnitude * 3, 1.0)
                z[i][j] = val
                z[j][i] = val

                hover = (
                    f"<b>{la} × {lb}</b><br>"
                    f"Tipo: {r.interaction_type.upper()}<br>"
                    f"Efecto A solo: {r.effect_a_alone:+.2f}<br>"
                    f"Efecto B solo: {r.effect_b_alone:+.2f}<br>"
                    f"Esperado (aditivo): {r.expected_additive:+.2f}<br>"
                    f"<b>Real combinado: {r.actual_combined:+.2f}</b><br>"
                    f"Ratio: {r.interaction_ratio:.2f}x<br><br>"
                    f"<i>{r.interpretation[:200]}...</i>"
                )
                text[i][j] = hover
                text[j][i] = hover

                sym = type_symbols.get(r.interaction_type, "?")
                label = f"{sym} {r.interaction_type[:4]}\n{r.actual_combined:+.2f}"
                annotations[i][j] = label
                annotations[j][i] = label

        # Diagonal
        for i in range(n):
            z[i][i] = 0
            text[i][i] = f"<b>{all_names[i]}</b> (efecto individual)"

        # Colorscale: rojo (antagónico) → gris (aditivo) → verde (sinérgico)
        colorscale = [
            [0.0, "#c62828"],       # antagónico fuerte
            [0.25, "#ef9a9a"],      # antagónico leve
            [0.5, "#e0e0e0"],       # aditivo (neutro)
            [0.75, "#a5d6a7"],      # sinérgico leve
            [1.0, "#2e7d32"],       # sinérgico fuerte
        ]

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=all_names,
            y=all_names,
            text=text,
            hoverinfo="text",
            colorscale=colorscale,
            zmid=0,
            zmin=-1, zmax=1,
            showscale=True,
            colorbar=dict(
                title="Interacción",
                tickvals=[-0.8, 0, 0.8],
                ticktext=["Antagónico", "Aditivo", "Sinérgico"],
                len=0.6,
            ),
        ))

        # Añadir anotaciones de texto en las celdas
        for i in range(n):
            for j in range(n):
                if annotations[i][j]:
                    fig.add_annotation(
                        x=all_names[j], y=all_names[i],
                        text=annotations[i][j],
                        showarrow=False,
                        font=dict(size=9, color="black"),
                    )

        metric_label = {"bloom_mean": "Nivel Bloom", "autonomy_score": "Autonomía",
                        "pedagogical_value": "Valor pedagógico"}.get(metric, metric)

        fig.update_layout(
            title=dict(
                text=f"Matriz de Interacción entre Configuraciones — {metric_label}",
                font_size=14,
            ),
            height=height,
            margin=dict(l=120, r=30, t=50, b=80),
            paper_bgcolor="white",
            xaxis=dict(tickangle=-35, tickfont_size=10, side="bottom"),
            yaxis=dict(tickfont_size=10, autorange="reversed"),
        )

        return fig

    def render_detail_cards(self, metric: str = "bloom_mean"):
        """Tarjetas detalladas para cada interacción (Streamlit)."""
        if not STREAMLIT_AVAILABLE:
            return

        results = self.compute_interaction_effects(metric)

        type_icons = {
            "synergistic": "🟢", "antagonistic": "🔴",
            "additive": "⚪", "emergent": "🔵",
        }
        type_colors = {
            "synergistic": "#e8f5e9", "antagonistic": "#ffebee",
            "additive": "#f5f5f5", "emergent": "#e3f2fd",
        }

        for r in sorted(results, key=lambda x: abs(x.interaction_magnitude), reverse=True):
            icon = type_icons.get(r.interaction_type, "❓")
            bg = type_colors.get(r.interaction_type, "#f5f5f5")

            with st.container():
                st.markdown(
                    f"<div style='background:{bg};padding:12px;border-radius:8px;"
                    f"margin-bottom:8px;border-left:4px solid "
                    f"{'#2e7d32' if r.interaction_type == 'synergistic' else '#c62828' if r.interaction_type == 'antagonistic' else '#1565c0' if r.interaction_type == 'emergent' else '#9e9e9e'}'>"
                    f"<b>{icon} {r.label_a()} × {r.label_b()}</b> — "
                    f"<span style='text-transform:uppercase'>{r.interaction_type}</span>"
                    f"<br><br>"
                    f"<b>Efectos individuales:</b> "
                    f"{r.label_a()} = {r.effect_a_alone:+.2f} | "
                    f"{r.label_b()} = {r.effect_b_alone:+.2f}<br>"
                    f"<b>Esperado (aditivo):</b> {r.expected_additive:+.2f}<br>"
                    f"<b>Real combinado:</b> {r.actual_combined:+.2f} "
                    f"(ratio: {r.interaction_ratio:.2f}x)<br><br>"
                    f"<i>{r.interpretation}</i><br><br>"
                    f"<small>{r.theoretical_basis}</small><br><br>"
                    f"<b>{r.recommendation}</b>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ──────────────────────────────────────────────────────────────────
    # PANEL STREAMLIT COMPLETO
    # ──────────────────────────────────────────────────────────────────

    def render_streamlit_panel(self):
        """Panel completo para la pestaña 🧠 Análisis Profundo."""
        if not STREAMLIT_AVAILABLE:
            return

        st.subheader("🔀 Interacción entre Configuraciones")
        st.caption(
            "Qué pasa cuando múltiples configuraciones pedagógicas operan "
            "simultáneamente. Las configuraciones pueden amplificarse "
            "(sinérgicas), sabotearse (antagónicas), o producir efectos "
            "inesperados (emergentes)."
        )

        # Selector de métrica
        metric = st.selectbox(
            "Métrica de análisis",
            options=self.METRICS,
            format_func=lambda m: {"bloom_mean": "Nivel Bloom",
                                    "autonomy_score": "Autonomía epistémica",
                                    "pedagogical_value": "Valor pedagógico"}.get(m, m),
            key="config_interaction_metric",
        )

        # Heatmap
        fig = self.render_interaction_heatmap(metric)
        if fig:
            st.plotly_chart(fig, use_container_width=True)

        # Leyenda
        cols = st.columns(4)
        cols[0].markdown("🟢 **Sinérgico**: se amplifican")
        cols[1].markdown("🔴 **Antagónico**: se sabotean")
        cols[2].markdown("⚪ **Aditivo**: independientes")
        cols[3].markdown("🔵 **Emergente**: efecto inesperado")

        st.divider()
        st.markdown("##### Detalle por par de configuraciones")
        self.render_detail_cards(metric)


# ═══════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    analyzer = ConfigInteractionAnalyzer()
    matrix = analyzer.generate_demo_data()

    print("═" * 70)
    print("CONFIG INTERACTION ANALYZER — Demo")
    print("═" * 70)
    print()
    print('"Las configuraciones interactúan. El modo socrático con límite bajo')
    print(' de prompts produce un efecto antagónico que nadie esperaba."')
    print()

    type_icons = {"synergistic": "🟢", "antagonistic": "🔴",
                  "additive": "⚪", "emergent": "🔵"}

    for metric in ["bloom_mean", "autonomy_score", "pedagogical_value"]:
        ml = {"bloom_mean": "Bloom", "autonomy_score": "Autonomía",
              "pedagogical_value": "Valor ped."}.get(metric, metric)
        results = analyzer.compute_interaction_effects(metric)

        print(f"\n{'─' * 70}")
        print(f"  MÉTRICA: {ml}")
        print(f"{'─' * 70}")

        for r in sorted(results, key=lambda x: abs(x.interaction_magnitude), reverse=True):
            icon = type_icons.get(r.interaction_type, "?")
            print(f"\n  {icon} {r.label_a()} × {r.label_b()}")
            print(f"     Tipo: {r.interaction_type.upper()}")
            print(f"     Individual: A={r.effect_a_alone:+.3f}  B={r.effect_b_alone:+.3f}")
            print(f"     Esperado (A+B): {r.expected_additive:+.3f}")
            print(f"     Real combinado: {r.actual_combined:+.3f}")
            print(f"     Ratio: {r.interaction_ratio:.2f}x")
            if r.recommendation:
                print(f"     → {r.recommendation[:80]}")

    print(f"\n{'═' * 70}")
    print("  🟢 Sinérgico: socrático + alucinación (piensa + verifica)")
    print("  🔴 Antagónico: socrático + límite bajo (el diálogo se corta)")
    print("  ⚪ Aditivo: bloqueo + RAG (dimensiones ortogonales)")
    print("═" * 70)
