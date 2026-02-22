"""
PILOT DESIGN — ANÁLISIS AUTOMÁTICO DEL PILOTO
═══════════════════════════════════════════════════════════════════════
Eje 2 del contrato CP25/152: infraestructura de análisis lista para correr.

LO QUE HACE ESTE MÓDULO:
─────────────────────────
Recibe los logs exportados del sistema (CSV o JSON) y ejecuta
automáticamente el pipeline completo de análisis:

    1. Power analysis y diseño del estudio
    2. Estadística descriptiva del cohort
    3. Topic modeling (BERTopic sobre prompts)
    4. Clustering de perfiles de engagement (K-Means + análisis de silueta)
    5. Correlación uso-rendimiento (Spearman, no Pearson)
    6. Análisis de trayectoria de autonomía epistémica
    7. Informe de calibración docente
    8. Visualizaciones publicables (exportadas como HTML y PNG)

DISEÑO QUASI-EXPERIMENTAL:
────────────────────────────
Within-subject con medidas repetidas. El mismo estudiante es su propio
control: se comparan métricas en semanas con configuración A vs. B.
Esto elimina varianza entre sujetos y reduce el n necesario.

Power analysis (Cohen, 1988):
  - Cohen's d = 0.5 (efecto medio, conservador)
  - α = 0.05 (nivel de significación estándar)
  - Power = 0.80 (estándar en ciencias sociales)
  - n ≈ 34 por grupo (between-subject)
  - n ≈ 25 within-subject (medidas repetidas, correlación r=0.5 asumida)
  - Target: 60-80 estudiantes para robustecer subgrupos

INSTRUMENTOS VALIDADOS:
─────────────────────────
  - SUS (System Usability Scale, Brooke, 1996): 10 ítems, escala Likert 5
  - UMUX-LITE (Lewis et al., 2013): 2 ítems, correlación alta con SUS
  - NPS (Net Promoter Score): 1 ítem + justificación abierta
  - Escala de experiencia previa con IA: 5 ítems Likert (autoría)
  - Autonomía epistémica percibida: 6 ítems Likert (autoría, basada en Ryan & Deci)

STACK:
  - pandas: manipulación de datos
  - scipy.stats: Spearman, Wilcoxon, Mann-Whitney
  - sklearn: K-Means, silhouette, t-SNE
  - bertopic: topic modeling (BERTopic)
  - plotly: visualizaciones exportables
  - sentence_transformers: embeddings para BERTopic (fallback: TF-IDF)

FALLBACK sin dependencias pesadas:
  Si BERTopic o sentence_transformers no están disponibles, el módulo
  usa TF-IDF + LDA como alternativa. El pipeline sigue funcionando.

Autor: Diego Elvira Vásquez · Prototipo CP25/152 · Feb 2026
"""

from __future__ import annotations

import os
import json
import math
import csv
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── Imports opcionales (graceful degradation) ──
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from bertopic import BERTopic
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    LDA_AVAILABLE = True
except ImportError:
    LDA_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════
# DATACLASSES DE CONFIGURACIÓN DEL ESTUDIO
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StudyDesign:
    """Configuración del diseño quasi-experimental."""
    study_type: str = "within_subject"          # "within_subject" | "between_subject"
    n_conditions: int = 2                        # condición A (socrático ON) vs B (OFF)
    alpha: float = 0.05
    power: float = 0.80
    effect_size_d: float = 0.5                  # Cohen's d conservador
    assumed_correlation: float = 0.5            # correlación esperada entre medidas repetidas

    # Calculado en power_analysis()
    n_required_between: int = 0
    n_required_within: int = 0
    n_target: int = 60                          # objetivo real con margen de abandono

    # Ventanas de análisis
    baseline_weeks: int = 2
    intervention_weeks: int = 4
    followup_weeks: int = 2


@dataclass
class StudentRecord:
    """Un registro de estudiante para el análisis del piloto."""
    student_id: str
    course_id: str
    condition: str = "A"                        # "A" | "B" (para between-subject)

    # Métricas de uso
    total_prompts: int = 0
    total_sessions: int = 0
    mean_session_length_min: float = 0.0
    topics_covered: list = field(default_factory=list)

    # Métricas cognitivas (de cognitive_analyzer)
    bloom_mean: float = 0.0
    bloom_trajectory_slope: float = 0.0        # pendiente de Bloom a lo largo del tiempo
    bloom_ceiling: int = 1                      # nivel máximo alcanzado

    # Métricas de autonomía (de epistemic_autonomy)
    autonomy_final: float = 0.0
    autonomy_initial: float = 0.0
    autonomy_delta: float = 0.0
    dominant_epistemic_mode: str = "consumption"

    # Métricas semióticas (de interaction_semiotics)
    pedagogical_value_mean: float = 0.0
    hypothesis_test_rate: float = 0.0
    solution_request_rate: float = 0.0
    gaming_events: int = 0

    # Métricas de confianza (de trust_dynamics)
    trust_calibration: float = 0.0

    # Instrumentos de evaluación (autoreporte)
    sus_score: float = 0.0                     # 0-100
    umux_score: float = 0.0                    # 0-100
    nps_score: int = 0                         # -100 a 100
    ai_experience_level: int = 1               # 1-5

    # Rendimiento académico (opcional, del LMS)
    grade_pre: Optional[float] = None
    grade_post: Optional[float] = None
    grade_delta: Optional[float] = None

    # Asignado en el clustering
    cluster_id: int = -1
    cluster_label: str = ""


@dataclass
class PilotAnalysisReport:
    """Informe completo del análisis del piloto."""
    generated_at: str = ""
    n_students: int = 0
    n_complete: int = 0

    # Resultados del power analysis
    power_analysis: dict = field(default_factory=dict)

    # Estadísticas descriptivas
    descriptive_stats: dict = field(default_factory=dict)

    # Topics identificados
    topic_model_results: dict = field(default_factory=dict)

    # Clusters de engagement
    cluster_profiles: list = field(default_factory=list)

    # Correlaciones
    correlations: dict = field(default_factory=dict)

    # Tests de hipótesis
    hypothesis_tests: dict = field(default_factory=dict)

    # Calibración docente
    teacher_calibration_summary: dict = field(default_factory=dict)

    # Alertas de calidad del análisis
    quality_warnings: list = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# MÓDULO 1: POWER ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def power_analysis(design: StudyDesign) -> dict:
    """
    Calcula el tamaño muestral necesario.

    Fórmulas:
    Between-subject (Cohen, 1988):
      n_per_group = ((z_alpha + z_beta) / d) ** 2 * 2

    Within-subject (corrección Dunlap, 1994):
      n = n_between * (1 - r) / 2 + 1
      donde r es la correlación entre medidas repetidas.

    z_alpha/2 = 1.96 (α=0.05, two-tailed)
    z_beta = 0.84 (power=0.80)
    """
    z_alpha = 1.96    # α=0.05, two-tailed
    z_beta  = 0.8416  # power=0.80

    # Between-subject
    n_between = math.ceil(
        2 * ((z_alpha + z_beta) / design.effect_size_d) ** 2
    )
    design.n_required_between = n_between

    # Within-subject (corrección de Dunlap, 1994)
    n_within = math.ceil(
        n_between * (1 - design.assumed_correlation) / 2 + 1
    )
    # Añadir 20% de margen para abandono
    n_within_with_margin = math.ceil(n_within * 1.25)
    design.n_required_within = n_within_with_margin

    result = {
        "design": design.study_type,
        "alpha": design.alpha,
        "power": design.power,
        "effect_size_d": design.effect_size_d,
        "n_between_per_group": n_between,
        "n_within_required": n_within_with_margin,
        "n_target": design.n_target,
        "feasibility": (
            "VIABLE" if design.n_target >= n_within_with_margin
            else f"INSUFICIENTE (necesitas {n_within_with_margin}, tienes {design.n_target})"
        ),
        "interpretation": (
            f"Para detectar un efecto medio (d={design.effect_size_d}) con "
            f"power={design.power:.0%} y α={design.alpha}: "
            f"necesitas {n_between} por grupo (between-subject) o "
            f"{n_within_with_margin} estudiantes (within-subject con margen 25%). "
            f"Con {design.n_target} estudiantes el estudio es "
            f"{'viable' if design.n_target >= n_within_with_margin else 'inviable'}."
        ),
    }

    return result


# ═══════════════════════════════════════════════════════════════════════
# MÓDULO 2: ESTADÍSTICA DESCRIPTIVA
# ═══════════════════════════════════════════════════════════════════════

def compute_descriptive_stats(records: list[StudentRecord]) -> dict:
    """Estadísticas descriptivas del cohort."""
    if not records:
        return {"n": 0, "error": "Sin datos"}

    n = len(records)

    def stats_for(values):
        if not values:
            return {}
        n_v = len(values)
        mean = sum(values) / n_v
        variance = sum((x - mean) ** 2 for x in values) / n_v
        std = math.sqrt(variance)
        sorted_v = sorted(values)
        median = sorted_v[n_v // 2] if n_v % 2 else (sorted_v[n_v//2-1] + sorted_v[n_v//2]) / 2
        return {
            "n": n_v,
            "mean": round(mean, 3),
            "std": round(std, 3),
            "median": round(median, 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }

    return {
        "n_total": n,
        "bloom_mean": stats_for([r.bloom_mean for r in records]),
        "bloom_trajectory_slope": stats_for([r.bloom_trajectory_slope for r in records]),
        "autonomy_final": stats_for([r.autonomy_final for r in records]),
        "autonomy_delta": stats_for([r.autonomy_delta for r in records]),
        "pedagogical_value": stats_for([r.pedagogical_value_mean for r in records]),
        "total_prompts": stats_for([r.total_prompts for r in records]),
        "hypothesis_test_rate": stats_for([r.hypothesis_test_rate for r in records]),
        "solution_request_rate": stats_for([r.solution_request_rate for r in records]),
        "sus_score": stats_for([r.sus_score for r in records if r.sus_score > 0]),
        "nps_score": stats_for([r.nps_score for r in records]),

        # Distribución de modos epistémicos
        "epistemic_mode_distribution": dict(
            zip(*[["consumption", "verification", "exploration", "delegation"],
                  [sum(1 for r in records if r.dominant_epistemic_mode == m) / n
                   for m in ["consumption", "verification", "exploration", "delegation"]]])
        ),

        # Distribución de ai_experience_level
        "ai_experience_distribution": {
            level: sum(1 for r in records if r.ai_experience_level == level) / n
            for level in range(1, 6)
        },
    }


# ═══════════════════════════════════════════════════════════════════════
# MÓDULO 3: TOPIC MODELING
# ═══════════════════════════════════════════════════════════════════════

def run_topic_modeling(prompts: list[str], n_topics: int = 8) -> dict:
    """
    Topic modeling sobre los prompts del piloto.

    Estrategia:
    1. Si BERTopic disponible: usar sentence-transformers (mejor calidad)
    2. Fallback: TF-IDF + LDA (sin GPU, sin modelos grandes)

    Returns dict con topics, palabras clave por topic, y asignación por prompt.
    """
    if not prompts:
        return {"error": "Sin prompts para analizar"}

    method_used = None
    topics_result = {}

    # ── BERTopic (preferido) ──
    if BERTOPIC_AVAILABLE:
        try:
            topic_model = BERTopic(
                language="multilingual",
                nr_topics=n_topics,
                verbose=False,
                calculate_probabilities=False,
            )
            topics_raw, _ = topic_model.fit_transform(prompts)
            topic_info = topic_model.get_topic_info()

            topic_keywords = {}
            for _, row in topic_info.iterrows():
                tid = row["Topic"]
                if tid == -1:
                    continue
                words = topic_model.get_topic(tid)
                topic_keywords[f"topic_{tid}"] = [w for w, _ in words[:6]]

            topic_assignment = [f"topic_{t}" if t >= 0 else "noise" for t in topics_raw]
            method_used = "BERTopic (sentence-transformers)"

            topics_result = {
                "method": method_used,
                "n_topics_detected": len(topic_keywords),
                "topic_keywords": topic_keywords,
                "topic_assignment": topic_assignment,
                "topic_distribution": {
                    tid: topic_assignment.count(tid) / len(topic_assignment)
                    for tid in set(topic_assignment)
                },
            }

        except Exception as e:
            topics_result = {"error": f"BERTopic falló: {e}"}

    # ── LDA Fallback ──
    if not topics_result.get("topic_keywords") and LDA_AVAILABLE:
        try:
            vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words=None,   # sin stopwords en español en sklearn básico
                min_df=2,
                ngram_range=(1, 2),
            )
            X = vectorizer.fit_transform(prompts)
            feature_names = vectorizer.get_feature_names_out()

            lda = LatentDirichletAllocation(
                n_components=min(n_topics, len(prompts) // 3),
                random_state=42,
                max_iter=20,
            )
            lda.fit(X)

            topic_keywords = {}
            for i, component in enumerate(lda.components_):
                top_words = [feature_names[j] for j in component.argsort()[:-7:-1]]
                topic_keywords[f"topic_{i}"] = top_words

            topic_assignments = lda.transform(X).argmax(axis=1)
            topic_assignment = [f"topic_{t}" for t in topic_assignments]
            method_used = "TF-IDF + LDA (fallback)"

            topics_result = {
                "method": method_used,
                "n_topics_detected": len(topic_keywords),
                "topic_keywords": topic_keywords,
                "topic_assignment": topic_assignment,
                "topic_distribution": {
                    f"topic_{i}": (topic_assignments == i).sum() / len(topic_assignments)
                    for i in range(len(topic_keywords))
                },
            }

        except Exception as e:
            topics_result = {"error": f"LDA también falló: {e}", "method": "none"}
    elif not topics_result.get("topic_keywords"):
        topics_result = {
            "error": "BERTopic y sklearn no disponibles. pip install bertopic scikit-learn",
            "method": "none",
        }

    return topics_result


# ═══════════════════════════════════════════════════════════════════════
# MÓDULO 4: CLUSTERING DE PERFILES
# ═══════════════════════════════════════════════════════════════════════

def run_clustering(
    records: list[StudentRecord],
    k_range: tuple = (2, 6),
) -> dict:
    """
    K-Means con análisis de silueta para k óptimo.

    Features del clustering:
    - bloom_mean (normalizado)
    - autonomy_final (normalizado)
    - pedagogical_value_mean (normalizado)
    - hypothesis_test_rate
    - solution_request_rate
    - total_prompts (normalizado)
    - bloom_trajectory_slope (pendiente temporal)

    El clustering produce PERFILES DE ENGAGEMENT MULTINIVEL:
    No "estudiante X: 47 prompts" sino "verificadores intensivos",
    "delegadores esporádicos", "exploradores autónomos".

    Esto es lo que el briefing pide literalmente: "higher-level analytics
    that derive engagement profiles" (LAK 2026, Ortega-Arranz et al.)
    """
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn no disponible. pip install scikit-learn"}
    if len(records) < 4:
        return {"error": f"Insuficientes registros para clustering: {len(records)} < 4"}

    # Feature matrix
    feature_vectors = []
    for r in records:
        feature_vectors.append([
            r.bloom_mean,
            r.autonomy_final,
            r.pedagogical_value_mean,
            r.hypothesis_test_rate,
            r.solution_request_rate,
            min(r.total_prompts / 50.0, 1.0),      # normalizar
            r.bloom_trajectory_slope + 0.5,          # centrar en 0.5
        ])

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_vectors)

    # Análisis de silueta para k óptimo
    best_k = k_range[0]
    best_silhouette = -1
    silhouette_scores = {}

    for k in range(k_range[0], min(k_range[1] + 1, len(records))):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        silhouette_scores[k] = round(score, 3)
        if score > best_silhouette:
            best_silhouette = score
            best_k = k

    # Clustering con k óptimo
    kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = kmeans_final.fit_predict(X)

    # Asignar cluster a cada estudiante
    for i, record in enumerate(records):
        record.cluster_id = int(labels[i])

    # Construir perfiles por cluster
    cluster_profiles = []
    for cluster_id in range(best_k):
        members = [r for r in records if r.cluster_id == cluster_id]
        if not members:
            continue

        n_m = len(members)
        profile_bloom = sum(r.bloom_mean for r in members) / n_m
        profile_autonomy = sum(r.autonomy_final for r in members) / n_m
        profile_ped_value = sum(r.pedagogical_value_mean for r in members) / n_m
        profile_prompts = sum(r.total_prompts for r in members) / n_m
        profile_hyp = sum(r.hypothesis_test_rate for r in members) / n_m
        profile_sol = sum(r.solution_request_rate for r in members) / n_m

        # Generar etiqueta descriptiva
        label = _generate_cluster_label(
            bloom=profile_bloom,
            autonomy=profile_autonomy,
            ped_value=profile_ped_value,
            prompts=profile_prompts,
            hyp_rate=profile_hyp,
            sol_rate=profile_sol,
        )

        # Asignar label a cada miembro
        for member in members:
            member.cluster_label = label

        cluster_profiles.append({
            "cluster_id": cluster_id,
            "label": label,
            "n_members": n_m,
            "pct_cohort": round(n_m / len(records), 2),
            "bloom_mean": round(profile_bloom, 2),
            "autonomy_mean": round(profile_autonomy, 3),
            "pedagogical_value_mean": round(profile_ped_value, 2),
            "total_prompts_mean": round(profile_prompts, 1),
            "hypothesis_test_rate": round(profile_hyp, 2),
            "solution_request_rate": round(profile_sol, 2),
            "member_ids": [r.student_id for r in members],
        })

    # Reducción PCA para visualización 2D (si hay suficientes features)
    pca_coords = None
    if len(X[0]) >= 2:
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(X)
        pca_coords = {
            "x": [float(c[0]) for c in coords_2d],
            "y": [float(c[1]) for c in coords_2d],
            "cluster": [int(l) for l in labels],
            "student_ids": [r.student_id for r in records],
            "variance_explained": round(sum(pca.explained_variance_ratio_), 2),
        }

    return {
        "optimal_k": best_k,
        "silhouette_scores": silhouette_scores,
        "best_silhouette": round(best_silhouette, 3),
        "cluster_profiles": cluster_profiles,
        "pca_visualization": pca_coords,
        "interpretation": (
            f"Análisis de silueta identificó k={best_k} como el número óptimo de clusters "
            f"(score={best_silhouette:.2f}). "
            f"Perfiles emergentes: {', '.join(p['label'] for p in cluster_profiles)}."
        ),
    }


def _generate_cluster_label(
    bloom: float,
    autonomy: float,
    ped_value: float,
    prompts: float,
    hyp_rate: float,
    sol_rate: float,
) -> str:
    """
    Genera una etiqueta descriptiva del cluster basada en sus métricas.
    Las etiquetas son los 'perfiles de engagement multinivel' que el
    briefing pide explícitamente.
    """
    high_use = prompts > 15
    high_quality = ped_value > 0.55 and bloom > 3.0
    high_autonomy = autonomy > 0.55
    delegating = sol_rate > 0.5 or (autonomy < 0.25 and prompts < 8)
    verifying = hyp_rate > 0.25 and bloom < 3.5
    exploring = hyp_rate > 0.3 and autonomy > 0.5 and bloom > 3.5

    if exploring:
        return "Exploradores autónomos"
    elif high_use and high_quality:
        return "Verificadores intensivos"
    elif delegating and not high_use:
        return "Delegadores esporádicos"
    elif high_use and not high_quality:
        return "Consultores de volumen"
    elif verifying and not high_autonomy:
        return "Validadores dependientes"
    elif not high_use and high_quality:
        return "Reflexivos selectivos"
    else:
        return f"Perfil mixto (Bloom {bloom:.1f})"


# ═══════════════════════════════════════════════════════════════════════
# MÓDULO 5: CORRELACIONES Y TESTS DE HIPÓTESIS
# ═══════════════════════════════════════════════════════════════════════

def run_statistical_analysis(records: list[StudentRecord]) -> dict:
    """
    Spearman (no Pearson — los datos no serán normales).
    Por qué Spearman: los scores de Bloom, autonomía y valor pedagógico
    son ordinales o semi-continuos, la distribución será sesgada positivamente
    (floor effects en estudiantes poco activos), y outliers son esperables.
    La correlación de Pearson asume normalidad y es sensible a outliers.
    Spearman es el estándar en educational data mining (Baker, 2010).

    También: Wilcoxon signed-rank (datos pareados within-subject)
    y Mann-Whitney U (comparación entre condiciones si hay between-subject).
    """
    if not records:
        return {"error": "Sin datos"}

    correlations = {}
    tests = {}
    warnings_list = []

    # ── Spearman entre uso y métricas cognitivas ──
    pairs_to_correlate = [
        ("total_prompts", "bloom_mean", "Uso (prompts) vs. nivel Bloom"),
        ("total_prompts", "autonomy_final", "Uso (prompts) vs. autonomía final"),
        ("total_prompts", "pedagogical_value_mean", "Uso vs. valor pedagógico"),
        ("bloom_mean", "autonomy_final", "Bloom vs. autonomía (validación convergente)"),
        ("bloom_mean", "sus_score", "Bloom vs. satisfacción (SUS)"),
        ("autonomy_delta", "bloom_trajectory_slope", "Δ autonomía vs. pendiente Bloom"),
        ("hypothesis_test_rate", "autonomy_final", "Tasa hipótesis vs. autonomía"),
        ("solution_request_rate", "bloom_mean", "Solicitudes directas vs. Bloom (hipótesis: negativo)"),
    ]

    for var_x, var_y, label in pairs_to_correlate:
        x_vals = [getattr(r, var_x, None) for r in records]
        y_vals = [getattr(r, var_y, None) for r in records]

        # Filtrar None y ceros irrelevantes (p.ej. SUS no rellenado)
        pairs = [(x, y) for x, y in zip(x_vals, y_vals)
                 if x is not None and y is not None]

        if len(pairs) < 5:
            correlations[label] = {"error": f"n={len(pairs)} insuficiente"}
            continue

        xs, ys = zip(*pairs)

        if SCIPY_AVAILABLE:
            rho, p_val = scipy_stats.spearmanr(xs, ys)
            correlations[label] = {
                "rho": round(rho, 3),
                "p_value": round(p_val, 4),
                "n": len(xs),
                "significant": p_val < 0.05,
                "interpretation": _interpret_correlation(rho, p_val, label),
            }
        else:
            # Spearman manual (sin scipy)
            rho = _manual_spearman(xs, ys)
            correlations[label] = {
                "rho": round(rho, 3),
                "n": len(xs),
                "note": "scipy no disponible — p-value no calculado",
            }

    # ── Causalidad inversa — advertencia ──
    total_prompts_bloom_corr = correlations.get(
        "Uso (prompts) vs. nivel Bloom", {}
    ).get("rho", 0)
    if total_prompts_bloom_corr and total_prompts_bloom_corr > 0.4:
        warnings_list.append(
            "ADVERTENCIA METODOLÓGICA: la correlación positiva entre total_prompts y bloom_mean "
            "es igualmente compatible con dos hipótesis causales opuestas: "
            "(1) más uso → mejor aprendizaje; "
            "(2) mejores estudiantes hacen más preguntas. "
            "El diseño within-subject reduce pero no elimina esta ambigüedad. "
            "Reportar como correlación, no como causalidad."
        )

    # ── Tests de hipótesis principales (si hay datos de condición) ──
    conditions = set(r.condition for r in records)
    if len(conditions) == 2:
        cond_a = [r for r in records if r.condition == "A"]
        cond_b = [r for r in records if r.condition == "B"]

        for metric, label in [
            ("bloom_mean", "Bloom A vs. B"),
            ("autonomy_final", "Autonomía A vs. B"),
            ("pedagogical_value_mean", "Valor pedagógico A vs. B"),
        ]:
            a_vals = [getattr(r, metric) for r in cond_a]
            b_vals = [getattr(r, metric) for r in cond_b]

            if SCIPY_AVAILABLE and len(a_vals) >= 5 and len(b_vals) >= 5:
                stat, p = scipy_stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
                d = _cohens_d(a_vals, b_vals)
                tests[label] = {
                    "test": "Mann-Whitney U",
                    "U": round(stat, 1),
                    "p_value": round(p, 4),
                    "significant": p < 0.05,
                    "cohens_d": round(d, 3),
                    "effect_size_label": "grande" if abs(d) > 0.8 else ("medio" if abs(d) > 0.5 else "pequeño"),
                }

    if len(records) > 1 and SCIPY_AVAILABLE:
        # Wilcoxon para autonomy_initial vs autonomy_final (within-subject)
        initial = [r.autonomy_initial for r in records if r.autonomy_initial > 0]
        final = [r.autonomy_final for r in records if r.autonomy_final > 0]
        if len(initial) == len(final) and len(initial) >= 5:
            try:
                stat, p = scipy_stats.wilcoxon(initial, final)
                tests["Autonomía inicial vs. final (within)"] = {
                    "test": "Wilcoxon signed-rank",
                    "W": round(stat, 1),
                    "p_value": round(p, 4),
                    "significant": p < 0.05,
                    "note": "Test clave para H1: el chatbot incrementa la autonomía epistémica.",
                }
            except Exception:
                pass

    return {
        "correlations": correlations,
        "hypothesis_tests": tests,
        "warnings": warnings_list,
    }


def _interpret_correlation(rho: float, p_val: float, label: str) -> str:
    sig = "significativa (p<0.05)" if p_val < 0.05 else "no significativa"
    magnitude = "fuerte" if abs(rho) > 0.5 else ("moderada" if abs(rho) > 0.3 else "débil")
    direction = "positiva" if rho > 0 else "negativa"
    return f"Correlación {direction} {magnitude} ({sig}): ρ={rho:.3f}"


def _manual_spearman(xs, ys):
    """Spearman sin scipy — para fallback."""
    n = len(xs)
    if n < 2:
        return 0.0
    ranks_x = _rank(xs)
    ranks_y = _rank(ys)
    d_sq = sum((rx - ry) ** 2 for rx, ry in zip(ranks_x, ranks_y))
    return 1 - (6 * d_sq) / (n * (n ** 2 - 1))


def _rank(values):
    sorted_vals = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0] * len(values)
    for rank, (idx, _) in enumerate(sorted_vals):
        ranks[idx] = rank + 1
    return ranks


def _cohens_d(a, b):
    """Cohen's d para dos muestras independientes."""
    n_a, n_b = len(a), len(b)
    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / n_a
    var_b = sum((x - mean_b) ** 2 for x in b) / n_b
    pooled_std = math.sqrt((var_a + var_b) / 2)
    return (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════
# MÓDULO 6: VISUALIZACIONES PUBLICABLES
# ═══════════════════════════════════════════════════════════════════════

def build_cluster_scatter(clustering_result: dict) -> Optional["go.Figure"]:
    """
    Scatter plot 2D de los clusters (PCA).
    Publicable en paper sin modificaciones.
    """
    if not PLOTLY_AVAILABLE:
        return None

    pca = clustering_result.get("pca_visualization")
    profiles = clustering_result.get("cluster_profiles", [])
    if not pca:
        return None

    cluster_labels_map = {p["cluster_id"]: p["label"] for p in profiles}
    colors = px.colors.qualitative.Safe

    fig = go.Figure()

    for cluster_id in sorted(set(pca["cluster"])):
        mask = [c == cluster_id for c in pca["cluster"]]
        x_vals = [x for x, m in zip(pca["x"], mask) if m]
        y_vals = [y for y, m in zip(pca["y"], mask) if m]
        ids = [sid for sid, m in zip(pca["student_ids"], mask) if m]
        label = cluster_labels_map.get(cluster_id, f"Cluster {cluster_id}")

        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers+text",
            marker=dict(
                size=12,
                color=colors[cluster_id % len(colors)],
                opacity=0.75,
                line=dict(width=1, color="white"),
            ),
            text=[sid[:4] for sid in ids],
            textposition="top center",
            textfont=dict(size=7),
            name=label,
            hovertemplate=f"<b>{label}</b><br>Estudiante: %{{text}}<extra></extra>",
        ))

    fig.update_layout(
        title=f"Perfiles de engagement — Clustering K-Means (PCA, {pca['variance_explained']:.0%} varianza)",
        xaxis_title=f"PC1",
        yaxis_title=f"PC2",
        height=450,
        legend_title="Perfil",
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
        font=dict(size=12),
    )

    return fig


def build_bloom_trajectory_heatmap(records: list[StudentRecord]) -> Optional["go.Figure"]:
    """
    Heatmap de trayectorias de Bloom por estudiante.
    Muestra la distribución temporal de niveles cognitivos.
    """
    if not PLOTLY_AVAILABLE or not records:
        return None

    # Ordenar por cluster para agrupación visual
    records_sorted = sorted(records, key=lambda r: (r.cluster_id, r.bloom_mean))

    student_ids = [r.student_id[:6] for r in records_sorted]
    metrics = ["Bloom", "Autonomía", "Val. ped.", "H. test", "Prompts (norm.)"]

    data = []
    for r in records_sorted:
        data.append([
            r.bloom_mean / 6.0,
            r.autonomy_final,
            r.pedagogical_value_mean,
            r.hypothesis_test_rate,
            min(r.total_prompts / 50.0, 1.0),
        ])

    import plotly.graph_objects as go
    fig = go.Figure(go.Heatmap(
        z=[[row[i] for row in data] for i in range(len(metrics))],
        x=student_ids,
        y=metrics,
        colorscale="RdYlGn",
        zmid=0.5,
        colorbar=dict(title="Valor norm."),
        text=[[f"{row[i]:.2f}" for row in data] for i in range(len(metrics))],
        hovertemplate="Estudiante: %{x}<br>Métrica: %{y}<br>Valor: %{z:.2f}<extra></extra>",
    ))

    fig.update_layout(
        title="Mapa de engagement por estudiante (ordenado por cluster)",
        height=320,
        xaxis_title="Estudiante",
        xaxis=dict(tickangle=45, tickfont=dict(size=9)),
    )

    return fig


def build_correlation_matrix(stats_result: dict) -> Optional["go.Figure"]:
    """
    Matriz de correlaciones Spearman visualizada como heatmap.
    """
    if not PLOTLY_AVAILABLE:
        return None

    corrs = stats_result.get("correlations", {})
    if not corrs:
        return None

    # Extraer pares con datos válidos
    labels = []
    rhos = []
    for label, data in corrs.items():
        if "rho" in data:
            short = label.split(" vs. ")[0][:15]
            labels.append(label[:40])
            rhos.append(data["rho"])

    if not rhos:
        return None

    colors = ["#c62828" if r < 0 else "#2e7d32" for r in rhos]
    sizes = [abs(r) * 30 + 10 for r in rhos]

    fig = go.Figure(go.Bar(
        x=labels,
        y=rhos,
        marker_color=colors,
        text=[f"ρ={r:.2f}" + ("*" if corrs[l].get("significant") else "")
              for r, l in zip(rhos, labels)],
        textposition="outside",
    ))

    fig.add_hline(y=0, line=dict(color="#888", width=1))
    fig.add_hline(y=0.3, line=dict(color="#66BB6A", width=1, dash="dot"))
    fig.add_hline(y=-0.3, line=dict(color="#EF5350", width=1, dash="dot"))

    fig.update_layout(
        title="Correlaciones Spearman (ρ) — * p<0.05",
        yaxis=dict(title="ρ de Spearman", range=[-1.1, 1.1]),
        xaxis=dict(tickangle=45, tickfont=dict(size=10)),
        height=380,
        paper_bgcolor="white",
        plot_bgcolor="#fafafa",
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# PIPELINE COMPLETO
# ═══════════════════════════════════════════════════════════════════════

def run_full_analysis(
    records: list[StudentRecord],
    prompts_log: Optional[list[str]] = None,
    design: Optional[StudyDesign] = None,
    output_dir: Optional[str] = None,
) -> PilotAnalysisReport:
    """
    PUNTO DE ENTRADA PRINCIPAL.
    
    Ejecuta el pipeline completo de análisis del piloto y devuelve
    un PilotAnalysisReport con todos los resultados.

    Args:
        records: lista de StudentRecord (uno por estudiante)
        prompts_log: lista de todos los prompts del piloto (para topic modeling)
        design: configuración del diseño del estudio
        output_dir: directorio para exportar visualizaciones (HTML, JSON)

    Returns:
        PilotAnalysisReport con todos los análisis ejecutados
    """
    print("=" * 65)
    print("GENIE Learn — Pipeline de Análisis del Piloto")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    report = PilotAnalysisReport(generated_at=datetime.now().isoformat())
    report.n_students = len(records)
    report.n_complete = sum(1 for r in records if r.total_prompts > 3)

    if design is None:
        design = StudyDesign()

    # ── 1. Power analysis ──
    print("\n[1/6] Power analysis...")
    report.power_analysis = power_analysis(design)
    print(f"  → {report.power_analysis['interpretation'][:80]}...")
    print(f"  → Viabilidad: {report.power_analysis['feasibility']}")

    # ── 2. Estadísticas descriptivas ──
    print("\n[2/6] Estadísticas descriptivas...")
    report.descriptive_stats = compute_descriptive_stats(records)
    bloom_stats = report.descriptive_stats.get("bloom_mean", {})
    print(f"  → Bloom medio del cohort: {bloom_stats.get('mean', 'N/A')} "
          f"(σ={bloom_stats.get('std', 'N/A')})")
    auto_stats = report.descriptive_stats.get("autonomy_final", {})
    print(f"  → Autonomía final media: {auto_stats.get('mean', 'N/A')}")

    # ── 3. Topic modeling ──
    print("\n[3/6] Topic modeling sobre prompts...")
    if prompts_log:
        report.topic_model_results = run_topic_modeling(prompts_log)
        method = report.topic_model_results.get("method", "none")
        n_topics = report.topic_model_results.get("n_topics_detected", 0)
        print(f"  → Método: {method}")
        print(f"  → Topics identificados: {n_topics}")
        if "topic_keywords" in report.topic_model_results:
            for tid, words in list(report.topic_model_results["topic_keywords"].items())[:3]:
                print(f"     {tid}: {', '.join(words[:4])}")
    else:
        report.topic_model_results = {"note": "Sin prompts_log proporcionado."}
        print("  → Omitido (no se proporcionó prompts_log)")

    # ── 4. Clustering ──
    print("\n[4/6] Clustering de perfiles de engagement...")
    clustering_result = run_clustering(records)
    if "cluster_profiles" in clustering_result:
        report.cluster_profiles = clustering_result["cluster_profiles"]
        k = clustering_result["optimal_k"]
        sil = clustering_result["best_silhouette"]
        print(f"  → k óptimo: {k} (silhouette={sil:.2f})")
        for p in report.cluster_profiles:
            print(f"     [{p['label']}] n={p['n_members']} ({p['pct_cohort']:.0%}) | "
                  f"Bloom={p['bloom_mean']:.1f} | Autonomía={p['autonomy_mean']:.2f}")
    else:
        print(f"  → {clustering_result.get('error', 'Error desconocido')}")

    # ── 5. Correlaciones y tests ──
    print("\n[5/6] Análisis estadístico (Spearman + tests)...")
    stats_result = run_statistical_analysis(records)
    report.correlations = stats_result.get("correlations", {})
    report.hypothesis_tests = stats_result.get("hypothesis_tests", {})
    report.quality_warnings = stats_result.get("warnings", [])

    n_sig = sum(1 for c in report.correlations.values() if c.get("significant"))
    print(f"  → Correlaciones calculadas: {len(report.correlations)}")
    print(f"  → Significativas (p<0.05): {n_sig}")
    for label, data in report.correlations.items():
        if data.get("significant"):
            print(f"     ✓ {label[:50]}: ρ={data['rho']:.3f} (p={data['p_value']:.4f})")
    if report.quality_warnings:
        for w in report.quality_warnings:
            print(f"  ⚠️  {w[:90]}...")

    # ── 6. Exportar visualizaciones ──
    print("\n[6/6] Exportando visualizaciones...")
    if output_dir and PLOTLY_AVAILABLE:
        os.makedirs(output_dir, exist_ok=True)

        figs = {
            "cluster_scatter": build_cluster_scatter(clustering_result),
            "bloom_heatmap": build_bloom_trajectory_heatmap(records),
            "correlations": build_correlation_matrix(stats_result),
        }

        for name, fig in figs.items():
            if fig:
                path_html = os.path.join(output_dir, f"{name}.html")
                fig.write_html(path_html)
                print(f"  → {path_html}")

        # Exportar informe JSON
        report_dict = asdict(report)
        path_json = os.path.join(output_dir, "pilot_analysis_report.json")
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2, default=str)
        print(f"  → {path_json}")
    else:
        print("  → Omitido (output_dir no especificado o Plotly no disponible)")

    print("\n" + "=" * 65)
    print("ANÁLISIS COMPLETO.")
    print(f"Estudiantes analizados: {report.n_students} ({report.n_complete} con datos completos)")
    print("=" * 65)

    return report


# ═══════════════════════════════════════════════════════════════════════
# CARGA DE DATOS DESDE CSV/JSON
# ═══════════════════════════════════════════════════════════════════════

def load_records_from_csv(csv_path: str) -> list[StudentRecord]:
    """
    Carga StudentRecords desde un CSV exportado del sistema.

    Formato esperado: columnas con nombres que coinciden con los campos
    de StudentRecord (bloom_mean, autonomy_final, total_prompts, etc.)
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas requerido para cargar CSV. pip install pandas")

    df = pd.read_csv(csv_path)
    records = []

    for _, row in df.iterrows():
        record = StudentRecord(
            student_id=str(row.get("student_id", f"S{len(records)+1:03d}")),
            course_id=str(row.get("course_id", "UNKNOWN")),
        )

        # Mapear columnas disponibles
        for field_name in [
            "bloom_mean", "bloom_trajectory_slope", "autonomy_final", "autonomy_initial",
            "autonomy_delta", "pedagogical_value_mean", "total_prompts", "total_sessions",
            "hypothesis_test_rate", "solution_request_rate", "sus_score", "umux_score",
            "nps_score", "gaming_events", "trust_calibration", "ai_experience_level",
            "grade_pre", "grade_post",
        ]:
            if field_name in row and pd.notna(row[field_name]):
                try:
                    setattr(record, field_name, float(row[field_name]))
                except (ValueError, TypeError):
                    pass

        if "dominant_epistemic_mode" in row and pd.notna(row["dominant_epistemic_mode"]):
            record.dominant_epistemic_mode = str(row["dominant_epistemic_mode"])
        if "condition" in row:
            record.condition = str(row.get("condition", "A"))
        if "grade_pre" in row and "grade_post" in row:
            if pd.notna(row["grade_pre"]) and pd.notna(row["grade_post"]):
                record.grade_delta = float(row["grade_post"]) - float(row["grade_pre"])
        if "bloom_ceiling" in row:
            record.bloom_ceiling = int(row.get("bloom_ceiling", 1))

        records.append(record)

    return records


# ═══════════════════════════════════════════════════════════════════════
# GENERADOR DE DATOS SINTÉTICOS — para demo sin piloto real
# ═══════════════════════════════════════════════════════════════════════

def generate_demo_records(n: int = 40, seed: int = 42) -> tuple[list[StudentRecord], list[str]]:
    """
    Genera n estudiantes sintéticos con los 3-4 perfiles de engagement
    que el análisis debería descubrir.

    Returns (records, prompts_log)
    """
    import random
    random.seed(seed)

    records = []
    all_prompts = []

    profiles = [
        {   # Exploradores autónomos (25%)
            "label": "exploring",
            "bloom_range": (3.5, 5.5), "autonomy_range": (0.6, 0.9),
            "ped_value_range": (0.65, 0.95), "prompts_range": (10, 20),
            "hyp_range": (0.35, 0.60), "sol_range": (0.05, 0.20),
            "sus_range": (75, 95), "n": int(n * 0.25),
        },
        {   # Verificadores intensivos (30%)
            "label": "verifying",
            "bloom_range": (2.5, 3.8), "autonomy_range": (0.3, 0.55),
            "ped_value_range": (0.45, 0.70), "prompts_range": (18, 35),
            "hyp_range": (0.25, 0.45), "sol_range": (0.20, 0.40),
            "sus_range": (65, 85), "n": int(n * 0.30),
        },
        {   # Delegadores esporádicos (25%)
            "label": "delegating",
            "bloom_range": (1.2, 2.5), "autonomy_range": (0.1, 0.30),
            "ped_value_range": (0.10, 0.35), "prompts_range": (3, 12),
            "hyp_range": (0.02, 0.12), "sol_range": (0.55, 0.85),
            "sus_range": (40, 70), "n": int(n * 0.25),
        },
        {   # Consultores de volumen (20%)
            "label": "volume",
            "bloom_range": (1.8, 3.0), "autonomy_range": (0.2, 0.45),
            "ped_value_range": (0.25, 0.50), "prompts_range": (25, 50),
            "hyp_range": (0.08, 0.22), "sol_range": (0.40, 0.65),
            "sus_range": (55, 80), "n": n - int(n * 0.80),
        },
    ]

    def rand_in(r):
        return random.uniform(r[0], r[1])

    idx = 1
    for profile in profiles:
        for _ in range(profile["n"]):
            bloom = rand_in(profile["bloom_range"])
            autonomy = rand_in(profile["autonomy_range"])
            ped = rand_in(profile["ped_value_range"])
            prompts = int(rand_in(profile["prompts_range"]))

            record = StudentRecord(
                student_id=f"S{idx:03d}",
                course_id="PROG101",
                condition="A" if idx % 2 == 0 else "B",
                bloom_mean=round(bloom, 2),
                bloom_trajectory_slope=round(random.uniform(-0.05, 0.15), 3),
                bloom_ceiling=min(int(bloom + 1), 6),
                autonomy_final=round(autonomy, 3),
                autonomy_initial=round(max(0.05, autonomy - random.uniform(0, 0.25)), 3),
                autonomy_delta=round(random.uniform(-0.05, 0.2), 3),
                dominant_epistemic_mode={
                    "exploring": "exploration",
                    "verifying": "verification",
                    "delegating": "delegation",
                    "volume": "consumption",
                }[profile["label"]],
                pedagogical_value_mean=round(ped, 2),
                total_prompts=prompts,
                total_sessions=max(1, prompts // 4),
                hypothesis_test_rate=round(rand_in(profile["hyp_range"]), 2),
                solution_request_rate=round(rand_in(profile["sol_range"]), 2),
                gaming_events=int(rand_in((0, 3))),
                trust_calibration=round(random.uniform(-0.3, 0.4), 2),
                sus_score=round(rand_in(profile["sus_range"]), 1),
                umux_score=round(rand_in((profile["sus_range"][0] - 5, profile["sus_range"][1] + 5)), 1),
                nps_score=int(rand_in((-20, 80))),
                ai_experience_level=random.randint(1, 5),
                grade_pre=round(random.uniform(4, 8), 1),
                grade_post=round(random.uniform(4 + bloom * 0.4, 6 + bloom * 0.5), 1),
            )
            records.append(record)

            # Generar prompts sintéticos acordes con el perfil
            prompt_templates = {
                "exploring": [
                    "¿cuál es la diferencia entre recursión e iteración en términos de complejidad?",
                    "¿cuándo es mejor usar un array que una lista enlazada?",
                    "mi teoría es que el problema de eficiencia viene del O(n²) del bucle anidado",
                    "¿por qué el garbage collector no puede eliminar referencias circulares directamente?",
                ],
                "verifying": [
                    "¿está bien este bucle for? creo que es correcto",
                    "mi código usa un while, ¿es lo mismo que un for aquí?",
                    "es correcto que la recursión base sea cuando n==0 ¿verdad?",
                ],
                "delegating": [
                    "escribe un programa que ordene una lista",
                    "dado el siguiente enunciado: implementar función recursiva de fibonacci",
                    "dame el código del ejercicio",
                ],
                "volume": [
                    "¿por qué no me funciona esto?",
                    "qué es un array",
                    "cómo hago un for",
                    "explícame las funciones",
                ],
            }

            for _ in range(min(prompts, 5)):
                all_prompts.append(
                    random.choice(prompt_templates[profile["label"]])
                )

            idx += 1

    return records, all_prompts


if __name__ == "__main__":
    print("Generando datos de demo...")
    records, prompts = generate_demo_records(n=40)
    print(f"Registros generados: {len(records)}")
    print(f"Prompts generados: {len(prompts)}")

    report = run_full_analysis(
        records=records,
        prompts_log=prompts,
        design=StudyDesign(n_target=60),
        output_dir="./pilot_output",
    )

    print(f"\nInforme generado: {report.n_students} estudiantes, "
          f"{len(report.cluster_profiles)} clusters, "
          f"{len(report.correlations)} correlaciones calculadas.")
