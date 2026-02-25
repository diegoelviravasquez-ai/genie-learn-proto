"""
analytics/bridge.py — Perfiles, clusters, riesgo de abandono y similitud
=========================================================================
- get_student_profiles: agregación por estudiante y kappa_p
- get_clusters: KMeans + PCA, etiquetas Explorador/Verificador/Delegador/Moderado
- get_dropout_risk: RandomForest, dropout_risk 0-1
- get_similarity_matrix: cosine_similarity sobre features normalizadas
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity


def get_student_profiles(course_id: Optional[str] = None) -> pd.DataFrame:
    """
    Lee interacciones, agrupa por student_id y calcula métricas + kappa_p.
    Devuelve un DataFrame con una fila por estudiante.
    """
    from data.database import get_interactions_df

    df = get_interactions_df(course_id=course_id)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "student_id", "total_prompts", "bloom_mean", "bloom_max",
                "autonomy_score", "pct_socratic", "kappa_p",
            ]
        )

    def autonomy_score(series: pd.Series) -> float:
        return float(1.0 - series.mean())

    def pct_socratic(series: pd.Series) -> float:
        s = series.astype(str).str.lower()
        return float((s.str.contains("socratic", na=False)).mean())

    agg = df.groupby("student_id").agg(
        total_prompts=("id", "count"),
        bloom_mean=("bloom_level", "mean"),
        bloom_max=("bloom_level", "max"),
        autonomy_score=("copy_paste_score", autonomy_score),
        pct_socratic=("scaffolding_mode", pct_socratic),
    ).reset_index()

    # Normalizar bloom_mean a escala 0-1 para kappa_p (asumiendo 1-6)
    bloom_norm = (agg["bloom_mean"] - 1) / 5.0
    agg["kappa_p"] = (
        bloom_norm * 0.4
        + agg["autonomy_score"] * 0.4
        + agg["pct_socratic"] * 0.2
    )
    return agg


def get_clusters(df_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    KMeans (4 clusters), PCA a 2D, añade cluster, cluster_label, pca_x, pca_y.
    Etiquetas: Explorador, Verificador, Delegador, Moderado.
    """
    if df_profiles.empty or len(df_profiles) < 2:
        for col in ["cluster", "cluster_label", "pca_x", "pca_y"]:
            df_profiles[col] = np.nan if col in ("pca_x", "pca_y") else -1 if col == "cluster" else ""
        return df_profiles

    features = ["bloom_mean", "autonomy_score", "total_prompts"]
    X = df_profiles[features].fillna(0).astype(float)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_profiles = df_profiles.copy()
    df_profiles["cluster"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df_profiles["pca_x"] = coords[:, 0]
    df_profiles["pca_y"] = coords[:, 1]

    labels = {0: "Explorador", 1: "Verificador", 2: "Delegador", 3: "Moderado"}
    df_profiles["cluster_label"] = df_profiles["cluster"].map(labels)
    return df_profiles


def get_dropout_risk(df_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columna dropout_risk (0-1). Entrena RandomForest;
    si hay menos de 10 estudiantes, usa datos sintéticos para entrenar.
    """
    df_profiles = df_profiles.copy()
    features = ["bloom_mean", "autonomy_score", "total_prompts"]

    if df_profiles.empty:
        df_profiles["dropout_risk"] = np.nan
        return df_profiles

    X = df_profiles[features].fillna(0).astype(float)

    if len(df_profiles) < 10:
        np.random.seed(42)
        n_synth = 50
        X_train = np.column_stack([
            np.random.uniform(1, 6, n_synth),
            np.random.uniform(0, 1, n_synth),
            np.random.uniform(5, 100, n_synth),
        ])
        # Baja autonomía / bajo bloom / pocos prompts → más riesgo (clase 1)
        y_cont = (
            (1 - X_train[:, 1]) * 0.4
            + (1 - (X_train[:, 0] - 1) / 5) * 0.4
            + (1 - np.clip(X_train[:, 2] / 100, 0, 1)) * 0.2
        )
        y_train = (np.clip(y_cont + np.random.normal(0, 0.1, n_synth), 0, 1) > 0.5).astype(int)
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        clf.fit(X_train, y_train)
    else:
        # Con suficientes datos: target sintético (proxy de riesgo) binarizado
        y_cont = (
            (1 - df_profiles["autonomy_score"].values) * 0.4
            + (1 - (df_profiles["bloom_mean"].values - 1) / 5) * 0.4
            + (1 - np.clip(df_profiles["total_prompts"].values / 100, 0, 1)) * 0.2
        )
        y = (np.clip(y_cont, 0, 1) > 0.5).astype(int)
        clf = RandomForestClassifier(n_estimators=20, random_state=42)
        clf.fit(X, y)

    risk = clf.predict_proba(X)[:, 1]
    df_profiles["dropout_risk"] = np.clip(risk, 0.0, 1.0)
    return df_profiles


def get_similarity_matrix(df_profiles: pd.DataFrame) -> Tuple[np.ndarray, list]:
    """
    Matriz de similitud coseno sobre features normalizadas.
    Devuelve (matriz numpy, lista de student_ids).
    """
    if df_profiles.empty:
        return np.array([[]]), []

    features = ["bloom_mean", "autonomy_score", "total_prompts"]
    if not all(c in df_profiles.columns for c in features):
        ids = list(df_profiles.get("student_id", []))
        return np.eye(len(ids)), ids

    X = df_profiles[features].fillna(0).astype(float)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    matrix = cosine_similarity(X_norm)
    ids = list(df_profiles["student_id"].values)
    return matrix, ids
