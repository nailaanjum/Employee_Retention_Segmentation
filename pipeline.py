"""Sklearn pipelines for clustering and PCA visualization."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import FEATURE_COLUMNS, N_CLUSTERS, RANDOM_STATE


def build_clustering_pipeline() -> Pipeline:
    """Scale features, then assign K-Means cluster labels."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clusterer",
                KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE),
            ),
        ]
    )


def build_pca_pipeline() -> Pipeline:
    """Scale features, then project to two principal components."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=2, random_state=RANDOM_STATE)),
        ]
    )


def fit_pipelines(features: pd.DataFrame) -> tuple[Pipeline, Pipeline, pd.DataFrame]:
    """Fit clustering and PCA pipelines and return cluster centers."""
    clustering = build_clustering_pipeline()
    pca = build_pca_pipeline()

    clustering.fit(features)
    pca.fit(features)

    clusterer: KMeans = clustering.named_steps["clusterer"]
    centers = pd.DataFrame(
        clusterer.cluster_centers_,
        columns=FEATURE_COLUMNS,
    )
    return clustering, pca, centers


def transform_for_pca(pca_pipeline: Pipeline, features: pd.DataFrame) -> pd.DataFrame:
    """Return PC1/PC2 coordinates for visualization."""
    components = pca_pipeline.transform(features)
    return pd.DataFrame(components, columns=["PC1", "PC2"], index=features.index)


@dataclass(frozen=True, slots=True)
class PredictionResult:
    cluster_id: int
    cluster_name: str


def predict_cluster(
    clustering_pipeline: Pipeline,
    cluster_mapping: dict[str, str],
    employee: pd.DataFrame,
) -> PredictionResult:
    """Run the full clustering pipeline on one employee row."""
    cluster_id = int(clustering_pipeline.predict(employee)[0])
    cluster_name = cluster_mapping.get(str(cluster_id), "Unknown")
    return PredictionResult(cluster_id=cluster_id, cluster_name=cluster_name)
