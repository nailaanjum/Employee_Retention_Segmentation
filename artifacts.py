"""Load, train, and persist automated sklearn pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from config import (
    CENTERS_FILE,
    CLUSTERING_PIPELINE_FILE,
    DASHBOARD_FILE,
    FEATURE_COLUMNS,
    MAPPING_FILE,
    METADATA_FILE,
    PCA_PIPELINE_FILE,
)
from pipeline import fit_pipelines


@dataclass(frozen=True, slots=True)
class AppArtifacts:
    clustering: Pipeline
    pca: Pipeline
    metadata: dict
    cluster_mapping: dict[str, str]
    dashboard: pd.DataFrame
    cluster_centers: pd.DataFrame


def _pipeline_files_exist() -> bool:
    return CLUSTERING_PIPELINE_FILE.exists() and PCA_PIPELINE_FILE.exists()


def _load_json(path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


def _prepare_dashboard(raw: pd.DataFrame) -> pd.DataFrame:
    dashboard = raw.copy()
    dashboard["Attrition_Flag"] = dashboard["Attrition"].map({"Yes": 1, "No": 0})
    return dashboard


def train_and_save_artifacts() -> AppArtifacts:
    """Fit pipelines on dashboard data and write all production files."""
    dashboard = pd.read_csv(DASHBOARD_FILE)
    features = dashboard[list(FEATURE_COLUMNS)]

    clustering, pca, centers = fit_pipelines(features)

    joblib.dump(clustering, CLUSTERING_PIPELINE_FILE)
    joblib.dump(pca, PCA_PIPELINE_FILE)
    centers.to_csv(CENTERS_FILE, index=False)

    metadata = _load_json(METADATA_FILE) if METADATA_FILE.exists() else {}
    metadata.update(
        {
            "clustering_pipeline_file": CLUSTERING_PIPELINE_FILE.name,
            "pca_pipeline_file": PCA_PIPELINE_FILE.name,
            "feature_columns": list(FEATURE_COLUMNS),
        }
    )
    _save_json(METADATA_FILE, metadata)

    return AppArtifacts(
        clustering=clustering,
        pca=pca,
        metadata=metadata,
        cluster_mapping=_load_json(MAPPING_FILE),
        dashboard=_prepare_dashboard(dashboard),
        cluster_centers=centers,
    )


def ensure_artifacts() -> None:
    """Create pipeline files when missing."""
    if _pipeline_files_exist():
        return
    train_and_save_artifacts()


def load_artifacts() -> AppArtifacts:
    """Load pipelines and data, training first if needed."""
    ensure_artifacts()

    return AppArtifacts(
        clustering=joblib.load(CLUSTERING_PIPELINE_FILE),
        pca=joblib.load(PCA_PIPELINE_FILE),
        metadata=_load_json(METADATA_FILE),
        cluster_mapping=_load_json(MAPPING_FILE),
        dashboard=_prepare_dashboard(pd.read_csv(DASHBOARD_FILE)),
        cluster_centers=pd.read_csv(CENTERS_FILE),
    )
