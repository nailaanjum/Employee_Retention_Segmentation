"""Train and persist sklearn pipelines for the employee segmentation app."""

from __future__ import annotations

from artifacts import train_and_save_artifacts
from config import CLUSTERING_PIPELINE_FILE, PCA_PIPELINE_FILE


def main() -> None:
    artifacts = train_and_save_artifacts()
    agreement = (
        artifacts.clustering.predict(artifacts.dashboard[list(artifacts.metadata["feature_columns"])])
        == artifacts.dashboard["Cluster"].to_numpy()
    ).mean()

    print("Saved automated pipelines:")
    print(f"  - {CLUSTERING_PIPELINE_FILE.name}")
    print(f"  - {PCA_PIPELINE_FILE.name}")
    print(f"Cluster label agreement with dashboard data: {agreement:.1%}")


if __name__ == "__main__":
    main()
