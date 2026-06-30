"""Project paths, feature schema, and business constants."""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Final

BASE_DIR: Final = Path(__file__).parent

FEATURE_COLUMNS: Final = (
    "Age",
    "Gender",
    "DistanceFromHome",
    "JobLevel",
    "MonthlyIncome",
    "PerformanceRating",
    "JobSatisfaction",
)

CLUSTERING_PIPELINE_FILE: Final = BASE_DIR / "clustering_pipeline.pkl"
PCA_PIPELINE_FILE: Final = BASE_DIR / "pca_pipeline.pkl"
METADATA_FILE: Final = BASE_DIR / "model_metadata.json"
MAPPING_FILE: Final = BASE_DIR / "cluster_mapping.json"
DASHBOARD_FILE: Final = BASE_DIR / "employee_dashboard_data.csv"
CENTERS_FILE: Final = BASE_DIR / "cluster_centers.csv"

N_CLUSTERS: Final = 6
RANDOM_STATE: Final = 42

RETENTION_RECOMMENDATIONS: Final = {
    "Long commuters": (
        "Highest attrition risk. Offer remote or hybrid work and strengthen remote culture."
    ),
    "Men who dislike their jobs": (
        "Schedule manager check-ins and address job satisfaction drivers early."
    ),
    "High performers": (
        "Create promotion paths and senior opportunities to retain top talent."
    ),
    "Men who like their jobs": (
        "Maintain current engagement; watch for satisfaction or commute changes."
    ),
    "Female employees": (
        "Attrition is below average; study what is working and extend those practices."
    ),
    "Senior employees": (
        "Lowest attrition segment. Use as a benchmark for tenure and stability programs."
    ),
}


class Gender(IntEnum):
    MALE = 0
    FEMALE = 1


GENDER_LABELS: Final = {Gender.MALE: "Male", Gender.FEMALE: "Female"}
