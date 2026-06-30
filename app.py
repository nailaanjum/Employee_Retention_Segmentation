"""Employee retention segmentation dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from artifacts import AppArtifacts, load_artifacts
from config import FEATURE_COLUMNS, GENDER_LABELS, Gender, RETENTION_RECOMMENDATIONS
from pipeline import PredictionResult, predict_cluster, transform_for_pca

st.set_page_config(page_title="Employee Segmentation", page_icon="👥", layout="wide")


@st.cache_resource
def get_artifacts() -> AppArtifacts:
    return load_artifacts()


def attrition_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby("Cluster_Name", as_index=False)
        .agg(
            Employees=("Cluster_Name", "count"),
            Attrition_Rate=("Attrition_Flag", "mean"),
        )
        .sort_values("Attrition_Rate", ascending=False)
    )
    summary["Attrition_Rate"] = (summary["Attrition_Rate"] * 100).round(1)
    summary["Recommendation"] = summary["Cluster_Name"].map(RETENTION_RECOMMENDATIONS)
    return summary


def filter_dashboard(
    data: pd.DataFrame,
    segments: list[str],
    departments: list[str],
    attrition: str,
) -> pd.DataFrame:
    mask = data["Cluster_Name"].isin(segments) & data["Department"].isin(departments)
    filtered = data.loc[mask]
    if attrition != "All":
        filtered = filtered.loc[filtered["Attrition"] == attrition]
    return filtered


def employee_from_form(values: dict) -> pd.DataFrame:
    gender = Gender.FEMALE if values["gender_label"] == "Female" else Gender.MALE
    return pd.DataFrame(
        [
            {
                "Age": values["age"],
                "Gender": int(gender),
                "DistanceFromHome": values["distance"],
                "JobLevel": values["job_level"],
                "MonthlyIncome": values["income"],
                "PerformanceRating": values["performance"],
                "JobSatisfaction": values["satisfaction"],
            }
        ]
    )


def render_sidebar(artifacts: AppArtifacts) -> None:
    with st.sidebar:
        st.header("About")
        st.write("**Model:** K-Means (k=6)")
        st.write("**Pipeline:** `StandardScaler` → `KMeans`")
        st.write("**Employees:**", len(artifacts.dashboard))
        st.markdown("---")
        st.write("**Features used**")
        for feature in FEATURE_COLUMNS:
            st.write(f"- {feature}")
        st.info("Department is shown for context but is not used in clustering.")


def render_overview(data: pd.DataFrame) -> None:
    st.subheader("Segment attrition and recommendations")
    summary = attrition_summary(data)
    overall = data["Attrition_Flag"].mean() * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Total employees", len(data))
    c2.metric("Segments", data["Cluster"].nunique())
    c3.metric("Overall attrition", f"{overall:.1f}%")

    bar = px.bar(
        summary,
        x="Cluster_Name",
        y="Attrition_Rate",
        color="Attrition_Rate",
        color_continuous_scale="Reds",
        labels={"Cluster_Name": "Segment", "Attrition_Rate": "Attrition %"},
        title="Attrition rate by segment",
    )
    bar.add_hline(y=overall, line_dash="dash", annotation_text="Company average")
    st.plotly_chart(bar, use_container_width=True)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Segment size")
    st.plotly_chart(
        px.pie(data, names="Cluster_Name", title="Workforce distribution across segments"),
        use_container_width=True,
    )


def render_explorer(data: pd.DataFrame) -> None:
    st.subheader("Browse historical employees")
    c1, c2, c3 = st.columns(3)
    segments = sorted(data["Cluster_Name"].unique())
    departments = sorted(data["Department"].unique())

    filtered = filter_dashboard(
        data,
        c1.multiselect("Segment", segments, default=segments),
        c2.multiselect("Department", departments, default=departments),
        c3.selectbox("Attrition", ["All", "Yes", "No"]),
    )

    st.write(f"Showing **{len(filtered)}** employees")
    st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered data",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="filtered_employees.csv",
        mime="text/csv",
    )


def render_pca_map(artifacts: AppArtifacts) -> None:
    st.subheader("PCA visualization (Round 2)")
    st.write(
        "Automated pipeline: `StandardScaler` → `PCA`. "
        "Clusters come from the separate clustering pipeline."
    )

    features = artifacts.dashboard[list(FEATURE_COLUMNS)]
    components = transform_for_pca(artifacts.pca, features)
    map_df = artifacts.dashboard.join(components)

    color_by = st.radio(
        "Color points by",
        ["Cluster_Name", "Department", "Attrition"],
        horizontal=True,
    )
    st.plotly_chart(
        px.scatter(
            map_df,
            x="PC1",
            y="PC2",
            color=color_by,
            hover_data=list(FEATURE_COLUMNS) + ["Department", "Attrition"],
            title="Employee segments in PCA space",
            opacity=0.75,
        ),
        use_container_width=True,
    )

    explained = artifacts.pca.named_steps["pca"].explained_variance_ratio_
    st.write(
        f"Explained variance: PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%} "
        f"(total {explained.sum():.1%})"
    )


def render_prediction_form(artifacts: AppArtifacts, ranges: dict) -> None:
    st.subheader("Classify a new employee")

    with st.form("predict_form"):
        left, right = st.columns(2)
        with left:
            age = st.slider("Age", *ranges["Age"], 30)
            gender_label = st.selectbox("Gender", list(GENDER_LABELS.values()))
            distance = st.slider("Distance from home", *ranges["DistanceFromHome"], 5)
            job_level = st.slider("Job level", *ranges["JobLevel"], 2)
        with right:
            income = st.slider("Monthly income", *ranges["MonthlyIncome"], ranges["MonthlyIncome"][0])
            performance = st.slider(
                "Performance rating",
                *ranges["PerformanceRating"],
                ranges["PerformanceRating"][0],
            )
            satisfaction = st.slider(
                "Job satisfaction",
                *ranges["JobSatisfaction"],
                ranges["JobSatisfaction"][0],
            )
        submitted = st.form_submit_button("Predict segment", type="primary")

    if not submitted:
        return

    employee = employee_from_form(
        {
            "age": age,
            "gender_label": gender_label,
            "distance": distance,
            "job_level": job_level,
            "income": income,
            "performance": performance,
            "satisfaction": satisfaction,
        }
    )
    result = predict_cluster(
        artifacts.clustering,
        artifacts.cluster_mapping,
        employee,
    )
    render_prediction_result(artifacts, employee, result)


def render_prediction_result(
    artifacts: AppArtifacts,
    employee: pd.DataFrame,
    result: PredictionResult,
) -> None:
    history = artifacts.dashboard.loc[artifacts.dashboard["Cluster"] == result.cluster_id]
    attrition = history["Attrition_Flag"].mean() * 100

    st.success(f"Predicted segment: **{result.cluster_name}** (cluster {result.cluster_id})")
    c1, c2, c3 = st.columns(3)
    c1.metric("Segment", result.cluster_name)
    c2.metric("Historical peers", len(history))
    c3.metric("Segment attrition", f"{attrition:.1f}%")

    st.markdown("### Recommended action")
    st.write(RETENTION_RECOMMENDATIONS.get(result.cluster_name, "No recommendation available."))

    st.markdown("### Input profile")
    display = employee.assign(
        Gender=employee["Gender"].map({int(k): v for k, v in GENDER_LABELS.items()})
    )
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("### Segment center (standardized features)")
    center = artifacts.cluster_centers.iloc[result.cluster_id]
    st.plotly_chart(
        px.bar(
            x=center.values,
            y=center.index,
            orientation="h",
            labels={"x": "Standardized value", "y": "Feature"},
            title=f"Cluster center: {result.cluster_name}",
        ),
        use_container_width=True,
    )


def main() -> None:
    artifacts = get_artifacts()
    ranges = artifacts.metadata["numeric_ranges"]

    st.title("Employee Retention Segmentation")
    st.caption(
        "Automated sklearn pipelines for segmentation, PCA visualization, and retention insights."
    )
    render_sidebar(artifacts)

    overview, explorer, pca_map, predict = st.tabs(
        ["Segment Overview", "Employee Explorer", "PCA Map", "Predict Segment"]
    )
    with overview:
        render_overview(artifacts.dashboard)
    with explorer:
        render_explorer(artifacts.dashboard)
    with pca_map:
        render_pca_map(artifacts)
    with predict:
        render_prediction_form(artifacts, ranges)


if __name__ == "__main__":
    main()
