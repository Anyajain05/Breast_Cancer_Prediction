from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from ml_utils import (
    MODEL_PATH,
    TARGET_COLUMN,
    get_top_correlations,
    load_bundle,
    load_data,
    prepare_prediction_frame,
    save_bundle,
    train_best_model,
)


st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 24px;
            background: linear-gradient(135deg, rgba(16, 24, 40, 0.96), rgba(88, 28, 135, 0.88));
            color: white;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.22);
            margin-bottom: 1rem;
        }
        .metric-card {
            background: white;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 10px 30px rgba(15, 23, 42, 0.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def load_or_train_bundle():
    dataset = load_data()
    if MODEL_PATH.exists():
        return load_bundle()
    bundle = train_best_model(dataset)
    save_bundle(bundle)
    return bundle


def label_for_prediction(prediction: int) -> str:
    return "Malignant" if prediction == 1 else "Benign"


def pretty_name(column_name: str) -> str:
    return column_name.replace("_", " ").title()


def render_stat_card(title: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div style="font-size: 0.84rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em;">{title}</div>
            <div style="font-size: 1.55rem; font-weight: 700; color: #0f172a; margin-top: 0.35rem;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    df = load_dataset()
    bundle = load_or_train_bundle()

    feature_columns = bundle.feature_columns
    numeric_df = df[feature_columns]

    st.markdown(
        """
        <div class="hero">
            <h1 style="margin: 0; font-size: 2.2rem;">Breast Cancer Prediction Dashboard</h1>
            <p style="margin: 0.55rem 0 0; font-size: 1rem; max-width: 58rem; line-height: 1.5;">
                Explore the dataset with matplotlib and seaborn, compare multiple machine learning algorithms,
                and make a prediction from a patient feature profile trained on the Wisconsin breast cancer dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_stat_card("Rows", f"{df.shape[0]:,}")
    with col2:
        render_stat_card("Features", f"{len(feature_columns)}")
    with col3:
        malignant_count = int((df[TARGET_COLUMN] == 1).sum())
        render_stat_card("Malignant", f"{malignant_count:,}")
    with col4:
        benign_count = int((df[TARGET_COLUMN] == 0).sum())
        render_stat_card("Benign", f"{benign_count:,}")

    tab_overview, tab_eda, tab_models, tab_predict = st.tabs([
        "Overview",
        "EDA",
        "Model Comparison",
        "Predict",
    ])

    with tab_overview:
        overview_col1, overview_col2 = st.columns([1.15, 0.85])
        with overview_col1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)
        with overview_col2:
            st.subheader("Target Distribution")
            fig, ax = plt.subplots(figsize=(5.5, 4))
            sns.countplot(data=df, x=TARGET_COLUMN, palette=["#1d4ed8", "#be123c"], ax=ax)
            ax.set_xticklabels(["Benign", "Malignant"])
            ax.set_xlabel("")
            ax.set_ylabel("Count")
            st.pyplot(fig, clear_figure=True)

    with tab_eda:
        st.subheader("Exploratory Data Analysis")
        corr_col, dist_col = st.columns([1.25, 0.75])

        with corr_col:
            st.write("Correlation with the target label")
            top_correlations = get_top_correlations(df, top_n=12)
            corr_fig, corr_ax = plt.subplots(figsize=(8, 5))
            sns.barplot(x=top_correlations.values, y=top_correlations.index, palette="viridis", ax=corr_ax)
            corr_ax.set_xlabel("Correlation with diagnosis")
            corr_ax.set_ylabel("")
            corr_ax.set_xlim(-1, 1)
            st.pyplot(corr_fig, clear_figure=True)

        with dist_col:
            selected_feature = st.selectbox("Inspect a feature", feature_columns, index=0)
            dist_fig, dist_ax = plt.subplots(figsize=(6, 5))
            sns.histplot(data=df, x=selected_feature, hue=TARGET_COLUMN, kde=True, palette=["#1d4ed8", "#be123c"], ax=dist_ax)
            dist_ax.set_xlabel(pretty_name(selected_feature))
            dist_ax.set_ylabel("Count")
            st.pyplot(dist_fig, clear_figure=True)

        heatmap_fig, heatmap_ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, ax=heatmap_ax)
        heatmap_ax.set_title("Feature Correlation Heatmap")
        st.pyplot(heatmap_fig, clear_figure=True)

    with tab_models:
        st.subheader("Machine Learning Model Comparison")
        metrics_df = pd.DataFrame(bundle.metrics).T.reset_index().rename(columns={"index": "model"})
        st.dataframe(metrics_df.sort_values("roc_auc", ascending=False), use_container_width=True)

        st.markdown("### Best Model")
        best_model = bundle.model_name
        best_metrics = bundle.holdout_metrics
        best_col1, best_col2, best_col3 = st.columns(3)
        with best_col1:
            render_stat_card("Selected model", best_model)
        with best_col2:
            render_stat_card("Holdout accuracy", f"{best_metrics['accuracy']:.3f}")
        with best_col3:
            render_stat_card("Holdout ROC AUC", f"{best_metrics['roc_auc']:.3f}")

        st.info(
            "The app trains and caches the best sklearn pipeline automatically if no saved artifact is present."
        )

    with tab_predict:
        st.subheader("Single Prediction")
        st.caption("Enter a patient profile or keep the default median values from the dataset.")

        feature_medians = df[feature_columns].median()
        feature_mins = df[feature_columns].min()
        feature_maxs = df[feature_columns].max()

        with st.form("prediction_form"):
            input_values: dict[str, float] = {}
            columns = st.columns(3)
            for index, feature in enumerate(feature_columns):
                column = columns[index % 3]
                value = float(feature_medians[feature])
                min_value = float(feature_mins[feature])
                max_value = float(feature_maxs[feature])
                step = max((max_value - min_value) / 1000.0, 0.0001)
                input_values[feature] = column.number_input(
                    pretty_name(feature),
                    min_value=min_value,
                    max_value=max_value,
                    value=value,
                    step=step,
                    format="%.6f",
                )

            submitted = st.form_submit_button("Predict Diagnosis")

        if submitted:
            input_frame = prepare_prediction_frame(input_values, feature_columns)
            prediction = int(bundle.pipeline.predict(input_frame)[0])
            probability = float(bundle.pipeline.predict_proba(input_frame)[0][1])

            result_col1, result_col2 = st.columns(2)
            with result_col1:
                st.success(f"Predicted diagnosis: {label_for_prediction(prediction)}")
            with result_col2:
                st.metric("Malignant probability", f"{probability:.2%}")

            st.write("Input values used for the prediction")
            st.dataframe(input_frame, use_container_width=True)


if __name__ == "__main__":
    main()