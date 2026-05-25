from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_PATH = Path("data.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "breast_cancer_model.joblib"
FEATURES_PATH = ARTIFACT_DIR / "feature_columns.joblib"

TARGET_COLUMN = "diagnosis"
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelBundle:
    model_name: str
    pipeline: Any
    feature_columns: list[str]
    metrics: dict[str, dict[str, float]]
    holdout_metrics: dict[str, float]


def load_data(csv_path: str | Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [str(column).strip().strip('"') for column in df.columns]
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", case=False, regex=True)]
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(str).str.strip().map({"B": 0, "M": 1})
    df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def build_model_pipelines() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    SVC(probability=True, random_state=RANDOM_STATE),
                ),
            ]
        ),
        "K-Nearest Neighbors": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", KNeighborsClassifier(n_neighbors=7)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=None,
                        min_samples_split=2,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
    }


def evaluate_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    rows: list[dict[str, float | str]] = []

    for model_name, pipeline in build_model_pipelines().items():
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        row: dict[str, float | str] = {"model": model_name}
        for metric_name in scoring:
            row[f"cv_{metric_name}"] = float(scores[f"test_{metric_name}"].mean())
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)
    return results


def train_best_model(df: pd.DataFrame) -> ModelBundle:
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    model_metrics_df = evaluate_models(X_train, y_train)
    best_model_name = str(model_metrics_df.iloc[0]["model"])
    best_pipeline = build_model_pipelines()[best_model_name]
    best_pipeline.fit(X_train, y_train)

    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    holdout_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    metrics = {
        row["model"]: {
            key.replace("cv_", ""): float(row[key])
            for key in row.index
            if key.startswith("cv_")
        }
        for _, row in model_metrics_df.iterrows()
    }

    return ModelBundle(
        model_name=best_model_name,
        pipeline=best_pipeline,
        feature_columns=list(X.columns),
        metrics=metrics,
        holdout_metrics=holdout_metrics,
    )


def save_bundle(bundle: ModelBundle, artifacts_dir: str | Path = ARTIFACT_DIR) -> None:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifacts_dir / MODEL_PATH.name)
    joblib.dump(bundle.feature_columns, artifacts_dir / FEATURES_PATH.name)


def load_bundle(model_path: str | Path = MODEL_PATH) -> ModelBundle:
    return joblib.load(model_path)


def get_top_correlations(df: pd.DataFrame, target_column: str = TARGET_COLUMN, top_n: int = 10) -> pd.Series:
    numeric_df = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
    correlations = numeric_df.corrwith(df[target_column]).sort_values(key=lambda s: s.abs(), ascending=False)
    return correlations.head(top_n)


def prepare_prediction_frame(input_data: dict[str, float], feature_columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame([input_data])
    frame = frame.reindex(columns=feature_columns)
    return frame