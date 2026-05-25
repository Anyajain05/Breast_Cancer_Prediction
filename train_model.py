from __future__ import annotations

import pandas as pd

from ml_utils import ARTIFACT_DIR, MODEL_PATH, load_data, save_bundle, train_best_model


def main() -> None:
    df = load_data()
    bundle = train_best_model(df)
    save_bundle(bundle)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(bundle.metrics).T.reset_index().rename(columns={"index": "model"})
    metrics_df.to_csv(ARTIFACT_DIR / "model_metrics.csv", index=False)

    summary_path = ARTIFACT_DIR / "model_summary.txt"
    summary_path.write_text(
        "Best model: {model}\n\nHoldout metrics:\n{metrics}\n".format(
            model=bundle.model_name,
            metrics=pd.Series(bundle.holdout_metrics).to_string(),
        ),
        encoding="utf-8",
    )

    print(f"Saved model bundle to {MODEL_PATH}")
    print(f"Best model: {bundle.model_name}")
    print(pd.DataFrame(bundle.metrics).T)
    print("Holdout metrics:")
    for metric_name, metric_value in bundle.holdout_metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()