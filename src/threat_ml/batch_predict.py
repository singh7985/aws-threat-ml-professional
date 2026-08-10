from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from threat_ml.predict_event import (
    calculate_rule_score,
    convert_decision_to_anomaly_score,
    determine_risk_level,
    is_anomalous,
    read_anomaly_threshold,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "test_features.csv"

MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"

FEATURE_MANIFEST_PATH = PROJECT_ROOT / "models" / "feature_manifest.json"

DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs" / "batch_prediction"


def load_model() -> IsolationForest:
    """Load the trained Isolation Forest model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH}\nRun: python -m threat_ml.train_model"
        )

    model = joblib.load(MODEL_PATH)

    if not isinstance(model, IsolationForest):
        raise TypeError("The stored model is not an IsolationForest.")

    return model


def load_manifest() -> dict[str, Any]:
    """Load the feature manifest written at training time."""

    if not FEATURE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "feature_manifest.json was not found. Run the model training program first."
        )

    manifest: dict[str, Any] = json.loads(
        FEATURE_MANIFEST_PATH.read_text(
            encoding="utf-8",
        )
    )

    if not isinstance(manifest.get("feature_columns"), list):
        raise ValueError("The feature manifest does not contain a valid feature_columns list.")

    return manifest


def load_feature_columns() -> list[str]:
    """Load the feature order used during training."""

    return [str(column) for column in load_manifest()["feature_columns"]]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a CSV dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Input dataset was not found: {path}")

    dataset = pd.read_csv(path)

    if dataset.empty:
        raise ValueError(f"The input dataset is empty: {path}")

    return dataset


def validate_dataset(
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    """Confirm that every required feature exists."""

    missing_columns = [column for column in feature_columns if column not in dataset.columns]

    if missing_columns:
        raise ValueError(f"The dataset is missing required features: {missing_columns}")


def prepare_feature_matrix(
    dataset: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Create the numerical matrix used by the model."""

    features = dataset[feature_columns].copy()

    features = features.fillna(0)

    try:
        features = features.astype(float)
    except ValueError as error:
        raise ValueError("One or more ML feature columns contain non-numeric values.") from error

    return features


def score_dataset(
    model: IsolationForest,
    dataset: pd.DataFrame,
    feature_columns: list[str],
    anomaly_threshold: float,
) -> pd.DataFrame:
    """Run ML and rule-based scoring for every row.

    Scored against the calibrated cutoff so batch results match what the CLI and
    the Lambda produce for the same rows.
    """

    feature_matrix = prepare_feature_matrix(
        dataset=dataset,
        feature_columns=feature_columns,
    )

    decision_values = model.decision_function(feature_matrix)

    prediction_rows: list[dict[str, Any]] = []

    for position, (_, row) in enumerate(dataset.iterrows()):
        payload = {feature: float(row[feature]) for feature in feature_columns}

        decision_value = float(decision_values[position])

        model_flagged_anomaly = is_anomalous(decision_value, anomaly_threshold)

        anomaly_score = convert_decision_to_anomaly_score(decision_value, anomaly_threshold)

        rule_score, reasons = calculate_rule_score(payload)

        final_risk_score = min(
            1.0,
            0.70 * anomaly_score + 0.30 * rule_score,
        )

        risk_level = determine_risk_level(final_risk_score)

        output_row: dict[str, Any] = {}

        metadata_columns = [
            "event_id",
            "timestamp",
            "principal_id",
            "event_name",
            "service",
            "region",
            "source_ip",
            "label",
            "label_binary",
            "attack_type",
        ]

        for column in metadata_columns:
            if column in dataset.columns:
                value = row[column]

                if pd.isna(value):
                    output_row[column] = None
                else:
                    output_row[column] = value

        output_row.update(
            {
                "model_prediction": ("suspicious" if model_flagged_anomaly else "normal"),
                "model_flagged_anomaly": model_flagged_anomaly,
                "model_decision_value": round(
                    decision_value,
                    6,
                ),
                "anomaly_score": round(
                    anomaly_score,
                    4,
                ),
                "rule_score": round(
                    rule_score,
                    4,
                ),
                "final_risk_score": round(
                    final_risk_score,
                    4,
                ),
                "risk_level": risk_level,
                "reasons": reasons,
            }
        )

        prediction_rows.append(output_row)

    return pd.DataFrame(prediction_rows)


def build_summary(
    predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Create a summary of a prediction run."""

    risk_counts = predictions["risk_level"].value_counts().to_dict()

    summary: dict[str, Any] = {
        "total_events": len(predictions),
        "low_risk_events": int(risk_counts.get("LOW", 0)),
        "medium_risk_events": int(risk_counts.get("MEDIUM", 0)),
        "high_risk_events": int(risk_counts.get("HIGH", 0)),
        "ml_flagged_anomalies": int(predictions["model_flagged_anomaly"].sum()),
        "average_anomaly_score": round(
            float(predictions["anomaly_score"].mean()),
            4,
        ),
        "average_final_risk_score": round(
            float(predictions["final_risk_score"].mean()),
            4,
        ),
    }

    if "label_binary" in predictions.columns:
        valid_rows = predictions[predictions["label_binary"].notna()].copy()

        if not valid_rows.empty:
            actual_labels = valid_rows["label_binary"].astype(int)

            predicted_labels = valid_rows["risk_level"].isin(["MEDIUM", "HIGH"]).astype(int)

            summary["evaluation"] = {
                "precision": round(
                    float(
                        precision_score(
                            actual_labels,
                            predicted_labels,
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "recall": round(
                    float(
                        recall_score(
                            actual_labels,
                            predicted_labels,
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "f1_score": round(
                    float(
                        f1_score(
                            actual_labels,
                            predicted_labels,
                            zero_division=0,
                        )
                    ),
                    4,
                ),
                "confusion_matrix": (
                    confusion_matrix(
                        actual_labels,
                        predicted_labels,
                        labels=[0, 1],
                    ).tolist()
                ),
            }

    return summary


def save_results(
    predictions: pd.DataFrame,
    summary: dict[str, Any],
    output_directory: Path,
) -> None:
    """Save predictions, incidents and summary files."""

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_path = output_directory / "predictions.csv"

    incidents_path = output_directory / "incidents.jsonl"

    high_risk_path = output_directory / "high_risk_incidents.csv"

    summary_path = output_directory / "summary.json"

    csv_predictions = predictions.copy()

    csv_predictions["reasons"] = csv_predictions["reasons"].apply(json.dumps)

    csv_predictions.to_csv(
        predictions_path,
        index=False,
    )

    incidents = predictions[predictions["risk_level"].isin(["MEDIUM", "HIGH"])].copy()

    with incidents_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in incidents.to_dict(orient="records"):
            file.write(
                json.dumps(
                    record,
                    default=str,
                )
                + "\n"
            )

    high_risk_incidents = predictions[predictions["risk_level"] == "HIGH"].copy()

    high_risk_incidents["reasons"] = high_risk_incidents["reasons"].apply(json.dumps)

    high_risk_incidents.to_csv(
        high_risk_path,
        index=False,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nFiles created:")

    print(f"- {predictions_path}")
    print(f"- {incidents_path}")
    print(f"- {high_risk_path}")
    print(f"- {summary_path}")


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=("Run batch threat predictions for an AWS activity dataset.")
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=("Input CSV file. Defaults to data/processed/test_features.csv"),
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for prediction results.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the batch prediction workflow."""

    arguments = parse_arguments()

    print("Loading model...")
    model = load_model()

    print("Loading feature manifest...")
    manifest = load_manifest()
    feature_columns = [str(column) for column in manifest["feature_columns"]]
    anomaly_threshold = read_anomaly_threshold(manifest)
    print(f"Anomaly threshold: {anomaly_threshold:.6f}")

    print(f"Loading dataset: {arguments.input}")
    dataset = load_dataset(arguments.input)

    validate_dataset(
        dataset=dataset,
        feature_columns=feature_columns,
    )

    print(f"Scoring {len(dataset)} events...")

    predictions = score_dataset(
        model=model,
        dataset=dataset,
        feature_columns=feature_columns,
        anomaly_threshold=anomaly_threshold,
    )

    summary = build_summary(predictions)

    save_results(
        predictions=predictions,
        summary=summary,
        output_directory=(arguments.output_directory),
    )

    print("\nBatch prediction completed.")

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
