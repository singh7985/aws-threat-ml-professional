from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_features.csv"

TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_features.csv"

MODEL_DIRECTORY = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIRECTORY / "model.joblib"
FEATURE_MANIFEST_PATH = MODEL_DIRECTORY / "feature_manifest.json"
METRICS_PATH = MODEL_DIRECTORY / "metrics.json"


FEATURE_COLUMNS = [
    "hour_of_day",
    "is_weekend",
    "failed_call",
    "sensitive_operation",
    "unusual_hour",
    "external_ip",
    "outside_home_region",
    "calls_last_5m",
    "failed_calls_last_5m",
    "unique_services_last_1h",
    "iam_calls_last_1h",
    "api_risk_score",
]

TARGET_COLUMN = "label_binary"


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate one dataset."""

    if not path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {path}")

    dataset = pd.read_csv(path)

    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in dataset.columns]

    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {missing_columns}")

    return dataset


def prepare_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Prepare numerical model features."""

    features = dataset[FEATURE_COLUMNS].copy()

    features = features.fillna(0)
    features = features.astype(float)

    return features


def train_model(
    training_data: pd.DataFrame,
) -> IsolationForest:
    """Train Isolation Forest using normal events only."""

    normal_training_data = training_data.loc[training_data[TARGET_COLUMN] == 0]

    normal_features = prepare_features(normal_training_data)

    if normal_features.empty:
        raise ValueError("No normal training records were found.")

    print(f"Total training rows: {len(training_data)}")
    print(f"Normal training rows: {len(normal_features)}")

    model = IsolationForest(
        n_estimators=300,
        contamination=0.10,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(normal_features)

    return model


def evaluate_model(
    model: IsolationForest,
    testing_data: pd.DataFrame,
) -> dict[str, object]:
    """Evaluate the model using the testing dataset."""

    test_features = prepare_features(testing_data)

    actual_labels = testing_data[TARGET_COLUMN].astype(int).to_numpy()

    raw_predictions = model.predict(test_features)

    # Isolation Forest returns:
    #  1 for normal
    # -1 for anomaly
    predicted_labels = (raw_predictions == -1).astype(int)

    # Higher score means more suspicious.
    anomaly_scores = -model.decision_function(test_features)

    precision = precision_score(
        actual_labels,
        predicted_labels,
        zero_division=0,
    )

    recall = recall_score(
        actual_labels,
        predicted_labels,
        zero_division=0,
    )

    f1 = f1_score(
        actual_labels,
        predicted_labels,
        zero_division=0,
    )

    roc_auc = roc_auc_score(
        actual_labels,
        anomaly_scores,
    )

    pr_auc = average_precision_score(
        actual_labels,
        anomaly_scores,
    )

    matrix = confusion_matrix(
        actual_labels,
        predicted_labels,
    )

    metrics: dict[str, object] = {
        "testing_rows": len(testing_data),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": matrix.tolist(),
    }

    return metrics


def save_artifacts(
    model: IsolationForest,
    metrics: dict[str, object],
) -> None:
    """Save the model and supporting files."""

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_PATH)

    feature_manifest = {
        "model_name": "AWS Threat Isolation Forest",
        "model_version": "0.1.0",
        "algorithm": "IsolationForest",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "normal_label": 0,
        "suspicious_label": 1,
    }

    FEATURE_MANIFEST_PATH.write_text(
        json.dumps(
            feature_manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    print("Loading training dataset...")

    training_data = load_dataset(TRAIN_DATA_PATH)

    print("Loading testing dataset...")

    testing_data = load_dataset(TEST_DATA_PATH)

    print("Training Isolation Forest model...")

    model = train_model(training_data)

    print("Evaluating model...")

    metrics = evaluate_model(
        model,
        testing_data,
    )

    save_artifacts(
        model,
        metrics,
    )

    print("\nTraining completed successfully.")

    print(f"\nModel saved to:\n{MODEL_PATH}")

    print(f"\nFeature manifest saved to:\n{FEATURE_MANIFEST_PATH}")

    print(f"\nMetrics saved to:\n{METRICS_PATH}")

    print("\nModel performance:")

    for metric_name in [
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "pr_auc",
    ]:
        metric_value = metrics[metric_name]
        print(f"{metric_name}: {metric_value:.4f}")

    print(
        "confusion_matrix:",
        metrics["confusion_matrix"],
    )


if __name__ == "__main__":
    main()
