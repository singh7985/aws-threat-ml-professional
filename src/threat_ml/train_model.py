from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

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

# Share of the training set held out to choose the decision cutoff. The model is
# fitted without these rows so the threshold is not tuned on data the forest has
# already seen.
CALIBRATION_FRACTION = 0.20

RANDOM_STATE = 42


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


def split_for_calibration(
    training_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the training set into a fitting part and a calibration part.

    Stratified on the label so the calibration slice keeps enough suspicious
    rows to locate a meaningful cutoff.
    """

    labels = training_data[TARGET_COLUMN].astype(int)

    # Stratification needs at least two members of each class.
    stratify = labels if labels.nunique() > 1 and labels.value_counts().min() >= 2 else None

    fitting_data, calibration_data = train_test_split(
        training_data,
        test_size=CALIBRATION_FRACTION,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    return fitting_data, calibration_data


def choose_anomaly_threshold(
    model: IsolationForest,
    calibration_data: pd.DataFrame,
) -> float:
    """Choose the decision cutoff that maximises F1 on held-out labelled data.

    Returned in suspicion space (``-decision_function``): an event is anomalous
    when its suspicion score is greater than or equal to the threshold.

    This replaces the ``contamination`` parameter as the effective cutoff.
    ``contamination`` flags a fixed fraction of every batch regardless of what
    the scores look like, which manufactures false positives on clean traffic.
    """

    labels = calibration_data[TARGET_COLUMN].astype(int).to_numpy()

    # Without both classes there is no curve to optimise. Refuse rather than
    # returning the uncalibrated boundary, which would silently ship a model
    # scoring the way the old one did.
    if len(np.unique(labels)) < 2:
        raise ValueError(
            "The calibration slice contains only one class, so no decision "
            "threshold can be chosen. Provide labelled normal and suspicious rows."
        )

    scores = -model.decision_function(prepare_features(calibration_data))

    precision, recall, thresholds = precision_recall_curve(labels, scores)

    # precision_recall_curve returns one more point than thresholds.
    precision, recall = precision[:-1], recall[:-1]

    denominator = np.clip(precision + recall, 1e-12, None)
    f1_scores = 2.0 * precision * recall / denominator

    return float(thresholds[int(np.argmax(f1_scores))])


def evaluate_model(
    model: IsolationForest,
    testing_data: pd.DataFrame,
    anomaly_threshold: float,
) -> dict[str, object]:
    """Evaluate the model using the testing dataset."""

    test_features = prepare_features(testing_data)

    actual_labels = testing_data[TARGET_COLUMN].astype(int).to_numpy()

    # Higher score means more suspicious.
    anomaly_scores = -model.decision_function(test_features)

    # Scored against the calibrated cutoff rather than IsolationForest.predict(),
    # so the reported metrics describe the model as it is actually served.
    predicted_labels = (anomaly_scores >= anomaly_threshold).astype(int)

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
        "anomaly_threshold": float(anomaly_threshold),
        "flagged_rows": int(predicted_labels.sum()),
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
    anomaly_threshold: float,
) -> None:
    """Save the model and supporting files."""

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_PATH)

    feature_manifest = {
        "model_name": "AWS Threat Isolation Forest",
        "model_version": "0.2.0",
        "algorithm": "IsolationForest",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "normal_label": 0,
        "suspicious_label": 1,
        # Cutoff in suspicion space (-decision_function). Serving reads this so
        # the deployed model uses the same boundary it was evaluated against.
        "anomaly_threshold": float(anomaly_threshold),
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

    print("Splitting a calibration slice out of the training data...")

    fitting_data, calibration_data = split_for_calibration(training_data)

    print("Training Isolation Forest model...")

    model = train_model(fitting_data)

    print("Choosing the decision threshold on held-out labelled data...")

    anomaly_threshold = choose_anomaly_threshold(model, calibration_data)

    print(f"Calibration rows: {len(calibration_data)}")
    print(f"Chosen anomaly threshold: {anomaly_threshold:.6f}")

    print("Evaluating model...")

    metrics = evaluate_model(
        model,
        testing_data,
        anomaly_threshold,
    )

    save_artifacts(
        model,
        metrics,
        anomaly_threshold,
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
