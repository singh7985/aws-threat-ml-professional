import argparse
import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

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

# Held out from fitting so the decision cutoff is chosen on rows the forest has
# not already seen. Mirrors src/threat_ml/train_model.py.
CALIBRATION_FRACTION = 0.20
RANDOM_STATE = 42


def choose_anomaly_threshold(model, calibration_data):
    """Pick the cutoff that maximises F1 on held-out labelled data.

    Returned in suspicion space (``-decision_function``). This replaces
    ``contamination`` as the effective boundary: contamination flags a fixed
    fraction of every batch regardless of the score distribution, which
    manufactures false positives on clean traffic.
    """
    labels = calibration_data[TARGET_COLUMN].astype(int).to_numpy()

    if len(np.unique(labels)) < 2:
        raise ValueError(
            "The calibration slice contains only one class, so no decision "
            "threshold can be chosen. Provide labelled normal and suspicious rows."
        )

    features = calibration_data[FEATURE_COLUMNS].copy().fillna(0).astype(float)
    scores = -model.decision_function(features)

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    precision, recall = precision[:-1], recall[:-1]

    denominator = np.clip(precision + recall, 1e-12, None)
    f1_scores = 2.0 * precision * recall / denominator

    return float(thresholds[int(np.argmax(f1_scores))])


def main():
    parser = argparse.ArgumentParser()
    
    # SageMaker passes specific environment variables
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model/"))
    parser.add_argument("--train-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train/"))
    
    args = parser.parse_args()

    # Load Training Data
    train_path = Path(args.train_dir)
    
    # Try looking specifically for train_features.csv or just a csv generically
    target_file = train_path / "train_features.csv"
    if not target_file.exists():
        csv_files = list(train_path.glob("*.csv"))
        if csv_files:
            target_file = csv_files[0]
        else:
            raise FileNotFoundError(f"No CSV mapping found inside {train_path}")

    logger.info(f"Loading training data from: {target_file}")
    training_data = pd.read_csv(target_file)
    
    # Hold out a labelled slice for threshold calibration before fitting.
    labels = training_data[TARGET_COLUMN].astype(int)
    stratify = labels if labels.nunique() > 1 and labels.value_counts().min() >= 2 else None
    fitting_data, calibration_data = train_test_split(
        training_data,
        test_size=CALIBRATION_FRACTION,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    # Pre-select normal occurrences (0) due to IsolationForest strategy
    normal_training_data = fitting_data.loc[fitting_data[TARGET_COLUMN] == 0]

    # Filter features exactly as originally developed
    features = normal_training_data[FEATURE_COLUMNS].copy()
    features = features.fillna(0).astype(float)

    logger.info(f"Total rows evaluated: {len(training_data)}. Normal rows for fit: {len(features)}")
    logger.info(f"Calibration rows held out: {len(calibration_data)}")

    if features.empty:
        raise ValueError("No normal training records were found!")

    # Establish and train IsolationForest algorithm
    model = IsolationForest(
        n_estimators=300,
        contamination=0.10,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    
    logger.info("Fitting model...")
    model.fit(features)

    anomaly_threshold = choose_anomaly_threshold(model, calibration_data)
    logger.info(f"Chosen anomaly threshold: {anomaly_threshold:.6f}")


    # Persist artifacts into the generic Model Directory explicitly routed to S3
    model_output_path = Path(args.model_dir)
    model_output_path.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_output_path / "model.joblib")
    logger.info("Saved model.joblib successfully")
    
    feature_manifest = {
        "model_name": "AWS Threat Isolation Forest",
        "model_version": "0.2.0",
        "algorithm": "IsolationForest",
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "normal_label": 0,
        "suspicious_label": 1,
        # Read by the evaluation step and by serving, so both use the same
        # boundary the model was calibrated against.
        "anomaly_threshold": anomaly_threshold,
    }
    
    (model_output_path / "feature_manifest.json").write_text(
        json.dumps(feature_manifest, indent=2), encoding="utf-8"
    )
    logger.info("Saved feature_manifest.json successfully")
    
    training_metadata = {
        "training_timestamp": datetime.now(UTC).isoformat(),
        "total_rows_evaluated": len(training_data),
        "normal_training_rows": len(features),
        "calibration_rows": len(calibration_data),
        "anomaly_threshold": anomaly_threshold,
        "environment": "SageMaker",
    }
    
    (model_output_path / "training_metadata.json").write_text(
        json.dumps(training_metadata, indent=2), encoding="utf-8"
    )
    logger.info("Saved training_metadata.json successfully")
    
    logger.info("Training pipeline completely finished logic!")

if __name__ == "__main__":
    main()
