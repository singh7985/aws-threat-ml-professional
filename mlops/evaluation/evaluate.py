import json
import logging
import tarfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

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

def main() -> None:
    # SageMaker Processing job mounts inputs here
    model_dir = Path("/opt/ml/processing/model")
    test_dir = Path("/opt/ml/processing/test")
    output_dir = Path("/opt/ml/processing/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Test Data
    target_file = test_dir / "test_features.csv"
    if not target_file.exists():
        csv_files = list(test_dir.glob("*.csv"))
        if csv_files:
            target_file = csv_files[0]
        else:
            raise FileNotFoundError(f"No CSV test file found inside {test_dir}")

    logger.info(f"Loading test data from: {target_file}")
    testing_data = pd.read_csv(target_file)
    
    # Extract features
    test_features = testing_data[FEATURE_COLUMNS].copy()
    test_features = test_features.fillna(0).astype(float)
    actual_labels = testing_data[TARGET_COLUMN].astype(int).to_numpy()

    # 2. Extract and Load Model
    # SageMaker TrainingStep often outputs model.tar.gz that ProcessingStep unpacks, 
    # but strictly depends on if it was extracted. We look for joblib or tar.gz.
    model_path = model_dir / "model.joblib"
    if not model_path.exists():
        tar_path = model_dir / "model.tar.gz"
        if tar_path.exists():
            logger.info("Extracting model.tar.gz...")
            with tarfile.open(tar_path) as tar:
                tar.extractall(path=model_dir, filter="data")

    if not model_path.exists():
        raise FileNotFoundError(f"model.joblib not found in {model_dir}")

    logger.info("Loading model...")
    model = joblib.load(model_path)

    # The training step writes the calibrated cutoff into the manifest packed
    # alongside the model. Evaluating at that same cutoff is what makes these
    # metrics describe the model as it will actually be served -- scoring with
    # IsolationForest.predict() would measure the contamination boundary instead.
    # No fallback: evaluating at an uncalibrated cutoff would report metrics for
    # a model nobody will actually serve, and the quality gate would then pass or
    # fail on the wrong numbers. Better to fail the pipeline step outright.
    manifest_path = model_dir / "feature_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"feature_manifest.json not found in {model_dir}. The training step must "
            "write it alongside the model so evaluation uses the calibrated cutoff."
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "anomaly_threshold" not in manifest:
        raise ValueError(
            "feature_manifest.json is missing 'anomaly_threshold'. The model artifact "
            "predates threshold calibration and must be retrained."
        )
    anomaly_threshold = float(manifest["anomaly_threshold"])

    logger.info(f"Scoring at anomaly threshold: {anomaly_threshold:.6f}")

    # 3. Score Data
    logger.info("Generating predictions...")

    # Anomaly scores (higher = more suspicious)
    anomaly_scores = -model.decision_function(test_features)

    predicted_labels = (anomaly_scores >= anomaly_threshold).astype(int)

    # 4. Calculate Metrics
    precision = precision_score(actual_labels, predicted_labels, zero_division=0)
    recall = recall_score(actual_labels, predicted_labels, zero_division=0)
    f1 = f1_score(actual_labels, predicted_labels, zero_division=0)
    roc_auc = roc_auc_score(actual_labels, anomaly_scores)
    pr_auc = average_precision_score(actual_labels, anomaly_scores)
    
    logger.info(f"Metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # 5. Format and Save Evaluation Report
    # Note: We provide exactly the structure expected by the ConditionStep in create_pipeline.py
    # ("metrics.precision.value") alongside the requested "classification_metrics" summary.
    report_dict = {
        "anomaly_threshold": float(anomaly_threshold),
        "metrics": {
            "precision": {"value": float(precision)},
            "recall": {"value": float(recall)},
            "f1": {"value": float(f1)},
            "roc_auc": {"value": float(roc_auc)},
            "pr_auc": {"value": float(pr_auc)}
        },
        "classification_metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc)
        }
    }

    eval_path = output_dir / "evaluation.json"
    eval_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    logger.info(f"Saved evaluation report to {eval_path}")

if __name__ == "__main__":
    main()
