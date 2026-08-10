from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
FEATURE_MANIFEST_PATH = PROJECT_ROOT / "models" / "feature_manifest.json"

# Key the training step writes into feature_manifest.json. There is no default:
# see read_anomaly_threshold for why a fallback is deliberately absent.
ANOMALY_THRESHOLD_KEY = "anomaly_threshold"


def load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON object from a file."""

    if not path.exists():
        raise FileNotFoundError(f"Input file was not found: {path}")

    try:
        content = path.read_text(encoding="utf-8")
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"Input file contains invalid JSON: {error}") from error

    if not isinstance(payload, dict):
        raise ValueError("The input JSON must contain one JSON object.")

    return payload


def load_feature_manifest() -> list[str]:
    """Load the exact feature order used during training."""

    if not FEATURE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Feature manifest was not found. Run the model-training program first."
        )

    manifest = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))

    feature_columns = manifest.get("feature_columns")

    if not isinstance(feature_columns, list):
        raise ValueError("feature_manifest.json does not contain a valid feature_columns list.")

    return [str(column) for column in feature_columns]


def read_anomaly_threshold(manifest: dict[str, Any]) -> float:
    """Read the calibrated cutoff from a feature manifest.

    Raises if the value is absent or unusable. There is deliberately **no**
    fallback to an uncalibrated default: a scorer that quietly reverted to the
    contamination boundary would raise roughly five times as many false alerts
    with nothing in the logs to explain why. Failing here makes the problem
    immediate and visible -- in Lambda the message lands in the DLQ and the
    alarm fires, instead of the system degrading in silence.
    """

    if ANOMALY_THRESHOLD_KEY not in manifest:
        raise ValueError(
            f"feature_manifest.json is missing '{ANOMALY_THRESHOLD_KEY}'. "
            "The model artifact predates threshold calibration and must be "
            "retrained with: python -m threat_ml.train_model"
        )

    value = manifest[ANOMALY_THRESHOLD_KEY]

    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"'{ANOMALY_THRESHOLD_KEY}' must be numeric. Received: {value!r}"
        ) from error

    if not math.isfinite(threshold):
        raise ValueError(f"'{ANOMALY_THRESHOLD_KEY}' must be finite. Received: {value!r}")

    return threshold


def load_anomaly_threshold() -> float:
    """Load the calibrated cutoff from the on-disk feature manifest."""

    if not FEATURE_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            "Feature manifest was not found. Run the model-training program first."
        )

    manifest = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))

    return read_anomaly_threshold(manifest)


def load_model() -> IsolationForest:
    """Load the trained Isolation Forest model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model was not found. Run: python -m threat_ml.train_model")

    model = joblib.load(MODEL_PATH)

    if not isinstance(model, IsolationForest):
        raise TypeError("The saved model is not an IsolationForest.")

    return model


def prepare_input(
    payload: dict[str, Any],
    feature_columns: list[str],
) -> pd.DataFrame:
    """Validate and prepare one feature row."""

    missing_columns = [column for column in feature_columns if column not in payload]

    if missing_columns:
        raise ValueError(f"The input is missing required features: {missing_columns}")

    prepared_row: dict[str, float] = {}

    for column in feature_columns:
        value = payload[column]

        try:
            prepared_row[column] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Feature '{column}' must be numeric. Received: {value!r}") from error

    return pd.DataFrame(
        [prepared_row],
        columns=feature_columns,
    )


def convert_decision_to_anomaly_score(
    decision_value: float,
    anomaly_threshold: float,
) -> float:
    """
    Convert the Isolation Forest decision value to 0-1.

    Lower model decision values represent more unusual events.

    The curve is centred on ``anomaly_threshold`` -- the calibrated cutoff
    chosen at training time, expressed in suspicion space (higher = more
    suspicious, i.e. ``-decision_function``). Centring it there means a score of
    exactly 0.5 is the decision boundary, so ``anomaly_score >= 0.5`` and "the
    model flagged this" are the same statement.

    The threshold is required rather than defaulted: an accidental call without
    it would silently score against the uncalibrated boundary.
    """

    exponent = max(
        -60.0,
        min(60.0, 6.0 * (decision_value + anomaly_threshold)),
    )

    score = 1.0 / (1.0 + math.exp(exponent))

    return float(score)


def is_anomalous(
    decision_value: float,
    anomaly_threshold: float,
) -> bool:
    """Decide whether one decision value clears the calibrated cutoff.

    Equivalent to ``-decision_value >= anomaly_threshold``. Used instead of
    ``IsolationForest.predict()``, whose cutoff is fixed by the ``contamination``
    parameter rather than by the observed score distribution.
    """

    return bool(-decision_value >= anomaly_threshold)


def calculate_rule_score(
    payload: dict[str, Any],
) -> tuple[float, list[str]]:
    """Calculate a deterministic security-rule score."""

    score = 0.0
    reasons: list[str] = []

    if float(payload["sensitive_operation"]) >= 1:
        score += 0.20
        reasons.append("The event performed a sensitive operation.")

    if float(payload["unusual_hour"]) >= 1:
        score += 0.15
        reasons.append("The event happened at an unusual time.")

    if float(payload["external_ip"]) >= 1:
        score += 0.20
        reasons.append("The activity came from an external IP.")

    if float(payload["outside_home_region"]) >= 1:
        score += 0.10
        reasons.append("The activity occurred outside the home Region.")

    if float(payload["failed_call"]) >= 1:
        score += 0.10
        reasons.append("The AWS API call failed.")

    api_risk = max(
        0.0,
        min(1.0, float(payload["api_risk_score"])),
    )

    score += 0.25 * api_risk

    if api_risk >= 0.70:
        reasons.append("The AWS API action has a high security risk.")

    return min(score, 1.0), reasons


def determine_risk_level(risk_score: float) -> str:
    """Convert the final score into a risk level."""

    if risk_score >= 0.70:
        return "HIGH"

    if risk_score >= 0.40:
        return "MEDIUM"

    return "LOW"


def predict_event(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Predict the risk of one AWS security event."""

    model = load_model()
    feature_columns = load_feature_manifest()
    anomaly_threshold = load_anomaly_threshold()

    features = prepare_input(
        payload=payload,
        feature_columns=feature_columns,
    )

    decision_value = float(model.decision_function(features)[0])

    # The calibrated cutoff replaces IsolationForest.predict(), whose boundary is
    # fixed by `contamination` rather than by the observed score distribution.
    model_flagged_anomaly = is_anomalous(decision_value, anomaly_threshold)

    anomaly_score = convert_decision_to_anomaly_score(decision_value, anomaly_threshold)

    rule_score, reasons = calculate_rule_score(payload)

    final_risk_score = min(
        1.0,
        0.70 * anomaly_score + 0.30 * rule_score,
    )

    risk_level = determine_risk_level(final_risk_score)

    model_label = "suspicious" if model_flagged_anomaly else "normal"

    if model_flagged_anomaly:
        reasons.insert(
            0,
            "The machine-learning model marked the behavior as unusual.",
        )

    if not reasons:
        reasons.append("No major security-warning rules were triggered.")

    return {
        "model_prediction": model_label,
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


def parse_arguments() -> argparse.Namespace:
    """Read command-line arguments."""

    parser = argparse.ArgumentParser(
        description=("Predict whether one AWS event is normal or suspicious.")
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to one JSON feature file.",
    )

    return parser.parse_args()


def main() -> None:
    """Run prediction from the command line."""

    arguments = parse_arguments()

    payload = load_json_file(arguments.input)

    result = predict_event(payload)

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
