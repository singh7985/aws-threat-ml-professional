from __future__ import annotations

import pandas as pd

from threat_ml.batch_predict import (
    build_summary,
)


def create_predictions() -> pd.DataFrame:
    """Create a small prediction result dataset."""

    return pd.DataFrame(
        [
            {
                "risk_level": "LOW",
                "model_flagged_anomaly": False,
                "anomaly_score": 0.20,
                "final_risk_score": 0.15,
                "label_binary": 0,
            },
            {
                "risk_level": "HIGH",
                "model_flagged_anomaly": True,
                "anomaly_score": 0.90,
                "final_risk_score": 0.87,
                "label_binary": 1,
            },
            {
                "risk_level": "MEDIUM",
                "model_flagged_anomaly": True,
                "anomaly_score": 0.60,
                "final_risk_score": 0.58,
                "label_binary": 1,
            },
        ]
    )


def test_build_summary_counts_events() -> None:
    predictions = create_predictions()

    summary = build_summary(predictions)

    assert summary["total_events"] == 3
    assert summary["low_risk_events"] == 1
    assert summary["medium_risk_events"] == 1
    assert summary["high_risk_events"] == 1


def test_build_summary_counts_ml_anomalies() -> None:
    predictions = create_predictions()

    summary = build_summary(predictions)

    assert summary["ml_flagged_anomalies"] == 2


def test_build_summary_contains_evaluation() -> None:
    predictions = create_predictions()

    summary = build_summary(predictions)

    assert "evaluation" in summary
    assert summary["evaluation"]["precision"] == 1.0
    assert summary["evaluation"]["recall"] == 1.0
    assert summary["evaluation"]["f1_score"] == 1.0


def test_build_summary_confusion_matrix() -> None:
    predictions = create_predictions()

    summary = build_summary(predictions)

    assert summary["evaluation"]["confusion_matrix"] == [[1, 0], [0, 2]]
