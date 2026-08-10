"""Tests for threshold calibration.

The decision cutoff is chosen at training time from held-out labelled data and
carried to serving in ``feature_manifest.json``. These tests cover the choice
itself, the scoring functions that consume it, and the backward-compatible
fallback for models trained before calibration existed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest

from threat_ml.predict_event import (
    convert_decision_to_anomaly_score,
    is_anomalous,
    read_anomaly_threshold,
)
from threat_ml.train_model import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    choose_anomaly_threshold,
    evaluate_model,
    split_for_calibration,
)


def build_dataset(rows: int = 400, suspicious_ratio: float = 0.1) -> pd.DataFrame:
    """Build a separable synthetic dataset in the real feature schema."""
    rng = np.random.default_rng(42)
    n_suspicious = int(rows * suspicious_ratio)
    n_normal = rows - n_suspicious

    normal = pd.DataFrame(
        {column: rng.normal(0.0, 0.5, n_normal) for column in FEATURE_COLUMNS}
    )
    normal[TARGET_COLUMN] = 0

    suspicious = pd.DataFrame(
        {column: rng.normal(6.0, 0.5, n_suspicious) for column in FEATURE_COLUMNS}
    )
    suspicious[TARGET_COLUMN] = 1

    return (
        pd.concat([normal, suspicious], ignore_index=True)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )


def fit_model(data: pd.DataFrame) -> IsolationForest:
    normals = data.loc[data[TARGET_COLUMN] == 0, FEATURE_COLUMNS].astype(float)
    model = IsolationForest(n_estimators=100, contamination=0.10, random_state=42)
    model.fit(normals)
    return model


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------


def test_split_holds_out_a_calibration_slice() -> None:
    data = build_dataset()

    fitting, calibration = split_for_calibration(data)

    assert len(fitting) + len(calibration) == len(data)
    assert len(calibration) == pytest.approx(len(data) * 0.20, abs=2)
    # Stratified, so the calibration slice must contain both classes.
    assert calibration[TARGET_COLUMN].nunique() == 2


def test_split_is_deterministic() -> None:
    data = build_dataset()

    first, _ = split_for_calibration(data)
    second, _ = split_for_calibration(data)

    assert first.index.tolist() == second.index.tolist()


def test_chosen_threshold_beats_contamination_boundary() -> None:
    """The whole point: calibration must improve precision over predict()."""
    data = build_dataset()
    fitting, calibration = split_for_calibration(data)
    model = fit_model(fitting)

    threshold = choose_anomaly_threshold(model, calibration)

    labels = data[TARGET_COLUMN].astype(int).to_numpy()
    features = data[FEATURE_COLUMNS].astype(float)
    scores = -model.decision_function(features)

    calibrated = (scores >= threshold).astype(int)
    contamination = (model.predict(features) == -1).astype(int)

    def precision(pred: np.ndarray) -> float:
        flagged = pred.sum()
        return float((pred & labels).sum() / flagged) if flagged else 0.0

    assert precision(calibrated) > precision(contamination)


def test_threshold_refuses_when_only_one_class_present() -> None:
    """Training must fail rather than silently ship an uncalibrated model."""
    data = build_dataset()
    model = fit_model(data)
    normals_only = data.loc[data[TARGET_COLUMN] == 0]

    with pytest.raises(ValueError, match="only one class"):
        choose_anomaly_threshold(model, normals_only)


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def test_anomaly_score_is_half_at_the_threshold() -> None:
    """A score of 0.5 must mean exactly 'on the decision boundary'."""
    for threshold in (0.0, 0.08, -0.15, 0.42):
        score = convert_decision_to_anomaly_score(-threshold, threshold)
        assert score == pytest.approx(0.5, abs=1e-9)


def test_anomaly_score_agrees_with_the_flag() -> None:
    """anomaly_score >= 0.5 and is_anomalous must be the same statement."""
    threshold = 0.08
    for decision in np.linspace(-1.0, 1.0, 41):
        score = convert_decision_to_anomaly_score(float(decision), threshold)
        assert (score >= 0.5) == is_anomalous(float(decision), threshold)


def test_anomaly_score_increases_as_events_get_more_unusual() -> None:
    threshold = 0.08
    scores = [convert_decision_to_anomaly_score(d, threshold) for d in (0.5, 0.1, -0.1, -0.5)]

    assert scores == sorted(scores)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_is_anomalous_at_the_exact_boundary() -> None:
    # suspicion == threshold counts as anomalous.
    assert is_anomalous(-0.08, 0.08)
    assert not is_anomalous(-0.079, 0.08)


def test_raising_the_threshold_flags_fewer_events() -> None:
    decisions = list(np.linspace(-0.5, 0.5, 101))

    flagged_low = sum(is_anomalous(float(d), 0.0) for d in decisions)
    flagged_high = sum(is_anomalous(float(d), 0.25) for d in decisions)

    assert flagged_high < flagged_low


# ---------------------------------------------------------------------------
# Manifest handling and backward compatibility
# ---------------------------------------------------------------------------


def test_manifest_threshold_is_read() -> None:
    assert read_anomaly_threshold({"anomaly_threshold": 0.125}) == 0.125
    assert read_anomaly_threshold({"anomaly_threshold": "0.25"}) == 0.25


@pytest.mark.parametrize(
    ("manifest", "expected_message"),
    [
        ({}, "missing 'anomaly_threshold'"),
        ({"model_version": "0.1.0"}, "missing 'anomaly_threshold'"),
        ({"anomaly_threshold": None}, "must be numeric"),
        ({"anomaly_threshold": "abc"}, "must be numeric"),
        ({"anomaly_threshold": float("nan")}, "must be finite"),
        ({"anomaly_threshold": float("inf")}, "must be finite"),
    ],
)
def test_manifest_without_usable_threshold_raises(
    manifest: dict[str, object], expected_message: str
) -> None:
    """No silent fallback: an uncalibrated artifact must refuse to score.

    A pre-calibration manifest (model_version 0.1.0, no threshold key) is exactly
    what sits in stale CDK asset directories, so this is the real failure mode.
    """
    with pytest.raises(ValueError, match=expected_message):
        read_anomaly_threshold(manifest)


def test_scoring_functions_require_an_explicit_threshold() -> None:
    """Neither scorer may be callable without a threshold."""
    with pytest.raises(TypeError):
        convert_decision_to_anomaly_score(0.1)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        is_anomalous(0.1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Evaluation reporting
# ---------------------------------------------------------------------------


def test_evaluate_model_reports_the_threshold_it_used() -> None:
    data = build_dataset()
    model = fit_model(data)

    metrics = evaluate_model(model, data, anomaly_threshold=0.08)

    assert metrics["anomaly_threshold"] == 0.08
    assert metrics["flagged_rows"] >= 0
    assert 0.0 <= float(metrics["precision"]) <= 1.0  # type: ignore[arg-type]


def test_evaluate_model_threshold_changes_the_verdict() -> None:
    data = build_dataset()
    model = fit_model(data)

    permissive = evaluate_model(model, data, anomaly_threshold=-1.0)
    strict = evaluate_model(model, data, anomaly_threshold=1.0)

    assert int(permissive["flagged_rows"]) > int(strict["flagged_rows"])  # type: ignore[arg-type]
