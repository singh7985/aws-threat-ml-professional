"""Tests for the CLI, settings, dataset loading, and the artifact-loading paths.

These modules had no coverage: they are thin, but they sit on the startup path
for training, batch scoring, and the container, so a breakage here fails
everything downstream while looking like a configuration problem.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.ensemble import IsolationForest

from threat_ml import batch_predict, cli, load_data, predict_event, train_model
from threat_ml.schemas import IncidentPrediction
from threat_ml.settings import Settings, get_settings

FEATURE_COLUMNS = train_model.FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_generate_writes_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "events.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        ["threat-ml", "generate", "--output", str(output), "--normal", "4", "--suspicious", "2"],
    )

    assert cli.main() == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 6


def test_cli_requires_a_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["threat-ml"])
    with pytest.raises(SystemExit):
        cli.main()


def test_cli_rejects_unknown_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["threat-ml", "nonsense"])
    with pytest.raises(SystemExit):
        cli.main()


def test_cli_parser_defaults() -> None:
    args = cli.build_parser().parse_args(["generate", "--output", "x.jsonl"])
    assert (args.normal, args.suspicious, args.seed) == (100, 10, 42)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.app_env in {"dev", "test", "prod"}
    assert settings.aws_region
    assert settings.model_path == Path("models/model.joblib")


def test_settings_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.app_env == "prod"
    assert settings.aws_region == "eu-west-1"


def test_settings_reject_invalid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "staging")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _write_dataset(path: Path, rows: int = 6) -> None:
    frame = pd.DataFrame({column: np.arange(rows, dtype=float) for column in FEATURE_COLUMNS})
    frame[train_model.TARGET_COLUMN] = [0, 1] * (rows // 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_load_datasets_reads_both_splits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    train, test = tmp_path / "train.csv", tmp_path / "test.csv"
    _write_dataset(train)
    _write_dataset(test, rows=4)
    monkeypatch.setattr(load_data, "TRAIN_DATA_PATH", train)
    monkeypatch.setattr(load_data, "TEST_DATA_PATH", test)

    training, testing = load_data.load_datasets()

    assert (len(training), len(testing)) == (6, 4)


def test_load_datasets_reports_missing_training_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(load_data, "TRAIN_DATA_PATH", tmp_path / "absent.csv")
    monkeypatch.setattr(load_data, "TEST_DATA_PATH", tmp_path / "absent2.csv")

    with pytest.raises(FileNotFoundError, match="Training dataset"):
        load_data.load_datasets()


def test_train_model_load_dataset_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"hour_of_day": [1, 2]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        train_model.load_dataset(path)


def test_train_model_load_dataset_rejects_absent_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        train_model.load_dataset(tmp_path / "nope.csv")


def test_prepare_features_fills_and_casts() -> None:
    frame = pd.DataFrame({c: [None] for c in FEATURE_COLUMNS})
    prepared = train_model.prepare_features(frame)

    assert prepared.notna().all().all()
    assert all(prepared[c].dtype.kind == "f" for c in FEATURE_COLUMNS)


def test_train_model_requires_normal_rows() -> None:
    frame = pd.DataFrame({c: [1.0, 2.0] for c in FEATURE_COLUMNS})
    frame[train_model.TARGET_COLUMN] = [1, 1]

    with pytest.raises(ValueError, match="No normal training records"):
        train_model.train_model(frame)


# ---------------------------------------------------------------------------
# Artifact loading (predict_event / batch_predict)
# ---------------------------------------------------------------------------


@pytest.fixture
def artifact_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A models/ directory holding a real, calibrated artifact set."""
    model = IsolationForest(n_estimators=20, random_state=42)
    model.fit(np.zeros((20, len(FEATURE_COLUMNS))))
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "feature_manifest.json").write_text(
        json.dumps(
            {
                "model_version": "0.2.0",
                "feature_columns": FEATURE_COLUMNS,
                "anomaly_threshold": 0.05,
            }
        ),
        encoding="utf-8",
    )
    for module in (predict_event, batch_predict):
        monkeypatch.setattr(module, "MODEL_PATH", tmp_path / "model.joblib")
        monkeypatch.setattr(module, "FEATURE_MANIFEST_PATH", tmp_path / "feature_manifest.json")
    return tmp_path


def test_load_model_and_manifest(artifact_dir: Path) -> None:
    assert isinstance(predict_event.load_model(), IsolationForest)
    assert predict_event.load_feature_manifest() == FEATURE_COLUMNS
    assert predict_event.load_anomaly_threshold() == 0.05


def test_load_anomaly_threshold_requires_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(predict_event, "FEATURE_MANIFEST_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        predict_event.load_anomaly_threshold()


def test_load_model_reports_absent_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(predict_event, "MODEL_PATH", tmp_path / "missing.joblib")
    with pytest.raises(FileNotFoundError, match="Trained model"):
        predict_event.load_model()


def test_load_model_rejects_a_non_isolation_forest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    joblib.dump({"not": "a model"}, tmp_path / "m.joblib")
    monkeypatch.setattr(predict_event, "MODEL_PATH", tmp_path / "m.joblib")
    with pytest.raises(TypeError, match="not an IsolationForest"):
        predict_event.load_model()


def test_load_json_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert predict_event.load_json_file(path) == {"a": 1}


def test_load_json_file_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        predict_event.load_json_file(path)


def test_load_json_file_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "list.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="one JSON object"):
        predict_event.load_json_file(path)


def test_load_json_file_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        predict_event.load_json_file(tmp_path / "nope.json")


def test_predict_event_end_to_end(artifact_dir: Path) -> None:
    payload = dict.fromkeys(FEATURE_COLUMNS, 0.0)
    result = predict_event.predict_event(payload)

    assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH"}
    assert 0.0 <= result["anomaly_score"] <= 1.0
    assert isinstance(result["model_flagged_anomaly"], bool)
    assert result["reasons"]


# ---------------------------------------------------------------------------
# Batch prediction internals
# ---------------------------------------------------------------------------


def _feature_frame(rows: int = 5) -> pd.DataFrame:
    frame = pd.DataFrame({c: np.linspace(0, 1, rows) for c in FEATURE_COLUMNS})
    frame["label_binary"] = [0, 1] * (rows // 2) + [0] * (rows % 2)
    frame["event_id"] = [f"evt-{i}" for i in range(rows)]
    return frame


def test_batch_validate_dataset_detects_missing_features() -> None:
    with pytest.raises(ValueError, match="missing required features"):
        batch_predict.validate_dataset(pd.DataFrame({"a": [1]}), FEATURE_COLUMNS)


def test_batch_load_dataset_rejects_empty_and_missing(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    pd.DataFrame({c: [] for c in FEATURE_COLUMNS}).to_csv(empty, index=False)
    with pytest.raises(ValueError, match="empty"):
        batch_predict.load_dataset(empty)
    with pytest.raises(FileNotFoundError):
        batch_predict.load_dataset(tmp_path / "nope.csv")


def test_batch_prepare_matrix_rejects_non_numeric() -> None:
    frame = _feature_frame()
    frame["api_risk_score"] = "abc"
    with pytest.raises(ValueError, match="non-numeric"):
        batch_predict.prepare_feature_matrix(frame, FEATURE_COLUMNS)


def test_batch_score_dataset_and_save(artifact_dir: Path, tmp_path: Path) -> None:
    model = batch_predict.load_model()
    frame = _feature_frame(6)

    predictions = batch_predict.score_dataset(
        model=model, dataset=frame, feature_columns=FEATURE_COLUMNS, anomaly_threshold=0.05
    )

    assert len(predictions) == 6
    assert {"risk_level", "model_flagged_anomaly", "reasons"} <= set(predictions.columns)

    summary = batch_predict.build_summary(predictions)
    out = tmp_path / "out"
    batch_predict.save_results(predictions, summary, out)

    for name in ("predictions.csv", "incidents.jsonl", "high_risk_incidents.csv", "summary.json"):
        assert (out / name).exists(), name
    assert json.loads((out / "summary.json").read_text())["total_events"] == 6


def test_batch_load_manifest_requires_feature_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "feature_manifest.json"
    bad.write_text(json.dumps({"model_version": "0.2.0"}), encoding="utf-8")
    monkeypatch.setattr(batch_predict, "FEATURE_MANIFEST_PATH", bad)
    with pytest.raises(ValueError, match="feature_columns"):
        batch_predict.load_manifest()


# ---------------------------------------------------------------------------
# ThreatPredictor
# ---------------------------------------------------------------------------


def test_threat_predictor_uses_the_shared_calibrated_scorer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ThreatPredictor must not reintroduce a private scoring formula."""
    from threat_ml import predict as predict_module

    model = IsolationForest(n_estimators=20, random_state=42)
    model.fit(np.zeros((20, len(FEATURE_COLUMNS))))
    joblib.dump(model, tmp_path / "model.joblib")
    (tmp_path / "feature_manifest.json").write_text(
        json.dumps(
            {
                "model_version": "0.2.0",
                "feature_columns": FEATURE_COLUMNS,
                "anomaly_threshold": 0.05,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(predict_module, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(predict_module, "MANIFEST_PATH", tmp_path / "feature_manifest.json")

    predictor = predict_module.ThreatPredictor()
    assert predictor.anomaly_threshold == 0.05

    event: dict[str, Any] = {
        "event_id": "evt-1",
        "timestamp": "2026-08-07T03:15:00Z",
        "account_id": "111111111111",
        "principal_id": "developer-04",
        "event_name": "CreateAccessKey",
        "service": "iam",
        "region": "eu-west-1",
        "source_ip": "203.0.113.77",
        "user_agent": "aws-cli/2",
        "success": False,
        "sensitive_operation": True,
    }
    report = predictor.predict(event)

    assert isinstance(report, IncidentPrediction)
    assert 0.0 <= report.anomaly_score <= 1.0
    assert 0.0 <= report.risk_score <= 1.0
    assert report.risk_level in {"LOW", "MEDIUM", "HIGH"}
    assert report.model_version == "0.2.0"


def test_threat_predictor_requires_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from threat_ml import predict as predict_module

    monkeypatch.setattr(predict_module, "MODEL_PATH", tmp_path / "missing.joblib")
    monkeypatch.setattr(predict_module, "MANIFEST_PATH", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError):
        predict_module.ThreatPredictor()
