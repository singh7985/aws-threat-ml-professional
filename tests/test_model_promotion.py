import json

import joblib
import pytest
from botocore.stub import Stubber

from mlops.deployment.promote_model import (
    get_latest_approved_model,
    validate_artifacts,
)


def test_get_latest_approved_model(mocker):
    import boto3

    sm_client = boto3.client("sagemaker", region_name="us-east-1")
    stubber = Stubber(sm_client)

    expected_params = {
        "ModelPackageGroupName": "ThreatMLModels",
        "ModelApprovalStatus": "Approved",
        "SortBy": "CreationTime",
        "SortOrder": "Descending",
        "MaxResults": 1,
    }

    import datetime

    mock_response = {
        "ModelPackageSummaryList": [
            {
                "ModelPackageName": "mock-model-pkg",
                "ModelPackageArn": "arn:mock",
                "ModelApprovalStatus": "Approved",
                "CreationTime": datetime.datetime(2026, 8, 5),
                "ModelPackageStatus": "Completed",
            }
        ]
    }

    stubber.add_response("list_model_packages", mock_response, expected_params)

    with stubber:
        model = get_latest_approved_model(sm_client, "ThreatMLModels")
        assert model["ModelPackageArn"] == "arn:mock"
        assert model["ModelApprovalStatus"] == "Approved"


def test_get_latest_approved_model_none_found():
    import boto3

    sm_client = boto3.client("sagemaker", region_name="us-east-1")
    stubber = Stubber(sm_client)

    mock_response = {"ModelPackageSummaryList": []}

    stubber.add_response("list_model_packages", mock_response)

    with stubber, pytest.raises(ValueError, match="No 'Approved' models found"):
        get_latest_approved_model(sm_client, "ThreatMLModels")


def test_validate_artifacts_missing_files(tmp_path):
    # Empty dir throws missing file
    with pytest.raises(FileNotFoundError, match="is missing"):
        validate_artifacts(tmp_path)

    (tmp_path / "model.joblib").touch()
    with pytest.raises(FileNotFoundError, match=r"feature_manifest\.json is missing"):
        validate_artifacts(tmp_path)


def test_validate_artifacts_invalid_manifest(tmp_path):
    (tmp_path / "model.joblib").touch()
    (tmp_path / "training_metadata.json").touch()

    invalid_manifest = tmp_path / "feature_manifest.json"

    invalid_manifest.write_text(json.dumps({"wrong": "keys"}))
    with pytest.raises(ValueError, match="missing essential keys"):
        validate_artifacts(tmp_path)

    invalid_manifest.write_text("not json")
    with pytest.raises(ValueError, match="Feature manifest is invalid"):
        validate_artifacts(tmp_path)


def test_validate_artifacts_model_load_failure(tmp_path):
    (tmp_path / "training_metadata.json").touch()

    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text(json.dumps({"feature_columns": ["a", "b"], "algorithm": "Testing"}))

    bad_model = tmp_path / "model.joblib"
    bad_model.write_text("not a joblib")

    with pytest.raises(ValueError, match="Model failed to load via joblib"):
        validate_artifacts(tmp_path)


def test_validate_artifacts_prediction_failure(tmp_path):
    import numpy as np
    from sklearn.ensemble import IsolationForest

    (tmp_path / "training_metadata.json").touch()

    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text(
        json.dumps({"feature_columns": ["col_1", "col_2", "col_3"], "algorithm": "IsolationForest"})
    )

    # Train dummy model on 2 features, but manifest expects 3 features
    # This will throw exception during mock prediction natively!
    model = IsolationForest()
    model.fit(np.zeros((10, 2)))
    joblib.dump(model, tmp_path / "model.joblib")

    with pytest.raises(ValueError, match="Model prediction regression test failed:"):
        validate_artifacts(tmp_path)


def test_validate_artifacts_success(tmp_path):
    import numpy as np
    from sklearn.ensemble import IsolationForest

    (tmp_path / "training_metadata.json").touch()

    manifest = tmp_path / "feature_manifest.json"
    manifest.write_text(
        json.dumps({"feature_columns": ["col_1", "col_2"], "algorithm": "IsolationForest"})
    )

    model = IsolationForest()
    model.fit(np.zeros((10, 2)))
    joblib.dump(model, tmp_path / "model.joblib")

    # Should not raise any exceptions
    validate_artifacts(tmp_path)
