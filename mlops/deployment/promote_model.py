import argparse
import json
import logging
import shutil
import tarfile
import urllib.parse
from pathlib import Path
from typing import Any

import boto3
import joblib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAMBDA_MODEL_DIRECTORY = PROJECT_ROOT / "models"

REQUIRED_FILES = [
    "model.joblib",
    "feature_manifest.json",
    "training_metadata.json"
]

def get_latest_approved_model(sm_client: Any, model_package_group_name: str) -> dict[str, Any]:
    # Find models specifically marked Approved
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    if not response["ModelPackageSummaryList"]:
        raise ValueError(f"No 'Approved' models found in {model_package_group_name}")
        
    summary: dict[str, Any] = response["ModelPackageSummaryList"][0]
    return summary

def validate_artifacts(extract_dir: Path) -> None:
    logger.info("Validating artifacts...")
    
    # 1. Missing files
    for req_file in REQUIRED_FILES:
        if not (extract_dir / req_file).exists():
            raise FileNotFoundError(f"Required artifact {req_file} is missing.")
            
    # 2. Feature manifest invalid.
    # Parsing is guarded separately from the key check so the "missing essential
    # keys" error is not swallowed and re-wrapped as "manifest is invalid".
    manifest_path = extract_dir / "feature_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception as e:
        raise ValueError(f"Feature manifest is invalid: {e}") from e

    if not isinstance(manifest, dict):
        raise ValueError("Feature manifest is invalid: expected a JSON object.")

    if not manifest.get("feature_columns") or not manifest.get("algorithm"):
        raise ValueError("Feature manifest is missing essential keys (feature_columns, algorithm).")

    # 3. Model cannot load
    model_path = extract_dir / "model.joblib"
    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise ValueError(f"Model failed to load via joblib: {e}") from e

    # 4. Local prediction regression test
    # Execute a simple mock predict to prove it accepts exactly the lengths defined in manifest.
    try:
        import numpy as np
        # create mock features of correct length
        mock_inputs = np.zeros((1, len(manifest["feature_columns"])))
        model.predict(mock_inputs)
    except Exception as e:
        raise ValueError(f"Model prediction regression test failed: {e}") from e
        
    logger.info("Artifacts passed all validations.")

def promote_model() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-group", type=str, default="ThreatMLModels")
    args = parser.parse_args()

    sm_client = boto3.client("sagemaker", region_name="us-east-1")
    s3_client = boto3.client("s3")

    logger.info(f"Looking for latest officially Approved model in {args.model_group}")
    try:
        approved_model_summary = get_latest_approved_model(sm_client, args.model_group)
    except ValueError as e:
        logger.error(str(e))
        return
        
    model_arn = approved_model_summary["ModelPackageArn"]
    logger.info(f"Promoting model: {model_arn}")
    
    # Extract structural details about the package to find the S3 Model Data
    model_package_details = sm_client.describe_model_package(ModelPackageName=model_arn)
    
    inference_specification = model_package_details.get("InferenceSpecification")
    if not inference_specification:
        logger.error("No InferenceSpecification found attached to model package.")
        return
        
    containers = inference_specification.get("Containers", [])
    if not containers:
        logger.error("No containers specified in model package.")
        return
        
    model_s3_uri = containers[0].get("ModelDataUrl")
    if not model_s3_uri:
        logger.error("No ModelDataUrl linked to the approved model container.")
        return
        
    logger.info(f"Downloading generic Model artifact from: {model_s3_uri}")
    parsed_s3 = urllib.parse.urlparse(model_s3_uri)
    bucket = parsed_s3.netloc
    key = parsed_s3.path.lstrip("/")
    
    temp_dir = PROJECT_ROOT / "mlops" / "deployment" / ".temp_promotion"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_tar_path = temp_dir / "model.tar.gz"
    
    try:
        s3_client.download_file(bucket, key, str(temp_tar_path))
    except Exception as e:
        logger.error(f"Failed to download from S3: {e}")
        return
        
    logger.info(f"Extracting artifacts into {temp_dir}...")
    try:
        with tarfile.open(temp_tar_path) as tar:
            # filter="data" blocks absolute paths, ".." traversal, and special
            # files. Without it a crafted archive could write outside temp_dir.
            tar.extractall(path=str(temp_dir), filter="data")
    except tarfile.ReadError:
        # Occasionally SDK v2 packs it raw initially depending on Job specs instead of tar.gz.
        logger.warning("Could not open as tarfile. Validation will assume raw components copied directly.")

    # Run hard validations
    try:
        validate_artifacts(temp_dir)
    except Exception as e:
        logger.error(f"Promotion rejected due to validation failure: {e}")
        return
        
    # Copy explicitly out to the production Lambda folder 
    logger.info(f"Promotion checks passed! Copying files to {LAMBDA_MODEL_DIRECTORY}...")
    LAMBDA_MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    for req_file in REQUIRED_FILES:
        src = temp_dir / req_file
        dest = LAMBDA_MODEL_DIRECTORY / req_file
        shutil.copy2(src, dest)
        logger.info(f"Copied: {req_file}")
        
    # Cleanup temp explicitly
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    logger.info("Local environment updated successfully. Run 'npx cdk deploy ThreatMlScorer-dev' to map this natively to Lambda.")

if __name__ == "__main__":
    promote_model()
