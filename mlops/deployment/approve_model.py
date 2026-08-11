import argparse
import contextlib
import json
import logging
from typing import Any

import boto3

from mlops.quality_gates import evaluate_gates

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)

def get_latest_model_version(sm_client: Any, model_package_group_name: str) -> dict[str, Any]:
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    if not response["ModelPackageSummaryList"]:
        raise ValueError(f"No models found in {model_package_group_name}")
        
    summary: dict[str, Any] = response["ModelPackageSummaryList"][0]
    return summary

def approve_model() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-group", type=str, default="ThreatMLModels")
    parser.add_argument("--force", action="store_true", help="Force approval without prompting")
    args = parser.parse_args()

    sm_client = boto3.client("sagemaker", region_name="us-east-1")
    s3_client = boto3.client("s3")

    logger.info(f"Looking for latest model in group: {args.model_group}")
    try:
        latest_model = get_latest_model_version(sm_client, args.model_group)
    except ValueError as e:
        logger.error(str(e))
        return

    model_arn = latest_model["ModelPackageArn"]
    status = latest_model["ModelApprovalStatus"]
    
    logger.info(f"Found model: {model_arn}")
    logger.info(f"Current status: {status}")

    if status != "PendingManualApproval":
        logger.info(f"Model is not in 'PendingManualApproval' state (current: {status}). Exiting.")
        return

    # Fetch detailed model package information
    model_details = sm_client.describe_model_package(ModelPackageName=model_arn)
    
    # Try to find and parse metrics from the registered S3 URI if they exist
    metrics_s3_uri = None
    with contextlib.suppress(AttributeError):
        metrics_s3_uri = (
            model_details.get("ModelMetrics", {})
            .get("ModelQuality", {})
            .get("Statistics", {})
            .get("S3Uri")
        )


    if metrics_s3_uri:
        logger.info(f"Downloading evaluation metrics from {metrics_s3_uri}...")
        # Parse S3 URI
        bucket = metrics_s3_uri.split("/")[2]
        key = "/".join(metrics_s3_uri.split("/")[3:])
        
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            metrics_data = json.loads(response["Body"].read().decode("utf-8"))
            
            # Print exact nested metrics from evaluation.json based on previous structure
            metrics = metrics_data.get("classification_metrics", {})

            # Checked against the same thresholds the pipeline's CheckModelQuality
            # step used. If these two ever disagree, this script would warn about a
            # model the pipeline already passed.
            meets_thresholds, gate_lines = evaluate_gates(metrics)

            logger.info("--- Model Metrics ---")
            for line in gate_lines:
                logger.info(line)
            logger.info("---------------------")

            # Additional double check locally, just to be absolutely certain before manual prompt
            if not meets_thresholds:
                logger.error("WARNING: Metrics do not strictly meet the recommended thresholds!")
                
        except Exception as e:
            logger.error(f"Could not load or parse metrics from S3: {e}")
    else:
        logger.warning("No metrics S3 URI associated with this model package.")

    if not args.force:
        response = input(f"Approve model {model_arn.split('/')[-1]}? [y/N]: ")
        if response.lower() not in ["y", "yes"]:
            logger.info("Approval aborted by user.")
            return

    logger.info("Approving model...")
    sm_client.update_model_package(
        ModelPackageArn=model_arn,
        ModelApprovalStatus="Approved"
    )
    logger.info("Model marked as 'Approved' successfully.")

if __name__ == "__main__":
    approve_model()
