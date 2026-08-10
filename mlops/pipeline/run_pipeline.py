"""Register the pipeline definition with real SageMaker, and optionally run it.

Registering the definition and executing it are two separate AWS operations:

    Pipeline.upsert()  ->  CreatePipeline / UpdatePipeline   (no compute)
    Pipeline.start()   ->  StartPipelineExecution            (needs job quota)

Only the second one launches processing and training jobs, so the definition can
be kept current while instance quota is still pending.

This module must never run inside moto's ``mock_aws()``. Moto intercepts the
calls and returns a successful response with a fake account ARN, so a mocked
upsert looks like it worked while nothing reaches the account. The mocked path
belongs in create_pipeline.py's ``__main__`` block, which only builds and
validates the definition.

    # Update the definition only (safe while quota is pending):
    python -m mlops.pipeline.run_pipeline --bucket-name YOUR_BUCKET

    # Update and start a run (needs approved quota):
    python -m mlops.pipeline.run_pipeline --bucket-name YOUR_BUCKET --execute
"""

import argparse

import boto3

try:
    # Works when the repository root is on sys.path (python -m mlops.pipeline.run_pipeline).
    from mlops.pipeline.create_pipeline import get_pipeline
except ImportError:
    # Fallback for running the file directly from inside mlops/pipeline/.
    from create_pipeline import get_pipeline


def run():
    parser = argparse.ArgumentParser(
        description="Upsert the ThreatML SageMaker pipeline definition into AWS.",
    )
    parser.add_argument("--role-name", type=str, default="ThreatMlSageMakerExecutionRole")
    parser.add_argument("--bucket-name", type=str, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Start a pipeline execution after upserting. Off by default: "
            "execution consumes instance quota, upserting the definition does not."
        ),
    )
    args = parser.parse_args()

    sts = boto3.client("sts")
    identity = sts.get_caller_identity()
    account_id = identity["Account"]

    role = f"arn:aws:iam::{account_id}:role/{args.role_name}"

    from sagemaker.workflow.pipeline_context import PipelineSession

    sess = PipelineSession(default_bucket=args.bucket_name)

    # Print the resolved identity so a misconfigured profile is obvious before
    # anything is written to the account.
    print(f"Target account: {account_id} (as {identity['Arn']})")
    print(f"Synthesizing Pipeline with Role: {role} in Bucket: {args.bucket_name}")
    pipeline = get_pipeline(role=role, sagemaker_session=sess, default_bucket=args.bucket_name)

    print("Upserting Pipeline to AWS SageMaker...")
    response = pipeline.upsert(role_arn=role)
    print(f"Pipeline definition registered: {response.get('PipelineArn', pipeline.name)}")

    if not args.execute:
        print("\nDefinition updated. No execution started (pass --execute to start one).")
        print("Review it in SageMaker Studio -> Pipelines -> ThreatML-MLOps-Pipeline")
        return

    print("Executing Pipeline...")
    execution = pipeline.start()

    print(f"Started pipeline execution: {execution.arn}")
    print("You can track this in SageMaker Studio -> Pipelines -> ThreatML-MLOps-Pipeline")


if __name__ == "__main__":
    run()
