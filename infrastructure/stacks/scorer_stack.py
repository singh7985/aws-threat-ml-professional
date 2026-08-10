import os
from typing import Any

from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
    Tags,
)
from aws_cdk import (
    aws_dynamodb as dynamodb,
)
from aws_cdk import (
    aws_ecr_assets as ecr_assets,
)
from aws_cdk import (
    aws_iam as iam,
)
from aws_cdk import (
    aws_lambda as lambda_,
)
from aws_cdk import (
    aws_lambda_event_sources as eventsources,
)
from aws_cdk import (
    aws_sns as sns,
)
from aws_cdk import (
    aws_sqs as sqs,
)
from constructs import Construct


class ScorerStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        processing_queue: sqs.Queue,
        incidents_table: dynamodb.Table,
        alerts_topic: sns.Topic,
        project_name: str,
        environment_name: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        is_dev = environment_name == "dev"

        self.scorer_function = lambda_.DockerImageFunction(
            self,
            "ScorerFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                directory=os.path.join(os.path.dirname(__file__), "..", ".."),
                file="services/scorer/Dockerfile",
                # Must match the `architecture` below. This was previously passed
                # as build_args={"platform": ...}, which sets a Docker build ARG
                # -- the Dockerfile declares no such ARG, so it did nothing. On an
                # arm64 build host that silently produced an arm64 image for an
                # x86_64 function, which fails at runtime with "exec format error".
                platform=ecr_assets.Platform.LINUX_AMD64,
            ),
            architecture=lambda_.Architecture.X86_64,
            memory_size=3008,  # Machine learning needs a bit of memory/cpu
            # Cold start measures ~10s: pulling a ~320 MB image and importing
            # scikit-learn/pandas before the first score. Warm invocations finish
            # in single-digit milliseconds. 30s left only a 3x margin over an
            # init that can spike under concurrency, and Lambda bills actual
            # duration rather than the configured ceiling, so the headroom is free.
            # Stays well inside the queue's 720s visibility timeout, which must
            # remain at least 6x this value.
            timeout=Duration.seconds(60),
            # Every environment variable the function needs is declared here, in
            # the stack that owns the function. MonitoringStack used to inject
            # some of these via add_environment() across a stack boundary, which
            # made the function's configuration depend on an unrelated stack
            # being deployed.
            environment={
                "INCIDENTS_TABLE_NAME": incidents_table.table_name,
                "ALERTS_TOPIC_ARN": alerts_topic.topic_arn,
                "LOG_LEVEL": "DEBUG" if is_dev else "INFO",
                "POWERTOOLS_SERVICE_NAME": "scorer",
                "POWERTOOLS_METRICS_NAMESPACE": "IntelligentAwsThreatResponse",
            },
            tracing=lambda_.Tracing.ACTIVE,
        )

        # grant_read_write_data already covers PutItem/GetItem/UpdateItem, so a
        # second explicit grant() would only add a duplicate policy statement.
        incidents_table.grant_read_write_data(self.scorer_function)
        alerts_topic.grant_publish(self.scorer_function)

        # Hook SQS queue as an event source to trigger Lambda
        self.scorer_function.add_event_source(
            eventsources.SqsEventSource(
                processing_queue,
                batch_size=10,
                max_batching_window=Duration.seconds(5),
                report_batch_item_failures=True,
            )
        )

        # Add basic cloudwatch rights
        self.scorer_function.add_to_role_policy(
            iam.PolicyStatement(actions=["cloudwatch:PutMetricData"], resources=["*"])
        )

        for key, value in {
            "Project": project_name,
            "Environment": environment_name,
            "ManagedBy": "AWS-CDK",
        }.items():
            Tags.of(self).add(key, value)

        CfnOutput(self, "ScorerFunctionName", value=self.scorer_function.function_name)
