from typing import Any

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    Tags,
)
from aws_cdk import (
    aws_dynamodb as dynamodb,
)
from aws_cdk import (
    aws_s3 as s3,
)
from aws_cdk import (
    aws_sns as sns,
)
from aws_cdk import (
    aws_sns_subscriptions as subscriptions,
)
from aws_cdk import (
    aws_sqs as sqs,
)
from constructs import Construct


class FoundationStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        project_name: str,
        environment_name: str,
        alert_email: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        is_dev = environment_name == "dev"
        removal_policy = RemovalPolicy.DESTROY if is_dev else RemovalPolicy.RETAIN

        raw_events_bucket = s3.Bucket(
            self,
            "RawEventsBucket",
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            versioned=True,
            removal_policy=removal_policy,
            auto_delete_objects=is_dev,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="ExpireDevelopmentData",
                    enabled=is_dev,
                    expiration=Duration.days(30) if is_dev else None,
                    noncurrent_version_expiration=Duration.days(7) if is_dev else None,
                )
            ],
        )

        dead_letter_queue = sqs.Queue(
            self,
            "ProcessingDeadLetterQueue",
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            retention_period=Duration.days(14),
            removal_policy=removal_policy,
        )

        processing_queue = sqs.Queue(
            self,
            "ProcessingQueue",
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            visibility_timeout=Duration.minutes(12),
            retention_period=Duration.days(4),
            dead_letter_queue=sqs.DeadLetterQueue(
                max_receive_count=5,
                queue=dead_letter_queue,
            ),
            removal_policy=removal_policy,
        )

        incidents_table = dynamodb.Table(
            self,
            "IncidentsTable",
            partition_key=dynamodb.Attribute(
                name="incident_id",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            encryption=dynamodb.TableEncryption.AWS_MANAGED,
            point_in_time_recovery=not is_dev,
            removal_policy=removal_policy,
        )

        alerts_topic = sns.Topic(
            self,
            "AlertsTopic",
            display_name="Autonomous Cloud Threat ML Alerts",
        )
        if alert_email and alert_email != "replace-me@example.com":
            alerts_topic.add_subscription(subscriptions.EmailSubscription(alert_email))

        for key, value in {
            "Project": project_name,
            "Environment": environment_name,
            "ManagedBy": "AWS-CDK",
        }.items():
            Tags.of(self).add(key, value)

        CfnOutput(self, "RawEventsBucketName", value=raw_events_bucket.bucket_name)
        CfnOutput(self, "ProcessingQueueUrl", value=processing_queue.queue_url)
        CfnOutput(self, "DeadLetterQueueUrl", value=dead_letter_queue.queue_url)
        CfnOutput(self, "IncidentsTableName", value=incidents_table.table_name)
        CfnOutput(self, "AlertsTopicArn", value=alerts_topic.topic_arn)
