from typing import Any

from aws_cdk import (
    CfnOutput,
    Duration,
    RemovalPolicy,
    Stack,
    Tags,
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

        self.dead_letter_queue = sqs.Queue(
            self,
            "ProcessingDeadLetterQueue",
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            retention_period=Duration.days(14),
            removal_policy=removal_policy,
        )

        self.processing_queue = sqs.Queue(
            self,
            "ProcessingQueue",
            encryption=sqs.QueueEncryption.SQS_MANAGED,
            visibility_timeout=Duration.minutes(12),
            retention_period=Duration.days(4),
            dead_letter_queue=sqs.DeadLetterQueue(
                max_receive_count=5,
                queue=self.dead_letter_queue,
            ),
            removal_policy=removal_policy,
        )

        # The incidents table lives in DataStack (physical name
        # "intelligent-aws-threat-incidents"). This stack used to declare a second,
        # separate table that nothing ever read: app.py wires DataStack's table into
        # the scorer, so the one here was created, billed, and left permanently empty.
        # Removed rather than kept "just in case" -- two tables with the same purpose
        # is how incidents end up written to one and queried from the other.

        self.alerts_topic = sns.Topic(
            self,
            "AlertsTopic",
            display_name="Autonomous Cloud Threat ML Alerts",
        )
        if alert_email and alert_email != "replace-me@example.com":
            self.alerts_topic.add_subscription(subscriptions.EmailSubscription(alert_email))

        for key, value in {
            "Project": project_name,
            "Environment": environment_name,
            "ManagedBy": "AWS-CDK",
        }.items():
            Tags.of(self).add(key, value)

        CfnOutput(self, "RawEventsBucketName", value=raw_events_bucket.bucket_name)
        CfnOutput(self, "ProcessingQueueUrl", value=self.processing_queue.queue_url)
        CfnOutput(self, "DeadLetterQueueUrl", value=self.dead_letter_queue.queue_url)
        CfnOutput(self, "AlertsTopicArn", value=self.alerts_topic.topic_arn)
