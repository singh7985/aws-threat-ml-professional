from __future__ import annotations

from typing import Any

from aws_cdk import (
    CfnOutput,
    Duration,
    Stack,
)
from aws_cdk import (
    aws_cloudwatch as cloudwatch,
)
from aws_cdk import (
    aws_cloudwatch_actions as actions,
)
from aws_cdk import (
    aws_lambda as lambda_,
)
from aws_cdk import (
    aws_sns as sns,
)
from aws_cdk import (
    aws_sqs as sqs,
)
from constructs import Construct


class MonitoringStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        # IFunction is sufficient: this stack only reads metrics from the
        # function, it does not modify it.
        scorer_function: lambda_.IFunction,
        processing_queue: sqs.IQueue,
        dead_letter_queue: sqs.IQueue,
        alerts_topic: sns.Topic,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            scope,
            construct_id,
            **kwargs,
        )

        # This stack only observes the scorer; it no longer mutates the
        # function's environment. Those variables are declared in ScorerStack,
        # where the function is defined.
        alarm_action = actions.SnsAction(alerts_topic)

        lambda_errors_alarm = cloudwatch.Alarm(
            self,
            "LambdaErrorsAlarm",
            alarm_name=("threat-ml-lambda-errors-dev"),
            metric=scorer_function.metric_errors(
                period=Duration.minutes(5),
                statistic="sum",
            ),
            threshold=1,
            evaluation_periods=1,
            treat_missing_data=(cloudwatch.TreatMissingData.NOT_BREACHING),
        )

        dlq_alarm = cloudwatch.Alarm(
            self,
            "DeadLetterQueueAlarm",
            alarm_name="threat-ml-dlq-dev",
            metric=(
                dead_letter_queue.metric_approximate_number_of_messages_visible(
                    period=Duration.minutes(5),
                    statistic="maximum",
                )
            ),
            threshold=1,
            evaluation_periods=1,
            treat_missing_data=(cloudwatch.TreatMissingData.NOT_BREACHING),
        )

        queue_age_alarm = cloudwatch.Alarm(
            self,
            "QueueAgeAlarm",
            alarm_name="threat-ml-queue-age-dev",
            metric=(
                processing_queue.metric_approximate_age_of_oldest_message(
                    period=Duration.minutes(5),
                    statistic="maximum",
                )
            ),
            threshold=300,
            evaluation_periods=1,
            treat_missing_data=(cloudwatch.TreatMissingData.NOT_BREACHING),
        )

        for alarm in (
            lambda_errors_alarm,
            dlq_alarm,
            queue_age_alarm,
        ):
            alarm.add_alarm_action(alarm_action)

        high_risk_metric = cloudwatch.Metric(
            namespace=("IntelligentAwsThreatResponse"),
            metric_name="HighRiskIncidents",
            dimensions_map={
                "service": "scorer",
            },
            period=Duration.minutes(5),
            statistic="sum",
        )

        dashboard = cloudwatch.Dashboard(
            self,
            "OperationsDashboard",
            dashboard_name=("Intelligent-AWS-Threat-ML-dev"),
        )

        dashboard.add_widgets(
            cloudwatch.SingleValueWidget(
                title="Lambda invocations",
                metrics=[scorer_function.metric_invocations()],
            ),
            cloudwatch.SingleValueWidget(
                title="Lambda errors",
                metrics=[scorer_function.metric_errors()],
            ),
            cloudwatch.SingleValueWidget(
                title="High-risk incidents",
                metrics=[high_risk_metric],
            ),
        )

        dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Lambda duration",
                left=[
                    scorer_function.metric_duration(statistic="average"),
                    scorer_function.metric_duration(statistic="p99"),
                ],
            ),
            cloudwatch.GraphWidget(
                title="Queue health",
                left=[
                    processing_queue.metric_approximate_number_of_messages_visible(),
                    dead_letter_queue.metric_approximate_number_of_messages_visible(),
                ],
            ),
        )

        CfnOutput(
            self,
            "DashboardName",
            value=dashboard.dashboard_name,
        )
