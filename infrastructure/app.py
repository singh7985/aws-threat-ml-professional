#!/usr/bin/env python3
import os

import aws_cdk as cdk
from dotenv import load_dotenv
from stacks.data_stack import DataStack
from stacks.foundation_stack import FoundationStack
from stacks.monitoring_stack import MonitoringStack
from stacks.scorer_stack import ScorerStack

load_dotenv()
app = cdk.App()
environment_name = app.node.try_get_context("environment") or os.getenv("APP_ENV", "dev")
project_name = os.getenv("PROJECT_NAME", "autonomous-cloud-threat-ml")

env = cdk.Environment(
    account=os.getenv("CDK_DEFAULT_ACCOUNT"),
    region=os.getenv("CDK_DEFAULT_REGION") or os.getenv("AWS_REGION", "us-east-1"),
)

data = DataStack(
    app,
    f"ThreatMlData-{environment_name}",
    env=env,
)

foundation = FoundationStack(
    app,
    f"ThreatMlFoundation-{environment_name}",
    project_name=project_name,
    environment_name=environment_name,
    alert_email=os.getenv("ALERT_EMAIL", ""),
    env=env,
)

scorer = ScorerStack(
    app,
    f"ThreatMlScorer-{environment_name}",
    processing_queue=foundation.processing_queue,
    incidents_table=data.table,
    alerts_topic=foundation.alerts_topic,
    project_name=project_name,
    environment_name=environment_name,
    env=env,
)

monitoring = MonitoringStack(
    app,
    f"ThreatMlMonitoring-{environment_name}",
    scorer_function=scorer.scorer_function,
    processing_queue=foundation.processing_queue,
    dead_letter_queue=foundation.dead_letter_queue,
    alerts_topic=foundation.alerts_topic,
    env=env,
)

monitoring.add_stack_dependency(scorer)

app.synth()
