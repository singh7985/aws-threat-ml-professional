#!/usr/bin/env python3


import os

import aws_cdk as cdk
from dotenv import load_dotenv
from stacks.foundation_stack import FoundationStack

load_dotenv()
app = cdk.App()
environment_name = app.node.try_get_context("environment") or os.getenv("APP_ENV", "dev")
project_name = os.getenv("PROJECT_NAME", "autonomous-cloud-threat-ml")

FoundationStack(
    app,
    f"ThreatMlFoundation-{environment_name}",
    project_name=project_name,
    environment_name=environment_name,
    alert_email=os.getenv("ALERT_EMAIL", ""),
    env=cdk.Environment(
        account=os.getenv("CDK_DEFAULT_ACCOUNT"),
        region=os.getenv("CDK_DEFAULT_REGION") or os.getenv("AWS_REGION", "us-east-1"),
    ),
)

app.synth()
