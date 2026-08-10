import json

import boto3


def setup():
    iam = boto3.client("iam")
    role_name = "ThreatMlSageMakerExecutionRole"
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        iam.get_role(RoleName=role_name)
        print(f"Role {role_name} already exists.")
    except iam.exceptions.NoSuchEntityException:
        print(f"Creating role {role_name}...")
        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy)
        )
        
        # Attach permissions needed for SageMaker pipelines and related services
        policies = [
            "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            "arn:aws:iam::aws:policy/AmazonS3FullAccess", # We allow s3 write for outputs securely
        ]
        
        for policy in policies:
            print(f"Attaching {policy}...")
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn=policy
            )
        print("Role setup complete. Please wait 10 seconds for IAM propagation.")

if __name__ == "__main__":
    setup()
