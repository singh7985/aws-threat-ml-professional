# Dataset for Intelligent AWS Cloud Threat Detection and Automated Incident Response

## What the data means

Each row represents one action performed inside an AWS account, similar to a simplified CloudTrail event.

Examples:

- A developer lists S3 buckets during normal work hours.
- An ML engineer checks an EC2 instance.
- An unknown source creates an IAM key at 3 AM.
- Someone stops CloudTrail logging.
- A security group or S3 bucket is exposed publicly.

This is synthetic training data. It contains no real customer, employee, account, or IP information.

## Files

- `data/raw/security_events.jsonl` — best input for the current Python starter project.
- `data/raw/security_events.csv` — easy to inspect in Excel or VS Code.
- `data/processed/ml_features.csv` — numerical features for a first ML model.
- `data/processed/train_events.csv` and `test_events.csv` — time-based raw split.
- `data/processed/train_features.csv` and `test_features.csv` — time-based ML split.
- `docs/dataset_overview.xlsx` — data dictionary and visual summaries.

## Size

- 5,000 events
- 4,500 normal
- 500 suspicious
- 90% normal / 10% suspicious
- 80% training / 20% testing

## Copy it into the project

Copy the `data` folder into your project root. The main file should be:

`data/raw/security_events.jsonl`

## Inspect the first five records

PowerShell:

```powershell
Get-Content data\raw\security_events.jsonl -TotalCount 5
```

macOS/Linux/WSL:

```bash
head -n 5 data/raw/security_events.jsonl
```

## First ML target

For supervised learning, predict `label_binary`:

- `0` = normal
- `1` = suspicious

For Isolation Forest, train mostly or only on rows where `label_binary == 0`, then evaluate using the suspicious rows.
