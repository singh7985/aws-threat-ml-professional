import argparse
import shutil
from pathlib import Path


def main():
    # Parsed so the script tolerates SageMaker-injected arguments, even though
    # every path below is fixed by the container's mount points.
    argparse.ArgumentParser().parse_known_args()

    input_dir = Path("/opt/ml/processing/input")
    train_output = Path("/opt/ml/processing/train")
    test_output = Path("/opt/ml/processing/test")
    
    train_output.mkdir(parents=True, exist_ok=True)
    test_output.mkdir(parents=True, exist_ok=True)

    # In our specific local workflow for the initial prototype, we 
    # already explicitly uploaded `train_features.csv` and `test_features.csv` to 
    # the S3 bucket's `mlops/input` folder directly bypassing native splitting inside the pipeline.
    
    train_path = input_dir / "train_features.csv"
    test_path = input_dir / "test_features.csv"

    # Fail here rather than silently emitting empty output directories. A silent
    # no-op only surfaces later as a confusing "no CSV found" error in training.
    missing = [p.name for p in (train_path, test_path) if not p.exists()]
    if missing:
        available = sorted(p.name for p in input_dir.glob("*")) if input_dir.exists() else []
        raise FileNotFoundError(
            f"Missing required input file(s) {missing} in {input_dir}. Found: {available}"
        )

    shutil.copy(train_path, train_output / "train_features.csv")
    shutil.copy(test_path, test_output / "test_features.csv")

if __name__ == "__main__":
    main()
