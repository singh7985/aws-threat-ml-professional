from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_features.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_features.csv"


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the training and testing datasets."""

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"Training dataset was not found: {TRAIN_DATA_PATH}")

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f"Testing dataset was not found: {TEST_DATA_PATH}")

    training_data = pd.read_csv(TRAIN_DATA_PATH)
    testing_data = pd.read_csv(TEST_DATA_PATH)

    return training_data, testing_data


def main() -> None:
    training_data, testing_data = load_datasets()

    print("Dataset loaded successfully.")
    print(f"Training rows: {len(training_data)}")
    print(f"Testing rows: {len(testing_data)}")

    print("\nTraining columns:")
    print(training_data.columns.tolist())

    print("\nFirst five records:")
    print(training_data.head())


if __name__ == "__main__":
    main()
