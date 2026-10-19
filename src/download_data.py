from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


def download_data():
    """Download the official UCI Heart Disease dataset."""

    print("Downloading UCI Heart Disease dataset...")

    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()

    df = pd.concat([X, y], axis=1)

    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "heart_disease.csv"

    df.to_csv(output_path, index=False)

    print("\nDownload successful!")
    print("Dataset shape:", df.shape)
    print("Saved to:", output_path)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    download_data()
