
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer
from ucimlrepo import fetch_ucirepo

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def save_breast_cancer():
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy()
    df["target"] = df["target"].map(
        {0: "malignant", 1: "benign"}
    )

    path = DATA_DIR / "breast_cancer.csv"
    df.to_csv(path, index=False)

    print("\nBREAST CANCER")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Saved: {path}")


def save_heart_disease():
    dataset = fetch_ucirepo(id=45)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    df = features.copy()
    df["target"] = targets.iloc[:, 0].values

    # UCI represents missing values in some fields as '?'.
    df = df.replace("?", pd.NA)

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Target 0 means no disease; values 1-4 indicate disease.
    df["target"] = df["target"].map(
        lambda value: (
            "no_disease"
            if pd.notna(value) and value == 0
            else "disease"
            if pd.notna(value) and value > 0
            else pd.NA
        )
    )

    df = df.dropna(subset=["target"])
    path = DATA_DIR / "heart_disease.csv"
    df.to_csv(path, index=False)

    print("\nHEART DISEASE")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    print(f"Saved: {path}")


def save_diabetes():
    dataset = fetch_ucirepo(id=891)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    df = features.copy()
    df["target"] = targets.iloc[:, 0].values

    df["target"] = df["target"].map(
        {0: "no_diabetes", 1: "diabetes"}
    )

    path = DATA_DIR / "diabetes.csv"
    df.to_csv(path, index=False)

    print("\nDIABETES")
    print(f"Shape: {df.shape}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    print(f"Saved: {path}")


def main():
    print("Preparing CodeAlpha Task 4 datasets...")

    tasks = [
        ("Breast Cancer", save_breast_cancer),
        ("Heart Disease", save_heart_disease),
        ("Diabetes", save_diabetes),
    ]

    for name, task in tasks:
        try:
            task()
        except Exception as error:
            print(f"\nERROR while preparing {name}: {error}")
            print("We'll troubleshoot this dataset before proceeding.")

    print("\nDataset preparation finished. Check the results above.")


if __name__ == "__main__":
    main()
