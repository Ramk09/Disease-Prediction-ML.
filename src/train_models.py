
from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = {
    "breast_cancer": DATA_DIR / "breast_cancer.csv",
    "heart_disease": DATA_DIR / "heart_disease.csv",
    "diabetes": DATA_DIR / "diabetes.csv",
}

# Reuse existing saved models when available.
SKIP_EXISTING_MODELS = True


def build_preprocessor(X):
    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    return ColumnTransformer(transformers)


def build_models(dataset_name):
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        ),
    }

    if dataset_name == "diabetes":
        # Linear SVM scales better to the large CDC dataset.
        # Calibration supplies probabilities for ROC-AUC evaluation.
        models["svm"] = CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                random_state=42,
                max_iter=3000,
                dual="auto",
            ),
            method="sigmoid",
            cv=3,
        )
    else:
        # RBF SVM is retained for the smaller datasets.
        models["svm"] = SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=42,
        )

    return models


def train_dataset(dataset_name, csv_path):
    print(f"\n{'=' * 65}")
    print(f"DATASET: {dataset_name.upper()}")
    print("=" * 65)

    df = pd.read_csv(csv_path)

    if "target" not in df.columns:
        raise ValueError("The dataset must contain a 'target' column.")

    df = df.dropna(subset=["target"]).copy()

    X = df.drop(columns=["target"])
    y = df["target"].astype(str)

    # Remove columns that contain no usable values.
    X = X.dropna(axis=1, how="all")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    if len(label_encoder.classes_) < 2:
        raise ValueError("At least two target classes are required.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    results = []
    preprocessor = build_preprocessor(X_train)

    for model_name, estimator in build_models(dataset_name).items():
        model_path = MODEL_DIR / (
            f"{dataset_name}_{model_name}.joblib"
        )

        if SKIP_EXISTING_MODELS and model_path.exists():
            print(f"\nSkipping existing model: {model_name}")
            continue

        print(f"\nTraining {model_name}...")

        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("classifier", estimator),
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        metrics = {
            "dataset": dataset_name,
            "model": model_name,
            "accuracy": accuracy_score(y_test, predictions),
            "precision_weighted": precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "recall_weighted": recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
            "f1_weighted": f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0,
            ),
        }

        # ROC-AUC uses probabilities when supported.
        try:
            probabilities = pipeline.predict_proba(X_test)

            if len(label_encoder.classes_) == 2:
                metrics["roc_auc"] = roc_auc_score(
                    y_test,
                    probabilities[:, 1],
                )
            else:
                metrics["roc_auc"] = roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                )
        except (ValueError, AttributeError, IndexError):
            metrics["roc_auc"] = None

        # Save the trained pipeline and the metadata required by the app.
        joblib.dump(
            {
                "pipeline": pipeline,
                "label_encoder": label_encoder,
                "feature_columns": X.columns.tolist(),
                "class_names": label_encoder.classes_.tolist(),
            },
            model_path,
        )

        results.append(metrics)

        print(
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision_weighted']:.4f} | "
            f"Recall: {metrics['recall_weighted']:.4f} | "
            f"F1: {metrics['f1_weighted']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']}"
        )

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=range(len(label_encoder.classes_)),
        )

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=label_encoder.classes_,
        )
        display.plot(values_format="d", xticks_rotation=30)

        plt.title(f"{dataset_name} - {model_name}")
        plt.tight_layout()

        plot_path = OUTPUT_DIR / (
            f"{dataset_name}_{model_name}_confusion_matrix.png"
        )
        plt.savefig(plot_path, dpi=150)
        plt.close()

    return results


def main():
    all_results = []

    for dataset_name, csv_path in DATASETS.items():
        if not csv_path.exists():
            print(f"Missing dataset: {csv_path}")
            continue

        try:
            results = train_dataset(dataset_name, csv_path)
            all_results.extend(results)
        except Exception as error:
            print(f"\nERROR in {dataset_name}: {error}")
            print("Other datasets will still be attempted.")

    results_path = OUTPUT_DIR / "model_metrics.csv"

    # Preserve earlier metric rows for models skipped this run.
    if results_path.exists():
        previous = pd.read_csv(results_path)

        if all_results:
            new_results = pd.DataFrame(all_results)
            keys = ["dataset", "model"]

            previous = previous[
                ~previous.set_index(keys).index.isin(
                    new_results.set_index(keys).index
                )
            ]
            combined = pd.concat(
                [previous, new_results],
                ignore_index=True,
            )
        else:
            combined = previous

        combined.to_csv(results_path, index=False)

    elif all_results:
        pd.DataFrame(all_results).to_csv(
            results_path,
            index=False,
        )

    if results_path.exists():
        print("\nMODEL COMPARISON")
        results_df = pd.read_csv(results_path)

        print(
            results_df.sort_values(
                ["dataset", "f1_weighted"],
                ascending=[True, False],
            ).to_string(index=False)
        )

    print(f"\nModels directory: {MODEL_DIR}")
    print(f"Outputs directory: {OUTPUT_DIR}")
    print("Training run finished.")


if __name__ == "__main__":
    main()
