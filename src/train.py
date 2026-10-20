from pathlib import Path

import joblib
import mlflow
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import build_preprocessor


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

RANDOM_STATE = 42

DATA_PATH = Path(
    "data/processed/heart_disease_processed.csv"
)

MODEL_DIR = Path("models")

MODEL_PATH = (
    MODEL_DIR / "heart_disease_pipeline.joblib"
)

EXPERIMENT_NAME = (
    "heart-disease-classification"
)


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(model, X_test, y_test):
    """Evaluate a fitted binary classification pipeline."""

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    metrics = {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
    }

    return metrics


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

def train():
    """Train candidate models, track them with MLflow,
    and save the best production pipeline.
    """

    # -----------------------------------------------------
    # Validate dataset
    # -----------------------------------------------------

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    print(
        "Loading dataset:",
        DATA_PATH.resolve(),
    )

    df = pd.read_csv(DATA_PATH)

    if "target" not in df.columns:
        raise ValueError(
            "Processed dataset does not contain "
            "the 'target' column."
        )

    if "num" in df.columns:
        raise ValueError(
            "Target leakage detected: "
            "'num' must not be present."
        )

    print("Dataset shape:", df.shape)

    # -----------------------------------------------------
    # Features / target
    # -----------------------------------------------------

    X = df.drop(
        columns=["target"]
    )

    y = df["target"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    print(
        "Training samples:",
        len(X_train),
    )

    print(
        "Test samples:",
        len(X_test),
    )

    # -----------------------------------------------------
    # Candidate models
    #
    # Replace these values with your Task 2 best
    # hyperparameters if they are different.
    # -----------------------------------------------------

    models = {
        "logistic_regression": (
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=RANDOM_STATE,
            )
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }

    # -----------------------------------------------------
    # MLflow
    # -----------------------------------------------------

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model = None
    best_score = -1.0
    best_name = None

    # -----------------------------------------------------
    # Train all candidate models
    # -----------------------------------------------------

    for model_name, classifier in models.items():

        print(
            f"\nTraining: {model_name}"
        )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(),
                ),
                (
                    "classifier",
                    classifier,
                ),
            ]
        )

        with mlflow.start_run(
            run_name=model_name
        ):

            # ---------------------------------------------
            # Train
            # ---------------------------------------------

            pipeline.fit(
                X_train,
                y_train,
            )

            # ---------------------------------------------
            # Evaluate
            # ---------------------------------------------

            metrics = evaluate(
                pipeline,
                X_test,
                y_test,
            )

            # ---------------------------------------------
            # Log general parameters
            # ---------------------------------------------

            mlflow.log_param(
                "model_name",
                model_name,
            )

            mlflow.log_param(
                "random_state",
                RANDOM_STATE,
            )

            mlflow.log_param(
                "test_size",
                0.20,
            )

            # ---------------------------------------------
            # Log classifier parameters
            # ---------------------------------------------

            classifier_params = (
                classifier.get_params()
            )

            for key, value in (
                classifier_params.items()
            ):

                if isinstance(
                    value,
                    (
                        str,
                        int,
                        float,
                        bool,
                    ),
                ) or value is None:

                    mlflow.log_param(
                        f"classifier_{key}",
                        value,
                    )

            # ---------------------------------------------
            # Log metrics
            # ---------------------------------------------

            mlflow.log_metrics(
                metrics
            )

            # ---------------------------------------------
            # Save model using Joblib
            #
            # We intentionally use Joblib rather than
            # mlflow.sklearn.log_model() because the
            # installed MLflow version rejects numpy.dtype
            # as an untrusted type during model logging.
            # ---------------------------------------------

            run_model_path = (
                MODEL_DIR
                / f"{model_name}_pipeline.joblib"
            )

            joblib.dump(
                pipeline,
                run_model_path,
            )

            # ---------------------------------------------
            # Log Joblib model as MLflow artifact
            # ---------------------------------------------

            mlflow.log_artifact(
                str(run_model_path),
                artifact_path="models",
            )

            # ---------------------------------------------
            # Display results
            # ---------------------------------------------

            print(
                "Metrics:"
            )

            for metric_name, value in (
                metrics.items()
            ):
                print(
                    f"  {metric_name}: "
                    f"{value:.4f}"
                )

            print(
                "Model artifact:",
                run_model_path,
            )

            # ---------------------------------------------
            # Track best model
            # ---------------------------------------------

            if (
                metrics["roc_auc"]
                > best_score
            ):

                best_score = (
                    metrics["roc_auc"]
                )

                best_model = pipeline

                best_name = model_name

    # -----------------------------------------------------
    # Safety check
    # -----------------------------------------------------

    if best_model is None:
        raise RuntimeError(
            "Training completed without "
            "selecting a model."
        )

    # -----------------------------------------------------
    # Save production model
    # -----------------------------------------------------

    joblib.dump(
        best_model,
        MODEL_PATH,
    )

    # -----------------------------------------------------
    # Verify serialized model
    # -----------------------------------------------------

    loaded_model = joblib.load(
        MODEL_PATH
    )

    sample_prediction = (
        loaded_model.predict(
            X_test.iloc[[0]]
        )[0]
    )

    sample_probability = (
        loaded_model.predict_proba(
            X_test.iloc[[0]]
        )[0, 1]
    )

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print(
        "\n"
        "======================================"
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        "Best model:",
        best_name,
    )

    print(
        "Best ROC-AUC:",
        round(
            best_score,
            4,
        ),
    )

    print(
        "Production model:",
        MODEL_PATH.resolve(),
    )

    print(
        "Model file exists:",
        MODEL_PATH.exists(),
    )

    print(
        "Pipeline steps:",
        list(
            loaded_model.named_steps.keys()
        ),
    )

    print(
        "Test reload prediction:",
        int(sample_prediction),
    )

    print(
        "Test reload probability:",
        round(
            float(sample_probability),
            4,
        ),
    )


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    train()