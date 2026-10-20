import joblib
import pandas as pd


MODEL_PATH = "models/heart_disease_pipeline.joblib"
DATA_PATH = "data/processed/heart_disease_processed.csv"


def test_model_can_predict():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["target"])

    prediction = model.predict(
        X.iloc[[0]]
    )

    assert prediction[0] in [0, 1]


def test_probability_is_valid():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["target"])

    probability = model.predict_proba(
        X.iloc[[0]]
    )[0, 1]

    assert 0 <= probability <= 1