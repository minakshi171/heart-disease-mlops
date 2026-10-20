import pandas as pd


DATA_PATH = "data/processed/heart_disease_processed.csv"


def test_dataset_exists_and_not_empty():
    df = pd.read_csv(DATA_PATH)

    assert not df.empty
    assert len(df) > 0


def test_target_exists():
    df = pd.read_csv(DATA_PATH)

    assert "target" in df.columns


def test_target_is_binary():
    df = pd.read_csv(DATA_PATH)

    assert set(df["target"].unique()).issubset(
        {0, 1}
    )


def test_no_target_leakage_column():
    df = pd.read_csv(DATA_PATH)

    assert "num" not in df.columns