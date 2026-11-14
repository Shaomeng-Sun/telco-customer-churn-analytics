from telco_churn.ml.common import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    ID_COLUMN,
    load_modeling_data,
    make_train_validation_test_split,
)


def test_modeling_dataset_integrity():
    df = load_modeling_data()

    assert len(df) == 7043
    assert df[ID_COLUMN].nunique() == 7043

    assert set(FEATURE_COLUMNS).issubset(df.columns)

    assert df[TARGET_COLUMN].notna().all()
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})

    assert df[TARGET_COLUMN].nunique() == 2


def test_train_validation_test_splits():
    df = load_modeling_data()

    splits = make_train_validation_test_split(
        df,
        random_state=42,
    )

    train_ids = set(splits["X_train"].index)
    validation_ids = set(splits["X_validation"].index)
    test_ids = set(splits["X_test"].index)

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)

    total_rows = (
        len(splits["X_train"])
        + len(splits["X_validation"])
        + len(splits["X_test"])
    )

    assert total_rows == len(df)
