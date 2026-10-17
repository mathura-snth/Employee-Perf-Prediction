import pandas as pd

from src.config import DATA_PATH
from src.features import encode_features, select_feature_sets
from src.imbalance import create_imbalance
from src.split import scale, split_indices

TARGET = "Performance_Score"


def load_splits(with_job_title=False):
    """Run the whole preparation once and return the splits.
    """
    df = create_imbalance(pd.read_csv(DATA_PATH), TARGET)
    y = df[TARGET]
    sets = select_feature_sets(encode_features(df, TARGET), with_job_title)
    idx_train, idx_test = split_indices(y)

    data = {}
    for name, X in sets.items():
        X_train, X_test = X.iloc[idx_train], X.iloc[idx_test]
        X_train_s, X_test_s, _ = scale(X_train, X_test)
        data[name] = {
            "X_train": X_train, "X_test": X_test,
            "X_train_s": X_train_s, "X_test_s": X_test_s,
        }
    return data, y.iloc[idx_train], y.iloc[idx_test]
