import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import SEED


def split_indices(y, test_size=0.2, seed=SEED):
    return train_test_split(
        np.arange(len(y)), test_size=test_size, stratify=y, random_state=seed
    )


def scale(X_train, X_test):
    scaler = StandardScaler().fit(X_train)
    X_train_s = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_s = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    return X_train_s, X_test_s, scaler
