import pandas as pd

from src.config import SEED


def create_imbalance(df, target="Performance_Score", keep=None, seed=SEED):
    if keep is None:
        keep = {1: 0.02, 2: 0.20}

    parts = []
    for cls, group in df.groupby(target):
        n = int(len(group) * keep.get(cls, 1.0))
        parts.append(group.sample(n=n, random_state=seed))

    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
