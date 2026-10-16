import pandas as pd

EDUCATION_ORDER = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
ONE_HOT = ["Department", "Gender", "Job_Title"]


def encode_features(df, target="Performance_Score"):
    """Return a fully numeric feature matrix (target and Employee_ID removed).

    - Education_Level: ordinal, the order is real
    - Department, Gender, Job_Title: one-hot, they have no order
    - Hire_Date: parsed as a date, only the year is kept
    - Resigned: 0/1
    """
    X = df.drop(columns=[target, "Employee_ID"]).copy()
    X["Hire_Year"] = pd.to_datetime(X.pop("Hire_Date")).dt.year
    X["Education_Level"] = X["Education_Level"].map(EDUCATION_ORDER)
    X["Resigned"] = X["Resigned"].astype(int)
    return pd.get_dummies(X, columns=ONE_HOT, dtype=int)
