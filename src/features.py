import pandas as pd

EDUCATION_ORDER = {"High School": 0, "Bachelor": 1, "Master": 2, "PhD": 3}
ONE_HOT = ["Department", "Gender", "Job_Title"]


def encode_features(df, target="Performance_Score"):
    """
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


def select_feature_sets(X, with_job_title=False):
    """
    Selection uses no information from the target, so it can safely be done
    before the train/test split.
    - Hire_Year is dropped: it is redundant with Years_At_Company (corr -0.99).
    - Job_Title dummies are dropped by default (flat on their own, and they
      decode the salary). with_job_title=True keeps them, for the ablation.
    - "salary" keeps Monthly_Salary, "no_salary" removes it.
    """
    X = X.drop(columns=["Hire_Year"])
    if not with_job_title:
        X = X.drop(columns=[c for c in X.columns if c.startswith("Job_Title_")])
    return {"salary": X, "no_salary": X.drop(columns=["Monthly_Salary"])}