# Employee Performance Prediction under Class Imbalance

Multiclass classification of employee performance (score 1–5) on a heavily imbalanced dataset, comparing resampling, class weighting and tree-based models, with and without the one feature that carries most of the signal (`Monthly_Salary`).

> **Status:** work in progress. Steps 1–3 are done (loading, EDA, imbalance creation).

## Goals

- Build a clean, reproducible pipeline for a multiclass problem where accuracy is misleading.
- Compare imbalance-handling strategies on the same split: random undersampling, SMOTE, class weights, Balanced Random Forest, XGBoost with sample weights.
- Implement multinomial (softmax) logistic regression from scratch in NumPy and validate it against scikit-learn.
- Tune XGBoost with cross-validated search, scored on macro F1.

## Data

[Employee Performance and Productivity Data](https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data) (Kaggle, `mexwell`): 100,000 employees, 20 columns, target `Performance_Score` (1–5), roughly balanced in its original form.

The raw file is **not** versioned. Download it and save it as `data/raw.csv`:

```bash
pip install kaggle   # needs an API token in ~/.kaggle/kaggle.json
kaggle datasets download -d mexwell/employee-performance-and-productivity-data -p data --unzip
# then rename the extracted CSV to data/raw.csv
```

Because the original classes are balanced, we create an imbalance artificially (only a fraction of classes 1 and 2 is kept) so that imbalance-handling methods can be compared.

## Method

1. Loading and sanity checks (done)
2. Exploratory data analysis (done)
3. Artificial imbalance creation (done)
4. Encoding and correlation analysis
5. Feature selection (with and without `Monthly_Salary`)
6. Stratified train/test split, scaling fitted on train only
7. Baseline logistic regression
8. Imbalance handling (undersampling, SMOTE, class weights)
9. Softmax regression from scratch
10. Tree models (Balanced Random Forest, Random Forest, XGBoost)
11. XGBoost hyperparameter search
12. Comparison, feature importance, export of results

*Metrics: macro F1 as the main score, per-class recall and row-normalized confusion matrices. Accuracy is reported but not used to rank models.*

## Findings so far

- The original data has no missing values and no duplicates, and the scores are balanced (about 20,000 rows each).
- Only `Monthly_Salary` relates to the score. All other numeric features are flat (means per score within ±0.02 standard deviations) and all categorical shares stay between 0.19 and 0.21.
- Salary appears to be derived from the target: it takes 28 distinct values and follows, within each job title, a grid of the form `base(Job_Title) × (1 + 0.1 × Performance_Score)`. This is target leakage by construction, which is why every model is compared with and without salary.
- After the imbalance step: 64,271 rows, with 402 / 4,002 / 19,999 / 19,940 / 19,928 rows for scores 1 to 5.

## Project structure

```
.
├── data/raw.csv       # not versioned, see "Data"
├── notebooks/
├── src/               # config.py, imbalance.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```