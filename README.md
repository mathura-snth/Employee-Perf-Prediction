# Employee Performance Prediction under Class Imbalance

Multiclass classification of employee performance (score 1–5) on a heavily imbalanced dataset, comparing resampling, class weighting and tree-based models, with and without the one feature that carries most of the signal (`Monthly_Salary`).

> **Status:** work in progress.

## Goals

- Build a clean, reproducible pipeline for a multiclass problem where accuracy is misleading.
- Compare imbalance-handling strategies on the same split: random undersampling, SMOTE, class weights, Balanced Random Forest, XGBoost with sample weights.
- Implement multinomial (softmax) logistic regression from scratch in NumPy and validate it against scikit-learn.
- Tune XGBoost with cross-validated search, scored on macro F1.

## Data

[Employee Performance and Productivity Data](https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data) (Kaggle, `mexwell`): 100,000 employees, 20 columns, target `Performance_Score` (1–5), roughly balanced in its original form.

The raw file is **not** versioned. Download it and place it at `data/raw/`:

```bash
pip install kaggle   # needs an API token in ~/.kaggle/kaggle.json
kaggle datasets download -d mexwell/employee-performance-and-productivity-data -p data/raw --unzip
```

Because the original classes are balanced, we create an imbalance artificially (only a fraction of classes 1 and 2 is kept) so that imbalance-handling methods can be compared.

## Method

1. Loading and sanity checks
2. Exploratory data analysis
3. Artificial imbalance creation
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

## Project structure

```
.
├── data/raw/          # not versioned, see "Data"
├── notebooks/
├── src/
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```