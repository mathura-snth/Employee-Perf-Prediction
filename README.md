# Employee Performance Prediction under Class Imbalance

Multiclass classification of employee performance (score 1–5) on a heavily imbalanced dataset, comparing resampling, class weighting and tree-based models, with and without `Monthly_Salary`, the one feature that carries all the signal.

**Main result:** `Monthly_Salary` is derived from the target (target leakage). With it, XGBoost reaches 0.93 macro F1; without it, no model beats a plain logistic regression at about 0.19 macro F1 (chance level), and rebalancing only moves recall between classes.

## Goals

- Build a clean, reproducible pipeline for a multiclass problem where accuracy is misleading.
- Compare imbalance-handling strategies on the same split: random undersampling, SMOTE, class weights, Balanced Random Forest, XGBoost with sample weights.
- Implement multinomial (softmax) logistic regression from scratch in NumPy and validate it against scikit-learn.
- Tune XGBoost with cross-validated search, scored on macro F1.
- Report uncertainty (bootstrap intervals) and be explicit about the limits of the study.

## Data

[Employee Performance and Productivity Data](https://www.kaggle.com/datasets/mexwell/employee-performance-and-productivity-data) (Kaggle, `mexwell`): 100,000 employees, 20 columns, target `Performance_Score` (1–5), balanced in its original form (about 20,000 rows per class).

The raw file is **not** versioned. Download it and save it as `data/raw.csv`:

```bash
pip install kaggle   # needs an API token in ~/.kaggle/kaggle.json
kaggle datasets download -d mexwell/employee-performance-and-productivity-data -p data --unzip
# then rename the extracted CSV to data/raw.csv
```

Because the original classes are balanced, an imbalance is created artificially: 2% of class 1 and 20% of class 2 are kept, giving 64,271 rows (402 / 4,002 / 19,999 / 19,940 / 19,928 for scores 1 to 5).

## Method

1. Loading and sanity checks
2. Exploratory data analysis
3. Artificial imbalance creation
4. Encoding and correlation analysis
5. Feature selection: two sets, with and without `Monthly_Salary`
6. Stratified 80/20 split, scaling fitted on train only
7. Baseline logistic regression
8. Imbalance handling (class weights, undersampling, SMOTE)
9. Softmax regression from scratch (weighted cross-entropy + L2, gradient checked numerically, compared with scikit-learn)
10. Tree models (Balanced Random Forest, Random Forest, XGBoost)
11. XGBoost randomized search (5-fold stratified CV, macro F1)
12. Final comparison with 95% bootstrap intervals, per-class recall, feature importance, export to `results/results.json`

Metrics: macro F1 as the main score, per-class recall and row-normalized confusion matrices. Accuracy is reported but not used to rank models.

## Findings

- No missing values and no duplicates. Only `Monthly_Salary` relates to the score; all other features are flat across scores (means within ±0.02 standard deviations), and their correlation with the score is below 0.011.
- **Leakage:** salary takes 28 distinct values and equals `base(Job_Title) × (1 + 0.1 × Performance_Score)` (checked on all 7 job titles). Pearson correlation only reaches 0.35 because the relation depends on the job title. Salary alone is ambiguous across job titles, so models without `Job_Title` get 0.98 accuracy, not 1.00.
- Without salary, nothing is learnable: all models are at 0.15–0.20 macro F1. The data look synthetic (uniform features).
- Rebalancing does not add information. It trades precision for recall: class 1 recall rises (about 0.45) while the other classes drop, and macro F1 goes down without salary.
- With salary, weighting XGBoost raises class 1 recall (0.64 → 0.80, paired difference +0.16 [0.09, 0.25]) but lowers macro F1 (0.933 → 0.909, paired difference −0.024 [−0.043, −0.006]). Both effects are distinguishable from 0: weighting is a real trade-off, not a gain.
- Tuning XGBoost gains about 0.04 macro F1 in repeated CV with salary, but on the test set tuned vs weighted is +0.017 [−0.001, 0.035], not distinguishable from 0, and tuned (0.926) does not beat the unweighted default (0.933).

## Results

Test set: 12,855 rows, of which 80 are class 1. Macro F1 with 95% bootstrap interval (1,000 resamples of the test rows). Full table, recalls and paired differences are in `notebooks/12-comparison.ipynb` and `results/results.json`.

| Model | With salary | Without salary |
|---|---|---|
| LogReg | 0.215 [0.210, 0.220] | 0.195 [0.190, 0.200] |
| LogReg balanced | 0.190 [0.184, 0.196] | 0.146 [0.140, 0.152] |
| LogReg + undersampling | 0.205 [0.199, 0.211] | 0.168 [0.163, 0.174] |
| LogReg + SMOTE | 0.190 [0.183, 0.196] | 0.147 [0.141, 0.153] |
| Softmax from scratch (balanced) | 0.189 [0.184, 0.196] | 0.146 [0.140, 0.152] |
| Balanced RF | 0.467 [0.460, 0.475] | 0.183 [0.177, 0.188] |
| RF balanced | 0.606 [0.592, 0.623] | 0.192 [0.187, 0.197] |
| XGBoost unweighted | 0.933 [0.914, 0.950] | 0.192 [0.187, 0.197] |
| XGBoost weighted | 0.909 [0.890, 0.924] | 0.189 [0.182, 0.195] |
| XGBoost tuned | 0.926 [0.907, 0.943] | 0.192 [0.186, 0.197] |

Paired bootstrap differences (same resampled rows for both models; 95% interval):

| Feature set | A − B | Metric | Difference | Distinguishable from 0 |
|---|---|---|---|---|
| salary | tuned XGBoost − weighted XGBoost | macro F1 | +0.017 [−0.001, 0.035] | no |
| salary | weighted XGBoost − unweighted XGBoost | macro F1 | −0.024 [−0.043, −0.006] | yes |
| salary | weighted XGBoost − unweighted XGBoost | recall class 1 | +0.163 [0.088, 0.247] | yes |
| salary | tuned XGBoost − LogReg balanced | macro F1 | +0.736 [0.716, 0.753] | yes |
| no salary | tuned XGBoost − LogReg | macro F1 | −0.003 [−0.010, 0.004] | no |
| no salary | LogReg − LogReg balanced | macro F1 | +0.049 [0.043, 0.054] | yes |

Feature importance (tuned XGBoost, gain): `Monthly_Salary` 0.641, every other feature at or below 0.027.

## Limitations

- **Leakage:** the with-salary scores measure how well a model decodes how salary was built from the score, not how well performance can be predicted.
- **Synthetic-looking data:** features other than salary are uninformative, so the no-salary results say little about real employee data.
- **Small minority class:** class 1 has 80 test rows, so its recall moves in steps of 0.0125 and its 95% interval is about ±0.11 wide. Unpaired intervals on class 1 overlap, so class 1 differences are only claimed where the paired bootstrap supports them.
- **Artificial imbalance:** the imbalance is created by subsampling a balanced dataset, not observed.
- **Single split:** the comparison uses one stratified split; the intervals reflect test-set sampling only, not split-to-split variation.

## Project structure

```
.
├── data/raw.csv          # not versioned, see "Data"
├── notebooks/            # one notebook per step, each ends with a Notes cell
├── results/              # xgb_best_params.json, results.json
├── src/
│   ├── config.py         # seed, paths
│   ├── imbalance.py      # artificial imbalance
│   ├── features.py       # encoding, feature sets
│   ├── split.py          # stratified split, scaling
│   ├── data.py           # load_splits()
│   ├── metrics.py        # evaluation, bootstrap intervals
│   ├── softmax.py        # softmax regression from scratch
│   └── models.py         # class-weighted XGBoost wrapper
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On macOS, XGBoost needs OpenMP: `brew install libomp`. Run the notebooks in order from the `notebooks/` folder.

## Acknowledgements

Dataset: *Employee Performance and Productivity Data* by `mexwell` on Kaggle. This is an independent re-implementation and extension of the analyses people have published on it; check the dataset's license on its Kaggle page before redistributing anything derived from it.