# COMP9417 Group Project +bonus — Feature Learning Kernel Machines for Tabular Data

> **Submission note:** This project attempts the residual-weighted AGOP bonus extension.
> The `+bonus` marker is therefore present in the project title as required by the specification.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset Selection](#3-dataset-selection)
4. [Hyperparameter Configurations](#4-hyperparameter-configurations)
5. [Architecture: Zero-Leakage Data Pipeline](#5-architecture-zero-leakage-data-pipeline)
6. [Task A — Data Pipeline & Baseline Training](#6-task-a--data-pipeline--baseline-training)
7. [Task B — Interpretability Analysis](#7-task-b--interpretability-analysis)
8. [Task C — Sample-Size Scaling Experiment](#8-task-c--sample-size-scaling-experiment)
9. [Task D — Bonus: Residual-Weighted AGOP](#9-task-d--bonus-residual-weighted-agop)
10. [Step-by-Step Execution Guide](#10-step-by-step-execution-guide)
11. [Environment Setup](#11-environment-setup)
12. [Reproducibility Checklist](#12-reproducibility-checklist)

---

## 1. Project Overview

This project benchmarks **xRFM** (Recursive Feature Machine with adaptive tree structure)
against four baselines — XGBoost, Random Forest, MLP, and TabNet — across five tabular
datasets covering regression, multi-class classification, binary classification,
high-dimensional input, and mixed feature types.

The report accompanies this code and is structured to match the official specification:
Introduction → Methodology → Results → Discussion → Conclusion.

**Global random seed:** `42` (applied to all `train_test_split` calls, model initialisations,
and permutation importance loops).

---

## 2. Repository Structure

```
COMP9417-PR/
├── src/
│   ├── data_loader.py            # Downloads, cleans, and splits all 5 datasets
│   ├── train_trees_and_xrfm.py  # Trains xRFM, XGBoost, RF; saves preprocessor.pkl
│   ├── train_deep_learning.py   # Trains MLP, TabNet; loads preprocessor.pkl
│   └── __init__.py
├── notebooks/
│   ├── 01_Main_Results_Evaluation.ipynb   # Task A — main results table + figures
│   ├── 02_Interpretability_Plots.ipynb    # Task B — AGOP vs PCA vs MI vs PI
│   └── 03_Scaling_Experiment_Plots.ipynb  # Task C — scaling curves
├── data/                         # Created at runtime by data_loader.py
│   └── <dataset_name>/{train,val,test}/{X.csv, y.csv}
├── saved_models/                 # Created at runtime by training scripts
│   └── <dataset_name>/{xrfm.pkl, xgb.pkl, rf.pkl, mlp.pkl, tabnet/, preprocessor.pkl}
├── results/                      # Figures and CSV tables written by notebooks
├── requirements.txt
└── README.md
```

---

## 3. Dataset Selection

All five datasets are sourced from the **UCI Machine Learning Repository** via the
`ucimlrepo` Python interface. None overlap with the benchmark datasets used in the
xRFM paper (arxiv:2508.10053).

| # | Dataset | UCI ID | Task | n | d | Special property | Pre-processing applied |
|---|---------|--------|------|---|---|-----------------|------------------------|
| 1 | **Dry Bean** | 602 | Multi-class (7 classes) | 13,611 | 16 | Purely numerical; clear physical semantics — ideal for interpretability analysis | None beyond standardisation |
| 2 | **AI4I Predictive Maintenance** | 601 | Binary classification | 10,000 | 12 | **Mixed feature types** (numerical sensors + categorical machine type L/M/H) | Drop `UID`, `Product ID` (unique identifiers); OneHot-encode `Type` |
| 3 | **Online News Popularity** | 332 | Regression | 39,644 | 58 | **High-dimensional** (d > 50); noisy web-traffic statistics | Drop `url` (non-predictive string), `timedelta` (data-collection artefact) |
| 4 | **Bank Marketing** | 222 | Binary classification | 45,211 | 16 | **Severe class imbalance** (~12 % positive); mixed numerical + categorical features | Map target `{'yes': 1, 'no': 0}`; OneHot-encode 9 categorical columns |
| 5 | **Superconductivity** | 464 | Regression | 21,263 | 81 | **n > 10,000 and d > 50**; purely static physical/chemical attributes — no time-series leakage risk; purpose-built for the scaling experiment | None beyond standardisation |

**Data split:** 60 % train / 20 % validation / 20 % test for all datasets.
Classification tasks use stratified splits (`stratify=y`).
The validation set is used exclusively for hyperparameter tuning and early stopping;
the test set is touched only once for final metric reporting.

---

## 4. Hyperparameter Configurations

All hyperparameters were selected on the validation set. MLP and TabNet use
fixed locked configurations as required by the project rules (no grid search).

### xRFM

```python
rfm_params = {
    "model": {
        "kernel":         "l2",
        "bandwidth":      10.0,
        "exponent":       1.0,
        "diag":           False,       # full d×d AGOP matrix
        "bandwidth_mode": "constant",
    },
    "fit": {
        "reg":            1e-3,
        "iters":          3,
        "verbose":        False,
        "early_stop_rfm": True,
    },
}
xRFM(
    rfm_params     = rfm_params,
    tuning_metric  = "mse",            # MSE used universally to avoid a known
                                        # accuracy-metric dimension mismatch bug
                                        # in xRFM when val-set refilling triggers
    max_leaf_size  = 3000,
    n_trees        = 1,
    random_state   = 42,
)
```

### XGBoost

```python
XGBClassifier / XGBRegressor(
    n_estimators    = 300,
    max_depth       = 6,
    learning_rate   = 0.05,
    subsample       = 0.9,
    colsample_bytree= 0.9,
    random_state    = 42,
)
# eval_set=[(X_val, y_val)] triggers internal early stopping.
```

### Random Forest

```python
RandomForestClassifier / RandomForestRegressor(
    n_estimators = 300,
    random_state = 42,
    n_jobs       = -1,
)
```

### MLP (locked — no tuning)

```python
MLPClassifier / MLPRegressor(
    hidden_layer_sizes = (128, 64),
    alpha              = 0.001,
    max_iter           = 200,
    early_stopping     = True,
    random_state       = 42,
)
```

### TabNet (locked — no tuning)

```python
TabNetClassifier / TabNetRegressor(
    n_d                = 8,
    n_a                = 8,
    n_steps            = 3,
    gamma              = 1.3,
    max_epochs         = 50,
    patience           = 10,
    batch_size         = 1024,
    virtual_batch_size = 128,
    seed               = 42,
)
# eval_set=[(X_val, y_val)] is passed to .fit() to trigger early stopping.
```

---

## 5. Architecture: Zero-Leakage Data Pipeline

The pipeline enforces a strict **fit-on-train-only** rule at every stage:

```
data_loader.py
  └─ Downloads raw data from UCI
  └─ Applies dataset-specific cleaning (drop IDs, map targets)
  └─ Splits into train / val / test CSVs  ──────────► data/<dataset>/

train_trees_and_xrfm.py                               (Step 2)
  └─ Reads train CSV → fits StandardScaler + OneHotEncoder on X_train ONLY
  └─ Transforms X_val and X_test with the fitted objects (no re-fitting)
  └─ Saves preprocessor.pkl  ───────────────────────► saved_models/<dataset>/preprocessor.pkl
  └─ Trains xRFM, XGBoost, RF → saves .pkl models

train_deep_learning.py                                 (Step 3)
  └─ Loads preprocessor.pkl (inherits scaler/OHE — never re-fits)
  └─ Trains MLP, TabNet with identical feature matrices

notebooks/  (Steps 4a–4d)
  └─ Load all saved models + preprocessor.pkl
  └─ Transform test data → evaluate → write results/
```

**Key leakage-prevention decisions:**

- `StandardScaler` and `OneHotEncoder` objects are created and `.fit()` called
  inside `train_trees_and_xrfm.py` on `X_train` only. The fitted objects are
  serialised to `preprocessor.pkl` and loaded read-only by every subsequent script.
- `LabelEncoder` for Dry Bean (multiclass string labels → integers 0–6) is similarly
  fit on `y_train` only and saved as `label_encoder.pkl`.
- **Multiclass AUC for Dry Bean** is computed as
  `roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')`,
  consistent with the one-vs-rest convention stated in the report.

---

## 6. Task A — Data Pipeline & Baseline Training

**Script:** `src/train_trees_and_xrfm.py` + `src/train_deep_learning.py`

**Notebook:** `notebooks/01_Main_Results_Evaluation.ipynb`

The notebook loads all saved model `.pkl` / `.zip` artefacts and evaluates them on
the held-out test set. Metrics reported:

| Task type | Metrics |
|-----------|---------|
| Regression | RMSE |
| Binary classification | Accuracy, AUC-ROC |
| Multi-class classification | Accuracy, macro-averaged OvR AUC-ROC |
| All | Training time (s), per-sample inference time (ms) |

Results are written to `results/main_results_table.csv`.

---

## 7. Task B — Interpretability Analysis

**Notebook:** `notebooks/02_Interpretability_Plots.ipynb`
**Dataset:** Dry Bean (16 purely numerical features, 7 classes)

Four feature-importance methods are compared on identical scaled training data:

| Method | Type | What it measures |
|--------|------|-----------------|
| **AGOP diagonal** | Supervised, model-internal | Gradient-sensitivity of xRFM at each input coordinate |
| **PCA PC1 loadings** | Unsupervised | Direction of maximum input variance (ignores labels) |
| **Mutual Information** | Supervised, univariate | Statistical dependence between each feature and the target |
| **Permutation Importance** | Supervised, model-external | Drop in xRFM accuracy when each feature is randomly shuffled |

### Per-Leaf AGOP Aggregation

xRFM builds a binary tree and fits a separate RFM at each leaf. The AGOP is
therefore local: each leaf's RFM has its own learned Mahalanobis matrix `M`
(shape `d × d`) representing the gradient outer product averaged over that
leaf's training samples.

A global AGOP estimate is obtained via a **sample-count-weighted average** across
all leaves in all trees:

```
AGOP_global = Σ_ℓ  (n_ℓ / n_train) · diag(M_ℓ)
```

where `n_ℓ = leaf_node['train_indices'].numel()` is the number of training
samples routed to leaf `ℓ`. This is equivalent to the population-level AGOP
if leaves partition the feature space.

Implementation detail: `xrfm_model._collect_leaf_nodes(tree)` traverses the
fitted tree and returns all leaf node dicts. Each dict contains `'model'` (the
leaf RFM) and `'train_indices'` (a LongTensor of sample indices).

**Outputs:** `results/interpretability_comparison.png`,
`results/interpretability_rank_correlation.png`,
`results/interpretability_scores.csv`

---

## 8. Task C — Sample-Size Scaling Experiment

**Notebook:** `notebooks/03_Scaling_Experiment_Plots.ipynb`
**Dataset:** Superconductivity (n = 21,263, d = 81)

The training set is subsampled at fractions `[10%, 20%, 40%, 60%, 80%, 100%]`.
At each fraction, all models are trained from scratch with fixed hyperparameters
(no re-tuning). Test RMSE and wall-clock training time are recorded.

Models in the comparison: **xRFM, XGBoost, Random Forest, MLP** (the four models
from the main results table). SVM and KRR are additionally included as supplementary
curves to illustrate the classical O(n²) kernel scaling collapse.

**Outputs:** `results/scaling_rmse_vs_n.png`, `results/scaling_time_vs_n.png`,
`results/scaling_results.csv`

---

## 9. Task D — Bonus: Residual-Weighted AGOP

**Script:** `src/bonus_residual_agop.py`

Implements the residual-weighted AGOP:

```
AGOP_res(f) = Σᵢ wᵢ ∇f(xᵢ) ∇f(xᵢ)ᵀ  /  Σᵢ wᵢ
```

where `wᵢ = rᵢ²` and `rᵢ = yᵢ − f(xᵢ)` is the residual of the fitted model.

The script addresses all four required sub-parts of the bonus:
1. **(i) Conceptual justification** — residual weighting concentrates gradient
   information on hard-to-fit samples, potentially exposing better split directions.
2. **(ii) Implementation** — computes the residual-weighted AGOP and derives
   a split direction (top eigenvector) on a small dataset.
3. **(iii) Disagreement example** — identifies at least one dataset where the
   standard and residual-weighted AGOPs select different split directions.
4. **(iv) Performance comparison** — measures RMSE improvement on a held-out
   test set when the residual-weighted split criterion is used.

---

## 10. Step-by-Step Execution Guide

Run the following commands **in order** from the repository root.
Each step depends on the artefacts produced by the previous step.

### Step 1 — Download and split all datasets

```bash
python src/data_loader.py
```

Downloads all five datasets from the UCI repository via `ucimlrepo`, applies
dataset-specific cleaning (column drops, target mapping), and writes
`data/<dataset>/{train,val,test}/{X.csv,y.csv}`.

Expected runtime: 1–3 minutes (network-dependent).

### Step 2 — Train xRFM, XGBoost, and Random Forest

```bash
python src/train_trees_and_xrfm.py
```

For each dataset:
- Reads train/val/test CSVs.
- Fits `StandardScaler` + `OneHotEncoder` **on `X_train` only**.
- Saves `preprocessor.pkl` and (for Dry Bean) `label_encoder.pkl`.
- Trains xRFM, XGBoost, RF and saves each to `saved_models/<dataset>/<model>.pkl`.
- Appends training times to `results/training_times.csv`.

Expected runtime: 15–40 minutes (CPU-only).

### Step 3 — Train MLP and TabNet

```bash
python src/train_deep_learning.py
```

For each dataset:
- Loads the **existing** `preprocessor.pkl` from Step 2 (no re-fitting).
- Trains MLP (sklearn) and TabNet (pytorch-tabnet) with locked hyperparameters.
- Saves MLP to `saved_models/<dataset>/mlp.pkl` and TabNet to
  `saved_models/<dataset>/tabnet/model.zip`.
- Appends training times to `results/training_times_dl.csv`.

Expected runtime: 20–60 minutes (CPU-only; GPU accelerates TabNet significantly).

### Step 4 — Run the Jupyter Notebooks

Launch Jupyter and run the notebooks in order. Each notebook is self-contained
and re-loadable without re-running the training scripts (all models are loaded
from `saved_models/`).

```bash
jupyter notebook
```

| Order | Notebook | What it produces |
|-------|----------|-----------------|
| 4a | `notebooks/01_Main_Results_Evaluation.ipynb` | `results/main_results_table.csv`, bar charts |
| 4b | `notebooks/02_Interpretability_Plots.ipynb` | `results/interpretability_comparison.png`, rank-correlation heatmap |
| 4c | `notebooks/03_Scaling_Experiment_Plots.ipynb` | `results/scaling_rmse_vs_n.png`, `results/scaling_time_vs_n.png` |

Run each notebook via **Kernel → Restart & Run All** to ensure a clean state.

### Step 5 — Bonus analysis (optional)

```bash
python src/bonus_residual_agop.py
```

Runs the full residual-weighted AGOP analysis and writes comparison figures to
`results/bonus_agop_*.png`.

---

## 11. Environment Setup

```bash
# Create and activate the conda environment
conda create -n comp9417 python=3.10 -y
conda activate comp9417

# Install all dependencies
pip install numpy pandas scikit-learn matplotlib seaborn joblib
pip install xgboost
pip install xrfm
pip install pytorch-tabnet
pip install ucimlrepo
pip install jupyter scipy
```

> **GPU note:** `xrfm` and `pytorch-tabnet` will automatically use CUDA if
> a compatible GPU is available. All results in the report were produced on CPU
> to ensure reproducibility across environments.

Python version: **3.10**  
All packages use versions available as of April 2026 (see `requirements.txt` for
pinned versions if exact reproducibility is needed).

---

## 12. Reproducibility Checklist

- [x] Global random seed `42` applied to all `train_test_split`, model initialisations,
      permutation importance loops, and TabNet seed.
- [x] Stratified splits for all classification tasks (`stratify=y`).
- [x] `StandardScaler` and `OneHotEncoder` fit on `X_train` only; `transform` applied
      to val and test without any re-fitting.
- [x] Test set accessed **only once** per model, in the evaluation notebooks.
- [x] Multiclass AUC for Dry Bean: `roc_auc_score(..., multi_class='ovr', average='macro')`.
- [x] All trained model artefacts are deterministic given the same seed (no CUDA
      non-determinism in the reported experiments).
- [x] Notebook outputs are committed with executed cells so reviewers can inspect
      results without re-running training.
