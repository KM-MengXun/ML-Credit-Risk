# =========================== DETERMINISM HEADER (TOP) ===========================
# Place ABOVE any numpy/pandas/scikit/xgboost/catboost imports.
import os, random

SEED = 42                           # single source of truth
DETERMINISTIC = True                # flip to False for speed over strict repeatability
THREADS = 1 if DETERMINISTIC else max(1, os.cpu_count() // 2)

# Make numeric libs single-threaded to avoid nondeterministic reductions
os.environ["OMP_NUM_THREADS"]        = "1" if DETERMINISTIC else str(THREADS)
os.environ["OPENBLAS_NUM_THREADS"]   = "1" if DETERMINISTIC else str(THREADS)
os.environ["MKL_NUM_THREADS"]        = "1" if DETERMINISTIC else str(THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" if DETERMINISTIC else str(THREADS)
os.environ["NUMEXPR_NUM_THREADS"]    = "1" if DETERMINISTIC else str(THREADS)

# Python & NumPy RNG
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
# ===============================================================================

import gc
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

# ====================== UNIFIED REPORTING HELPERS ======================

def _ks_from_proba(y_true, proba):
    fpr, tpr, thr = roc_curve(y_true, proba)
    ks_vals = tpr - fpr
    i = int(np.argmax(ks_vals))
    return float(ks_vals[i]), float(thr[i])

def _evaluate_block(y_true, proba, thr):
    auc   = float(roc_auc_score(y_true, proba))
    prauc = float(average_precision_score(y_true, proba))
    fpr, tpr, _ = roc_curve(y_true, proba)
    ks = float(np.max(tpr - fpr))
    pred = (proba >= thr).astype(int)
    cm = confusion_matrix(y_true, pred)
    rpt = classification_report(y_true, pred, digits=4)
    return auc, prauc, ks, cm, rpt

def print_model_report(
    name,
    cv_best_params,
    cv_best_score,
    val_proba, y_val,
    test_proba, y_test,
):
    print(f"\n=== {name} CV ===")
    if cv_best_params is not None:
        print(f"Best params: {cv_best_params}")
    else:
        print("Best params: (n/a)")
    if cv_best_score is not None:
        print(f"CV ROC-AUC (mean): {cv_best_score:.6f}")
    else:
        print("CV ROC-AUC (mean): (n/a)")

    ks_val, thr_ks = _ks_from_proba(y_val, val_proba)
    val_auc, val_prauc, _, val_cm, val_rpt = _evaluate_block(y_val, val_proba, thr_ks)

    print(f"\n=== {name} VALIDATION ===")
    print(f"Validation ROC-AUC: {val_auc:.4f}")
    print(f"Validation PR-AUC : {val_prauc:.4f}")
    print(f"Validation KS     : {ks_val:.4f} @ threshold {thr_ks:.4f}")
    print("\nValidation Confusion Matrix:")
    print(val_cm)
    print("\nValidation Classification Report:")
    print(val_rpt)

    test_auc, test_prauc, ks_test, test_cm, test_rpt = _evaluate_block(y_test, test_proba, thr_ks)

    print(f"\n=== {name} TEST (using validation KS threshold) ===")
    print(f"Test ROC-AUC: {test_auc:.4f}")
    print(f"Test PR-AUC : {test_prauc:.4f}")
    print(f"Test KS     : {ks_test:.4f}")
    print("\nTest Confusion Matrix:")
    print(test_cm)
    print("\nTest Classification Report:")
    print(test_rpt)

    return thr_ks
# ======================================================================


# --- file paths (edit TEST_PATH if needed) ---
#  r"H:\\git\\ML-Credit-Risk\\Dataset\\GiveMeSomeCredit\\cs-training.csv"
#  r"D:\\Github\\ML-Credit-Risk\\Dataset\\GiveMeSomeCredit\\cs-training.csv"

# === 1. Load Data & Data Pre-processing ===
#  File location for developers:
#  NOTE: Keep your original path; adjust if running elsewhere.
df = pd.read_csv(r"H:\\git\\ML-Credit-Risk\\Dataset\\GiveMeSomeCredit\\cs-training.csv")

# drop stray index col (assign back)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# keep ages 20–100 and take a real copy (avoids chained-assignment issues)
df = df[(df["age"] >= 20) & (df["age"] <= 100)].copy()

# MonthlyIncome: keep 0s, impute only NAs by age-group median ===
df["NoIncomeFlag"] = (df["MonthlyIncome"] == 0).astype("int8")

bins   = [20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-100"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

median_by_group = (
    df.loc[df["MonthlyIncome"] > 0]
      .groupby("age_group", observed=False)["MonthlyIncome"]
      .median()
)

mi_na_mask = df["MonthlyIncome"].isna()
df.loc[mi_na_mask, "MonthlyIncome"] = df.loc[mi_na_mask, "age_group"].map(median_by_group)
df["IncomeImputedFlag"] = mi_na_mask.astype("int8")
df = df.drop(columns=["age_group"])

# DebtRatio: coerce, remove inf, and cap per new rules (0-income=10, imputed=2, normal=10) ===
dr = pd.to_numeric(df["DebtRatio"], errors="coerce")
dr = dr.mask(np.isinf(dr), np.nan)  # remove ±inf safely

# Masks from existing columns/flags
pos_income    = df["MonthlyIncome"] > 0
was_imputed   = df["IncomeImputedFlag"].eq(1)  # MonthlyIncome was NA and then imputed by median
zero_income   = df["NoIncomeFlag"].eq(1)       # MonthlyIncome == 0 (flag set before imputation)
normal_income = pos_income & (~was_imputed)    # positive and not imputed

# Caps
REALISTIC_CAP = 10.0  # for 0-income and normal positive-income rows
IMPUTED_CAP   = 2.0   # for rows whose MonthlyIncome was imputed

# Start from a clean series
dr_series = pd.Series(dr, index=df.index)

# Apply caps with lower bound at 0 (avoid negatives)
dr_series.loc[was_imputed]   = np.clip(dr_series.loc[was_imputed],   0.0, IMPUTED_CAP)
dr_series.loc[zero_income]   = np.clip(dr_series.loc[zero_income],   0.0, REALISTIC_CAP)
dr_series.loc[normal_income] = np.clip(dr_series.loc[normal_income], 0.0, REALISTIC_CAP)

# Fill any remaining NaNs with the median of observed values
dr_series = dr_series.fillna(np.nanmedian(dr_series.values))
df["DebtRatio"] = dr_series.astype(float)

# NumberOfDependents: flag missing, top-code, impute median, large-family flag ===
dep_na_mask = df["NumberOfDependents"].isna()
df["DependentsMissingFlag"] = dep_na_mask.astype("int8")

# top-code extremes at 5 before imputation
dep_series = df["NumberOfDependents"].copy()
dep_series = dep_series.where(dep_series.isna() | (dep_series <= 5), 5)
dep_median = int(dep_series.median(skipna=True))
dep_series = dep_series.fillna(dep_median).astype("int8")

df["NumberOfDependents"] = dep_series
df["LargeFamilyFlag"] = (df["NumberOfDependents"] >= 5).astype("int8")

# Number of Times Past Due Clean up
cols_past_due = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate"
]

for col in cols_past_due:
    # Ensure numeric
    df[col] = pd.to_numeric(df[col], errors="coerce")
    # Missing flag for special codes
    df[f"{col}_MissingFlag"] = df[col].isin([96, 98]).astype(int)
    # Replace special codes with NaN
    df[col] = df[col].replace({96: np.nan, 98: np.nan})
    # Calculate median from valid (non-NaN) values
    median_val = df[col].median(skipna=True)
    # Fill NaN with median
    df[col] = df[col].fillna(median_val)
    # Cap extreme real values
    df[col] = np.where(df[col] > 10, 10, df[col]).astype(int)

# === 2. Split the dataset ===
# Suppose df is already cleaned
TARGET = "SeriousDlqin2yrs"
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

# First split into train+val and test (stratified)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=SEED
)

# ---------------- Persist splits so future runs use identical rows --------------
split_dir = Path("./_splits"); split_dir.mkdir(exist_ok=True)
f_train = split_dir / "train_idx.npy"
f_val   = split_dir / "val_idx.npy"
f_test  = split_dir / "test_idx.npy"

if all(f.exists() for f in (f_train, f_val, f_test)):
    # Rebuild splits from saved indices (guaranteed identical)
    idx_train = np.load(f_train, allow_pickle=False)
    idx_val   = np.load(f_val,   allow_pickle=False)
    idx_test  = np.load(f_test,  allow_pickle=False)
    X_train, y_train = X.loc[idx_train], y.loc[idx_train]
    X_val,   y_val   = X.loc[idx_val],   y.loc[idx_val]
    X_test,  y_test  = X.loc[idx_test],  y.loc[idx_test]
else:
    # First run: save the indices we just generated with the fixed random_state
    np.save(f_train, X_train.index.values)
    np.save(f_val,   X_val.index.values)
    np.save(f_test,  X_test.index.values)
# ------------------------------------------------------------------------------

# ======================================================================================
# L1 Logistic Regression (with scaling)
pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        penalty="l1",
        solver="saga",            # supports L1 on large datasets
        max_iter=5000,
        class_weight="balanced",  # handle imbalance
        random_state=SEED,
        n_jobs=THREADS
    ))
])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

param_grid = {
    "clf__C": np.logspace(-3, 1, 16)  # 16 candidates
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,            # keep deterministic; set >1 if you want faster CV
    verbose=1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_

# VALIDATION probabilities from CV-best model
val_proba_logit = best_model.predict_proba(X_val)[:, 1]

# Retrain on TRAIN+VAL with best C for TEST probabilities
best_model_final = GridSearchCV(
    estimator=pipe,
    param_grid={"clf__C": [grid.best_params_["clf__C"]]},
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
).fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val])).best_estimator_

test_proba_logit = best_model_final.predict_proba(X_test)[:, 1]

_ = print_model_report(
    name="L1-Logit",
    cv_best_params=grid.best_params_,
    cv_best_score=grid.best_score_,
    val_proba=val_proba_logit, y_val=y_val,
    test_proba=test_proba_logit, y_test=y_test
)

# ======================================================================================
# XGBoost
from xgboost import XGBClassifier

# Reduce footprint: use float32 (no copy if already float32)
X_train_xgb = X_train.astype(np.float32, copy=False)
X_val_xgb   = X_val.astype(np.float32,   copy=False)
X_test_xgb  = X_test.astype(np.float32,  copy=False)

# CV setup (smaller grid + 3-fold to save memory)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
pos, neg = int((y_train == 1).sum()), int((y_train == 0).sum())
scale_pos_weight = neg / max(pos, 1)

pipe_xgb = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("xgb", XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        learning_rate=0.05,
        n_estimators=600,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1.0,
        reg_lambda=1.0,
        reg_alpha=0.0,
        scale_pos_weight=scale_pos_weight,
        n_jobs=THREADS,
        random_state=SEED,
        verbosity=0
    ))
])

param_grid_xgb = {
    "xgb__n_estimators": [400, 800],
    "xgb__max_depth": [4, 6],
    "xgb__learning_rate": [0.05, 0.1],
    "xgb__min_child_weight": [1.0, 3.0],
}

grid_xgb = GridSearchCV(
    estimator=pipe_xgb,
    param_grid=param_grid_xgb,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,
    verbose=1,
    refit=True
)

grid_xgb.fit(X_train_xgb, y_train)
best_xgb = grid_xgb.best_estimator_

# VALIDATION proba
val_proba_xgb = best_xgb.predict_proba(X_val_xgb)[:, 1]

# Refit on TRAIN+VAL (no extra CV)
best_xgb_final = grid_xgb.best_estimator_
X_trv = pd.concat([X_train_xgb, X_val_xgb], copy=False)
y_trv = pd.concat([y_train, y_val], copy=False)
best_xgb_final.fit(X_trv, y_trv)
del X_trv, y_trv
gc.collect()

# TEST proba
test_proba_xgb = best_xgb_final.predict_proba(X_test_xgb)[:, 1]

_ = print_model_report(
    name="XGBoost",
    cv_best_params=grid_xgb.best_params_,
    cv_best_score=grid_xgb.best_score_,
    val_proba=val_proba_xgb, y_val=y_val,
    test_proba=test_proba_xgb, y_test=y_test
)

# ======================================================================================
# CatBoost
from catboost import CatBoostClassifier

X_train_cb = X_train.copy()
X_val_cb   = X_val.copy()
X_test_cb  = X_test.copy()

# Class weight for imbalance
pos = int((y_train == 1).sum())
neg = int((y_train == 0).sum())
class_weights = [1.0, neg / max(pos, 1)]

# Small param grid
param_grid_cb = [
    {"depth": 6,  "learning_rate": 0.05, "l2_leaf_reg": 3.0},
    {"depth": 8,  "learning_rate": 0.05, "l2_leaf_reg": 3.0},
    {"depth": 6,  "learning_rate": 0.1,  "l2_leaf_reg": 3.0},
]

best_cb = None
best_auc = -np.inf
best_params = None

# Simple manual CV loop for small grid
for params in param_grid_cb:
    cb = CatBoostClassifier(
        task_type="CPU",
        iterations=2000,
        early_stopping_rounds=100,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=class_weights,
        random_seed=SEED,
        thread_count=THREADS,
        verbose=False,
        **params
    )
    cb.fit(X_train_cb, y_train, eval_set=(X_val_cb, y_val), use_best_model=True)
    vproba = cb.predict_proba(X_val_cb)[:, 1]
    vauc = roc_auc_score(y_val, vproba)
    if vauc > best_auc:
        best_auc = vauc
        best_cb = cb
        best_params = params

# VALIDATION proba from best model
val_proba_cb = best_cb.predict_proba(X_val_cb)[:, 1]

# Retrain FINAL on TRAIN+VAL
cb_final = CatBoostClassifier(
    task_type="CPU",
    iterations=2000,
    early_stopping_rounds=100,
    loss_function="Logloss",
    eval_metric="AUC",
    class_weights=class_weights,
    random_seed=SEED,
    thread_count=THREADS,
    verbose=False,
    **best_params
)
X_trv_cb = pd.concat([X_train_cb, X_val_cb], copy=False)
y_trv_cb = pd.concat([y_train, y_val], copy=False)
cb_final.fit(X_trv_cb, y_trv_cb, use_best_model=False)

# TEST proba
test_proba_cb = cb_final.predict_proba(X_test_cb)[:, 1]

_ = print_model_report(
    name="CatBoost",
    cv_best_params=best_params,
    cv_best_score=best_auc,   # using best validation AUC from the simple grid loop
    val_proba=val_proba_cb, y_val=y_val,
    test_proba=test_proba_cb, y_test=y_test
)

gc.collect()

# ======================================================================================
# Random Forest
from sklearn.ensemble import RandomForestClassifier

X_train_rf = X_train.astype(np.float32, copy=False)
X_val_rf   = X_val.astype(np.float32,   copy=False)
X_test_rf  = X_test.astype(np.float32,  copy=False)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

pipe_rf = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=max(1, os.cpu_count() // 2),
        random_state=SEED
    ))
])

param_grid_rf = {
    "rf__n_estimators": [400, 800],
    "rf__max_depth": [None, 12],
    "rf__min_samples_leaf": [1, 5],
    "rf__max_features": ["sqrt", 0.5],
}

grid_rf = GridSearchCV(
    estimator=pipe_rf,
    param_grid=param_grid_rf,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,
    verbose=1,
    refit=True
)

grid_rf.fit(X_train_rf, y_train)
best_rf = grid_rf.best_estimator_

# VALIDATION proba
val_proba_rf = best_rf.predict_proba(X_val_rf)[:, 1]

# Refit on TRAIN+VAL
best_rf_final = grid_rf.best_estimator_
X_trv_rf = pd.concat([X_train_rf, X_val_rf], copy=False)
y_trv_rf = pd.concat([y_train, y_val],       copy=False)
best_rf_final.fit(X_trv_rf, y_trv_rf)
del X_trv_rf, y_trv_rf
gc.collect()

# TEST proba
test_proba_rf = best_rf_final.predict_proba(X_test_rf)[:, 1]

_ = print_model_report(
    name="Random Forest",
    cv_best_params=grid_rf.best_params_,
    cv_best_score=grid_rf.best_score_,
    val_proba=val_proba_rf, y_val=y_val,
    test_proba=test_proba_rf, y_test=y_test
)

# ======================================================================================
# Decision Tree
from sklearn.tree import DecisionTreeClassifier

X_train_dt = X_train.astype(np.float32, copy=False)
X_val_dt   = X_val.astype(np.float32,   copy=False)
X_test_dt  = X_test.astype(np.float32,  copy=False)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

pipe_dt = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("dt", DecisionTreeClassifier(
        criterion="gini",
        class_weight="balanced",
        random_state=SEED
    ))
])

param_grid_dt = {
    "dt__max_depth": [None, 6, 12, 20],
    "dt__min_samples_split": [2, 10],
    "dt__min_samples_leaf": [1, 5, 20],
    "dt__max_features": [None, "sqrt"],
    "dt__ccp_alpha": [0.0, 0.001]
}

grid_dt = GridSearchCV(
    estimator=pipe_dt,
    param_grid=param_grid_dt,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,
    verbose=1,
    refit=True
)

grid_dt.fit(X_train_dt, y_train)
best_dt = grid_dt.best_estimator_

# VALIDATION proba
val_proba_dt = best_dt.predict_proba(X_val_dt)[:, 1]

# Refit on TRAIN+VAL
best_dt_final = grid_dt.best_estimator_
X_trv_dt = pd.concat([X_train_dt, X_val_dt], copy=False)
y_trv_dt = pd.concat([y_train, y_val],      copy=False)
best_dt_final.fit(X_trv_dt, y_trv_dt)
del X_trv_dt, y_trv_dt
gc.collect()

# TEST proba
test_proba_dt = best_dt_final.predict_proba(X_test_dt)[:, 1]

_ = print_model_report(
    name="Decision Tree",
    cv_best_params=grid_dt.best_params_,
    cv_best_score=grid_dt.best_score_,
    val_proba=val_proba_dt, y_val=y_val,
    test_proba=test_proba_dt, y_test=y_test
)

# Optional: top feature importances
fi_dt = best_dt_final.named_steps["dt"].feature_importances_
top_dt = pd.Series(fi_dt, index=X.columns).sort_values(ascending=False).head(15)
print("\nTop 15 Decision Tree feature importances:\n", top_dt)

# ================================================================================
# === PLOTS: ROC / PR / KS on TEST for 5 models ===
import matplotlib.pyplot as plt

probas = {
    "L1-Logit":     test_proba_logit,
    "XGBoost":      test_proba_xgb,
    "CatBoost":     test_proba_cb,
    "RandomForest": test_proba_rf,
    "DecisionTree": test_proba_dt,
}

# Summary table: AUC / PR-AUC / KS (max TPR-FPR)
summary_rows = []
ks_curves = {}  # store (thr, ks_vals) for plotting
for name, p in probas.items():
    fpr, tpr, thr = roc_curve(y_test, p)
    ks_vals = tpr - fpr
    ks = float(np.max(ks_vals))
    thr_ks = float(thr[np.argmax(ks_vals)])
    auc = float(roc_auc_score(y_test, p))
    pr_auc = float(average_precision_score(y_test, p))
    summary_rows.append([name, auc, pr_auc, ks, thr_ks])
    ks_curves[name] = (thr, ks_vals)

summary = pd.DataFrame(summary_rows, columns=["Model", "ROC-AUC", "PR-AUC", "KS", "KS_threshold"]) \
            .sort_values("ROC-AUC", ascending=False)
print("\n=== TEST summary ===")
print(summary.to_string(index=False))

# ROC curve (combined)
plt.figure()
for name, p in probas.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves on TEST")
plt.legend()
plt.tight_layout()
plt.savefig("roc_test.png", dpi=200)
plt.show()

# PR curve (combined)
plt.figure()
for name, p in probas.items():
    prec, rec, _ = precision_recall_curve(y_test, p)
    plt.plot(rec, prec, label=f"{name} (PR-AUC={average_precision_score(y_test, p):.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curves on TEST")
plt.legend()
plt.tight_layout()
plt.savefig("pr_test.png", dpi=200)
plt.show()

# KS curve (combined)
plt.figure()
for name, (thr, ks_vals) in ks_curves.items():
    plt.plot(thr, ks_vals, label=f"{name}")
plt.xlabel("Threshold")
plt.ylabel("KS = TPR - FPR")
plt.title("KS Curves on TEST")
plt.legend()
plt.tight_layout()
plt.savefig("ks_test.png", dpi=200)
plt.show()


# ================================================================================
# QRM ADD-ON: Portfolio VaR & ES from PDs (Monte Carlo, Gaussian copula)
# Place this block directly AFTER your existing code (after the ROC/PR/KS plots).
# ================================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import norm

rng = np.random.default_rng(SEED)

# ---------- 1) Portfolio setup (adjust if you have real EAD/LGD) ----------
N = len(next(iter(probas.values())))            # number of test accounts
EAD_const = 1000.0                               # constant exposure per account ($)
# Fixed LGD per account from Beta(2,5) ~ mean 0.286
LGD = rng.beta(2.0, 5.0, size=N).astype(np.float64)

# Effective per-account loss weight
W = EAD_const * LGD  # shape (N,)

# ---------- 2) Bin accounts by PD ----------
def make_pd_bins(p, w, n_bins=200):
    """Return per-bin counts, avg PD, and avg exposure-per-default occurrence."""
    p = np.asarray(p, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = len(p)
    n_bins = int(min(n_bins, max(1, n)))  # cap at N
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    n_b = np.array([len(b) for b in bins], dtype=np.int32)
    n_b = np.where(n_b == 0, 1, n_b)
    p_b = np.array([p[b].mean() for b in bins], dtype=np.float64)
    e_b = np.array([w[b].sum() / max(len(b), 1) for b in bins], dtype=np.float64)
    return n_b, p_b, e_b

# ---------- 3) Simulator ----------
def simulate_losses_binomial(n_b, p_b, e_b, sims=20000, rho=0.0, batch=2000, seed=SEED):
    """Monte Carlo portfolio loss using per-bin binomial draws.
       rho=0: independent defaults; rho>0: one-factor Gaussian copula."""
    rng = np.random.default_rng(seed)
    a_b = norm.ppf(np.clip(p_b, 1e-12, 1-1e-12))
    B = len(n_b)
    out = np.empty(sims, dtype=np.float64)
    i = 0
    while i < sims:
        m = min(batch, sims - i)
        if rho <= 0.0:
            D = rng.binomial(n=n_b, p=p_b, size=(m, B)).astype(np.float64)
        else:
            F = rng.standard_normal(size=m)  # common factor
            z = (a_b[None, :] - sqrt(rho) * F[:, None]) / sqrt(max(1e-12, 1.0 - rho))
            p_cond = norm.cdf(z)
            D = rng.binomial(n=n_b[None, :], p=p_cond).astype(np.float64)
        out[i:i+m] = D.dot(e_b)
        i += m
    return out

def var_es(x, alpha=0.99):
    """Return (VaR_alpha, ES_alpha) for array x of losses."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = x.size
    k = int(np.ceil(alpha * n)) - 1
    k = max(0, min(k, n-1))
    VaR = x[k]
    ES = x[k:].mean() if k < n else x[-1]
    return float(VaR), float(ES)

# ---------- 4) Run sims & summary ----------
rhos = [0.0, 0.1, 0.3, 0.5]
alpha_levels = [0.975, 0.99]
rows = []
cdf_for_plot = {}

for model_name, p in probas.items():
    p = np.asarray(p, dtype=np.float64)
    n_b, p_b, e_b = make_pd_bins(p, W, n_bins=200)
    for rho in rhos:
        losses = simulate_losses_binomial(n_b, p_b, e_b, sims=20000, rho=rho, batch=2000, seed=SEED+int(1000*rho))
        el = float(losses.mean())
        for a in alpha_levels:
            VaR, ES = var_es(losses, alpha=a)
            rows.append([model_name, rho, a, el, VaR, ES])
        if abs(rho - 0.30) < 1e-9:
            cdf_for_plot[model_name] = np.sort(losses)

qrm_df = pd.DataFrame(rows, columns=["Model", "Rho", "Alpha", "ExpectedLoss", "VaR", "ES"])
print("\n=== Portfolio Risk (Monte Carlo, 20k sims per rho) ===")
print(qrm_df.sort_values(["Rho", "Alpha", "VaR"], ascending=[True, True, False]).to_string(index=False))
qrm_df.to_csv("qrm_var_es_summary.csv", index=False)

# ---------- 5) Plot overlay loss CDFs at rho=0.30 ----------
plt.figure()
for name, xs in cdf_for_plot.items():
    n = xs.size
    u = np.linspace(0, 1, n, endpoint=False)
    plt.plot(xs, u, label=name)
plt.xlabel("Portfolio Loss")
plt.ylabel("Empirical CDF")
plt.title("Loss CDF on TEST (rho = 0.30)")
plt.legend()
plt.tight_layout()
plt.savefig("loss_cdf_rho0.30.png", dpi=200)
plt.show()

print("\nSaved: qrm_var_es_summary.csv, loss_cdf_rho0.30.png")
# ================================================================================
