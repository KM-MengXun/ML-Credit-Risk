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




import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer


# --- file paths (edit TEST_PATH if needed) ---
#   r"H:\git\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv"
#   r"D:\Github\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv"

# === 1. Load Data & Data Pre-processing ===
#  File location for developers:

df = pd.read_csv(r"H:\git\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv")

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

# Export cleaned data 
# df.to_csv(r"F:\Waterloo\Actsc\Actsc 445\Project\git\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training-clean.csv", index=False)

# === 2. Split the dataset ===
from sklearn.model_selection import train_test_split

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
from pathlib import Path
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


from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)

# L1 needs scaling for stable coefficients; StandardScaler is fine for all-numeric data
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

param_grid = {
    "clf__C": np.logspace(-3, 1, 9)  # 0.001 ... 10
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best C:", grid.best_params_["clf__C"])
print("CV ROC-AUC (mean):", grid.best_score_)

# ===== 6) Validate on VAL set =====
val_proba = best_model.predict_proba(X_val)[:, 1]
val_auc   = roc_auc_score(y_val, val_proba)
val_prauc = average_precision_score(y_val, val_proba)

# KS statistic (max TPR-FPR)
fpr, tpr, thr = roc_curve(y_val, val_proba)
ks_vals = tpr - fpr
ks = np.max(ks_vals)
thr_ks = thr[np.argmax(ks_vals)]

print(f"Validation ROC-AUC: {val_auc:.4f}")
print(f"Validation PR-AUC : {val_prauc:.4f}")
print(f"Validation KS     : {ks:.4f} @ threshold {thr_ks:.4f}")

# Choose threshold by max KS on validation
val_pred = (val_proba >= thr_ks).astype(int)
print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, val_pred))
print("\nValidation Classification Report:")
print(classification_report(y_val, val_pred, digits=4))

# ===== 7) Retrain on TRAIN+VAL with best hyperparams, then evaluate on TEST =====
best_model_final = GridSearchCV(
    estimator=pipe,
    param_grid={"clf__C": [grid.best_params_["clf__C"]]},
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
).fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val])).best_estimator_

test_proba = best_model_final.predict_proba(X_test)[:, 1]
test_auc   = roc_auc_score(y_test, test_proba)
test_prauc = average_precision_score(y_test, test_proba)
fpr_t, tpr_t, thr_t = roc_curve(y_test, test_proba)
ks_t = np.max(tpr_t - fpr_t)

# Use the threshold chosen on validation (thr_ks) for fair final metrics
test_pred = (test_proba >= thr_ks).astype(int)

print("\n=== TEST RESULTS (using validation KS threshold) ===")
print(f"Test ROC-AUC: {test_auc:.4f}")
print(f"Test PR-AUC : {test_prauc:.4f}")
print(f"Test KS     : {ks_t:.4f}")
print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))
print("\nTest Classification Report:")
print(classification_report(y_test, test_pred, digits=4))


# ======================================================================================
# XGBoost
import os, gc, numpy as np, pandas as pd
from xgboost import XGBClassifier

# 0) Reduce footprint: use float32 (no copy if already float32)
X_train = X_train.astype(np.float32, copy=False)
X_val   = X_val.astype(np.float32,   copy=False)
X_test  = X_test.astype(np.float32,  copy=False)

# 1) CV setup (smaller grid + 3-fold to save memory)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
pos, neg = int((y_train == 1).sum()), int((y_train == 0).sum())
scale_pos_weight = neg / max(pos, 1)

pipe_xgb = Pipeline([
    # trees don't need scaling, but we must impute missing values
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

# compact grid (8 combos)
param_grid_xgb = {
    "xgb__n_estimators": [400, 800],
    "xgb__max_depth": [4, 6],
    "xgb__learning_rate": [0.05, 0.1],
    "xgb__min_child_weight": [1.0, 3.0],
}

grid_xgb = GridSearchCV(
    estimator=pipe_xgb,
    param_grid=param_grid_xgb,
    scoring="roc_auc",     # keep identical to your L1 setup
    cv=cv,
    n_jobs=1,              # <<< single process to avoid pickling big data
    verbose=1,
    refit=True
)

# 2) Fit CV
grid_xgb.fit(X_train, y_train)
best_xgb = grid_xgb.best_estimator_
print("Best params:", grid_xgb.best_params_)
print("CV ROC-AUC :", grid_xgb.best_score_)
gc.collect()

# 3) Validate on VAL — get KS threshold
val_proba = best_xgb.predict_proba(X_val)[:, 1]
val_auc   = roc_auc_score(y_val, val_proba)
val_prauc = average_precision_score(y_val, val_proba)
fpr, tpr, thr = roc_curve(y_val, val_proba)
ks_vals = tpr - fpr
ks = float(np.max(ks_vals))
thr_ks = float(thr[np.argmax(ks_vals)])

print(f"Validation ROC-AUC: {val_auc:.4f}")
print(f"Validation PR-AUC : {val_prauc:.4f}")
print(f"Validation KS     : {ks:.4f} @ threshold {thr_ks:.4f}")

val_pred = (val_proba >= thr_ks).astype(int)
print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, val_pred))
print("\nValidation Classification Report:")
print(classification_report(y_val, val_pred, digits=4))
del val_proba, fpr, tpr, thr, ks_vals
gc.collect()

# 4) Refit on TRAIN+VAL with best hyperparams (no extra CV)
best_xgb_final = grid_xgb.best_estimator_
X_trv = pd.concat([X_train, X_val], copy=False)
y_trv = pd.concat([y_train, y_val], copy=False)
best_xgb_final.fit(X_trv, y_trv)
del X_trv, y_trv
gc.collect()

# 5) Final TEST evaluation (use validation KS threshold)
test_proba = best_xgb_final.predict_proba(X_test)[:, 1]
test_auc   = roc_auc_score(y_test, test_proba)
test_prauc = average_precision_score(y_test, test_proba)
fpr_t, tpr_t, thr_t = roc_curve(y_test, test_proba)
ks_t = float(np.max(tpr_t - fpr_t))

test_pred = (test_proba >= thr_ks).astype(int)

print("\n=== TEST RESULTS (using validation KS threshold) ===")
print(f"Test ROC-AUC: {test_auc:.4f}")
print(f"Test PR-AUC : {test_prauc:.4f}")
print(f"Test KS     : {ks_t:.4f}")
print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))
print("\nTest Classification Report:")
print(classification_report(y_test, test_pred, digits=4))


# ======================================================================================
# CatBoost
from catboost import CatBoostClassifier
import gc

# Convert to float32 (CatBoost handles its own NaN imputation)
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
    val_proba = cb.predict_proba(X_val_cb)[:, 1]
    val_auc = roc_auc_score(y_val, val_proba)
    if val_auc > best_auc:
        best_auc = val_auc
        best_cb = cb
        best_params = params

print("Best CatBoost params:", best_params)
print("Validation ROC-AUC:", best_auc)

# Compute validation PR-AUC and KS
val_proba = best_cb.predict_proba(X_val_cb)[:, 1]
val_prauc = average_precision_score(y_val, val_proba)
fpr, tpr, thr = roc_curve(y_val, val_proba)
ks_vals = tpr - fpr
ks = float(np.max(ks_vals))
thr_ks_cb = float(thr[np.argmax(ks_vals)])
print(f"Validation PR-AUC: {val_prauc:.4f}")
print(f"Validation KS    : {ks:.4f} @ threshold {thr_ks_cb:.4f}")

val_pred = (val_proba >= thr_ks_cb).astype(int)
print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, val_pred))
print("\nValidation Classification Report:")
print(classification_report(y_val, val_pred, digits=4))

# Retrain on TRAIN+VAL
cb_final = CatBoostClassifier(
    task_type="CPU",   # or "GPU"
    iterations=2000,
    early_stopping_rounds=100,
    loss_function="Logloss",
    eval_metric="AUC",
    class_weights=class_weights,
    random_seed=42,
    verbose=False,
    **best_params
)
X_trv_cb = pd.concat([X_train_cb, X_val_cb], copy=False)
y_trv_cb = pd.concat([y_train, y_val], copy=False)
cb_final.fit(X_trv_cb, y_trv_cb, use_best_model=False)

# Test set evaluation
test_proba = cb_final.predict_proba(X_test_cb)[:, 1]
test_auc   = roc_auc_score(y_test, test_proba)
test_prauc = average_precision_score(y_test, test_proba)
fpr_t, tpr_t, thr_t = roc_curve(y_test, test_proba)
ks_t = float(np.max(tpr_t - fpr_t))

test_pred = (test_proba >= thr_ks_cb).astype(int)

print("\n=== TEST RESULTS (CatBoost, using validation KS threshold) ===")
print(f"Test ROC-AUC: {test_auc:.4f}")
print(f"Test PR-AUC : {test_prauc:.4f}")
print(f"Test KS     : {ks_t:.4f}")
print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))
print("\nTest Classification Report:")
print(classification_report(y_test, test_pred, digits=4))
gc.collect()


# ======================================================================================
# Random Forest
from sklearn.ensemble import RandomForestClassifier
import os, gc

# Trees don't need scaling, but they can't handle NaNs -> impute
X_train_rf = X_train.astype(np.float32, copy=False)
X_val_rf   = X_val.astype(np.float32,   copy=False)
X_test_rf  = X_test.astype(np.float32,  copy=False)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

pipe_rf = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        n_jobs=THREADS,           # <- deterministic threads
        random_state=SEED         # <- single source of truth
    ))
])

# compact grid (8 combos)
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
    n_jobs=1,         # single process to avoid pickling large data during CV
    verbose=1,
    refit=True
)

# 1) Fit CV
grid_rf.fit(X_train_rf, y_train)
best_rf = grid_rf.best_estimator_
print("Best RF params:", grid_rf.best_params_)
print("CV ROC-AUC    :", grid_rf.best_score_)

# 2) Validate — pick KS threshold
val_proba = best_rf.predict_proba(X_val_rf)[:, 1]
val_auc   = roc_auc_score(y_val, val_proba)
val_prauc = average_precision_score(y_val, val_proba)
fpr, tpr, thr = roc_curve(y_val, val_proba)
ks_vals = tpr - fpr
ks = float(np.max(ks_vals))
thr_ks_rf = float(thr[np.argmax(ks_vals)])

print(f"Validation ROC-AUC: {val_auc:.4f}")
print(f"Validation PR-AUC : {val_prauc:.4f}")
print(f"Validation KS     : {ks:.4f} @ threshold {thr_ks_rf:.4f}")

val_pred = (val_proba >= thr_ks_rf).astype(int)
print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, val_pred))
print("\nValidation Classification Report:")
print(classification_report(y_val, val_pred, digits=4))
del val_proba, fpr, tpr, thr, ks_vals
gc.collect()

# 3) Refit on TRAIN+VAL with best hyperparams
best_rf_final = grid_rf.best_estimator_
X_trv_rf = pd.concat([X_train_rf, X_val_rf], copy=False)
y_trv_rf = pd.concat([y_train, y_val],       copy=False)
best_rf_final.fit(X_trv_rf, y_trv_rf)
del X_trv_rf, y_trv_rf
gc.collect()

# 4) Final TEST evaluation (use validation KS threshold)
test_proba = best_rf_final.predict_proba(X_test_rf)[:, 1]
test_auc   = roc_auc_score(y_test, test_proba)
test_prauc = average_precision_score(y_test, test_proba)
fpr_t, tpr_t, thr_t = roc_curve(y_test, test_proba)
ks_t = float(np.max(tpr_t - fpr_t))
test_pred = (test_proba >= thr_ks_rf).astype(int)

print("\n=== TEST RESULTS (Random Forest, using validation KS threshold) ===")
print(f"Test ROC-AUC: {test_auc:.4f}")
print(f"Test PR-AUC : {test_prauc:.4f}")
print(f"Test KS     : {ks_t:.4f}")
print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))
print("\nTest Classification Report:")
print(classification_report(y_test, test_pred, digits=4))


# ======================================================================================
# Random Forest
from sklearn.ensemble import RandomForestClassifier
import os, gc

# Trees don't need scaling, but they can't handle NaNs -> impute
X_train_rf = X_train.astype(np.float32, copy=False)
X_val_rf   = X_val.astype(np.float32,   copy=False)
X_test_rf  = X_test.astype(np.float32,  copy=False)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

pipe_rf = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced_subsample",          # handle imbalance
        n_jobs=max(1, os.cpu_count() // 2),
        random_state=42
    ))
])

# compact grid (8 combos)
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
    n_jobs=1,         # single process to avoid pickling large data during CV
    verbose=1,
    refit=True
)

# 1) Fit CV
grid_rf.fit(X_train_rf, y_train)
best_rf = grid_rf.best_estimator_
print("Best RF params:", grid_rf.best_params_)
print("CV ROC-AUC    :", grid_rf.best_score_)

# 2) Validate — pick KS threshold
val_proba = best_rf.predict_proba(X_val_rf)[:, 1]
val_auc   = roc_auc_score(y_val, val_proba)
val_prauc = average_precision_score(y_val, val_proba)
fpr, tpr, thr = roc_curve(y_val, val_proba)
ks_vals = tpr - fpr
ks = float(np.max(ks_vals))
thr_ks_rf = float(thr[np.argmax(ks_vals)])

print(f"Validation ROC-AUC: {val_auc:.4f}")
print(f"Validation PR-AUC : {val_prauc:.4f}")
print(f"Validation KS     : {ks:.4f} @ threshold {thr_ks_rf:.4f}")

val_pred = (val_proba >= thr_ks_rf).astype(int)
print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, val_pred))
print("\nValidation Classification Report:")
print(classification_report(y_val, val_pred, digits=4))
del val_proba, fpr, tpr, thr, ks_vals
gc.collect()

# 3) Refit on TRAIN+VAL with best hyperparams
best_rf_final = grid_rf.best_estimator_
X_trv_rf = pd.concat([X_train_rf, X_val_rf], copy=False)
y_trv_rf = pd.concat([y_train, y_val],       copy=False)
best_rf_final.fit(X_trv_rf, y_trv_rf)
del X_trv_rf, y_trv_rf
gc.collect()

# 4) Final TEST evaluation (use validation KS threshold)
test_proba = best_rf_final.predict_proba(X_test_rf)[:, 1]
test_auc   = roc_auc_score(y_test, test_proba)
test_prauc = average_precision_score(y_test, test_proba)
fpr_t, tpr_t, thr_t = roc_curve(y_test, test_proba)
ks_t = float(np.max(tpr_t - fpr_t))
test_pred = (test_proba >= thr_ks_rf).astype(int)

print("\n=== TEST RESULTS (Random Forest, using validation KS threshold) ===")
print(f"Test ROC-AUC: {test_auc:.4f}")
print(f"Test PR-AUC : {test_prauc:.4f}")
print(f"Test KS     : {ks_t:.4f}")
print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))
print("\nTest Classification Report:")
print(classification_report(y_test, test_pred, digits=4))


# ======================================================================================
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
import gc

# Trees don't need scaling; still impute for any NaNs
X_train_dt = X_train.astype(np.float32, copy=False)
X_val_dt   = X_val.astype(np.float32,   copy=False)
X_test_dt  = X_test.astype(np.float32,  copy=False)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

pipe_dt = Pipeline([
    ("imp", SimpleImputer(strategy="median")),
    ("dt", DecisionTreeClassifier(
        criterion="gini",
        class_weight="balanced",
        random_state=42
    ))
])

param_grid_dt = {
    "dt__max_depth": [None, 6, 12, 20],
    "dt__min_samples_split": [2, 10],
    "dt__min_samples_leaf": [1, 5, 20],
    "dt__max_features": [None, "sqrt"],
    "dt__ccp_alpha": [0.0, 0.001]  # minimal cost-complexity pruning
}

grid_dt = GridSearchCV(
    estimator=pipe_dt,
    param_grid=param_grid_dt,
    scoring="roc_auc",
    cv=cv,
    n_jobs=1,        # avoid heavy pickling across processes
    verbose=1,
    refit=True
)

# 1) Fit CV
grid_dt.fit(X_train_dt, y_train)
best_dt = grid_dt.best_estimator_
print("Best DT params:", grid_dt.best_params_)
print("CV ROC-AUC    :", grid_dt.best_score_)

# 2) Validate — pick KS threshold
val_proba = best_dt.predict_proba(X_val_dt)[:, 1]
val_auc   = roc_auc_score(y_val, val_proba)
val_prauc = average_precision_score(y_val, val_proba)
fpr, tpr, thr = roc_curve(y_val, val_proba)
ks_vals = tpr - fpr
ks = float(np.max(ks_vals))
thr_ks_dt = float(thr[np.argmax(ks_vals)])

print(f"Validation ROC-AUC: {val_auc:.4f}")
print(f"Validation PR-AUC : {val_prauc:.4f}")
print(f"Validation KS     : {ks:.4f} @ threshold {thr_ks_dt:.4f}")

val_pred = (val_proba >= thr_ks_dt).astype(int)
print("\nValidation Confusion Matrix:")
print(confusion_matrix(y_val, val_pred))
print("\nValidation Classification Report:")
print(classification_report(y_val, val_pred, digits=4))
del val_proba, fpr, tpr, thr, ks_vals
gc.collect()

# 3) Refit on TRAIN+VAL with best hyperparams
best_dt_final = grid_dt.best_estimator_
X_trv_dt = pd.concat([X_train_dt, X_val_dt], copy=False)
y_trv_dt = pd.concat([y_train, y_val],      copy=False)
best_dt_final.fit(X_trv_dt, y_trv_dt)
del X_trv_dt, y_trv_dt
gc.collect()

# 4) Final TEST evaluation (use validation KS threshold)
test_proba = best_dt_final.predict_proba(X_test_dt)[:, 1]
test_auc   = roc_auc_score(y_test, test_proba)
test_prauc = average_precision_score(y_test, test_proba)
fpr_t, tpr_t, thr_t = roc_curve(y_test, test_proba)
ks_t = float(np.max(tpr_t - fpr_t))
test_pred = (test_proba >= thr_ks_dt).astype(int)

print("\n=== TEST RESULTS (Decision Tree, using validation KS threshold) ===")
print(f"Test ROC-AUC: {test_auc:.4f}")
print(f"Test PR-AUC : {test_prauc:.4f}")
print(f"Test KS     : {ks_t:.4f}")
print("\nTest Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))
print("\nTest Classification Report:")
print(classification_report(y_test, test_pred, digits=4))

# Optional: top feature importances
fi_dt = best_dt_final.named_steps["dt"].feature_importances_
top_dt = pd.Series(fi_dt, index=X.columns).sort_values(ascending=False).head(15)
print("\nTop 15 Decision Tree feature importances:\n", top_dt)
