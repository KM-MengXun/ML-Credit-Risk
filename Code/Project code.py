import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer


# --- file paths (edit TEST_PATH if needed) ---
#   r"F:\Waterloo\Actsc\Actsc 445\Project\git\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv"
#   r"D:\Github\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv"

# === 1. Load Data & Data Pre-processing ===
#  File location for developers:

df = pd.read_csv(r"F:\Waterloo\Actsc\Actsc 445\Project\git\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv")

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
df.to_csv(r"F:\Waterloo\Actsc\Actsc 445\Project\git\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training-clean.csv", index=False)

# === 2. Split the dataset ===
from sklearn.model_selection import train_test_split

# Suppose df is already cleaned
TARGET = "SeriousDlqin2yrs"
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)


# First split into train+val and test (stratified)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Then split train+val into train and validation (stratified again)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
)


# === 2. Split the dataset ===
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
        random_state=42,
        n_jobs=-1
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