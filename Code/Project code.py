import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer


# --- file paths (edit TEST_PATH if needed) ---
TRAIN_PATH = r"D:\Github\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv"
TEST_PATH  = r"D:\Github\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-test.csv"
#   r"F:\Waterloo\Actsc\Actsc 445\Project\git\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv"

# --- shared bins/labels & caps ---
bins   = [20, 30, 40, 50, 60, 70, 80, 90, 100]
labels = ["20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-100"]
REALISTIC_CAP = 10.0  # for zero-income and normal positive-income rows
IMPUTED_CAP   = 2.0   # for rows whose MonthlyIncome was imputed

# =========================
# 1) TRAIN: your original flow (minimal taps to save params)
# =========================
df_train = pd.read_csv(TRAIN_PATH)

# drop stray index col (assign back)
df_train = df_train.drop(columns=["Unnamed: 0"], errors="ignore")

# keep ages 20–100 and take a real copy (avoids chained-assignment issues)
df_train = df_train[(df_train["age"] >= 20) & (df_train["age"] <= 100)].copy()

# MonthlyIncome: keep 0s, impute only NAs by age-group median ===
df_train["NoIncomeFlag"] = (df_train["MonthlyIncome"] == 0).astype("int8")

df_train["age_group"] = pd.cut(df_train["age"], bins=bins, labels=labels, right=False)

median_by_group_train = (
    df_train.loc[df_train["MonthlyIncome"] > 0]
            .groupby("age_group", observed=False)["MonthlyIncome"]
            .median()
)

# global positive-income median (fallback for empty groups)
global_med_pos_train = df_train.loc[df_train["MonthlyIncome"] > 0, "MonthlyIncome"].median()

mi_na_mask_tr = df_train["MonthlyIncome"].isna()
# minimal change: add fallback to TRAIN global median (prevents leftover NA)
df_train.loc[mi_na_mask_tr, "MonthlyIncome"] = (
    df_train.loc[mi_na_mask_tr, "age_group"].map(median_by_group_train)
            .fillna(global_med_pos_train)
)
df_train["IncomeImputedFlag"] = mi_na_mask_tr.astype("int8")
df_train = df_train.drop(columns=["age_group"])

# DebtRatio: coerce, remove inf, and cap per new rules (0-income=10, imputed=2, normal=10) ===
dr_tr = pd.to_numeric(df_train["DebtRatio"], errors="coerce")
dr_tr = dr_tr.mask(np.isinf(dr_tr), np.nan)  # remove ±inf safely

# Masks from existing columns/flags
pos_income_tr    = df_train["MonthlyIncome"] > 0
was_imputed_tr   = df_train["IncomeImputedFlag"].eq(1)  # MonthlyIncome was NA and then imputed by median
zero_income_tr   = df_train["NoIncomeFlag"].eq(1)       # MonthlyIncome == 0 (flag set before imputation)
normal_income_tr = pos_income_tr & (~was_imputed_tr)    # positive and not imputed

# Start from a clean series
dr_series_tr = pd.Series(dr_tr, index=df_train.index)

# Apply caps with lower bound at 0 (avoid negatives)
dr_series_tr.loc[was_imputed_tr]   = np.clip(dr_series_tr.loc[was_imputed_tr],   0.0, IMPUTED_CAP)
dr_series_tr.loc[zero_income_tr]   = np.clip(dr_series_tr.loc[zero_income_tr],   0.0, REALISTIC_CAP)
dr_series_tr.loc[normal_income_tr] = np.clip(dr_series_tr.loc[normal_income_tr], 0.0, REALISTIC_CAP)

# per your statement: DebtRatio has no NA after coercion/capping
assert not dr_series_tr.isna().any(), "DebtRatio has NA after coercion/capping in TRAIN."
df_train["DebtRatio"] = dr_series_tr.astype(float)

# NumberOfDependents: flag missing, top-code, impute median, large-family flag ===
dep_na_mask_tr = df_train["NumberOfDependents"].isna()
df_train["DependentsMissingFlag"] = dep_na_mask_tr.astype("int8")

# top-code extremes at 5 before imputation
dep_series_tr = df_train["NumberOfDependents"].copy()
dep_series_tr = dep_series_tr.where(dep_series_tr.isna() | (dep_series_tr <= 5), 5)

# save TRAIN dependents median (to reuse on TEST)
_dep_med = dep_series_tr.median(skipna=True)
dep_median_train = int(_dep_med) if not pd.isna(_dep_med) else 0

dep_series_tr = dep_series_tr.fillna(dep_median_train).astype("int8")

df_train["NumberOfDependents"] = dep_series_tr
df_train["LargeFamilyFlag"] = (df_train["NumberOfDependents"] >= 5).astype("int8")

# =========================
# 2) TEST: same flow, but reuse TRAIN-fitted params
# =========================
df_test = pd.read_csv(TEST_PATH)

# drop stray index col (assign back)
df_test = df_test.drop(columns=["Unnamed: 0"], errors="ignore")

# keep ages 20–100 and take a real copy (avoids chained-assignment issues)
df_test = df_test[(df_test["age"] >= 20) & (df_test["age"] <= 100)].copy()

# MonthlyIncome: keep 0s, impute only NAs by TRAIN age-group median ===
df_test["NoIncomeFlag"] = (df_test["MonthlyIncome"] == 0).astype("int8")

df_test["age_group"] = pd.cut(df_test["age"], bins=bins, labels=labels, right=False)

mi_na_mask_te = df_test["MonthlyIncome"].isna()
# reuse TRAIN medians; add fallback to TRAIN global median
df_test.loc[mi_na_mask_te, "MonthlyIncome"] = (
    df_test.loc[mi_na_mask_te, "age_group"].map(median_by_group_train)
           .fillna(global_med_pos_train)
)
df_test["IncomeImputedFlag"] = mi_na_mask_te.astype("int8")
df_test = df_test.drop(columns=["age_group"])

# DebtRatio: coerce, remove inf, and cap per new rules ===
dr_te = pd.to_numeric(df_test["DebtRatio"], errors="coerce")
dr_te = dr_te.mask(np.isinf(dr_te), np.nan)

pos_income_te    = df_test["MonthlyIncome"] > 0
was_imputed_te   = df_test["IncomeImputedFlag"].eq(1)
zero_income_te   = df_test["NoIncomeFlag"].eq(1)
normal_income_te = pos_income_te & (~was_imputed_te)

dr_series_te = pd.Series(dr_te, index=df_test.index)
dr_series_te.loc[was_imputed_te]   = np.clip(dr_series_te.loc[was_imputed_te],   0.0, IMPUTED_CAP)
dr_series_te.loc[zero_income_te]   = np.clip(dr_series_te.loc[zero_income_te],   0.0, REALISTIC_CAP)
dr_series_te.loc[normal_income_te] = np.clip(dr_series_te.loc[normal_income_te], 0.0, REALISTIC_CAP)

# per your statement: DebtRatio has no NA after coercion/capping
assert not dr_series_te.isna().any(), "DebtRatio has NA after coercion/capping in TEST."
df_test["DebtRatio"] = dr_series_te.astype(float)

# NumberOfDependents: flag missing, top-code, impute TRAIN median, large-family flag ===
dep_na_mask_te = df_test["NumberOfDependents"].isna()
df_test["DependentsMissingFlag"] = dep_na_mask_te.astype("int8")

dep_series_te = df_test["NumberOfDependents"].copy()
dep_series_te = dep_series_te.where(dep_series_te.isna() | (dep_series_te <= 5), 5)
# reuse TRAIN dependents median
dep_series_te = dep_series_te.fillna(dep_median_train).astype("int8")

df_test["NumberOfDependents"] = dep_series_te
df_test["LargeFamilyFlag"] = (df_test["NumberOfDependents"] >= 5).astype("int8")

print("Processed shapes:", df_train.shape, df_test.shape)

 
# Export cleaned data
# df.to_csv(r"F:\Waterloo\Actsc\Actsc 445\Project\git\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training-clean.csv", index=False)


