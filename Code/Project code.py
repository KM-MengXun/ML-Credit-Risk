import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer


# --- file paths (edit TEST_PATH if needed) ---
#   r"F:\Waterloo\Actsc\Actsc 445\Project\git\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv"
#   r"D:\Github\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv"

# === 1. Load Data & Data Pre-processing ===
#  File location for developers:
#   r"D:\Github\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv"

df = pd.read_csv(r"D:\Github\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv")

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
 
# Export cleaned data
# df.to_csv(r"F:\Waterloo\Actsc\Actsc 445\Project\git\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training-clean.csv", index=False)


