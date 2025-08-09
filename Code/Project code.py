import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
# from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

# from imblearn.over_sampling import SMOTE
# from imblearn.pipeline import Pipeline as ImbPipeline

# === 1. Load Data & Data Pre-processing ===
#  File location for developers:
#   r"D:\Github\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv"
#   r"F:\Waterloo\Actsc\Actsc 445\Project\git\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv"
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

# DebtRatio: coerce, remove inf, cap (separate zero-income cases) ===
dr = pd.to_numeric(df["DebtRatio"], errors="coerce")
dr = dr.mask(np.isinf(dr), np.nan)  # remove ±inf safely

pos_income = df["MonthlyIncome"] > 0
REALISTIC_CAP = 10.0   # tune if needed
HARD_CAP      = 1_000.0  # protects solvers for zero-income rows

dr = np.where(pos_income, np.clip(dr, 0, REALISTIC_CAP), np.clip(dr, 0, HARD_CAP))
# fill any remaining NaNs in the series with median of observed values
dr_series = pd.Series(dr, index=df.index)
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
df.to_csv(r"F:\Waterloo\Actsc\Actsc 445\Project\git\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training-clean.csv", index=False)







# # Features & target
# X = df.drop(columns=["SeriousDlqin2yrs"])
# y = df["SeriousDlqin2yrs"]

# # === 2. SMOTE + Logistic Regression Pipeline (no nested 
#     ("imputer", SimpleImputer(strategy="median")),
#     ("scaler", StandardScaler()),
#     ("smote", SMOTE(random_state=42)),
#     ("clf", LogisticRegression(
#         penalty="l1",
#         solver="saga",
#         max_iter=5000,
#         class_weight="balanced",
#         random_state=42
#     ))
# ])

# # === 3. Hyperparameter Grid ===
# param_grid = {
#     "clf__C": np.logspace(-3, 1, 10)  # C: inverse of regularization strength
# }

# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # === 4. Grid Search CV ===
# grid_search = GridSearchCV(
#     model_pipe,
#     param_grid,
#     scoring="roc_auc",
#     cv=cv,
#     n_jobs=-1,
#     verbose=1
# )

# # Train/Validation Split
# X_train, X_val, y_train, y_val = train_test_split(
#     X, y, test_size=0.3, stratify=y, random_state=42
# )

# # Fit model
# grid_search.fit(X_train, y_train)

# # === 5. Evaluation ===
# print("✅ Best C:", grid_search.best_params_)
# print("✅ CV AUC:", grid_search.best_score_)

# # Holdout evaluation
# y_pred_proba = grid_search.best_estimator_.predict_proba(X_val)[:, 1]
# auc = roc_auc_score(y_val, y_pred_proba)
# print("✅ Hold-out AUC:", auc)

# # Confusion matrix
# y_pred = (y_pred_proba >= 0.5).astype(int)
# print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
# print("\nClassification Report:\n", classification_report(y_val, y_pred, digits=4))

# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc

# # 计算 ROC 曲线
# fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
# roc_auc = auc(fpr, tpr)

# # 画图
# plt.figure(figsize=(8, 6))
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
# plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label="Random guess")
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("Receiver Operating Characteristic (ROC)")
# plt.legend(loc="lower right")
# plt.grid(True)
# plt.tight_layout()
# plt.show()