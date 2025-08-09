# %%
import pandas as pd
import numpy as np

df_train = pd.read_csv(r"D:\Github\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv")
df_train = df_train[(df_train['age'] > 0) & (df_train['age'] <= 100)]
df_train_sorted = df_train.sort_values(by='age', ascending=True)
df_train['MonthlyIncome'] = df_train['MonthlyIncome'].fillna(df_train['MonthlyIncome'].median())

# %%

# 🚀 Lasso Logistic Regression (baseline model)

from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# === 1. Load Data ===
df_train = pd.read_csv(r"D:\Github\ML-Credit-Risk\Dataset\GiveMeSomeCredit\cs-training.csv")
df_train.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')

# Features & target
X = df_train.drop(columns=["SeriousDlqin2yrs"])
y = df_train["SeriousDlqin2yrs"]

# === 2. SMOTE + Logistic Regression Pipeline (no nested pipeline) ===
model_pipe = ImbPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("clf", LogisticRegression(
        penalty="l1",
        solver="saga",
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    ))
])

# === 3. Hyperparameter Grid ===
param_grid = {
    "clf__C": np.logspace(-3, 1, 10)  # C: inverse of regularization strength
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === 4. Grid Search CV ===
grid_search = GridSearchCV(
    model_pipe,
    param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

# Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Fit model
grid_search.fit(X_train, y_train)

# === 5. Evaluation ===
print("✅ Best C:", grid_search.best_params_)
print("✅ CV AUC:", grid_search.best_score_)

# Holdout evaluation
y_pred_proba = grid_search.best_estimator_.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_proba)
print("✅ Hold-out AUC:", auc)

# Confusion matrix
y_pred = (y_pred_proba >= 0.5).astype(int)
print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred))
print("\nClassification Report:\n", classification_report(y_val, y_pred, digits=4))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 计算 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
roc_auc = auc(fpr, tpr)

# 画图
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label="Random guess")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()