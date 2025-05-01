import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("pr_data_04272025.csv")

# Clean has_milestone: remove rows with invalid values and convert to numeric
invalid_values = ['has_milestone']  # Add other invalid values if found
data = data[~data["has_milestone"].astype(str).isin(invalid_values)]

# Convert numeric columns to numeric, coercing errors to NaN
numeric_cols = [
    "additions", "deletions", "changed_files", "comments", "commits",
    "author_account_age_days", "author_public_repos", "author_merged_prs",
    "title_length", "description_length", "pr_age_days"
]
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Convert has_milestone to numeric (handle all string variations case-insensitively)
data["has_milestone"] = data["has_milestone"].astype(str).str.lower().replace({
    'true': 1, 'false': 0, '1': 1, '0': 0, True: 1, False: 0
}).astype(int)

# Drop rows with NaN values
data = data.dropna()

# Features (exclude requested_reviewers_count, include has_milestone)
features = [
    "additions", "deletions", "changed_files", "comments", "commits",
    "author_account_age_days", "author_public_repos", "author_merged_prs",
    "has_milestone", "title_length", "description_length", "pr_age_days",
    "has_comments", "has_pr_age"
]

# Create binary features for sparse columns
data["has_comments"] = (data["comments"] > 0).astype(int)
data["has_pr_age"] = (data["pr_age_days"] > 0).astype(int)

X = data[features].copy()
y = data["merged"]

# Preprocessing
# Cap outliers at 95th percentile
for col in ["additions", "deletions", "changed_files", "comments", "commits",
            "author_merged_prs", "description_length", "pr_age_days",
            "author_public_repos"]:
    cap = X[col].quantile(0.95)
    X[col] = X[col].clip(upper=cap)

# Clip negative values to 0 for log-transform
for col in ["pr_age_days"]:
    X[col] = X[col].clip(lower=0)

# Log-transform skewed features (add 1 to avoid log(0))
for col in ["additions", "deletions", "changed_files", "comments", "commits",
            "author_merged_prs", "description_length", "pr_age_days"]:
    X[col] = np.log1p(X[col])

# Verify all features are numeric
print("Feature Data Types Before Scaling:")
print(X.dtypes)

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data (80/20) and preserve indices
X_train, X_test, y_train, y_test, X_train_indices, X_test_indices = train_test_split(
    X_scaled, y, X.index, test_size=0.2, random_state=42
)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # Likelihood scores
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Feature importance
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": clf.feature_importances_
}).sort_values(by="Importance", ascending=False)
print("\nFeature Importance:")
print(importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=importance)
plt.title("Feature Importance")
plt.show()

# Example predictions with likelihood
results = pd.DataFrame({
    "PR Number": data.loc[X_test_indices, "pr_number"],
    "Actual Merged": y_test.values,
    "Predicted Merged": y_pred,
    "Merge Likelihood": y_proba
})
print("\nSample Predictions:")
print(results.head())

# Plot likelihood score distribution
plt.figure(figsize=(8, 4))
sns.histplot(y_proba, bins=20, kde=True)
plt.title("Distribution of Merge Likelihood Scores")
plt.xlabel("Likelihood Score")
plt.ylabel("Count")
plt.show()




# (venv) chase@chase-Lenovo-YOGA-C930-13IKB:~/pr-merge-predictor$ python3 random_forest_04272025.py 
# /home/chase/pr-merge-predictor/random_forest_04272025.py:27: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`
#   data["has_milestone"] = data["has_milestone"].astype(str).str.lower().replace({
# Feature Data Types Before Scaling:
# additions                  float64
# deletions                  float64
# changed_files              float64
# comments                   float64
# commits                    float64
# author_account_age_days      int64
# author_public_repos          int64
# author_merged_prs          float64
# has_milestone                int64
# title_length                 int64
# description_length         float64
# pr_age_days                float64
# has_comments                 int64
# has_pr_age                   int64
# dtype: object
# Classification Report:
#               precision    recall  f1-score   support

#        False       0.78      0.68      0.73        57
#         True       0.83      0.89      0.86       101

#     accuracy                           0.82       158
#    macro avg       0.81      0.79      0.80       158
# weighted avg       0.81      0.82      0.81       158

# ROC-AUC Score: 0.8656418273406288

# Feature Importance:
#                     Feature  Importance
# 7         author_merged_prs    0.165199
# 5   author_account_age_days    0.112780
# 3                  comments    0.106643
# 6       author_public_repos    0.087676
# 10       description_length    0.086134
# 12             has_comments    0.080444
# 9              title_length    0.074492
# 0                 additions    0.070570
# 1                 deletions    0.063945
# 4                   commits    0.052380
# 2             changed_files    0.044569
# 11              pr_age_days    0.032261
# 8             has_milestone    0.012426
# 13               has_pr_age    0.010482
# /home/chase/pr-merge-predictor/random_forest_04272025.py:102: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
#   plt.show()

# Sample Predictions:
#     PR Number Actual Merged Predicted Merged  Merge Likelihood
# 511     29932          True             True              1.00
# 39       1098          True             True              1.00
# 211     19636         False            False              0.44
# 199     19675          True             True              0.90
# 235     19565         False             True              1.00
# /home/chase/pr-merge-predictor/random_forest_04272025.py:120: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
#   plt.show()



