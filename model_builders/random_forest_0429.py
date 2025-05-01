import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set pandas option to avoid FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# Load data
data = pd.read_csv("datasets/pr_data_04272025.csv")

# Convert numeric columns to numeric, coercing errors to NaN
numeric_cols = [
    "additions", "deletions", "changed_files", "comments", "commits",
    "author_account_age_days", "author_public_repos", "author_merged_prs",
    "title_length", "description_length", "pr_age_days"
]
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# Convert merged to boolean (handle string variations case-insensitively)
data["merged"] = data["merged"].astype(str).str.lower().replace({
    'true': True, 'false': False, '1': True, '0': False, True: True, False: False
}).astype(bool)

# Clean data: remove rows with invalid values in has_milestone (if present)
invalid_values = ['has_milestone']  # Add other invalid values if found
data = data[~data["has_milestone"].astype(str).isin(invalid_values)]

# Drop rows with NaN values
data = data.dropna()

# Features (exclude requested_reviewers_count, has_milestone, has_pr_age)
features = [
    "additions", "deletions", "changed_files", "comments", "commits",
    "author_account_age_days", "author_public_repos", "author_merged_prs",
    "title_length", "description_length", "pr_age_days", "has_comments"
]

# Create binary feature for sparse column
data["has_comments"] = (data["comments"] > 0).astype(int)

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

# Train Random Forest with custom class weights
clf = RandomForestClassifier(n_estimators=100, class_weight={False: 2, True: 1}, random_state=42)
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
plt.savefig("feature_importance.png")  # Save plot
plt.close()

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
plt.savefig("likelihood_distribution.png")  # Save plot
plt.close()


# (venv) chase@chase-Lenovo-YOGA-C930-13IKB:~/pr-merge-predictor$ python3 random_forest_04292025.py 
# Feature Data Types Before Scaling:
# additions                  float64
# deletions                  float64
# changed_files              float64
# comments                   float64
# commits                    float64
# author_account_age_days    float64
# author_public_repos        float64
# author_merged_prs          float64
# title_length               float64
# description_length         float64
# pr_age_days                float64
# has_comments                 int64
# dtype: object
# Classification Report:
#               precision    recall  f1-score   support

#        False       0.74      0.68      0.71        57
#         True       0.83      0.86      0.84       101

#     accuracy                           0.80       158
#    macro avg       0.78      0.77      0.78       158
# weighted avg       0.80      0.80      0.80       158

# ROC-AUC Score: 0.8694632621156853

# Feature Importance:
#                     Feature  Importance
# 7         author_merged_prs    0.165546
# 5   author_account_age_days    0.122081
# 3                  comments    0.117621
# 9        description_length    0.094307
# 6       author_public_repos    0.087756
# 8              title_length    0.082100
# 0                 additions    0.071546
# 11             has_comments    0.068525
# 1                 deletions    0.061516
# 4                   commits    0.048762
# 2             changed_files    0.045763
# 10              pr_age_days    0.034476

# Sample Predictions:
#     PR Number  Actual Merged  Predicted Merged  Merge Likelihood
# 511     29932           True              True              1.00
# 39       1098           True              True              1.00
# 211     19636          False             False              0.40
# 199     19675           True              True              0.91
# 235     19565          False              True              1.00




# Changes Made to Model Script

#     Fixed FutureWarning:
#         Added pd.set_option('future.no_silent_downcasting', True) to opt into the new pandas behavior for replace.
#         Why: Prevents the deprecation warning and ensures compatibility with future pandas versions.
#     Dropped Low-Importance Features:
#         Removed has_milestone (importance=0.0124) and has_pr_age (0.0105) from features.
#         Why: Low contribution to predictions, simplifying the model and reducing noise.
#     Adjusted Class Weights:
#         Changed class_weight="balanced" to class_weight={False: 2, True: 1} to emphasize the minority class (False, 33.3%).
#         Why: Improves recall for False (currently 0.68), addressing the imbalance.
#     Saved Plots:
#         Replaced plt.show() with plt.savefig() to save plots (feature_importance.png, likelihood_distribution.png) in a non-interactive environment.
#         Why: Avoids FigureCanvasAgg warnings and allows you to view plots in ~/pr-merge-predictor.
#     Retained Other Fixes:
#         Kept pd.to_numeric() for numeric columns and cleaning for has_milestone to handle string values.
#         Dropped NaN rows to remove the 1 problematic row.
#         Excluded requested_reviewers_count (94.4% zeros).
#         Capped author_merged_prs and log-transformed skewed features (e.g., additions mean=4503.70, median=17).
#         Why: Ensures robust preprocessing for the 791-row dataset.

# Why This Optimizes the Model

#     Performance: The previous F1-score (0.73 for False, 0.86 for True) and ROC-AUC (0.8656) are strong, but custom class weights aim to boost False recall.
#     Feature Selection: Dropping has_milestone and has_pr_age reduces noise, focusing on strong predictors (author_merged_prs, comments, description_length).
#     Future-Proofing: The pandas option fixes the FutureWarning, ensuring compatibility.
#     Visualization: Saved plots improve usability in your terminal environment.