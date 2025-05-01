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
data = pd.read_csv("pr_data_0429.csv")

# Drop rows with NaN values (none expected, but included for robustness)
data = data.dropna()

# Verify columns exist
required_columns = ['comments', 'additions', 'changed_files', 'description_length']
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"Column '{col}' not found in dataset. Check pr_data_04302025.csv.")

# Features (exclude requested_reviewers_count, has_milestone, has_pr_age, has_linked_issue)
features = [
    "additions", "deletions", "changed_files", "comments", "commits",
    "author_account_age_days", "author_public_repos", "author_merged_prs",
    "title_length", "description_length", "pr_age_days", "has_comments"
]

# Create binary feature
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

# Evaluate on training set
y_train_pred = clf.predict(X_train)
print("Training Set Classification Report:")
print(classification_report(y_train, y_train_pred))

# Predict and evaluate on test set
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]  # Likelihood scores
print("\nTest Set Classification Report:")
print(classification_report(y_test, y_pred))
print("Test Set ROC-AUC Score:", roc_auc_score(y_test, y_proba))

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

# (venv) chase@chase-Lenovo-YOGA-C930-13IKB:~/pr-merge-predictor$ python3 random_forest_04302025.py 
# Feature Data Types Before Scaling:
# additions                  float64
# deletions                  float64
# changed_files              float64
# comments                   float64
# commits                    float64
# author_account_age_days      int64
# author_public_repos          int64
# author_merged_prs          float64
# title_length                 int64
# description_length         float64
# pr_age_days                float64
# has_comments                 int64
# dtype: object
# Training Set Classification Report:
#               precision    recall  f1-score   support

#        False       1.00      1.00      1.00       342
#         True       1.00      1.00      1.00       524

#     accuracy                           1.00       866
#    macro avg       1.00      1.00      1.00       866
# weighted avg       1.00      1.00      1.00       866


# Test Set Classification Report:
#               precision    recall  f1-score   support

#        False       0.81      0.77      0.79       102
#         True       0.81      0.83      0.82       115

#     accuracy                           0.81       217
#    macro avg       0.81      0.80      0.81       217
# weighted avg       0.81      0.81      0.81       217

# Test Set ROC-AUC Score: 0.917306052855925

# Feature Importance:
#                     Feature  Importance
# 7         author_merged_prs    0.218645
# 5   author_account_age_days    0.138069
# 6       author_public_repos    0.112506
# 9        description_length    0.089620
# 8              title_length    0.086147
# 0                 additions    0.068536
# 3                  comments    0.063244
# 1                 deletions    0.060224
# 10              pr_age_days    0.054671
# 2             changed_files    0.043659
# 4                   commits    0.043393
# 11             has_comments    0.021287

# Sample Predictions:
#      PR Number  Actual Merged  Predicted Merged  Merge Likelihood
# 56        1075           True              True              0.97
# 428      32968          False              True              0.57
# 589      29798           True             False              0.42
# 558      29858           True              True              1.00
# 51        1080           True              True              0.97

# Applying the ML Troubleshooting Decision Tree

# Using the decision tree framework, let’s evaluate the results and decide next steps:
# Step 1: Is the overall performance acceptable?

#     Question: Are the F1-score (>0.75) and ROC-AUC (>0.80) good enough?
#     Results:
#         F1-score: 0.79 (False, meets target), 0.82 (True, meets target).
#         ROC-AUC: 0.9173 (excellent, above 0.80).
#         Accuracy: 0.81 (solid, balanced by 59%/41% split).
#     Evaluation: The model meets your target (F1 >0.75 for both classes), with a strong ROC-AUC and balanced performance. The False class’s F1-score improved from 0.71 to 0.79, and recall from 0.68 to 0.77, due to the larger dataset (1083 rows) and more False examples (444 vs. 263). Minor errors (e.g., PR 32968) suggest room for marginal gains.
#     Decision: Performance is acceptable, but we can refine for incremental improvements. Proceed to optional actions in Step 1.

# Step 2: Optional Refinements (since performance is acceptable)

#     Question: Can we improve performance marginally or ensure robustness?
#     Results:
#         Training Performance: Perfect F1=1.00 indicates overfitting, as test F1-scores (0.79, 0.82) are significantly lower. This is common with Random Forest on 866 training rows without constraints (e.g., max_depth).
#         Test Performance: F1=0.79 for False is good, but recall (0.77) could be pushed higher to reduce false positives (e.g., PR 32968).
#         Feature Importance: title_length (0.0861) and has_comments (0.0213) are moderate-to-low, suggesting potential noise. author_merged_prs (0.2186) and author_account_age_days (0.1381) dominate.
#     Evaluation: Overfitting is a concern (training F1=1.00 vs. test F1=0.79), and low-importance features may add noise. The 59%/41% split is well-handled by {False: 2, True: 1}, but further tweaks could stabilize performance.
#     Actions:
#         Action 1: Reduce Overfitting:
#             Add max_depth=10 or min_samples_split=5 to Random Forest to limit tree complexity.
#             Use cross-validation (e.g., 5-fold) to stabilize test performance.
#         Action 2: Remove Low-Importance Features:
#             Drop has_comments (importance=0.0213) and possibly title_length (0.0861) if it drops <0.05 in future runs.
#         Action 3: Fine-Tune Hyperparameters:
#             Grid search for n_estimators, max_depth, min_samples_split to optimize F1-score for False.
#         Action 4: Try XGBoost:
#             If F1-score for False doesn’t improve to >0.80, use XGBoost with scale_pos_weight=2 for better imbalance handling.
#         Action 5: Add Features:
#             Reintroduce has_linked_issue by adding description to collect_pr_data.py (requires regeneration, ~1 hour).
#             Add has_large_pr (additions + deletions > median) to capture large PRs, which are less likely to merge.
#     Decision: Apply Action 1 (reduce overfitting) and Action 2 (remove has_comments) first, as they don’t require regeneration. Test and monitor title_length.