import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set pandas option to avoid FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# Load data
data = pd.read_csv("datasets/pr_data_0429.csv")

# Drop rows with NaN values (none expected, but included for robustness)
data = data.dropna()

# Verify columns exist
required_columns = ['comments', 'additions', 'description_length']
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"Column '{col}' not found in dataset. Check pr_data_04302025.csv.")

# Features (exclude requested_reviewers_count, has_milestone, has_pr_age, has_comments, changed_files, commits, deletions, title_length)
features = [
    "additions", "comments", "author_account_age_days", "author_public_repos",
    "author_merged_prs", "description_length", "pr_age_days", "has_large_pr"
]

# Create binary feature
median_size = (data["additions"] + data["deletions"]).median()
data["has_large_pr"] = ((data["additions"] + data["deletions"]) > median_size).astype(int)

X = data[features].copy()
y = data["merged"]

# Preprocessing
# Cap outliers at 95th percentile
for col in ["additions", "comments", "author_merged_prs", "description_length", "pr_age_days",
            "author_public_repos"]:
    cap = X[col].quantile(0.95)
    X[col] = X[col].clip(upper=cap)

# Clip negative values to 0 for log-transform
for col in ["pr_age_days"]:
    X[col] = X[col].clip(lower=0)

# Log-transform skewed features (add 1 to avoid log(0))
for col in ["additions", "comments", "author_merged_prs", "description_length", "pr_age_days"]:
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

# Train Random Forest with stronger constraints to reduce overfitting
clf = RandomForestClassifier(
    n_estimators=100,
    class_weight={False: 2, True: 1},
    max_depth=6,  # Further limit tree depth
    min_samples_split=15,  # Increase minimum samples to split
    min_samples_leaf=10,  # Increase minimum samples per leaf
    random_state=42
)
clf.fit(X_train, y_train)

# Cross-validation (5-fold) for stability
cv_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='f1_macro')
print("Cross-Validation F1-Macro Scores:", cv_scores)
print("Mean CV F1-Macro Score:", cv_scores.mean())

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

# (venv) chase@chase-Lenovo-YOGA-C930-13IKB:~/pr-merge-predictor$ 
# Feature Data Types Before Scaling:
# additions                  float64
# comments                   float64
# author_account_age_days      int64
# author_public_repos          int64
# author_merged_prs          float64
# description_length         float64
# pr_age_days                float64
# has_large_pr                 int64
# dtype: object
# Cross-Validation F1-Macro Scores: [0.47908568 0.69735007 0.6414966  0.5466407  0.46203008]
# Mean CV F1-Macro Score: 0.5653206240147575
# Training Set Classification Report:
#               precision    recall  f1-score   support

#        False       0.72      0.85      0.78       342
#         True       0.89      0.78      0.83       524

#     accuracy                           0.81       866
#    macro avg       0.80      0.82      0.81       866
# weighted avg       0.82      0.81      0.81       866


# Test Set Classification Report:
#               precision    recall  f1-score   support

#        False       0.73      0.85      0.79       102
#         True       0.85      0.72      0.78       115

#     accuracy                           0.78       217
#    macro avg       0.79      0.79      0.78       217
# weighted avg       0.79      0.78      0.78       217

# Test Set ROC-AUC Score: 0.8668371696504689

# Feature Importance:
#                    Feature  Importance
# 4        author_merged_prs    0.315139
# 2  author_account_age_days    0.220586
# 3      author_public_repos    0.147302
# 5       description_length    0.089092
# 1                 comments    0.084050
# 0                additions    0.068570
# 6              pr_age_days    0.067694
# 7             has_large_pr    0.007567

# Sample Predictions:
#      PR Number  Actual Merged  Predicted Merged  Merge Likelihood
# 56        1075           True              True          0.837685
# 428      32968          False             False          0.307426
# 589      29798           True             False          0.207933
# 558      29858           True              True          0.903969
# 51        1080           True              True          0.785323


# Applying the ML Troubleshooting Decision Tree

# Using the decision tree framework, let’s evaluate the results and plan next steps:
# Step 1: Is the overall performance acceptable?

#     Question: Are the F1-score (>0.75) and ROC-AUC (>0.80) good enough?
#     Results:
#         F1-score: 0.79 (False, meets target), 0.78 (True, meets target).
#         ROC-AUC: 0.8668 (excellent).
#         Accuracy: 0.78 (balanced by 59%/41% split).
#     Evaluation: The model meets your target for both classes, with a strong F1-score for False (0.79, recall=0.87) and True (0.78, recall=0.72). The ROC-AUC (0.8668) supports reliable likelihood scores. However, overfitting (training F1=0.78-0.83 vs. test F1=0.78-0.79) and low cross-validation F1-macro (0.565) indicate generalization issues, suggesting the test performance may overestimate real-world results. The drop in performance (F1=0.81 to 0.79 for False, 0.80 to 0.78 for True, ROC-AUC=0.8943 to 0.8668) and ineffective has_large_pr (0.0076) show that stricter constraints and the new feature didn’t fully resolve issues.
#     Decision: Performance is acceptable, but overfitting and low CV scores warrant refinement for robustness. Proceed to optional actions in Step 1.

# Step 2: Optional Refinements (since performance is acceptable)

#     Question: Can we improve performance marginally or ensure robustness?
#     Results:
#         Overfitting: Training F1=0.78-0.83 vs. test F1=0.78-0.79 shows reduced overfitting compared to F1=0.86-0.89, but it persists. The low CV F1-macro (0.565) confirms poor generalization.
#         Feature Importance: has_large_pr (0.0076) is ineffective and should be removed. comments (0.0841), additions (0.0686), and pr_age_days (0.0677) are moderate but may add noise.
#         Class Balance: The 59%/41% split is well-handled by {False: 2, True: 1}, with no need for SMOTE/undersampling, as F1=0.79 for False is strong.
#     Evaluation: The model is effective but overfits, and low CV scores indicate unstable generalization. Removing has_large_pr, pruning moderate features, or switching to a more robust algorithm (e.g., XGBoost) can improve CV F1-macro and stability.
#     Actions:
#         Action 1: Further Reduce Overfitting:
#             Increase min_samples_split to 20 and min_samples_leaf to 15.
#             Reduce max_depth to 5.
#         Action 2: Remove Low-Importance Features:
#             Drop has_large_pr (importance=0.0076). Consider dropping comments (0.0841), additions (0.0686), or pr_age_days (0.0677) if CV F1-macro remains low.
#         Action 3: Fine-Tune Hyperparameters:
#             Grid search for n_estimators, max_depth, min_samples_split, min_samples_leaf to optimize CV F1-macro (>0.75).
#         Action 4: Try XGBoost:
#             Switch to XGBoost with scale_pos_weight=2 to improve generalization and CV F1-macro, as Random Forest struggles with stability.
#         Action 5: Add Features:
#             Add has_linked_issue by including description in collect_pr_data.py (requires regeneration, ~1 hour).
#     Decision: Apply Action 2 (remove has_large_pr) and Action 4 (try XGBoost) first, as they address low CV F1-macro without regeneration. If CV F1-macro remains <0.75, proceed to Action 5 (add has_linked_issue).