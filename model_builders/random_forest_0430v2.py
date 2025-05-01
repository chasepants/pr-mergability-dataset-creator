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
required_columns = ['comments', 'additions', 'changed_files', 'description_length']
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"Column '{col}' not found in dataset. Check pr_data_04302025.csv.")

# Features (exclude requested_reviewers_count, has_milestone, has_pr_age, has_comments)
features = [
    "additions", "deletions", "changed_files", "comments", "commits",
    "author_account_age_days", "author_public_repos", "author_merged_prs",
    "title_length", "description_length", "pr_age_days"
]

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

# Train Random Forest with constraints to reduce overfitting
clf = RandomForestClassifier(
    n_estimators=100,
    class_weight={False: 2, True: 1},
    max_depth=10,  # Limit tree depth
    min_samples_split=5,  # Require minimum samples to split
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


# (venv) chase@chase-Lenovo-YOGA-C930-13IKB:~/pr-merge-predictor$ python3 model_builders/random_forest_04302025_v2.py 
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
# dtype: object
# Cross-Validation F1-Macro Scores: [0.63807935 0.72132477 0.55678105 0.58204334 0.53200903]
# Mean CV F1-Macro Score: 0.6060475093249037
# Training Set Classification Report:
#               precision    recall  f1-score   support

#        False       0.98      0.98      0.98       342
#         True       0.99      0.99      0.99       524

#     accuracy                           0.99       866
#    macro avg       0.99      0.99      0.99       866
# weighted avg       0.99      0.99      0.99       866


# Test Set Classification Report:
#               precision    recall  f1-score   support

#        False       0.79      0.83      0.81       102
#         True       0.84      0.80      0.82       115

#     accuracy                           0.82       217
#    macro avg       0.82      0.82      0.82       217
# weighted avg       0.82      0.82      0.82       217

# Test Set ROC-AUC Score: 0.9157715260017051

# Feature Importance:
#                     Feature  Importance
# 7         author_merged_prs    0.258960
# 5   author_account_age_days    0.136878
# 6       author_public_repos    0.122038
# 8              title_length    0.084643
# 9        description_length    0.075530
# 3                  comments    0.073591
# 0                 additions    0.063645
# 1                 deletions    0.052844
# 10              pr_age_days    0.051676
# 4                   commits    0.045262
# 2             changed_files    0.034933

# Sample Predictions:
#      PR Number  Actual Merged  Predicted Merged  Merge Likelihood
# 56        1075           True              True          0.948997
# 428      32968          False             False          0.489990
# 589      29798           True             False          0.417622
# 558      29858           True              True          0.998653
# 51        1080           True              True          0.933194


# Applying the ML Troubleshooting Decision Tree

# Using the decision tree framework, let’s evaluate the results and plan next steps:
# Step 1: Is the overall performance acceptable?

#     Question: Are the F1-score (>0.75) and ROC-AUC (>0.80) good enough?
#     Results:
#         F1-score: 0.81 (False, exceeds target), 0.82 (True, exceeds target).
#         ROC-AUC: 0.9158 (excellent).
#         Accuracy: 0.82 (balanced by 59%/41% split).
#     Evaluation: The model meets your target for both classes, with significant improvement in False class F1-score (0.81 vs. 0.71) and recall (0.83 vs. 0.68), thanks to the 1083-row dataset and 41% False split. The high ROC-AUC confirms reliable likelihood scores. However, overfitting (training F1=0.99 vs. test F1=0.81) and low cross-validation F1-macro scores (mean=0.606) suggest the test performance may be optimistic and unstable across folds.
#     Decision: Performance is acceptable, but overfitting and low CV scores warrant refinement for robustness. Proceed to optional actions in Step 1.

# Step 2: Optional Refinements (since performance is acceptable)

#     Question: Can we improve performance marginally or ensure robustness?
#     Results:
#         Overfitting: Training F1=0.99 vs. test F1=0.81 indicates overfitting, despite max_depth=10 and min_samples_split=5. The low CV F1-macro (0.606) confirms instability.
#         Feature Importance: title_length (0.0846) is moderate, close to the 0.05 threshold for removal. changed_files (0.0349) and commits (0.0453) are low, adding potential noise.
#         Class Balance: The 59%/41% split is well-handled by {False: 2, True: 1}, with no need for SMOTE/undersampling.
#     Evaluation: The model is strong but overfits, and low CV scores suggest generalization issues. Pruning low-importance features and further constraining the model can stabilize performance.
#     Actions:
#         Action 1: Further Reduce Overfitting:
#             Increase min_samples_split (e.g., 10) or reduce max_depth (e.g., 8).
#             Increase min_samples_leaf (e.g., 5) to require more samples per leaf.
#         Action 2: Remove Low-Importance Features:
#             Drop changed_files (0.0349) and commits (0.0453). Monitor title_length (0.0846) for removal if <0.05.
#         Action 3: Fine-Tune Hyperparameters:
#             Grid search for n_estimators, max_depth, min_samples_split, min_samples_leaf to optimize F1-macro CV score.
#         Action 4: Try XGBoost:
#             If CV F1-macro remains <0.75, use XGBoost with scale_pos_weight=2 for better generalization.
#         Action 5: Add Features:
#             Add has_linked_issue by including description in collect_pr_data.py (requires regeneration, ~1 hour).
#             Add has_large_pr (additions + deletions > median) to capture large PRs.
#     Decision: Apply Action 1 (reduce overfitting) and Action 2 (remove changed_files, commits) first, as they don’t require regeneration. Test and monitor title_length.