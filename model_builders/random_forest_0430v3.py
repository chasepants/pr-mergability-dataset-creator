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

# Features (exclude requested_reviewers_count, has_milestone, has_pr_age, has_comments, changed_files, commits)
features = [
    "additions", "deletions", "comments",
    "author_account_age_days", "author_public_repos", "author_merged_prs",
    "title_length", "description_length", "pr_age_days"
]

X = data[features].copy()
y = data["merged"]

# Preprocessing
# Cap outliers at 95th percentile
for col in ["additions", "deletions", "comments",
            "author_merged_prs", "description_length", "pr_age_days",
            "author_public_repos"]:
    cap = X[col].quantile(0.95)
    X[col] = X[col].clip(upper=cap)

# Clip negative values to 0 for log-transform
for col in ["pr_age_days"]:
    X[col] = X[col].clip(lower=0)

# Log-transform skewed features (add 1 to avoid log(0))
for col in ["additions", "deletions", "comments",
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

# Train Random Forest with stronger constraints to reduce overfitting
clf = RandomForestClassifier(
    n_estimators=100,
    class_weight={False: 2, True: 1},
    max_depth=8,  # Further limit tree depth
    min_samples_split=10,  # Increase minimum samples to split
    min_samples_leaf=5,  # Require minimum samples per leaf
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

# (venv) chase@chase-Lenovo-YOGA-C930-13IKB:~/pr-merge-predictor$ python3 model_builders/random_forest_0430v3.py 
# Feature Data Types Before Scaling:
# additions                  float64
# deletions                  float64
# comments                   float64
# author_account_age_days      int64
# author_public_repos          int64
# author_merged_prs          float64
# title_length                 int64
# description_length         float64
# pr_age_days                float64
# dtype: object
# Cross-Validation F1-Macro Scores: [0.49281829 0.71103952 0.6037801  0.51212167 0.4686907 ]
# Mean CV F1-Macro Score: 0.5576900564088494
# Training Set Classification Report:
#               precision    recall  f1-score   support

#        False       0.80      0.92      0.86       342
#         True       0.94      0.85      0.89       524

#     accuracy                           0.88       866
#    macro avg       0.87      0.89      0.88       866
# weighted avg       0.89      0.88      0.88       866


# Test Set Classification Report:
#               precision    recall  f1-score   support

#        False       0.75      0.87      0.81       102
#         True       0.87      0.75      0.80       115

#     accuracy                           0.81       217
#    macro avg       0.81      0.81      0.81       217
# weighted avg       0.81      0.81      0.81       217

# Test Set ROC-AUC Score: 0.8942881500426257

# Feature Importance:
#                    Feature  Importance
# 5        author_merged_prs    0.306487
# 3  author_account_age_days    0.160080
# 4      author_public_repos    0.144957
# 7       description_length    0.082906
# 2                 comments    0.075505
# 6             title_length    0.069240
# 0                additions    0.056479
# 8              pr_age_days    0.054533
# 1                deletions    0.049812

# Sample Predictions:
#      PR Number  Actual Merged  Predicted Merged  Merge Likelihood
# 56        1075           True              True          0.843774
# 428      32968          False             False          0.455952
# 589      29798           True             False          0.261607
# 558      29858           True              True          0.979555
# 51        1080           True              True          0.885080


# Applying the ML Troubleshooting Decision Tree

# Using the decision tree framework, let’s evaluate the results and plan next steps:
# Step 1: Is the overall performance acceptable?

#     Question: Are the F1-score (>0.75) and ROC-AUC (>0.80) good enough?
#     Results:
#         F1-score: 0.81 (False, meets target), 0.80 (True, meets target).
#         ROC-AUC: 0.8943 (excellent).
#         Accuracy: 0.81 (balanced by 59%/41% split).
#     Evaluation: The model meets your target for both classes, with a strong F1-score for False (0.81, recall=0.87) and True (0.80, recall=0.75). The ROC-AUC (0.8943) is robust, though slightly down from 0.9158, likely due to stricter constraints. However, overfitting (training F1=0.86-0.89 vs. test F1=0.80-0.81) and low cross-validation F1-macro (0.557) indicate instability and potential generalization issues, suggesting the test performance may be optimistic.
#     Decision: Performance is acceptable, but overfitting and low CV scores warrant refinement for robustness. Proceed to optional actions in Step 1.

# Step 2: Optional Refinements (since performance is acceptable)

#     Question: Can we improve performance marginally or ensure robustness?
#     Results:
#         Overfitting: Training F1=0.86-0.89 vs. test F1=0.80-0.81 shows reduced overfitting compared to F1=0.99, but it’s still significant. The low CV F1-macro (0.557) confirms poor generalization across folds.
#         Feature Importance: title_length (0.0692) is moderate but close to the 0.05 threshold for removal. deletions (0.0498) is below 0.05, a candidate for pruning.
#         Class Balance: The 59%/41% split is well-handled by {False: 2, True: 1}, with no need for SMOTE/undersampling, as F1=0.81 for False is strong.
#     Evaluation: The model is effective but overfits, and low CV scores suggest unstable generalization. Further constraining the model, pruning low-importance features, or trying a different algorithm (e.g., XGBoost) can improve robustness and CV performance.
#     Actions:
#         Action 1: Further Reduce Overfitting:
#             Increase min_samples_split to 15 and min_samples_leaf to 10.
#             Reduce max_depth to 6.
#         Action 2: Remove Low-Importance Features:
#             Drop deletions (importance=0.0498) and title_length (0.0692, close to 0.05).
#         Action 3: Fine-Tune Hyperparameters:
#             Grid search for n_estimators, max_depth, min_samples_split, min_samples_leaf to optimize CV F1-macro (>0.75).
#         Action 4: Try XGBoost:
#             Use XGBoost with scale_pos_weight=2 to improve generalization and CV F1-macro, as Random Forest struggles with stability.
#         Action 5: Add Features:
#             Add has_linked_issue by including description in collect_pr_data.py (requires regeneration, ~1 hour).
#             Add has_large_pr (additions + deletions > median) to capture large PRs, using existing columns.
#     Decision: Apply Action 1 (reduce overfitting), Action 2 (remove deletions, title_length), and Action 5 (add has_large_pr) first, as they don’t require regeneration. If CV F1-macro remains <0.75, proceed to Action 4 (XGBoost).