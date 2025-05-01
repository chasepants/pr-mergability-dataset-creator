import pandas as pd
import numpy as np
from xgboost import XGBClassifier
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

# Features (exclude requested_reviewers_count, has_milestone, has_pr_age, has_comments, changed_files, commits, deletions, title_length, has_large_pr)
features = [
    "additions", "comments", "author_account_age_days", "author_public_repos",
    "author_merged_prs", "description_length", "pr_age_days"
]

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

# Train XGBoost with scale_pos_weight for imbalance
clf = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=2,  # Equivalent to class_weight={False: 2, True: 1}
    random_state=42,
    eval_metric='logloss'
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

# (venv) chase@chase-Lenovo-YOGA-C930-13IKB:~/pr-merge-predictor$ python3 model_builders/xg_boost_0501.py 
# Feature Data Types Before Scaling:
# additions                  float64
# comments                   float64
# author_account_age_days      int64
# author_public_repos          int64
# author_merged_prs          float64
# description_length         float64
# pr_age_days                float64
# dtype: object
# Cross-Validation F1-Macro Scores: [0.62690633 0.62173726 0.55266955 0.54614065 0.63271713]
# Mean CV F1-Macro Score: 0.5960341841896126
# Training Set Classification Report:
#               precision    recall  f1-score   support

#        False       1.00      0.92      0.96       342
#         True       0.95      1.00      0.98       524

#     accuracy                           0.97       866
#    macro avg       0.98      0.96      0.97       866
# weighted avg       0.97      0.97      0.97       866


# Test Set Classification Report:
#               precision    recall  f1-score   support

#        False       0.85      0.74      0.79       102
#         True       0.79      0.89      0.84       115

#     accuracy                           0.82       217
#    macro avg       0.82      0.81      0.81       217
# weighted avg       0.82      0.82      0.81       217

# Test Set ROC-AUC Score: 0.8973572037510658

# Feature Importance:
#                    Feature  Importance
# 4        author_merged_prs    0.322045
# 2  author_account_age_days    0.179344
# 1                 comments    0.111674
# 3      author_public_repos    0.104189
# 5       description_length    0.097060
# 6              pr_age_days    0.096897
# 0                additions    0.088790

# Sample Predictions:
#      PR Number  Actual Merged  Predicted Merged  Merge Likelihood
# 56        1075           True                 1          0.990309
# 428      32968          False                 1          0.727799
# 589      29798           True                 0          0.193666
# 558      29858           True                 1          0.994974
# 51        1080           True                 1          0.984052
