import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

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
        raise KeyError(f"Column '{col}' not found in dataset. Check pr_data_0429.csv.")

# Features (exclude requested_reviewers_count, has_milestone, has_pr_age, has_comments, changed_files, commits, deletions, title_length, has_large_pr)
features = [
    "additions", "comments", "author_account_age_days", "author_public_repos",
    "author_merged_prs", "description_length", "pr_age_days"
]

X = data[features].copy()
y = data["merged"]

# Preprocessing
# Cap outliers at 95th percentile
quantiles = {}
for col in ["additions", "comments", "author_merged_prs", "description_length", "pr_age_days",
            "author_public_repos"]:
    quantiles[col] = X[col].quantile(0.95)
    X[col] = X[col].clip(upper=quantiles[col])

# Save quantiles for testing
with open("models/quantiles_0501.pkl", "wb") as f:
    pickle.dump(quantiles, f)

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

# Save scaler for testing
with open("models/scaler_0501.pkl", "wb") as f:
    pickle.dump(scaler, f)

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

# Save model in binary mode (overwrite, not append)
with open("models/xg_boost_0501", "wb") as f:
    pickle.dump(clf, f)