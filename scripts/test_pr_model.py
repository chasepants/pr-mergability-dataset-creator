import argparse
from datetime import datetime
import os
from pathlib import Path
import pickle
import requests
import pandas as pd
import numpy as np

BASE_URL = "https://api.github.com"

# HELPER FUNCTIONS
def get_commit_count(commits_url, headers):
    """Fetch the number of commits in a PR."""
    response = requests.get(commits_url, headers=headers)
    if response.status_code != 200:
        return 0
    return len(response.json())

def get_comments_count(comments_url, review_comments_url, headers):
    """Fetch total comment count (issue comments + review comments)."""
    total_comments = 0
    response = requests.get(comments_url, headers=headers)
    if response.status_code == 200:
        total_comments += len(response.json())
    response = requests.get(review_comments_url, headers=headers)
    if response.status_code == 200:
        total_comments += len(response.json())
    return total_comments

def get_user_stats(username, headers):
    """Fetch user stats: account age, public repos, merged PRs."""
    url = f"{BASE_URL}/users/{username}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Warning: Failed to fetch user {username}: {response.status_code}")
        return {"created_at": "", "public_repos": 0, "merged_prs": 0}

    user_data = response.json()
    created_at = user_data.get("created_at", "")
    public_repos = user_data.get("public_repos", 0)

    search_url = f"{BASE_URL}/search/issues?q=is:pr is:merged author:{username}"
    search_response = requests.get(search_url, headers=headers)
    merged_prs = search_response.json().get("total_count", 0) if search_response.status_code == 200 else 0

    age_days = 0
    if created_at:
        try:
            created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
            age_days = (datetime.now() - created_date).days
        except ValueError:
            pass

    return {
        "account_age_days": age_days,
        "public_repos": public_repos,
        "merged_prs": merged_prs
    }

# Preprocess input data to match training
def preprocess_input(data, scaler, training_quantiles):
    # Select only the features used in training
    model_features = [
        "additions", "comments", "author_account_age_days", "author_public_repos",
        "author_merged_prs", "description_length", "pr_age_days"
    ]
    data = data[model_features].copy()

    # Cap outliers at training dataset's 95th percentile
    for col in ["additions", "comments", "author_merged_prs", "description_length", "pr_age_days", "author_public_repos"]:
        cap = training_quantiles[col]
        data[col] = data[col].clip(upper=cap)

    # Clip negative values
    data["pr_age_days"] = data["pr_age_days"].clip(lower=0)

    # Log-transform skewed features
    for col in ["additions", "comments", "author_merged_prs", "description_length", "pr_age_days"]:
        data[col] = np.log1p(data[col])

    # Scale features using the training scaler
    X_scaled = scaler.transform(data)
    return X_scaled

# Load envs
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("Please update the env variable GITHUB_TOKEN")
    exit(1)

# Load arguments
parser = argparse.ArgumentParser(description="Predict PR merge likelihood using a trained model")
parser.add_argument("model", type=str, help="Path to the file containing your pickled model")
parser.add_argument("scaler", type=str, help="Path to the file containing your pickled scaler")
parser.add_argument("quantiles", type=str, help="Path to the file containing your pickled quantiles")
parser.add_argument("owner", type=str, help="Owner or organization of the repo")
parser.add_argument("repo", type=str, help="Name of the repo")
parser.add_argument("pr_number", type=int, help="Pull request number you'd like to evaluate")

args = parser.parse_args()

model_path = args.model
scaler_path = args.scaler
quantiles_path = args.quantiles
owner = args.owner
repo = args.repo
pr_number = args.pr_number

# Load model, scaler, and quantiles
model_file = Path(model_path)
scaler_file = Path(scaler_path)
quantile_file = Path(quantiles_path)

if not model_file.exists():
    print(f"File '{model_path}' does not exist.")
    exit(1)
if not scaler_file.exists():
    print(f"Scaler file '{scaler_path}' does not exist. Please retrain to save scaler.")
    exit(1)
if not quantile_file.exists():
    print(f"Quantile file '{quantiles_path}' does not exist. Please retrain to save quantiles.")
    exit(1)

with model_file.open('rb') as f:
    loaded_model = pickle.load(f)
with scaler_file.open('rb') as f:
    scaler = pickle.load(f)
with quantile_file.open('rb') as f:
    training_quantiles = pickle.load(f)

# Fetch pull request
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

url = f"{BASE_URL}/repos/{owner}/{repo}/pulls/{pr_number}"
response = requests.get(url, headers=headers)
if response.status_code != 200:
    raise Exception(f"Error fetching PRs from {repo}: {response.status_code}")

pr_data = response.json()

if not pr_data:
    print("PR could not be found on GitHub")
    exit(1)

# Extract features
commits = get_commit_count(pr_data["commits_url"], headers) if pr_data.get("commits_url") else 0
comments = get_comments_count(pr_data.get("comments_url", ""), pr_data.get("review_comments_url", ""), headers)

username = pr_data["user"]["login"] if pr_data.get("user") else "unknown"
user_stats = get_user_stats(username, headers) if username != "unknown" else {
    "account_age_days": 0, "public_repos": 0, "merged_prs": 0
}

pr_age_days = 0
created_at = pr_data.get("created_at", "")
if created_at:
    try:
        created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        # Use current date for open PRs, closed_at for closed PRs
        end_date = datetime.strptime(pr_data["closed_at"], "%Y-%m-%dT%H:%M:%SZ") if pr_data.get("closed_at") else datetime.now()
        pr_age_days = (end_date - created_date).days
    except ValueError:
        pass

features = {
    "pr_number": pr_data["number"],
    "additions": pr_data.get("additions", 0),
    "deletions": pr_data.get("deletions", 0),
    "changed_files": pr_data.get("changed_files", 0),
    "comments": comments,
    "commits": commits,
    "author": pr_data["user"]["login"] if pr_data.get("user") else "unknown",
    "author_account_age_days": user_stats["account_age_days"],
    "author_public_repos": user_stats["public_repos"],
    "author_merged_prs": user_stats["merged_prs"],
    "has_milestone": pr_data["milestone"] is not None,
    "requested_reviewers_count": len(pr_data.get("requested_reviewers", [])),
    "title_length": len(str(pr_data.get("title", ""))),
    "description_length": len(str(pr_data.get("body", ""))),
    "pr_age_days": pr_age_days
}

# Convert features to DataFrame and preprocess
features_df = pd.DataFrame([features])
X_pr_scaled = preprocess_input(features_df, scaler, training_quantiles)

# Make prediction
prediction = loaded_model.predict(X_pr_scaled)[0]
probability = loaded_model.predict_proba(X_pr_scaled)[0]
print(f"PR {pr_number} Prediction: {'Merged' if prediction else 'Not Merged'}")
print(f"Merge Likelihood: {probability[1]:.4f} (Not Merged: {probability[0]:.4f})")



# EXAMPLE Output
# (venv) chase@chase-Lenovo-YOGA-C930-13IKB:~/pr-merge-predictor$ python3 scripts/test_pr_model.py models/xg_boost_0501 models/scaler_0501.pkl models/quantiles_0501.pkl expressjs express 6464
# Found file 'models/xg_boost_0501'.
# PR 6464 Prediction: Not Merged
# Merge Likelihood: 0.3428 (Not Merged: 0.6572)