import argparse
from datetime import datetime
import os
from pathlib import Path
import pickle
import requests

BASE_URL = "https://api.github.com"

# HELPER FUNCTIONS
def get_commit_count(commits_url):
    """Fetch the number of commits in a PR."""
    response = requests.get(commits_url, headers=headers)
    if response.status_code != 200:
        return 0
    return len(response.json())

def get_comments_count(comments_url, review_comments_url):
    """Fetch total comment count (issue comments + review comments)."""
    total_comments = 0
    response = requests.get(comments_url, headers=headers)
    if response.status_code == 200:
        total_comments += len(response.json())
    response = requests.get(review_comments_url, headers=headers)
    if response.status_code == 200:
        total_comments += len(response.json())
    return total_comments

def get_user_stats(username):
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

# Load envs
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("Please update the env variable GITHUB_TOKEN")
    exit

# Load arguments
parser = argparse.ArgumentParser(description="A script that uses ")
parser.add_argument("model", type=str, help="Path to the file containing your pickled model")
parser.add_argument("owner", type=str, help="Owner or organization of the repo")
parser.add_argument("repo", type=str, help="Name of the repo")
parser.add_argument("pr_number", type=int, help="Pull request number you'd like to evaluate")

args = parser.parse_args()

model_path = args.model
owner = args.owner
repo = args.repo
pr_number = args.pr_number

model_file = Path(model_path)
if model_file.exists():
    print(f"Found file '{model_path}'.")
else:
    print(f"File '{model_path}' does not exist.")
    exit

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

if pr_data is None:
    print("Pr could not be found on github")
    exit

commits = get_commit_count(pr_data["commits_url"]) if pr_data.get("commits_url") else 0
comments = get_comments_count(pr_data.get("comments_url", ""), pr_data.get("review_comments_url", ""))

username = pr_data["user"]["login"] if pr_data.get("user") else "unknown"
user_stats = get_user_stats(username) if username != "unknown" else {
    "account_age_days": 0, "public_repos": 0, "merged_prs": 0
}

pr_age_days = 0
created_at = pr_data.get("created_at", "")
closed_at = pr_data.get("closed_at", None)
if created_at and closed_at:
    try:
        created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        closed_date = datetime.strptime(closed_at, "%Y-%m-%dT%H:%M:%SZ")
        pr_age_days = (closed_date - created_date).days
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

loaded_model = pickle.load(model_file.open('rb'))

print(loaded_model.predict(features))