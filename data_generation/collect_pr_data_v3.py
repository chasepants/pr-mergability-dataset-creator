import requests
import csv
import time
import os
from dotenv import load_dotenv
from datetime import datetime

# Load .env file
load_dotenv()

# Read environment variables
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPOSITORIES = os.getenv("REPOSITORIES")

# Validate environment variables
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable is not set")
if not REPOSITORIES:
    raise ValueError("REPOSITORIES environment variable is not set")

# Parse repositories into a list
repositories = [repo.strip() for repo in REPOSITORIES.split(",")]

# GitHub API base URL
BASE_URL = "https://api.github.com"

# Headers for API authentication
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

def get_pull_requests(repo, max_prs=5):
    """Fetch up to max_prs closed pull requests from a repository."""
    url = f"{BASE_URL}/repos/{repo}/pulls?state=closed&per_page={max_prs}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching PRs from {repo}: {response.status_code}")
    return response.json()[:max_prs]

def get_detailed_pr(repo, pr_number):
    """Fetch detailed PR data."""
    url = f"{BASE_URL}/repos/{repo}/pulls/{pr_number}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Warning: Failed to fetch detailed PR {pr_number}: {response.status_code}")
        return None
    return response.json()

def get_commit_count(repo, commits_url):
    """Fetch the number of commits in a PR."""
    response = requests.get(commits_url, headers=headers)
    if response.status_code != 200:
        return 0
    return len(response.json())

def get_comments_count(repo, pr_number, comments_url, review_comments_url):
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

def is_abandoned(pr, comments, commits):
    """Check if a PR is abandoned (no recent activity, no commits/comments)."""
    updated_at = pr.get("updated_at", "")
    if not updated_at:
        return True
    try:
        updated_date = datetime.strptime(updated_at, "%Y-%m-%dT%H:%M:%SZ")
        days_since_update = (datetime.now() - updated_date).days
        return days_since_update > 180 and comments == 0 and commits == 0
    except ValueError:
        return True

def extract_features(pr, repo):
    """Extract features from a pull request."""
    detailed_pr = get_detailed_pr(repo, pr["number"])
    if not detailed_pr:
        return None

    commits = get_commit_count(repo, pr["commits_url"]) if pr.get("commits_url") else 0
    comments = get_comments_count(
        repo, pr["number"], pr.get("comments_url", ""), pr.get("review_comments_url", "")
    )

    if is_abandoned(pr, comments, commits):
        print(f"Skipping abandoned PR {pr['number']} from {repo}")
        return None

    username = pr["user"]["login"] if pr.get("user") else "unknown"
    user_stats = get_user_stats(username) if username != "unknown" else {
        "account_age_days": 0, "public_repos": 0, "merged_prs": 0
    }

    pr_age_days = 0
    created_at = pr.get("created_at", "")
    closed_at = pr.get("closed_at", None)
    if created_at and closed_at:
        try:
            created_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
            closed_date = datetime.strptime(closed_at, "%Y-%m-%dT%H:%M:%SZ")
            pr_age_days = (closed_date - created_date).days
        except ValueError:
            pass

    features = {
        "pr_number": pr["number"],
        "additions": detailed_pr.get("additions", 0),
        "deletions": detailed_pr.get("deletions", 0),
        "changed_files": detailed_pr.get("changed_files", 0),
        "comments": comments,
        "commits": commits,
        "author": pr["user"]["login"] if pr.get("user") else "unknown",
        "author_account_age_days": user_stats["account_age_days"],
        "author_public_repos": user_stats["public_repos"],
        "author_merged_prs": user_stats["merged_prs"],
        "has_milestone": pr["milestone"] is not None,
        "requested_reviewers_count": len(pr.get("requested_reviewers", [])),
        "title_length": len(str(pr.get("title", ""))),
        "description_length": len(str(detailed_pr.get("body", ""))),
        "pr_age_days": pr_age_days
    }
    return features

def is_merged(pr):
    """Check if a pull request was merged."""
    return pr["merged_at"] is not None

def main():
    """Collect pull request data and save it to a CSV file."""
    with open("pr_data_0430.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "pr_number", "additions", "deletions", "changed_files", "comments",
            "commits", "author", "author_account_age_days", "author_public_repos",
            "author_merged_prs", "has_milestone", "requested_reviewers_count",
            "title_length", "description_length", "pr_age_days", "merged"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for repo in repositories:
            print(f"Collecting data from {repo}...")
            try:
                prs = get_pull_requests(repo, max_prs=300)
                for pr in prs:
                    features = extract_features(pr, repo)
                    if features and "author" in features and "dependabot" in features["author"].lower():
                        print(f"Skipping Dependabot PR {features['pr_number']} from {repo}")
                        continue
                    if features:
                        merged = is_merged(pr)
                        writer.writerow({**features, "merged": merged})
                    time.sleep(1)  # Respect rate limits
            except Exception as e:
                print(f"Failed to process {repo}: {e}")

if __name__ == "__main__":
    main()