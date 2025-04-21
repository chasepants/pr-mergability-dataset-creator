import requests
import csv
import time
import os
from dotenv import load_dotenv

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

def get_pull_requests(repo, max_prs=50):
    """Fetch up to max_prs pull requests from a repository."""
    url = f"{BASE_URL}/repos/{repo}/pulls?state=all&per_page={max_prs}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error fetching PRs from {repo}: {response.status_code}")
    return response.json()[:max_prs]

def get_detailed_pr(repo, pr_number):
    """Fetch detailed PR data for fields like additions/deletions."""
    url = f"{BASE_URL}/repos/{repo}/pulls/{pr_number}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Warning: Failed to fetch detailed PR {pr_number}: {response.status_code}")
        return None
    return response.json()

def get_changed_files(repo, pr_number):
    """Fetch list of changed files for a PR."""
    url = f"{BASE_URL}/repos/{repo}/pulls/{pr_number}/files"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Warning: Failed to fetch files for PR {pr_number}: {response.status_code}")
        return []
    return [file["filename"] for file in response.json()]

def get_commit_count(repo, commits_url):
    """Fetch the number of commits in a PR."""
    response = requests.get(commits_url, headers=headers)
    if response.status_code != 200:
        return 0
    return len(response.json())

def extract_features(pr, repo):
    """Extract features from a pull request."""
    detailed_pr = get_detailed_pr(repo, pr["number"])
    if not detailed_pr:
        return None

    commits = get_commit_count(repo, pr["commits_url"]) if pr.get("commits_url") else 0
    changed_files_list = get_changed_files(repo, pr["number"])

    features = {
        "pr_number": pr["number"],
        "additions": detailed_pr.get("additions", 0),
        "deletions": detailed_pr.get("deletions", 0),
        "changed_files": detailed_pr.get("changed_files", 0),
        "comments": pr.get("comments", 0) + pr.get("review_comments", 0),
        "commits": commits,
        "author": pr["user"]["login"] if pr.get("user") else "unknown",
        "description": detailed_pr.get("body", "") or "",
        "changed_files_list": ",".join(changed_files_list) if changed_files_list else ""
    }
    return features

def is_merged(pr):
    """Check if a pull request was merged."""
    return pr["merged_at"] is not None

def main():
    """Collect pull request data and save it to a CSV file."""
    with open("pr_data.csv", "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["pr_number", "additions", "deletions", "changed_files", 
                      "comments", "commits", "author", "description", 
                      "changed_files_list", "merged"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for repo in repositories:
            print(f"Collecting data from {repo}...")
            try:
                prs = get_pull_requests(repo, max_prs=100)
                for pr in prs:
                    features = extract_features(pr, repo)
                    if features["author"].find("dependabot") > -1:
                        print(features["author"])
                        break # ignore dependabot PR commits

                    if features:
                        merged = is_merged(pr)
                        writer.writerow({**features, "merged": merged})
                    time.sleep(1)  # Respect rate limits
            except Exception as e:
                print(f"Failed to process {repo}: {e}")

if __name__ == "__main__":
    main()