import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from wordcloud import WordCloud

# Set seaborn style for better visuals
sns.set_style("whitegrid")

# Read the dataset
data = pd.read_csv("pr_data_0429.csv")

# Fix merged column (handle unexpected 'merged' value)
data["merged"] = data["merged"].replace({"merged": False, "True": True, "False": False}).astype(bool)
data["has_milestone"] = data["has_milestone"].replace({"True": True, "False": False}).astype(bool)

# Define numeric columns based on latest dataset
numeric_cols = [
    "additions", "deletions", "changed_files", "comments", "commits",
    "author_account_age_days", "author_public_repos", "author_merged_prs",
    "requested_reviewers_count", "pr_age_days", "title_length", "description_length"
]

# Convert numeric columns to numeric type, coercing errors to NaN
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# 1. Dataset Overview
print("=== Dataset Overview ===")
print("Shape:", data.shape)
print("\nInfo:")
data.info()
print("\nSummary Statistics:")
print(data.describe())
print("\nMissing Values:")
print(data.isnull().sum())

# 2. Target Variable Distribution
print("\n=== Target Variable Distribution (merged) ===")
plt.figure(figsize=(6, 4))
sns.countplot(x="merged", data=data)
plt.title("Distribution of Merged PRs")
plt.xlabel("Merged (True/False)")
plt.ylabel("Count")
plt.show()
print("Proportion of Merged PRs:")
print(data["merged"].value_counts(normalize=True))

# 3. Numeric Feature Distributions
print("\n=== Numeric Feature Distributions ===")
for col in numeric_cols:
    plt.figure(figsize=(12, 5))
    
    # Histogram with log scale for skewed features
    plt.subplot(1, 2, 1)
    if col in ["additions", "deletions", "comments", "commits", "changed_files"]:
        non_zero_data = data[(data[col] > 0) & (data[col].notna())][col]
        if not non_zero_data.empty:
            sns.histplot(non_zero_data, bins=30, kde=True, log_scale=True)
            plt.title(f"Histogram of {col} (Non-Zero, Log Scale)")
            plt.gca().xaxis.set_major_formatter(ticker.LogFormatterMathtext())
        else:
            sns.histplot(data[col].dropna(), bins=30, kde=True)
            plt.title(f"Histogram of {col} (All Zeros or NaN)")
            plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
            plt.gca().xaxis.get_major_formatter().set_scientific(False)
    else:
        sns.histplot(data[col].dropna(), bins=30, kde=True)
        plt.title(f"Histogram of {col}")
        plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())
        plt.gca().xaxis.get_major_formatter().set_scientific(False)
    plt.xlabel(col)
    plt.ylabel("Count")
    
    # Box plot with outlier annotation
    plt.subplot(1, 2, 2)
    sns.boxplot(y=data[col].dropna())
    plt.title(f"Box Plot of {col}")
    
    if data[col].notna().any():
        q1, q3 = data[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        outliers = data[data[col] > upper_bound][col]
        if not outliers.empty:
            plt.text(0.95, 0.95, f"Outliers: {len(outliers)}", 
                     transform=plt.gca().transAxes, ha='right', va='top')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nSummary for {col}:")
    print(f"Mean: {data[col].mean():.2f}")
    print(f"Median: {data[col].median():.2f}")
    print(f"Zero Count: {(data[col] == 0).sum()}/{len(data)}")
    print(f"NaN Count: {data[col].isna().sum()}/{len(data)}")
    print(f"Outliers (> Q3 + 1.5*IQR): {len(outliers)}")

# 4. Boolean Feature Distributions
boolean_cols = ["has_milestone"]
print("\n=== Boolean Feature Distributions ===")
for col in boolean_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=data)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()
    print(f"Proportion of {col}:")
    print(data[col].value_counts(normalize=True))

# 5. Feature-Target Relationships
print("\n=== Feature-Target Relationships ===")
for col in numeric_cols:
    print(f"\nFeature: {col}")
    stats = data.groupby("merged")[col].agg(["count", "mean", "median", "std", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
    stats.columns = ["Count", "Mean", "Median", "Std", "25th Percentile", "75th Percentile"]
    print("Summary Statistics by Merged Status:")
    print(stats)
    print("Non-Zero Counts by Merged Status:")
    print(data[data[col] > 0].groupby("merged")[col].count())
    
    plt.figure(figsize=(8, 4))
    sns.boxplot(x="merged", y=col, data=data)
    plt.title(f"{col} by Merged Status")
    plt.xlabel("Merged")
    plt.ylabel(col)
    plt.show()

for col in boolean_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue="merged", data=data)
    plt.title(f"{col} by Merged Status")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()