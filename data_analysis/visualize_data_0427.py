import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from wordcloud import WordCloud

# Set seaborn style for better visuals
sns.set_style("whitegrid")

# Read the dataset
data = pd.read_csv("pr_data_04272025.csv")

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


# === Dataset Overview ===
# Shape: (791, 16)

# Info:
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 791 entries, 0 to 790
# Data columns (total 16 columns):
#  #   Column                     Non-Null Count  Dtype  
# ---  ------                     --------------  -----  
#  0   pr_number                  791 non-null    object 
#  1   additions                  790 non-null    float64
#  2   deletions                  790 non-null    float64
#  3   changed_files              790 non-null    float64
#  4   comments                   790 non-null    float64
#  5   commits                    790 non-null    float64
#  6   author                     791 non-null    object 
#  7   author_account_age_days    790 non-null    float64
#  8   author_public_repos        790 non-null    float64
#  9   author_merged_prs          790 non-null    float64
#  10  has_milestone              791 non-null    bool   
#  11  requested_reviewers_count  790 non-null    float64
#  12  title_length               790 non-null    float64
#  13  description_length         790 non-null    float64
#  14  pr_age_days                790 non-null    float64
#  15  merged                     791 non-null    bool   
# dtypes: bool(2), float64(12), object(2)
# memory usage: 88.2+ KB

# Summary Statistics:
#           additions     deletions  ...  description_length  pr_age_days
# count  7.900000e+02  7.900000e+02  ...          790.000000   790.000000
# mean   4.503697e+03  6.447722e+03  ...         1257.602532     7.048101
# std    8.701828e+04  1.127908e+05  ...         5073.067374    20.189099
# min    0.000000e+00  0.000000e+00  ...            1.000000     0.000000
# 25%    3.000000e+00  1.000000e+00  ...           83.000000     0.000000
# 50%    1.700000e+01  6.000000e+00  ...          393.000000     0.000000
# 75%    8.000000e+01  3.000000e+01  ...         1067.000000     3.000000
# max    2.261619e+06  2.365750e+06  ...        59997.000000   191.000000

# [8 rows x 12 columns]

# Missing Values:
# pr_number                    0
# additions                    1
# deletions                    1
# changed_files                1
# comments                     1
# commits                      1
# author                       0
# author_account_age_days      1
# author_public_repos          1
# author_merged_prs            1
# has_milestone                0
# requested_reviewers_count    1
# title_length                 1
# description_length           1
# pr_age_days                  1
# merged                       0
# dtype: int64

# === Target Variable Distribution (merged) ===
# Proportion of Merged PRs:
# merged
# True     0.666245
# False    0.333755
# Name: proportion, dtype: float64

# === Numeric Feature Distributions ===

# Summary for additions:
# Mean: 4503.70
# Median: 17.00
# Zero Count: 25/791
# NaN Count: 1/791
# Outliers (> Q3 + 1.5*IQR): 99

# Summary for deletions:
# Mean: 6447.72
# Median: 6.00
# Zero Count: 106/791
# NaN Count: 1/791
# Outliers (> Q3 + 1.5*IQR): 126

# Summary for changed_files:
# Mean: 27.98
# Median: 2.00
# Zero Count: 6/791
# NaN Count: 1/791
# Outliers (> Q3 + 1.5*IQR): 110

# Summary for comments:
# Mean: 2.69
# Median: 1.00
# Zero Count: 224/791
# NaN Count: 1/791
# Outliers (> Q3 + 1.5*IQR): 72

# Summary for commits:
# Mean: 2.92
# Median: 1.00
# Zero Count: 6/791
# NaN Count: 1/791
# Outliers (> Q3 + 1.5*IQR): 138

# Summary for author_account_age_days:
# Mean: 3960.44
# Median: 4410.00
# Zero Count: 0/791
# NaN Count: 1/791
# Outliers (> Q3 + 1.5*IQR): 0

# Summary for author_public_repos:
# Mean: 103.88
# Median: 65.00
# Zero Count: 43/791
# NaN Count: 1/791
# Outliers (> Q3 + 1.5*IQR): 56

# Summary for author_merged_prs:
# Mean: 157871.63
# Median: 669.00
# Zero Count: 100/791
# NaN Count: 1/791
# Outliers (> Q3 + 1.5*IQR): 119

# Summary for requested_reviewers_count:
# Mean: 0.07
# Median: 0.00
# Zero Count: 747/791
# NaN Count: 1/791
# Outliers (> Q3 + 1.5*IQR): 43

# Summary for pr_age_days:
# Mean: 7.05
# Median: 0.00
# Zero Count: 485/791
# NaN Count: 1/791
# Outliers (> Q3 + 1.5*IQR): 130

# Summary for title_length:
# Mean: 46.59
# Median: 44.00
# Zero Count: 0/791
# NaN Count: 1/791
# Outliers (> Q3 + 1.5*IQR): 14

# Summary for description_length:
# Mean: 1257.60
# Median: 393.00
# Zero Count: 0/791
# NaN Count: 1/791
# Outliers (> Q3 + 1.5*IQR): 52

# === Boolean Feature Distributions ===
# Proportion of has_milestone:
# has_milestone
# False    0.661188
# True     0.338812
# Name: proportion, dtype: float64

# === Feature-Target Relationships ===

# Feature: additions
# Summary Statistics by Merged Status:
#         Count          Mean  ...  25th Percentile  75th Percentile
# merged                       ...                                  
# False     263  13263.577947  ...              2.5             82.0
# True      527    132.068311  ...              4.0             76.0

# [2 rows x 6 columns]
# Non-Zero Counts by Merged Status:
# merged
# False    246
# True     519
# Name: additions, dtype: int64

# Feature: deletions
# Summary Statistics by Merged Status:
#         Count          Mean  ...  25th Percentile  75th Percentile
# merged                       ...                                  
# False     263  18744.110266  ...              1.0             30.0
# True      527    311.193548  ...              1.0             30.0

# [2 rows x 6 columns]
# Non-Zero Counts by Merged Status:
# merged
# False    218
# True     466
# Name: deletions, dtype: int64

# Feature: changed_files
# Summary Statistics by Merged Status:
#         Count       Mean  Median         Std  25th Percentile  75th Percentile
# merged                                                                        
# False     263  71.456274     2.0  513.495160              1.0              6.0
# True      527   6.286528     2.0   14.841093              1.0              5.0
# Non-Zero Counts by Merged Status:
# merged
# False    257
# True     527
# Name: changed_files, dtype: int64

# Feature: comments
# Summary Statistics by Merged Status:
#         Count      Mean  Median       Std  25th Percentile  75th Percentile
# merged                                                                     
# False     263  3.178707     2.0  3.576109              1.0              4.0
# True      527  2.453510     1.0  3.998719              0.0              3.0
# Non-Zero Counts by Merged Status:
# merged
# False    245
# True     321
# Name: comments, dtype: int64

# Feature: commits
# Summary Statistics by Merged Status:
#         Count      Mean  Median       Std  25th Percentile  75th Percentile
# merged                                                                     
# False     263  3.110266     1.0  6.528213              1.0              2.0
# True      527  2.817837     1.0  4.050657              1.0              3.0
# Non-Zero Counts by Merged Status:
# merged
# False    257
# True     527
# Name: commits, dtype: int64

# Feature: author_account_age_days
# Summary Statistics by Merged Status:
#         Count         Mean  ...  25th Percentile  75th Percentile
# merged                      ...                                  
# False     263  3364.996198  ...           2056.0           4863.0
# True      527  4257.590133  ...           3581.0           5413.0

# [2 rows x 6 columns]
# Non-Zero Counts by Merged Status:
# merged
# False    263
# True     527
# Name: author_account_age_days, dtype: int64

# Feature: author_public_repos
# Summary Statistics by Merged Status:
#         Count        Mean  Median         Std  25th Percentile  75th Percentile
# merged                                                                         
# False     263   83.969582    40.0  134.368210              8.0             86.0
# True      527  113.812144    67.0  166.970299             23.0            139.0
# Non-Zero Counts by Merged Status:
# merged
# False    237
# True     510
# Name: author_public_repos, dtype: int64

# Feature: author_merged_prs
# Summary Statistics by Merged Status:
#         Count           Mean  ...  25th Percentile  75th Percentile
# merged                        ...                                  
# False     263  314688.730038  ...             11.0            832.0
# True      527   79611.867173  ...            198.0           1302.5

# [2 rows x 6 columns]
# Non-Zero Counts by Merged Status:
# merged
# False    221
# True     469
# Name: author_merged_prs, dtype: int64

# Feature: requested_reviewers_count
# Summary Statistics by Merged Status:
#         Count      Mean  Median       Std  25th Percentile  75th Percentile
# merged                                                                     
# False     263  0.057034     0.0  0.303570              0.0              0.0
# True      527  0.075901     0.0  0.323256              0.0              0.0
# Non-Zero Counts by Merged Status:
# merged
# False    11
# True     32
# Name: requested_reviewers_count, dtype: int64

# Feature: pr_age_days
# Summary Statistics by Merged Status:
#         Count      Mean  Median        Std  25th Percentile  75th Percentile
# merged                                                                      
# False     263  8.539924     0.0  23.500474              0.0              3.0
# True      527  6.303605     0.0  18.293313              0.0              3.0
# Non-Zero Counts by Merged Status:
# merged
# False    112
# True     193
# Name: pr_age_days, dtype: int64

# Feature: title_length
# Summary Statistics by Merged Status:
#         Count       Mean  Median        Std  25th Percentile  75th Percentile
# merged                                                                       
# False     263  46.851711    45.0  20.885476             32.0             58.5
# True      527  46.459203    43.0  21.310705             31.0             57.0
# Non-Zero Counts by Merged Status:
# merged
# False    263
# True     527
# Name: title_length, dtype: int64

# Feature: description_length
# Summary Statistics by Merged Status:
#         Count         Mean  ...  25th Percentile  75th Percentile
# merged                      ...                                  
# False     263  2192.365019  ...            157.0           1426.0
# True      527   791.108159  ...             75.0            874.0

# [2 rows x 6 columns]
# Non-Zero Counts by Merged Status:
# merged
# False    263
# True     527
# Name: description_length, dtype: int64
