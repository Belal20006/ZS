import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_theme(style="whitegrid")

print("=== PART 1: DATA EXPLORATION ===")
# 1. Load the dataset
df = pd.read_csv('StudentsPerformance.csv')
print("\n1. Dataset loaded successfully.")

# 2. Display first 5 rows
print("\n2. First 5 rows:")
print(df.head())

# 3. Display last 5 rows
print("\n3. Last 5 rows:")
print(df.tail())

# 4. Display shape
print(f"\n4. Shape of dataset: {df.shape}")

# 5. Display column names
print(f"\n5. Column names: {list(df.columns)}")

# 6. Display dataset info
print("\n6. Dataset Info:")
df.info()

# 7. Descriptive statistics
print("\n7. Descriptive statistics:")
print(df.describe())

# 8. Check missing values
print("\n8. Missing values count:")
missing_vals = df.isna().sum()
print(missing_vals)

# 9. Check duplicate rows
print(f"\n9. Duplicate rows count: {df.duplicated().sum()}")


print("\n=== PART 2: DATA CLEANING & FILTERING ===")
df_clean = df.copy()

# 1. Fill missing values if any
# Numerical columns mean
num_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if df_clean[col].isna().sum() > 0:
        mean_val = df_clean[col].mean()
        df_clean[col].fillna(mean_val, inplace=True)
        print(f"Filled missing values in numerical column '{col}' with mean: {mean_val:.2f}")

# Categorical columns mode
cat_cols = df_clean.select_dtypes(include=['object']).columns
for col in cat_cols:
    if df_clean[col].isna().sum() > 0:
        mode_val = df_clean[col].mode()[0]
        df_clean[col].fillna(mode_val, inplace=True)
        print(f"Filled missing values in categorical column '{col}' with mode: {mode_val}")

# 2. Remove duplicate rows
init_len = len(df_clean)
df_clean.drop_duplicates(inplace=True)
print(f"Duplicates removed: {init_len - len(df_clean)}")

# 3. Filter math score > 80
math_gt_80 = df_clean[df_clean['math score'] > 80]
print(f"3. Students with Math score > 80: {len(math_gt_80)}")

# 4. Filter reading score < 50
read_lt_50 = df_clean[df_clean['reading score'] < 50]
print(f"4. Students with Reading score < 50: {len(read_lt_50)}")

# 5. Display female students
female_students = df_clean[df_clean['gender'] == 'female']
print(f"5. Total Female students: {len(female_students)}")

# 6. Writing score above average
avg_writing = df_clean['writing score'].mean()
df_high_writing = df_clean[df_clean['writing score'] > avg_writing]
print(f"6. Average writing score: {avg_writing:.2f}. Students with Writing score > average: {len(df_high_writing)}")

# 7. Create Average Score column
df_clean['Average Score'] = (df_clean['math score'] + df_clean['reading score'] + df_clean['writing score']) / 3
print(f"7. 'Average Score' column created. Overall mean score: {df_clean['Average Score'].mean():.2f}")

# 8. Create Performance column
def categorize_performance(score):
    if score > 80:
        return 'Excellent'
    elif score >= 60:
        return 'Good'
    else:
        return 'Needs Improvement'

df_clean['Performance'] = df_clean['Average Score'].apply(categorize_performance)
print("8. 'Performance' column created.")

# 9. Count students per category
perf_counts = df_clean['Performance'].value_counts()
print("\n9. Performance category counts:")
print(perf_counts)

# 10. Top 10 students by Average Score
print("\n10. Top 10 students based on Average Score:")
top_10 = df_clean.sort_values(by='Average Score', ascending=False).head(10)
print(top_10[['gender', 'race/ethnicity', 'parental level of education', 'math score', 'reading score', 'writing score', 'Average Score', 'Performance']])


print("\n=== PART 4: INSIGHTS & ANSWERS ===")
avg_math = df_clean['math score'].mean()
avg_read = df_clean['reading score'].mean()
avg_write = df_clean['writing score'].mean()

print(f"Average Math Score: {avg_math:.2f}")
print(f"Average Reading Score: {avg_read:.2f}")
print(f"Average Writing Score: {avg_write:.2f}")

highest_subject = max([('Math', avg_math), ('Reading', avg_read), ('Writing', avg_write)], key=lambda x: x[1])
print(f"\nQ1. Subject with highest average score: {highest_subject[0]} ({highest_subject[1]:.2f})")

corr_matrix = df_clean[['math score', 'reading score', 'writing score']].corr()
read_write_corr = corr_matrix.loc['reading score', 'writing score']
print(f"\nQ2. Correlation between Reading and Writing scores: {read_write_corr:.4f}")

math_by_gender = df_clean.groupby('gender')['math score'].mean()
print(f"\nQ3. Average Math score by Gender:")
print(math_by_gender)

# Checking outliers in reading score using IQR
Q1 = df_clean['reading score'].quantile(0.25)
Q3 = df_clean['reading score'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_read = df_clean[(df_clean['reading score'] < lower_bound) | (df_clean['reading score'] > upper_bound)]
print(f"\nQ4. Reading Score Outliers (IQR method: Lower {lower_bound:.2f}, Upper {upper_bound:.2f}): {len(outliers_read)} outlier(s) found.")
if len(outliers_read) > 0:
    print(outliers_read[['gender', 'reading score']])
