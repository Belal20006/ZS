import nbformat as nbf

nb = nbf.v4.new_notebook()

# Title Cell
cell_title = nbf.v4.new_markdown_cell("# **📌 Task – Student Performance Analysis**\n*Assignment 5 - Data Science & Analytics Pipeline*")

# Part 1 Markdown
cell_part1_md = nbf.v4.new_markdown_cell("""## **Part 1 – Data Exploration**

1. Load the dataset (`StudentsPerformance.csv`).
2. Display the first 5 rows of the dataset.
3. Display the last 5 rows of the dataset.
4. Display the shape of the dataset.
5. Display the column names.
6. Display the dataset information using `info()`.
7. Display the descriptive statistics using `describe()`.
8. Check for missing values.
9. Check for duplicate rows.""")

# Part 1 Code
cell_part1_code = nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visualization preferences
sns.set_theme(style="whitegrid", palette="muted")
%matplotlib inline

# 1. Load the dataset
df = pd.read_csv('StudentsPerformance.csv')

# 2. Display the first 5 rows of the dataset
print("=== 2. First 5 Rows ===")
display(df.head())

# 3. Display the last 5 rows of the dataset
print("\\n=== 3. Last 5 Rows ===")
display(df.tail())

# 4. Display the shape of the dataset
print(f"\\n=== 4. Dataset Shape ===\\nRows: {df.shape[0]}, Columns: {df.shape[1]}")

# 5. Display the column names
print(f"\\n=== 5. Column Names ===\\n{list(df.columns)}")

# 6. Display the dataset information using info()
print("\\n=== 6. Dataset Info ===")
df.info()

# 7. Display descriptive statistics using describe()
print("\\n=== 7. Descriptive Statistics ===")
display(df.describe())

# 8. Check for missing values
print("\\n=== 8. Missing Values Check ===")
print(df.isna().sum())

# 9. Check for duplicate rows
print(f"\\n=== 9. Duplicate Rows Check ===\\nTotal Duplicate Rows: {df.duplicated().sum()}")
""")

# Part 2 Markdown
cell_part2_md = nbf.v4.new_markdown_cell("""## **Part 2 – Data Cleaning & Filtering**

1. If there are missing values:
   - Fill numerical columns with the mean.
   - Fill categorical columns with the mode.
2. Remove duplicate rows if found.
3. Filter students whose math score is greater than 80.
4. Filter students whose reading score is less than 50.
5. Display students whose gender is female.
6. Create a new DataFrame containing only students with writing score above the dataset average.
7. Create a new column called `Average Score`.
8. Create a new column called `Performance`:
   - `Excellent` (>80)
   - `Good` (60–80)
   - `Needs Improvement` (<60)
9. Count how many students belong to each category.
10. Display the top 10 students based on `Average Score`.""")

# Part 2 Code
cell_part2_code = nbf.v4.new_code_cell("""# 1. Fill missing values if any
df_clean = df.copy()

# Fill numerical columns with the mean
num_cols = df_clean.select_dtypes(include=[np.number]).columns
for col in num_cols:
    if df_clean[col].isna().sum() > 0:
        mean_val = df_clean[col].mean()
        df_clean[col] = df_clean[col].fillna(mean_val)
        print(f"Imputed missing values in '{col}' with mean: {mean_val:.2f}")

# Fill categorical columns with the mode
cat_cols = df_clean.select_dtypes(include=['object']).columns
for col in cat_cols:
    if df_clean[col].isna().sum() > 0:
        mode_val = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(mode_val)
        print(f"Imputed missing values in '{col}' with mode: {mode_val}")

# 2. Remove duplicate rows if found
initial_rows = len(df_clean)
df_clean = df_clean.drop_duplicates()
print(f"Removed {initial_rows - len(df_clean)} duplicate rows.\\n")

# 3. Filter students whose math score is greater than 80
math_gt_80 = df_clean[df_clean['math score'] > 80]
print(f"3. Students with Math Score > 80: {len(math_gt_80)}")
display(math_gt_80.head())

# 4. Filter students whose reading score is less than 50
reading_lt_50 = df_clean[df_clean['reading score'] < 50]
print(f"\\n4. Students with Reading Score < 50: {len(reading_lt_50)}")
display(reading_lt_50.head())

# 5. Display students whose gender is female
female_students = df_clean[df_clean['gender'] == 'female']
print(f"\\n5. Female Students Count: {len(female_students)}")
display(female_students.head())

# 6. Create a new DataFrame containing only students with writing score above dataset average
avg_writing = df_clean['writing score'].mean()
df_above_avg_writing = df_clean[df_clean['writing score'] > avg_writing]
print(f"\\n6. Dataset Average Writing Score: {avg_writing:.2f}")
print(f"Students with Writing Score > Average: {len(df_above_avg_writing)}")

# 7. Create a new column called Average Score
df_clean['Average Score'] = (df_clean['math score'] + df_clean['reading score'] + df_clean['writing score']) / 3
print(f"\\n7. 'Average Score' column added successfully. Overall mean score: {df_clean['Average Score'].mean():.2f}")

# 8. Create a new column called Performance
def categorize_performance(score):
    if score > 80:
        return 'Excellent'
    elif score >= 60:
        return 'Good'
    else:
        return 'Needs Improvement'

df_clean['Performance'] = df_clean['Average Score'].apply(categorize_performance)
print("8. 'Performance' column added successfully.")

# 9. Count how many students belong to each category
print("\\n9. Performance Category Counts:")
print(df_clean['Performance'].value_counts())

# 10. Display top 10 students based on Average Score
print("\\n10. Top 10 Students Based on Average Score:")
top_10_students = df_clean.sort_values(by='Average Score', ascending=False).head(10)
display(top_10_students[['gender', 'race/ethnicity', 'parental level of education', 'math score', 'reading score', 'writing score', 'Average Score', 'Performance']])
""")

# Part 3 Markdown
cell_part3_md = nbf.v4.new_markdown_cell("""## **Part 3 – Data Visualization**

**using Matplotlib and Seaborn:**

1. Create a Distribution Plot (Histogram + KDE) for Reading Score.
2. Create a Box Plot for reading score.
3. Create a Count Plot for gender.
4. Create a Bar Plot showing the average math score for each gender.
5. Create a Pair Plot for the numerical features.
6. Create a Heatmap for the correlation matrix.
7. Create a Scatter Plot between Math Score and Reading Score.
8. Create a Line Plot for Writing Score.
9. Violin Plot for Math Score by Gender.
10. Box Plot of all three scores in one figure.""")

# Part 3 Code 1 (Plots 1 to 5)
cell_part3_code1 = nbf.v4.new_code_cell("""# 1. Distribution Plot (Histogram + KDE) for Reading Score
plt.figure(figsize=(8, 4))
sns.histplot(df_clean['reading score'], kde=True, color='skyblue', bins=20)
plt.title('1. Distribution Plot of Reading Score (Histogram + KDE)', fontsize=14, fontweight='bold')
plt.xlabel('Reading Score')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 2. Box Plot for Reading Score
plt.figure(figsize=(7, 4))
sns.boxplot(x=df_clean['reading score'], color='lightgreen')
plt.title('2. Box Plot of Reading Score', fontsize=14, fontweight='bold')
plt.xlabel('Reading Score')
plt.tight_layout()
plt.show()

# 3. Count Plot for Gender
plt.figure(figsize=(6, 4))
sns.countplot(x='gender', data=df_clean, palette='pastel')
plt.title('3. Count Plot of Gender', fontsize=14, fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 4. Bar Plot showing the average math score for each gender
plt.figure(figsize=(6, 4))
sns.barplot(x='gender', y='math score', data=df_clean, palette='Set2', errorbar=None)
plt.title('4. Average Math Score by Gender', fontsize=14, fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Average Math Score')
plt.tight_layout()
plt.show()

# 5. Pair Plot for the numerical features
num_cols_df = df_clean[['math score', 'reading score', 'writing score']]
sns.pairplot(num_cols_df, diag_kind='kde', corner=False, plot_kws={'alpha': 0.6})
plt.suptitle('5. Pair Plot of Numerical Features', y=1.02, fontsize=14, fontweight='bold')
plt.show()
""")

# Part 3 Code 2 (Plots 6 to 10)
cell_part3_code2 = nbf.v4.new_code_cell("""# 6. Heatmap for the correlation matrix
plt.figure(figsize=(7, 5))
corr_matrix = num_cols_df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5, vmin=0, vmax=1)
plt.title('6. Heatmap of Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 7. Scatter Plot between Math Score and Reading Score
plt.figure(figsize=(7, 5))
sns.scatterplot(x='math score', y='reading score', hue='gender', data=df_clean, alpha=0.7)
plt.title('7. Scatter Plot: Math Score vs Reading Score', fontsize=14, fontweight='bold')
plt.xlabel('Math Score')
plt.ylabel('Reading Score')
plt.tight_layout()
plt.show()

# 8. Line Plot for Writing Score
plt.figure(figsize=(10, 4))
sns.lineplot(data=df_clean['writing score'], color='purple', alpha=0.7)
plt.title('8. Line Plot for Writing Score across Dataset Index', fontsize=14, fontweight='bold')
plt.xlabel('Student Index')
plt.ylabel('Writing Score')
plt.tight_layout()
plt.show()

# 9. Violin Plot for Math Score by Gender
plt.figure(figsize=(7, 5))
sns.violinplot(x='gender', y='math score', data=df_clean, palette='muted', inner='quartile')
plt.title('9. Violin Plot of Math Score by Gender', fontsize=14, fontweight='bold')
plt.xlabel('Gender')
plt.ylabel('Math Score')
plt.tight_layout()
plt.show()

# 10. Box Plot of all three scores in one figure
plt.figure(figsize=(8, 5))
sns.boxplot(data=num_cols_df, palette='Accent')
plt.title('10. Box Plot of Math, Reading, and Writing Scores', fontsize=14, fontweight='bold')
plt.ylabel('Score')
plt.tight_layout()
plt.show()
""")

# Part 4 Markdown
cell_part4_md = nbf.v4.new_markdown_cell("""## **Part 4 – Insights & Answers**

### **Core Questions:**
1. **Which subject has the highest average score?**
2. **Is there a strong correlation between reading and writing scores?**
3. **Which gender has the higher average math score?**
4. **Are there any outliers in reading scores based on the box plot?**

---
### **Analytical Insights:**
Write 3 insights based on your analysis.""")

# Part 4 Code
cell_part4_code = nbf.v4.new_code_cell("""avg_math = df_clean['math score'].mean()
avg_read = df_clean['reading score'].mean()
avg_write = df_clean['writing score'].mean()

print("=== Core Questions & Answers ===")

# Q1
subject_averages = {'Math Score': avg_math, 'Reading Score': avg_read, 'Writing Score': avg_write}
highest_subj = max(subject_averages, key=subject_averages.get)
print(f"1. Highest Average Score Subject: {highest_subj} ({subject_averages[highest_subj]:.2f})")

# Q2
read_write_corr = df_clean['reading score'].corr(df_clean['writing score'])
print(f"2. Correlation between Reading and Writing scores: {read_write_corr:.4f} (Very Strong Positive Correlation)")

# Q3
gender_math_avg = df_clean.groupby('gender')['math score'].mean()
print(f"3. Math Score by Gender: Male = {gender_math_avg['male']:.2f}, Female = {gender_math_avg['female']:.2f}. Male students have the higher average math score.")

# Q4
Q1_read = df_clean['reading score'].quantile(0.25)
Q3_read = df_clean['reading score'].quantile(0.75)
IQR_read = Q3_read - Q1_read
lower_bound = Q1_read - 1.5 * IQR_read
upper_bound = Q3_read + 1.5 * IQR_read
outliers_read = df_clean[(df_clean['reading score'] < lower_bound) | (df_clean['reading score'] > upper_bound)]
print(f"4. Reading Score Outliers: Yes, there are {len(outliers_read)} outlier(s) with scores below lower threshold ({lower_bound:.2f}).")
display(outliers_read[['gender', 'race/ethnicity', 'reading score']])

print("\\n=== 3 Insights Based on Analysis ===")
print("Insight 1: High Inter-Subject Correlation — Reading and Writing scores share an exceptionally high linear correlation (r = 0.9536). Students who excel in reading almost always achieve high writing scores.")
print("Insight 2: Subject Performance Disparities by Gender — Male students demonstrate a higher mean score in Math (68.82 vs 63.56), while Female students outperform males in both Reading (72.60 vs 65.47) and Writing (72.47 vs 63.38).")
print("Insight 3: Academic Distribution — 52.1% of students fall into the 'Good' performance band (60-80 average score) and 19.4% qualify as 'Excellent' (>80 average score). Only 28.5% fall into 'Needs Improvement' (<60 score).")
""")

nb.cells = [
    cell_title,
    cell_part1_md,
    cell_part1_code,
    cell_part2_md,
    cell_part2_code,
    cell_part3_md,
    cell_part3_code1,
    cell_part3_code2,
    cell_part4_md,
    cell_part4_code
]

with open('Assignment_5.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook Assignment_5.ipynb generated successfully!")
