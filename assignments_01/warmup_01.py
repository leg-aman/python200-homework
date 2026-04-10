import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
from scipy import stats
import seaborn as sns

############ Pandas Review ############

#  Pandas Q1 
data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)
# print(f"Num Rows: {len(df)}")
# print(f"Header:\n {df.head(2)}")
# print(df.dtypes)

# Pandas Q2
filtered_df = df[(df["passed"]) & (df["grade"] > 80)]
# print(filtered_df)

# Pandas Q3
df["grade_curved"] = df["grade"] + 5
# print(df)

# Pandas Q4
df["name_upper"] = df["name"].apply(str.upper)
# print(df[["name","name_upper"]])

# Pandas Q5
groupedBy_City = df.groupby("city")["grade"].mean()
# print(groupedBy_City)

# Pandas Q6
df["city"] = df["city"].replace("Austin","Houston") 
# print(df)

# Pandas Q7
sorted_df = df.sort_values("grade", ascending=False)
# print(sorted_df.head(3))

############ NumPy Review ############

# NumPy Q1
arr = np.array([10, 20, 30, 40, 50])
# print(arr)
# print(type(arr))
# print(arr.dtype)
# print(arr.shape)
# print(arr.ndim)

# NumPy Question 2
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
# print(arr)
# print(arr.shape)
# print(arr.size)

# NumPy Question 3
# print(arr[0:2,0:2])

# NumPy Question 4
arr_zeros = np.zeros((3,4))
arr_ones = np.ones((2,5))
# print(arr_zeros,arr_ones)

# NumPy Question 5
arr_test = np.arange(0, 50, 5)
# print(f"array: {arr_test} \n Shape: {arr.shape} \n Mean: {np.mean(arr_test)} \n Sum: {np.sum(arr_test)}\n Standard Deviation: {np.std(arr_test)}")

# NumPy Question 6
arr_random = np.random.normal(0,1,200)
# print(f"Mean: {np.mean(arr_random)}")
# print(f"Standard Deviation: {np.std(arr_random)}")

############ Matplot ############

# Matplotlib Question 1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
# plt.plot(x,y)
# plt.title("Squares")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# Matplotlib Question 2
subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
# plt.bar(subjects, scores)
# plt.title("Subject Scores")
# plt.xlabel("Subjects")
# plt.ylabel("Scores")
# plt.show()

# Matplotlib Question 3
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]
# plt.scatter(x1, y1, color="blue", label="Dataset 1")
# plt.scatter(x2, y2, color="red", label="Dataset 2")
# plt.title("Scatter Plot Comparison")
# plt.xlabel("X Axis")
# plt.ylabel("Y Axis")
# plt.legend()
# plt.show()
# Matplotlib Question 4
# plt.subplot(1, 2, 1)
# plt.plot(x,y)
# plt.title("Squares")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.subplot(1, 2, 2)
# plt.bar(subjects, scores)
# plt.title("Subject Scores")
# plt.xlabel("Subjects")
# plt.ylabel("Scores")
# plt.tight_layout()
# plt.show()

############ Descriptive Statistics Review ############

# Descriptive Stats Question 1
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
# print("Mean:", np.mean(data))
# print("Median:", np.median(data))
# print("Variance:", np.var(data))
# print("standard deviation:", np.std(data))

# Descriptive Stats Question 2
# random_data = np.random.normal(65, 10, 500)
# plt.hist(random_data, bins=20)
# plt.title("Distribution of Scores")
# plt.show()

# Descriptive Stats Question 3
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
# plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
# plt.title("Score Comparison")
# plt.ylabel("Score")
# plt.show()

# Descriptive Stats Question 4
normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
# plt.boxplot([normal_data, skewed_data], labels=["normally distributed", "exponential"])
# plt.title("Distribution Comparison")
# plt.ylabel("Score")
# plt.show()
# The skewed_data is more skewed because it has a long tail, whereas the normal_data is balanced and symmetric. 
# For the balanced data, the mean is the best measure of the center, but for the skewed data, the median is more accurate.

# Descriptive Stats Question 5
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
# print("Mean:", np.mean(data1))
# print("Median:", np.median(data1))
# print("Mode:", stats.mode(data1))
# print("Mean:", np.mean(data2))
# print("Median:", np.median(data2))
# print("Mode:", stats.mode(data2))
# The mean and median differ in data2 because the value 150 acts as a significant outlier. 
# While the mean is forced higher by this one large number, the median stays the same because 
# the middle position of the sorted list hasn't changed.

############ Hypothesis Testing Review ############

# Hypothesis Question 1
group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
t_stat, p_val = stats.ttest_ind(group_a, group_b)
# print("t-statistic:", t_stat)
# print("p-value:", p_val)

# Hypothesis Question 2
"""
if p_val < 0.05:
    print("The difference is statistically significant.")
else:
    print("No statistically significant difference detected.")
"""

# Hypothesis Question 3
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]
t_statistic, p_value = stats.ttest_rel(before, after)
#print(t_statistic, p_value)

# Hypothesis Question 4
scores = [72, 68, 75, 70, 69, 74, 71, 73]
scores_tstat, p_val = stats.ttest_1samp(scores, 70)
# print(scores_tstat, p_val)

# Hypothesis Question 5
res = stats.ttest_ind(group_a, group_b, alternative="less")
# print(res.pvalue)

# Hypothesis Question 6
# print("Group B's higher average score suggests a significant difference between the two groups.")
# print(f"Group A mean: {np.mean(group_a)}, Group B mean: {np.mean(group_b)}")
# print(f"t-statistic: {t_stat}, p-value: {p_val}")

############ Correlation Review ############

# Correlation Question 1
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
# I expect, the correlationit will be 1 because y is double of x. 
# Therefore there is a perfect positive linear relationship between the variables.
# corr_matrix = np.corrcoef(x, y)
# print(corr_matrix)

# Correlation Question 2
x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]
r, p = stats.pearsonr(x, y)
# print("Correlation:", round(r, 2))
# print("p-value:", round(p, 4))

# Correlation Question 3
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
corr = df.corr()
# print(corr)

# Correlation Question 4
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.scatter(x, y, color='red')
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
# plt.show()

# Correlation Question 5
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
# plt.show()

############ Pipelines ############

# Pipeline Question 1
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    values = pd.Series(arr)
    return values
def clean_data(series):
    return series.dropna()
def summarize_data(series):
    summary = {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }
    return summary
def data_pipeline(arr):
    values = create_series(arr)
    values_cleaned = clean_data(values)
    return summarize_data(values_cleaned)
summary = data_pipeline(arr)
for key, value in summary.items():
    print(f"{key}: {value}")    