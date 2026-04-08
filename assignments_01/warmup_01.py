import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats

# Pandas Q1 
# creating a DataFrame

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
# filtering a DataFrame

filtered_df = df[(df["passed"]) & (df["grade"] > 80)]
# print(filtered_df)

# Pandas Q3
# adding a new column

df["grade_curved"] = df["grade"] + 5
# print(df)

# Pandas Q4
# capitalize names

df["name_upper"] = df["name"].apply(str.upper)
# print(df[["name","name_upper"]])

# Pandas Q5
# Group the DataFrame by "city" and compute the mean grade for each city. 
# Print the result.

groupedBy_City = df.groupby("city")["grade"].mean()
# print(groupedBy_City)
# Pandas Q6
#  replace a value in a column

df["city"] = df["city"].replace("Austin","Houston") 
# print(df)

# Pandas Q7
# sorting data

sorted_df = df.sort_values("grade", ascending=False)
# print(sorted_df.head(3))

# =========================================================================
# NumPy Q1
# Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. 
# Print its shape, dtype, and ndim.

arr = np.array([10, 20, 30, 40, 50])
# print(arr)
# print(type(arr))
# print(arr.dtype)
# print(arr.shape)
# print(arr.ndim)

# NumPy Question 2
# Create the following 2D array and 
# print its shape and size (total number of elements).

arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
# print(arr)
# print(arr.shape)
# print(arr.size)

# NumPy Question 3
# Using the 2D array from Q2, slice out the top-left 2x2 block and print it.
#  The expected result is [[1, 2], [4, 5]].

# print(arr[0:2,0:2])

# NumPy Question 4
# Create a 3x4 array of zeros using a built-in command.
# Then create a 2x5 array of ones using a built-in command. Print both.

arr_zeros = np.zeros((3,4))
arr_ones = np.ones((2,5))
# print(arr_zeros,arr_ones)

# NumPy Question 5
# Create an array using np.arange(0, 50, 5). 
# First, think about what you expect it to look like. 
# Then, print the array, its shape, mean, sum, and standard deviation.

arr_test = np.arange(0, 50, 5)
# print(f"array: {arr_test} \n Shape: {arr.shape} \n Mean: {np.mean(arr_test)} \n Sum: {np.sum(arr_test)}\n Standard Deviation: {np.std(arr_test)}")

# NumPy Question 6
# Generate an array of 200 random values drawn from a normal distribution 
# with mean 0 and standard deviation 1 (use np.random.normal()). Print the mean and standard deviation of the result.

arr_random = np.random.normal(0,1,200)
# print(f"Mean: {np.mean(arr_random)}")
# print(f"Standard Deviation: {np.std(arr_random)}")

# =========================================================================
# Matplotlib Question 1
# Plot the following data as a line plot. Add a title "Squares",
#  x-axis label "x", and y-axis label "y"

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]
# plt.plot(x,y)
# plt.title("Squares")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()

# Matplotlib Question 2
# Create a bar plot for the following subject scores. 
# Add a title "Subject Scores" and label both axes.

subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]
# plt.bar(subjects, scores)
# plt.title("Subject Scores")
# plt.xlabel("Subjects")
# plt.ylabel("Scores")
# plt.show()

# Matplotlib Question 3
# Plot the two datasets below as a scatter plot on the same figure. 
# Use different colors for each, add a legend, and label both axes.

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
# Use plt.subplots() to create a figure with 1 row and 2 subplots side by side.
# In the left subplot, plot x vs y from Q1 as a line. In the right subplot, plot 
# the subjects and scores from Q2 as a bar plot. Add a title to each subplot and 
# call plt.tight_layout() before showing.

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

# Descriptive Statistics Review
# Descriptive Stats Question 1
# Given the list below, use NumPy to compute and print the mean, median, variance, and standard deviation. 
# Label each printed value.

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
# print("Mean:", np.mean(data))
# print("Median:", np.median(data))
# print("Variance:", np.var(data))
# print("standard deviation:", np.std(data))

# Descriptive Stats Question 2
# Generate 500 random values from a normal distribution with mean 65 and standard deviation 10 
# (use np.random.normal(65, 10, 500)). Plot a histogram with 20 bins. 
# Add a title "Distribution of Scores" and label both axes.

# random_data = np.random.normal(65, 10, 500)
# plt.hist(random_data, bins=20)
# plt.title("Distribution of Scores")
# plt.show()

# Descriptive Stats Question 3
# Create a boxplot comparing the two groups below. Label each box ("Group A" and "Group B") and 
# add a title "Score Comparison".
# Hint: pass labels=["Group A", "Group B"] to plt.boxplot().

group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
# plt.boxplot([group_a, group_b], labels=["Group A", "Group B"])
# plt.title("Score Comparison")
# plt.ylabel("Score")
# plt.show()

# Descriptive Stats Question 4
# You are given two datasets: one normally distributed and one 'exponential' distribution.

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
# plt.boxplot([normal_data, skewed_data], labels=["normally distributed", "exponential"])
# plt.title("Distribution Comparison")
# plt.ylabel("Score")
# plt.show()

# The skewed_data is more skewed because it has a long tail, whereas the normal_data is balanced and symmetric. 
# For the balanced data, the mean is the best measure of the center, but for the skewed data, the median is more accurate.

# Descriptive Stats Question 5
# Print the mean, median, and mode of the following:

data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]
# Why are the median and mean so different for data2? Add your answer as a comment in the code.

print("Mean:", np.mean(data1))
print("Median:", np.median(data1))
print("Mode:", stats.mode(data1))

print("Mean:", np.mean(data2))
print("Median:", np.median(data2))
print("Mode:", stats.mode(data2))

# The mean and median differ in data2 because the value 150 acts as a significant outlier. 
# While the mean is forced higher by this one large number, the median stays the same because 
# the middle position of the sorted list hasn't changed.