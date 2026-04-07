import pandas as pd
import numpy as np

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