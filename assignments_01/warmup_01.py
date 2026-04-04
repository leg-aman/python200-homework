import pandas as pd

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
print(groupedBy_City)
# Pandas Q6
#  replace a value in a column

df["city"] = df["city"].replace("Austin","Houston") 
# print(df)

# Pandas Q7
# sorting data

sorted_df = df.sort_values("grade", ascending=False)
# print(sorted_df.head(3))