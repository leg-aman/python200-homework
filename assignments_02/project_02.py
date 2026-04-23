from statistics import LinearRegression

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Task 1: Load and Explore
print("==============[ Task 1: Load and Explore ]==============")
df = pd.read_csv("assignments_02/student_performance_math.csv",sep=";")
print("Head: \n", df.head())
print("Shape: ", df.shape)
print("Data Type: ", df.dtypes)

plt.hist(df["G3"], bins=21 )
plt.title('Distribution of Final Math Grades')
plt.xlabel('Student')
plt.ylabel('Grade')
plt.grid()
plt.savefig('assignments_02/outputs/g3_distribution.png')
# plt.show()

# Task 2: Preprocess the Data
print("Shape before dropping rows: ", df.shape)
# drop rows with 0 value
new_df = df[df["G3"] != 0]
print("Shape after dropping rows: ", new_df.shape)
new_df.replace(['yes', 'no'], [1, 0], inplace=True)
new_df.replace(['M', 'F'], [1, 0], inplace=True)

print("*****[ All binary features converted into 0 & 1 ]***** \n",new_df[[
    "sex","schoolsup", "internet", "higher", "activities"
]]
)
# Correlation absences vs G3
corr = df["absences"].corr(df["G3"])
corr_filtered = new_df["absences"].corr(new_df["G3"])

print("\nCorrelation:", corr)
print("Correlation (filtered):", corr_filtered)

# Task 3: Exploratory Data Analysis
num_col = [ "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "absences", "freetime", "goout", "Walc"]

corr = new_df[num_col + ["G3"]].corr()["G3"].drop("G3").sort_values()
print("G3 correlation: ", corr)

# failure vs G3
plt.figure()
plt.scatter(new_df["failures"], new_df["G3"])
plt.xlabel("Failures")
plt.ylabel("G3")
plt.title("Failures vs Final Grade")
plt.savefig("assignments_02/outputs/failures_vs_g3.png")
plt.show()

# studytime vs G3
plt.figure()
plt.scatter(new_df["studytime"], new_df["G3"])
plt.xlabel("Study Time")
plt.ylabel("G3")
plt.title("Study Time vs Final Grade")
plt.savefig("assignments_02/outputs/studytime_vs_g3.png")
plt.show()

# Task 4: Baseline Model
X = new_df[["failures"]].values
y = new_df["G3"].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("Slope: ", model.coef_[0])
print("RMSE: ", rmse)
print("R2 ", r2)


# Task 5: Build the Full Model
feature_cols = ["failures", "Medu", "Fedu", "studytime", "higher", "schoolsup", "internet", "sex", "freetime", "activities", "traveltime"]

X = new_df[feature_cols]
y = new_df["G3"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

print("Train Score:", model.score(X_train, y_train))
print("Test Score:", model.score(X_test, y_test))

print("\nModel Results:")
results = pd.DataFrame(model.coef_, feature_cols, columns=['Coefficient'])
print(results)

# Task 6
plt.figure()
plt.scatter(y_pred, y_test)
plt.plot([0, 20], [0, 20], color="red")
plt.xlabel("predicted")
plt.ylabel("Actual")
plt.title("Predicated vs Actual (Full Model)")
plt.savefig("assignments_02/outputs/predicted_vs_actual.png")
plt.show()

# Neglected Feature: Add G1
feature_cols_g1 = feature_cols + ["G1"]

x = new_df[feature_cols_g1].values
y = new_df["G3"].values

X_train, X_test, y_train, y_test = train_test_split(
    X,y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

test_r2_g1 = model.score(X_test, y_test)
print("Model including G1")
print("Test R2: ", test_r2_g1)