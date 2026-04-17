import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# scikit-learn Question 1
# 1. setup the data
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# 2. create and fit the model
model = LinearRegression()
model.fit(years, salary)

# 3. predict for 4 and 8 years
# we reshape to (-1,1) because sklearn expects a 2D array for features
years_to_predict = np.array([4,8]).reshape(-1,1)
predictions = model.predict(years_to_predict)
line_predictions = model.predict(years)
print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"Prediction for 4 years: ${predictions[0]:,.2f}")
print(f"Prediction for 8 years: ${predictions[1]:,.2f}")

# plot
# plt.scatter(years, salary, color="blue", alpha=0.5, label="Data")
# plt.plot(years  , line_predictions, color="red", label="Linear fit")
# plt.xlabel("years")
# plt.ylabel("salary ($)")
# plt.legend()
# plt.show()

# scikit-learn Question 2
# x 1D array shape
# This is just a loose list of numbers. To a computer, it’s like a single row of data spilled out on the floor.
x = np.array([10,20,30,40,50])
print(f"x 1D array: {x}")
print(f"x 1D array shape: {x.shape}")

# x 2D array shape
# This is a Table. It has clear Rows (each person) and a clear Column (their years of experience).

x = np.array([10,20,30,40,50]).reshape(-1,1)
print(f"x 2D array: {x}")
print(f"x 2D array shape: {x.shape}")
# sklearn always expects a Table(Rows and Columns), even if that table only has one column.
# Scikit-Learn is built to handle many features at once (like Years, Age, and Education). It refuses to "guess" how your list of numbers should fit into a table.

# scikit-learn Question 3
# Setup data
X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
print(f"Data shape: {X_clusters.shape}")

# 1. Create and fit the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)

# 2. Predict cluster labels
labels = kmeans.predict(X_clusters)

# 3. Print requested values
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Points per cluster:", np.bincount(labels))

# 4. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left plot: Original unclustered data
ax1.scatter(X_clusters[:, 0], X_clusters[:, 1], color='red', s=60, alpha=0.7)
ax1.set_title("Original Data")
ax1.set_xlabel("Feature 1")
ax1.set_ylabel("Feature 2")

# Right plot: Clustered data with centers
ax2.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', s=60, alpha=0.7)
# Plotting the black X's for centers
ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            color='black', marker='x', s=200, linewidths=3, label="Centers")

ax2.set_title("KMeans Clustering (k=3)")
ax2.set_xlabel("Feature 1")
ax2.set_ylabel("Feature 2")

plt.tight_layout()
plt.show()
plt.savefig("assignments_02/outputs/kmeans_clusters.png")

# Linear Regression
np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Linear Regression Question 1
plt.scatter(age,cost, c=smoker, cmap="coolwarm")
plt.xlabel("age")
plt.ylabel("cost ($)")
plt.title("Medical Cost vs Age")
plt.show()
plt.savefig("assignments_02/outputs/cost_vs_age.png")
# As the figure suggests yes, there are two distnict groups.
# The smoker medical bill is higher than the non smoker specially as the age gets older.

# Linear Regression Question 2
from sklearn.model_selection import train_test_split

# 1. Reshape age into a 2D array (required for X)
X = age.reshape(-1, 1)
y = cost
# 2. Split into training and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Print the shapes of the resulting arrays
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

