import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# Preprocessing Question 1
# 1. split the data
X_train, X_test,y_train, y_test = train_test_split(
    X,y, test_size=0.2,stratify=y,random_state=42
)

# 2. print the shape
print("Q1: Shapes of datasets")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Preprocessing Question 2
# 1. Initialize the scaler
scaler = StandardScaler()

# 2. Fit and transform the training data, then transform the test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Print the means (should be basically 0)
print("Means of scaled training columns:", X_train_scaled.mean(axis=0))

# Comment: 
# We fit only on X_train to prevent "data leakage," ensuring the model doesn't get any information or hints from the test set before it's officially tested.

# KNN
# KNN Question 1
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# KNN Question 2
# 1. Initialize a new KNN model
knn_scaled = KNeighborsClassifier(n_neighbors=5)

# 2. Fit on the SCALED training data
knn_scaled.fit(X_train_scaled, y_train)

# 3. Predict using the SCALED test data
y_pred_scaled = knn_scaled.predict(X_test_scaled)

# 4. Print the accuracy
print(f"Scaled Accuracy Score: {accuracy_score(y_test, y_pred_scaled):.4f}")

# Comment: For the Iris dataset, scaling usually makes little to no difference because the features (lengths and widths in cm) are already in similar ranges; however, it is still best practice for KNN since it relies on measuring distances between points.

# KNN Question 3
# 1. Initialize the model
knn_cv = KNeighborsClassifier(n_neighbors=5)

# 2. Run 5-fold cross-validation
cv_scores = cross_val_score(knn_cv, X_train, y_train, cv=5)

# 3. Print the results
print(f"Scores for each fold: {cv_scores}")
print(f"Mean Accuracy: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")

# Comment: This result is more trustworthy because it tests the model on five different "mini-exams" instead of just one, ensuring the performance isn't just due to a lucky or unlucky random split of the data.

# KNN Question 4
# 1. Define the list of k values
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

# 2. Loop through each k, calculate CV score, and print
for k in k_values:
    knn_loop = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_loop, X_train, y_train, cv=5)
    print(f"k = {k:2d} | Mean CV Score: {scores.mean():.4f}")

# Comment: I would choose the k with the highest mean accuracy, but if two k-values are tied, I'd pick the larger one (like k=13 over k=1) because higher k-values make the model smoother and less likely to overfit to "weird" individual data points.

# Classifier Evaluation
# Classifier Evaluation Question 1

# 1. Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# 2. Set up the display
# We use the names from the iris dataset (setosa, versicolor, virginica)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

# 3. Plot and Save
disp.plot(cmap=plt.cm.Blues)
plt.title("KNN Confusion Matrix (Unscaled)")
plt.savefig('assignments_03/outputs/knn_confusion_matrix.png')
plt.show()

# Comment: The model most often confuses 'versicolor' and 'virginica' because these two species have very similar physical measurements compared to 'setosa', which is much more distinct.

# The sklearn API: Decision Trees
# Decision Trees Question 1
# 1. Initialize the Decision Tree
# max_depth=3 means the tree can only ask 3 questions before making a guess
dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)

# 2. Fit on unscaled training data
dt_model.fit(X_train, y_train)

# 3. Predict on the test set
y_dt_pred = dt_model.fit(X_train, y_train).predict(X_test)

# 4. Print results
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_dt_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_dt_pred))

# Comment 1: The Decision Tree accuracy is likely very similar to KNN for this dataset, 
# as both are strong classifiers for simple problems like the Iris dataset.

# Comment 2: Scaling would not affect the result for a Decision Tree because it splits 
# data based on "greater than" or "less than" values rather than measuring physical distances.

# Logistic Regression and Regularization
# Logistic Regression Question 1
# List of C values to test
c_values = [0.01, 1.0, 100]

for c in c_values:
    # 1. Initialize and train the model
    # We use scaled data because Logistic Regression prefers it!
    log_reg = (LogisticRegression(C=c, max_iter=1000, solver='lbfgs'))
    log_reg.fit(X_train_scaled, y_train)
    
    # 2. Calculate the total magnitude (size) of the coefficients
    coef_size = np.abs(log_reg.coef_).sum()
    
    # 3. Print the results
    print(f"C = {c:>4} | Total Coefficient Magnitude: {coef_size:.4f}")

# Comment: As C increases, the total coefficient magnitude also increases; 
# this shows that a smaller C applies stronger regularization, which "shrinks" 
# the weights to prevent the model from becoming too complex or over-relying on any single feature.

# PCA
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

# PCA Question 1
# 1. Print the shapes
print(f"X_digits shape: {X_digits.shape}") # Should be (1797, 64)
print(f"images shape:   {images.shape}")   # Should be (1797, 8, 8)

# 2. Create the subplot for digits 0-9
fig, axes = plt.subplots(1, 10, figsize=(15, 3))

for i in range(10):
    # Find the index of the first image that matches digit i
    index = np.where(y_digits == i)[0][0]
    
    # Show the image using the 8x8 version
    axes[i].imshow(images[index], cmap='gray_r')
    axes[i].set_title(f"Label: {i}")
    axes[i].axis('off') # Hide the x/y numbers on the sides

# 3. Save and show
plt.tight_layout()
plt.savefig('assignments_03/outputs/sample_digits.png')
plt.show()

# PCA Question 2
# 1. Initialize and fit PCA (keeping all 64 components for now)
pca = PCA()
pca.fit(X_digits)

# 2. Get the scores (the new "coordinates" for our data)
scores = pca.transform(X_digits)

# 3. Create the scatter plot using the first two Principal Components (PC1 and PC2)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10, alpha=0.7)

# 4. Add labels, colorbar, and save
plt.colorbar(scatter, label='Digit')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Projection of Handwritten Digits')
plt.savefig('assignments_03/outputs/pca_2d_projection.png')
plt.show()

# Comment: Yes, same-digit images tend to cluster together in this 2D space, showing that 
# PCA successfully captured enough information from the original 64 dimensions to 
# separate the different numbers based on their visual similarities.

# PCA Question 3
# 1. Calculate the cumulative (running total) variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# 2. Create the plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 65), cumulative_variance, marker='o', linestyle='--', markersize=4)

# 3. Add a horizontal line at 80% to make it easy to see
plt.axhline(y=0.80, color='r', linestyle='-')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)

# 4. Save and show
plt.savefig('assignments_03/outputs/pca_variance_explained.png')
plt.show()

# Comment: Looking at the plot, you need approximately 13 to 15 components to explain 80% of the variance, 
# which is a huge reduction from the original 64 dimensions!

# PCA Question 4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

# 1. Define our settings
n_list = [2, 5, 15, 40]
n_digits = 5

# 2. Create a grid: 5 rows (Original + 4 n-values) and 5 columns (the digits)
fig, axes = plt.subplots(5, n_digits, figsize=(10, 10))

for col in range(n_digits):
    # Row 0: Original Images
    axes[0, col].imshow(images[col], cmap='gray_r')
    axes[0, col].set_title(f"Orig (Idx {col})")
    axes[0, col].axis('off')
    
    # Rows 1-4: Reconstructions for each n in n_list
    for row_idx, n in enumerate(n_list):
        recon = reconstruct_digit(col, scores, pca, n)
        axes[row_idx + 1, col].imshow(recon, cmap='gray_r')
        axes[row_idx + 1, col].set_title(f"n={n}")
        axes[row_idx + 1, col].axis('off')

plt.tight_layout()
plt.savefig('assignments_03/outputs/pca_reconstructions.png')
plt.show()

# Comment: The digits usually become clearly recognizable around n=15. This matches 
# our variance curve, which started to level off around that same point, meaning 
# those first 15 components captured the "essence" of the shapes.
