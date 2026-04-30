# project_03.py

# --- Setup ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline


# --- Task 1: Load Data ---
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
columns = [f"feature_{i}" for i in range(57)] + ["spam_label"]

df = pd.read_csv(url, header=None, names=columns)

X = df.drop(columns=["spam_label"])
y = df["spam_label"]

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(y.value_counts(normalize=True))

# Comment:
# Dataset has ~4600 emails and is somewhat imbalanced (~60% not spam).
# This means accuracy alone can be misleading.


# --- Feature Exploration ---
features_to_plot = ["feature_15", "feature_52", "feature_56"]

for feature in features_to_plot:
    plt.figure()
    df.boxplot(column=feature, by="spam_label")
    plt.title(feature)
    plt.suptitle("")
    plt.savefig(f"assignments_03/outputs/{feature}_boxplot.png")
    plt.close()

# Comment:
# Spam emails tend to have higher values in these features.
# Many values are zero, showing sparse data.
# Feature scales vary widely → scaling is important.


# --- Task 2: Data Preparation ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Comment:
# Fit scaler only on training data to avoid data leakage.


# --- PCA ---
pca = PCA()
pca.fit(X_train_scaled)

cumulative = np.cumsum(pca.explained_variance_ratio_)

plt.plot(cumulative)
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.savefig("assignments_03/outputs/pca_variance.png")
plt.close()

n = np.argmax(cumulative >= 0.9) + 1
print("\nComponents needed for 90% variance:", n)

X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca = pca.transform(X_test_scaled)[:, :n]


# --- Task 3: Classifiers ---

# KNN
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)
print("\nKNN (Unscaled)")
print(accuracy_score(y_test, knn.predict(X_test)))
print(classification_report(y_test, knn.predict(X_test)))

knn.fit(X_train_scaled, y_train)
print("\nKNN (Scaled)")
print(accuracy_score(y_test, knn.predict(X_test_scaled)))

knn.fit(X_train_pca, y_train)
print("\nKNN (PCA)")
print(accuracy_score(y_test, knn.predict(X_test_pca)))


# Decision Tree depth testing
depths = [3, 5, 10, None]

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)
    
    print(f"\nDecision Tree depth={d}")
    print("Train:", accuracy_score(y_train, tree.predict(X_train)))
    print("Test:", accuracy_score(y_test, tree.predict(X_test)))

# Final tree choice
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

print("\nDecision Tree Final")
print(accuracy_score(y_test, tree.predict(X_test)))
print(classification_report(y_test, tree.predict(X_test)))

# Comment:
# As depth increases, training accuracy rises but test accuracy may drop → overfitting.


# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print("\nRandom Forest")
print(accuracy_score(y_test, rf.predict(X_test)))
print(classification_report(y_test, rf.predict(X_test)))


# Logistic Regression
lr = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')

lr.fit(X_train_scaled, y_train)
print("\nLogistic Regression (Scaled)")
print(accuracy_score(y_test, lr.predict(X_test_scaled)))

lr.fit(X_train_pca, y_train)
print("\nLogistic Regression (PCA)")
print(accuracy_score(y_test, lr.predict(X_test_pca)))


# --- Feature Importances (Random Forest) ---
importances = rf.feature_importances_
indices = np.argsort(importances)[-10:]

plt.barh(range(10), importances[indices])
plt.yticks(range(10), [X.columns[i] for i in indices])
plt.title("Top 10 Feature Importances (Random Forest)")
plt.savefig("assignments_03/outputs/feature_importances.png")
plt.close()


# --- Best Model Confusion Matrix ---
best_model = rf  # adjust if another performs better

y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.title("Best Model Confusion Matrix")
plt.savefig("assignments_03/outputs/best_model_confusion_matrix.png")
plt.close()

# Comment:
# In spam filtering, false positives (ham marked as spam) are usually more costly.


# --- Task 4: Cross Validation ---
models = {
    "KNN_scaled": KNeighborsClassifier(5),
    "DecisionTree": DecisionTreeClassifier(max_depth=5),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "LogReg": LogisticRegression(max_iter=1000, solver='liblinear')
}

for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\n{name}")
    print("Mean:", scores.mean())
    print("Std:", scores.std())

# Comment:
# Random Forest is typically the most stable (low variance).


# --- Task 5: Pipelines ---

# Tree pipeline (no scaling needed)
tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

tree_pipeline.fit(X_train, y_train)
print("\nTree Pipeline")
print(classification_report(y_test, tree_pipeline.predict(X_test)))


# Non-tree pipeline (with PCA)
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=n)),
    ("classifier", LogisticRegression(max_iter=1000, solver='liblinear'))
])

lr_pipeline.fit(X_train, y_train)
print("\nLogistic Regression Pipeline")
print(classification_report(y_test, lr_pipeline.predict(X_test)))

# Comment:
# Pipelines automate preprocessing and prevent mistakes.
# Tree pipeline skips scaling/PCA because trees don't need them.