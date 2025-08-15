# iris_classifier_portfolio.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------
# 1️⃣ Load dataset
# -------------------------
iris = datasets.load_iris()
X_full = iris.data
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

print("Feature names:", feature_names)
print("Target names:", target_names)

# -------------------------
# 2️⃣ Visualize dataset
# -------------------------
sns.set(style="whitegrid")
iris_df = sns.load_dataset("iris")
sns.pairplot(iris_df, hue="species", diag_kind="hist")
plt.suptitle("Iris Flower Dataset - Pair Plot", y=1.02)
plt.show()

# -------------------------
# 3️⃣ Split data
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

# -------------------------
# 4️⃣ Define models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

# -------------------------
# 5️⃣ Train, evaluate, confusion matrices
# -------------------------
accuracy_dict = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    accuracy_dict[name] = accuracy
    
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_')}.png")
    plt.show()

# -------------------------
# 6️⃣ Model performance summary table
# -------------------------
summary = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    summary.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 2),
        "Precision (macro avg)": round(report["macro avg"]["precision"], 2),
        "Recall (macro avg)": round(report["macro avg"]["recall"], 2),
        "F1-Score (macro avg)": round(report["macro avg"]["f1-score"], 2)
    })

df_summary = pd.DataFrame(summary)
print("\n=== Model Performance Summary ===")
print(df_summary)

# -------------------------
# 7️⃣ Accuracy comparison bar chart
# -------------------------
plt.figure(figsize=(6,4))
sns.barplot(x=list(accuracy_dict.keys()), y=list(accuracy_dict.values()), palette="viridis")
plt.ylim(0, 1.1)
plt.title("Comparison of Model Accuracies")
plt.ylabel("Accuracy")
plt.savefig("accuracy_comparison.png")
plt.show()

# -------------------------
# 8️⃣ Feature importance / coefficients
# -------------------------
# Logistic Regression coefficients
lr = models["Logistic Regression"]
lr.fit(X_train, y_train)
print("\nLogistic Regression Coefficients:")
for f, coef in zip(feature_names, lr.coef_[0]):
    print(f"{f}: {coef:.2f}")

# Decision Tree feature importance
dt = models["Decision Tree"]
dt.fit(X_train, y_train)
print("\nDecision Tree Feature Importance:")
for f, importance in zip(feature_names, dt.feature_importances_):
    print(f"{f}: {importance:.2f}")

# -------------------------
# 9️⃣ Decision boundary visualization (using 2 features)
# -------------------------
X = iris.data[:, 2:4]  # petal length & width
y = iris.target
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

plt.figure(figsize=(18, 5))
for i, (name, model) in enumerate(models.items()):
    model.fit(X_train2, y_train2)
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.subplot(1, 3, i+1)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=[target_names[i] for i in y],
                    palette="Set1", edgecolor="k")
    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")
    plt.title(f"{name} Decision Boundary")

plt.tight_layout()
plt.show()
