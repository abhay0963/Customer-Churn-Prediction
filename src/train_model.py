import json

import joblib

import pandas as pd
import matplotlib.pyplot as plt

from preprocess import load_and_preprocess_data

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------

X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data()

# ---------------------------------------------------
# Logistic Regression
# ---------------------------------------------------

print("\n" + "="*60)
print("LOGISTIC REGRESSION")
print("="*60)

lr_model = LogisticRegression(
    max_iter=3000,
    random_state=42
)
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_predictions)
lr_precision = precision_score(y_test, lr_predictions)
lr_recall = recall_score(y_test, lr_predictions)
lr_f1 = f1_score(y_test, lr_predictions)

print(f"Accuracy  : {lr_accuracy:.4f}")
print(f"Precision : {lr_precision:.4f}")
print(f"Recall    : {lr_recall:.4f}")
print(f"F1 Score  : {lr_f1:.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, lr_predictions))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    lr_predictions,
    cmap="Blues"
)

plt.title("Logistic Regression Confusion Matrix")
plt.show()

# ---------------------------------------------------
# Random Forest
# ---------------------------------------------------

print("\n" + "="*60)
print("RANDOM FOREST")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)

print(f"Accuracy  : {rf_accuracy:.4f}")
print(f"Precision : {rf_precision:.4f}")
print(f"Recall    : {rf_recall:.4f}")
print(f"F1 Score  : {rf_f1:.4f}")

print("\nClassification Report\n")
print(classification_report(y_test, rf_predictions))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    rf_predictions,
    cmap="Greens"
)

plt.title("Random Forest Confusion Matrix")
plt.show()

# ---------------------------------------------------
# Compare Models
# ---------------------------------------------------

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        lr_accuracy,
        rf_accuracy
    ],
    "Precision": [
        lr_precision,
        rf_precision
    ],
    "Recall": [
        lr_recall,
        rf_recall
    ],
    "F1 Score": [
        lr_f1,
        rf_f1
    ]
})

print("\n")
print("="*60)
print("MODEL COMPARISON")
print("="*60)

print(comparison)

# ---------------------------------------------------
# Save Best Model
# ---------------------------------------------------

if rf_accuracy > lr_accuracy:
    best_model = rf_model
    print("\nBest Model : Random Forest")
else:
    best_model = lr_model
    print("\nBest Model : Logistic Regression")

joblib.dump(best_model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Save Feature Names
with open("models/feature_names.json", "w") as file:
    json.dump(list(feature_names), file)

print("Feature names saved successfully!")

print("\nModel Saved Successfully!")

# ---------------------------------------------------
# Feature Importance
# ---------------------------------------------------

if isinstance(best_model, RandomForestClassifier):

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": best_model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 10 Important Features\n")

    print(importance.head(10))

    plt.figure(figsize=(10,6))

    plt.barh(
        importance.head(10)["Feature"],
        importance.head(10)["Importance"]
    )

    plt.title("Top 10 Important Features")

    plt.xlabel("Importance")

    plt.gca().invert_yaxis()

    plt.show()