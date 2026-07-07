"""
train_model.py
----------------
This file trains two Machine Learning models (Logistic Regression and
Random Forest) on the preprocessed Telco Customer Churn dataset,
evaluates them, compares their performance, and saves the best model
along with the scaler and feature names for later use in the
Streamlit app.

Author: Abhay | Anshika | Akshit | Riya
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from preprocess import get_preprocessed_data

# ---------------------------------------------------------
# Paths where model files will be saved
# ---------------------------------------------------------
MODEL_PATH = "models/churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.json"
RESULTS_PATH = "models/model_results.csv"
CONFUSION_MATRIX_PATH = "models/confusion_matrix.png"
FEATURE_IMPORTANCE_PATH = "models/feature_importance.png"


def evaluate_model(name, model, X_test, y_test):
    """Calculates evaluation metrics for a given trained model."""
    y_pred = model.predict(X_test)

    results = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
    }

    print(f"\n===== {name} =====")
    print(f"Accuracy  : {results['Accuracy']*100:.2f}%")
    print(f"Precision : {results['Precision']*100:.2f}%")
    print(f"Recall    : {results['Recall']*100:.2f}%")
    print(f"F1 Score  : {results['F1 Score']*100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    return results, cm, y_pred


def save_confusion_matrix_plot(cm, model_name, path=CONFUSION_MATRIX_PATH):
    """Saves a simple, clean confusion matrix plot as a PNG image."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])

    # Write the numbers inside each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)

    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix plot to {path}")


def save_feature_importance_plot(model, feature_names, path=FEATURE_IMPORTANCE_PATH, top_n=10):
    """
    Saves a bar chart of the most important features.
    Uses model.coef_ for Logistic Regression or
    model.feature_importances_ for Random Forest.
    """
    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        print("Model does not support feature importance.")
        return

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1], color="#4C72B0")
    ax.set_title(f"Top {top_n} Important Features")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved feature importance plot to {path}")


def main():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, feature_names, scaler = get_preprocessed_data()

    # ---------------------------------------------------------
    # Train Logistic Regression
    # ---------------------------------------------------------
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train, y_train)
    log_reg_results, log_reg_cm, _ = evaluate_model("Logistic Regression", log_reg, X_test, y_test)

    # ---------------------------------------------------------
    # Train Random Forest
    # ---------------------------------------------------------
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_results, rf_cm, _ = evaluate_model("Random Forest", rf, X_test, y_test)

    # ---------------------------------------------------------
    # Compare models and pick the best one (based on Accuracy)
    # ---------------------------------------------------------
    results_df = pd.DataFrame([log_reg_results, rf_results])
    results_df.to_csv(RESULTS_PATH, index=False)
    print("\nModel comparison saved to", RESULTS_PATH)
    print(results_df)

    if log_reg_results["Accuracy"] >= rf_results["Accuracy"]:
        best_model = log_reg
        best_name = "Logistic Regression"
        best_cm = log_reg_cm
    else:
        best_model = rf
        best_name = "Random Forest"
        best_cm = rf_cm

    print(f"\nSelected Best Model: {best_name}")

    # ---------------------------------------------------------
    # Save plots for the best model
    # ---------------------------------------------------------
    save_confusion_matrix_plot(best_cm, best_name)
    save_feature_importance_plot(best_model, feature_names)

    # ---------------------------------------------------------
    # Save model, scaler, and feature names
    # ---------------------------------------------------------
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(FEATURES_PATH, "w") as f:
        json.dump(feature_names, f)

    print(f"\nSaved trained model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")
    print(f"Saved feature names to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
