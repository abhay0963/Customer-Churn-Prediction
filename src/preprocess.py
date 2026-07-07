"""
preprocess.py
--------------
This file loads the Telco Customer Churn dataset, cleans it,
encodes categorical columns, scales numerical columns, and
splits the data into training and testing sets.

Author: Abhay | Anshika | Akshit | Riya
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# Columns that are not useful for prediction and are dropped
# ---------------------------------------------------------
COLUMNS_TO_DROP = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Churn Score",
    "CLTV",
    "Churn Reason",
    "Churn Value",
]

# Numeric columns that will be scaled using StandardScaler
NUMERIC_COLUMNS = ["Tenure Months", "Monthly Charges", "Total Charges"]

# Target column
TARGET_COLUMN = "Churn Label"


def load_data(file_path="data/Telco_customer_churn.xlsx"):
    """Loads the raw Excel dataset into a pandas DataFrame."""
    df = pd.read_excel(file_path)
    return df


def clean_data(df):
    """
    Cleans the raw dataset:
    - Drops unnecessary columns
    - Fixes 'Total Charges' data type
    - Fills missing values
    - Converts target column to 0/1
    """
    df = df.copy()

    # Drop columns that do not help in prediction
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    # 'Total Charges' sometimes contains blank strings, convert to numeric
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    # Fill any missing values with median (numeric) / mode (categorical)
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Convert target column: Yes -> 1, No -> 0
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    return df


def encode_data(df):
    """
    Encodes categorical columns using pandas get_dummies (One-Hot Encoding).
    drop_first=True is used to avoid the dummy variable trap.
    """
    df = df.copy()

    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


def scale_data(df, scaler=None, fit=True):
    """
    Scales numeric columns using StandardScaler.
    If fit=True, a new scaler is trained (used for training data).
    If fit=False, the given scaler is only used to transform (used for test/live data).
    """
    df = df.copy()

    if fit:
        scaler = StandardScaler()
        df[NUMERIC_COLUMNS] = scaler.fit_transform(df[NUMERIC_COLUMNS])
    else:
        df[NUMERIC_COLUMNS] = scaler.transform(df[NUMERIC_COLUMNS])

    return df, scaler


def get_preprocessed_data(file_path="data/Telco_customer_churn.xlsx", test_size=0.2, random_state=42):
    """
    Full preprocessing pipeline. Returns:
    X_train, X_test, y_train, y_test, feature_names, scaler
    """
    # Step 1: Load
    df = load_data(file_path)

    # Step 2: Clean
    df = clean_data(df)

    # Step 3: Separate target
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    # Step 4: Encode categorical columns
    X = encode_data(X)

    # Save the final list of feature names (used later in Streamlit app)
    feature_names = X.columns.tolist()

    # Step 5: Split BEFORE scaling to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Step 6: Scale numeric columns (fit only on training data)
    X_train, scaler = scale_data(X_train, fit=True)
    X_test, _ = scale_data(X_test, scaler=scaler, fit=False)

    return X_train, X_test, y_train, y_test, feature_names, scaler


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names, scaler = get_preprocessed_data()
    print("Preprocessing complete!")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("Total features after encoding:", len(feature_names))