import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data():

    # ----------------------------
    # Load Dataset
    # ----------------------------
    df = pd.read_excel("data/Telco_customer_churn.xlsx")

    print("=" * 60)
    print("TELCO CUSTOMER CHURN DATASET")
    print("=" * 60)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    # ----------------------------
    # Drop unnecessary columns
    # ----------------------------
    columns_to_drop = [
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
        "Churn Value"
    ]

    df.drop(columns=columns_to_drop, inplace=True)

    # ----------------------------
    # Handle Missing Values
    # ----------------------------
    df["Total Charges"] = pd.to_numeric(
        df["Total Charges"],
        errors="coerce"
    )

    df["Total Charges"] = df["Total Charges"].fillna(
        df["Total Charges"].median()
    )

    # ----------------------------
    # Encode Target Column
    # ----------------------------
    df["Churn Label"] = df["Churn Label"].map({
        "Yes": 1,
        "No": 0
    })

    # ----------------------------
    # Convert Categorical Features
    # ----------------------------
    df = pd.get_dummies(
        df,
        drop_first=True
    )
    # Scale Numerical Columns
    scaler = StandardScaler()

    numerical_columns = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges"
]

    df[numerical_columns] = scaler.fit_transform(
    df[numerical_columns]
)
    # ----------------------------
    # Features and Target
    # ----------------------------
    X = df.drop("Churn Label", axis=1)

    y = df["Churn Label"]

    # ----------------------------
    # Train Test Split
    # ----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining Data :", X_train.shape)
    print("Testing Data  :", X_test.shape)

    print("\nPreprocessing Completed Successfully!")

    return (
    X_train,
    X_test,
    y_train,
    y_test,
    X.columns,
    scaler
)


if __name__ == "__main__":

    load_and_preprocess_data()