"""
app.py
-------
Streamlit web application for the Telco Customer Churn Prediction project.

Pages:
    Home | Dataset | EDA | Model Performance | Prediction | About

Author: Abhay | Anshika | Akshit | Riya
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os

# ===========================================================
# PAGE CONFIG
# ===========================================================
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ===========================================================
# CONSTANTS / PATHS
# ===========================================================
DATA_PATH = "data/Telco_customer_churn.xlsx"
MODEL_PATH = "models/churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.json"
RESULTS_PATH = "models/model_results.csv"
CONFUSION_MATRIX_IMG = "models/confusion_matrix.png"
FEATURE_IMPORTANCE_IMG = "models/feature_importance.png"

NUMERIC_COLUMNS = ["Tenure Months", "Monthly Charges", "Total Charges"]


# ===========================================================
# CACHED LOADERS
# ===========================================================
@st.cache_data
def load_raw_data():
    df = pd.read_excel(DATA_PATH)
    return df


@st.cache_resource
def load_model_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(FEATURES_PATH, "r") as f:
        feature_names = json.load(f)
    return model, scaler, feature_names


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)


# ===========================================================
# HELPER: Encode a single user input into the model's feature format
# ===========================================================
def encode_user_input(raw_input, feature_names, scaler):
    """
    Converts raw user selections (dictionary) into the exact
    30-feature encoded format the model was trained on.

    Logic:
    - Start with all encoded features set to 0.
    - For every categorical value the user picked, check if
      "<column>_<value>" exists in the saved feature list.
      If it does, set it to 1 (this replicates pd.get_dummies
      with drop_first=True, since the "dropped" category is
      simply left as all zeros).
    - Numeric columns are scaled using the saved StandardScaler
      and placed directly into the vector.
    """
    encoded = {feature: 0 for feature in feature_names}

    # Handle categorical columns
    for col, value in raw_input.items():
        if col in NUMERIC_COLUMNS:
            continue
        dummy_col = f"{col}_{value}"
        if dummy_col in encoded:
            encoded[dummy_col] = 1

    # Handle numeric columns (scale them together, in the same order used in training)
    numeric_values = [[raw_input[col] for col in NUMERIC_COLUMNS]]
    scaled_values = scaler.transform(numeric_values)[0]

    for col, scaled_value in zip(NUMERIC_COLUMNS, scaled_values):
        encoded[col] = scaled_value

    # Build final DataFrame in the exact column order used during training
    final_row = pd.DataFrame([encoded])[feature_names]
    return final_row


# ===========================================================
# SIDEBAR NAVIGATION
# ===========================================================
st.sidebar.title("📊 Churn Prediction")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "📁 Dataset", "📈 EDA", "🧪 Model Performance", "🔮 Prediction", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.caption("MCA Machine Learning Assignment")
st.sidebar.caption("Built with Python, Scikit-learn & Streamlit")


# ===========================================================
# PAGE: HOME
# ===========================================================
if page == "🏠 Home":
    st.title("📊 Customer Churn Prediction")
    st.subheader("Predicting Telecom Customer Churn using Machine Learning")

    st.write(
        """
        Customer churn happens when a customer stops using a company's service.
        This project uses **Machine Learning** to predict whether a telecom
        customer is likely to churn, based on their account and service details.
        """
    )

    st.markdown("### 🔑 Key Highlights")

    try:
        df = load_raw_data()
        results_df = load_results()
        best_row = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", f"{df.shape[0]:,}")
        col2.metric("Churn Rate", f"{(df['Churn Label'] == 'Yes').mean() * 100:.1f}%")
        col3.metric("Best Model", best_row["Model"])
        col4.metric("Accuracy", f"{best_row['Accuracy'] * 100:.2f}%")
    except Exception:
        st.info("Run `train_model.py` first to generate model results.")

    st.markdown("### 🛠️ Tech Stack")
    st.write("Python • Pandas • NumPy • Scikit-learn • Matplotlib • Streamlit • Joblib")

    st.markdown("### 🧭 How to Use This App")
    st.write(
        """
        - **📁 Dataset** — Explore the raw dataset used for this project.
        - **📈 EDA** — Visualize churn patterns and customer behavior.
        - **🧪 Model Performance** — Compare model metrics and view charts.
        - **🔮 Prediction** — Enter customer details and predict churn instantly.
        """
    )


# ===========================================================
# PAGE: DATASET
# ===========================================================
elif page == "📁 Dataset":
    st.title("📁 Dataset Overview")

    df = load_raw_data()

    col1, col2 = st.columns(2)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.markdown("### 🔍 Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### 🧾 Column Information")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values
    })
    st.dataframe(info_df, use_container_width=True)

    st.markdown("### 📊 Statistical Summary (Numeric Columns)")
    st.dataframe(df.describe(), use_container_width=True)


# ===========================================================
# PAGE: EDA
# ===========================================================
elif page == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")

    df = load_raw_data()
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    # ---- Churn Distribution (Pie Chart) ----
    st.markdown("### 🥧 Churn Distribution")
    churn_counts = df["Churn Label"].value_counts()

    fig1, ax1 = plt.subplots(figsize=(4, 4))
    ax1.pie(
        churn_counts,
        labels=churn_counts.index,
        autopct="%1.1f%%",
        colors=["#4C72B0", "#DD8452"],
        startangle=90
    )
    ax1.set_title("Churn vs No Churn")
    st.pyplot(fig1)

    st.markdown("---")

    # ---- Histograms ----
    st.markdown("### 📊 Distribution of Numeric Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots()
        ax.hist(df["Tenure Months"].dropna(), bins=20, color="#4C72B0")
        ax.set_title("Tenure (Months)")
        ax.set_xlabel("Months")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots()
        ax.hist(df["Monthly Charges"].dropna(), bins=20, color="#55A868")
        ax.set_title("Monthly Charges")
        ax.set_xlabel("Charges ($)")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots()
        ax.hist(df["Total Charges"].dropna(), bins=20, color="#C44E52")
        ax.set_title("Total Charges")
        ax.set_xlabel("Charges ($)")
        ax.set_ylabel("Number of Customers")
        st.pyplot(fig)

    st.markdown("---")

    # ---- Bar Charts: Categorical vs Churn ----
    st.markdown("### 📊 Churn Rate by Category")

    def churn_rate_by_column(column):
        temp = df.groupby(column)["Churn Label"].apply(lambda x: (x == "Yes").mean() * 100)
        return temp.sort_values(ascending=False)

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Contract Type**")
        data = churn_rate_by_column("Contract")
        fig, ax = plt.subplots()
        ax.bar(data.index, data.values, color="#4C72B0")
        ax.set_ylabel("Churn Rate (%)")
        ax.tick_params(axis="x", rotation=15)
        st.pyplot(fig)

    with col2:
        st.write("**Internet Service**")
        data = churn_rate_by_column("Internet Service")
        fig, ax = plt.subplots()
        ax.bar(data.index, data.values, color="#55A868")
        ax.set_ylabel("Churn Rate (%)")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.write("**Payment Method**")
        data = churn_rate_by_column("Payment Method")
        fig, ax = plt.subplots()
        ax.bar(data.index, data.values, color="#C44E52")
        ax.set_ylabel("Churn Rate (%)")
        ax.tick_params(axis="x", rotation=25)
        st.pyplot(fig)

    with col4:
        st.write("**Senior Citizen**")
        data = churn_rate_by_column("Senior Citizen")
        fig, ax = plt.subplots()
        ax.bar(data.index, data.values, color="#8172B2")
        ax.set_ylabel("Churn Rate (%)")
        st.pyplot(fig)


# ===========================================================
# PAGE: MODEL PERFORMANCE
# ===========================================================
elif page == "🧪 Model Performance":
    st.title("🧪 Model Performance")

    try:
        results_df = load_results()

        st.markdown("### 📋 Model Comparison")
        st.dataframe(results_df, use_container_width=True)

        best_row = results_df.sort_values(by="Accuracy", ascending=False).iloc[0]

        st.markdown(f"### 🏆 Best Model: **{best_row['Model']}**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{best_row['Accuracy'] * 100:.2f}%")
        col2.metric("Precision", f"{best_row['Precision'] * 100:.2f}%")
        col3.metric("Recall", f"{best_row['Recall'] * 100:.2f}%")
        col4.metric("F1 Score", f"{best_row['F1 Score'] * 100:.2f}%")

    except FileNotFoundError:
        st.warning("model_results.csv not found. Please run `train_model.py` first.")

    st.markdown("---")

    st.markdown("### 🔲 Confusion Matrix")
    if os.path.exists(CONFUSION_MATRIX_IMG):
        st.image(CONFUSION_MATRIX_IMG, use_container_width=False)
    else:
        st.info("Confusion matrix image not found. Run `train_model.py` to generate it.")

    st.markdown("### ⭐ Feature Importance")
    if os.path.exists(FEATURE_IMPORTANCE_IMG):
        st.image(FEATURE_IMPORTANCE_IMG, use_container_width=False)
    else:
        st.info("Feature importance image not found. Run `train_model.py` to generate it.")


# ===========================================================
# PAGE: PREDICTION
# ===========================================================
elif page == "🔮 Prediction":
    st.title("🔮 Predict Customer Churn")
    st.write("Fill in the customer's details below and click **Predict** to see the result.")

    try:
        model, scaler, feature_names = load_model_artifacts()
    except FileNotFoundError:
        st.error("Model files not found. Please run `train_model.py` first to train and save the model.")
        st.stop()

    with st.form("prediction_form"):
        st.markdown("#### 👤 Customer Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        with col2:
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        with col3:
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

        st.markdown("#### 🌐 Subscribed Services")
        col4, col5, col6 = st.columns(3)

        with col4:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

        with col5:
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

        with col6:
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.markdown("#### 💳 Billing")
        col7, col8 = st.columns(2)
        with col7:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=500.0, value=70.0)
        with col8:
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0)

        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        raw_input = {
            "Gender": gender,
            "Senior Citizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure Months": tenure,
            "Phone Service": phone_service,
            "Multiple Lines": multiple_lines,
            "Internet Service": internet_service,
            "Online Security": online_security,
            "Online Backup": online_backup,
            "Device Protection": device_protection,
            "Tech Support": tech_support,
            "Streaming TV": streaming_tv,
            "Streaming Movies": streaming_movies,
            "Contract": contract,
            "Paperless Billing": paperless_billing,
            "Payment Method": payment_method,
            "Monthly Charges": monthly_charges,
            "Total Charges": total_charges,
        }

        input_row = encode_user_input(raw_input, feature_names, scaler)

        prediction = model.predict(input_row)[0]
        probability = model.predict_proba(input_row)[0][1]

        st.markdown("---")
        st.markdown("### 📢 Result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("⚠️ This customer is **likely to churn**.")
            else:
                st.success("✅ This customer is **likely to stay**.")

        with col2:
            st.metric("Churn Probability", f"{probability * 100:.2f}%")


# ===========================================================
# PAGE: ABOUT
# ===========================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.write(
        """
        This project was built as part of an **MCA Machine Learning Assignment**.
        It demonstrates a complete, beginner-friendly Machine Learning workflow —
        from data cleaning to model deployment — using only standard, well-known
        Python libraries.
        """
    )

    st.markdown("### 🎯 Objective")
    st.write("Predict whether a telecom customer is likely to churn based on their account and service usage details.")

    st.markdown("### 🛠️ Tech Stack")
    st.write("Python, Pandas, NumPy, Scikit-learn, Matplotlib, Streamlit, Joblib")

    st.markdown("### 🤖 Machine Learning Models")
    st.write("Logistic Regression, Random Forest")

    st.markdown("### 👥 Team")
    st.write("Abhay • Anshika • Akshit • Riya")

    st.markdown("### 🔗 Repository")
    st.write("https://github.com/abhay0963/Customer-Churn-Prediction")

    st.caption("Made with ❤️ using Streamlit")
