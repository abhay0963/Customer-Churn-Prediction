import streamlit as st
import pandas as pd
import joblib
import json

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ---------------------------------------------------
# Load Dataset
# ---------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_excel("data/Telco_customer_churn.xlsx")

df = load_data()

# ---------------------------------------------------
# Load Model
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.pkl")

model = load_model()

@st.cache_resource
def load_scaler():
    return joblib.load("models/scaler.pkl")

scaler = load_scaler()

# ---------------------------------------------------
# Load Feature Names
# ---------------------------------------------------

with open("models/feature_names.json", "r") as file:
    feature_names = json.load(file)

# ---------------------------------------------------
# Load Model Results
# ---------------------------------------------------

results = pd.read_csv("models/model_results.csv")

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "📁 Dataset",
        "📊 EDA",
        "🤖 Model Performance",
        "🔮 Predict Churn",
        "ℹ About"
    ]
)

# ===================================================
# HOME
# ===================================================

if page == "🏠 Home":

    st.title("📊 Customer Churn Prediction")

    st.markdown("""
This project predicts whether a telecom customer is likely to leave the company.

The project is built using Machine Learning with Logistic Regression and Random Forest classifiers.
""")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Customers",
        len(df)
    )

    col2.metric(
        "Features",
        len(feature_names)
    )

    best_model = results.sort_values(
        "Accuracy",
        ascending=False
    ).iloc[0]["Model"]

    best_accuracy = results.sort_values(
        "Accuracy",
        ascending=False
    ).iloc[0]["Accuracy"]

    col3.metric(
        "Best Accuracy",
        f"{best_accuracy*100:.2f}%"
    )

    st.subheader("Algorithms Used")

    st.write("• Logistic Regression")

    st.write("• Random Forest")

    st.subheader("Dataset")

    st.write(df.head())

# ===================================================
# DATASET
# ===================================================

elif page == "📁 Dataset":

    st.title("📁 Dataset Information")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10))

    st.subheader("Shape")

    st.write(df.shape)

    st.subheader("Columns")

    st.write(df.columns.tolist())

    st.subheader("Missing Values")

    st.dataframe(df.isnull().sum())

    st.subheader("Statistics")

    st.dataframe(df.describe())

# ===================================================
# EDA
# ===================================================

elif page == "📊 EDA":

    st.title("📊 Exploratory Data Analysis")

    st.subheader("Churn Distribution")

    st.bar_chart(
        df["Churn Label"].value_counts()
    )

    st.subheader("Gender Distribution")

    st.bar_chart(
        df["Gender"].value_counts()
    )

    st.subheader("Contract Type")

    st.bar_chart(
        df["Contract"].value_counts()
    )

    st.subheader("Internet Service")

    st.bar_chart(
        df["Internet Service"].value_counts()
    )

    st.subheader("Monthly Charges")

    st.bar_chart(
        df["Monthly Charges"]
    )

    st.subheader("Tenure")

    st.bar_chart(
        df["Tenure Months"]
    )
    

    # ===================================================
# MODEL PERFORMANCE
# ===================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    st.subheader("Model Comparison")

    st.dataframe(results)

    best_model = results.sort_values(
        by="Accuracy",
        ascending=False
    ).iloc[0]

    st.success(f"Best Model : {best_model['Model']}")

    col1, col2 = st.columns(2)

    col1.metric(
        "Accuracy",
        f"{best_model['Accuracy']*100:.2f}%"
    )

    col2.metric(
        "Precision",
        f"{best_model['Precision']*100:.2f}%"
    )

    col3, col4 = st.columns(2)

    col3.metric(
        "Recall",
        f"{best_model['Recall']*100:.2f}%"
    )

    col4.metric(
        "F1 Score",
        f"{best_model['F1 Score']*100:.2f}%"
    )

    # ===================================================
# PREDICT CHURN
# ===================================================

elif page == "🔮 Predict Churn":

    st.title("🔮 Predict Customer Churn")

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    senior = st.selectbox(
        "Senior Citizen",
        ["No", "Yes"]
    )

    partner = st.selectbox(
        "Partner",
        ["No", "Yes"]
    )

    dependents = st.selectbox(
        "Dependents",
        ["No", "Yes"]
    )

    phone = st.selectbox(
        "Phone Service",
        ["No", "Yes"]
    )

    internet = st.selectbox(
        "Internet Service",
        [
            "DSL",
            "Fiber optic",
            "No"
        ]
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    tenure = st.slider(
        "Tenure Months",
        0,
        72,
        12
    )

    monthly = st.number_input(
        "Monthly Charges",
        0.0,
        200.0,
        70.0
    )

    total = st.number_input(
        "Total Charges",
        0.0,
        10000.0,
        1000.0
    )

    if st.button("Predict"):

        input_data = dict.fromkeys(feature_names, 0)

        scaled = scaler.transform(
    [[
        tenure,
        monthly,
        total
    ]]
)

        input_data["Tenure Months"] = scaled[0][0]
        input_data["Monthly Charges"] = scaled[0][1]
        input_data["Total Charges"] = scaled[0][2]

        if gender == "Male":
            input_data["Gender_Male"] = 1

        if senior == "Yes":
            input_data["Senior Citizen_Yes"] = 1

        if partner == "Yes":
            input_data["Partner_Yes"] = 1

        if dependents == "Yes":
            input_data["Dependents_Yes"] = 1

        if phone == "Yes":
            input_data["Phone Service_Yes"] = 1

        if internet == "Fiber optic":
            input_data["Internet Service_Fiber optic"] = 1

        elif internet == "No":
            input_data["Internet Service_No"] = 1

        if contract == "One year":
            input_data["Contract_One year"] = 1

        elif contract == "Two year":
            input_data["Contract_Two year"] = 1

        prediction = model.predict(
            pd.DataFrame([input_data])
        )[0]

        probability = model.predict_proba(
            pd.DataFrame([input_data])
        )[0]

        st.subheader("Prediction Result")

        if prediction == 1:

            st.error("⚠ Customer is likely to Churn")

        else:

            st.success("✅ Customer is likely to Stay")

        st.write(
            f"Confidence : {max(probability)*100:.2f}%"
        )


    # ===================================================
# ABOUT
# ===================================================

elif page == "ℹ About":

    st.title("ℹ About Project")

    st.markdown("""
### Customer Churn Prediction

This project predicts whether a telecom customer is likely to churn using Machine Learning.

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib

### Machine Learning Algorithms

- Logistic Regression
- Random Forest

""")