# 📊 Customer Churn Prediction

A beginner-friendly, end-to-end **Machine Learning project** that predicts whether a telecom customer is likely to churn (leave the service), built as an **MCA Machine Learning Assignment**.

The project covers the full ML workflow — data cleaning, exploratory data analysis, model training, evaluation, and deployment through an interactive **Streamlit** web app.

---

## 📌 Table of Contents

- [Introduction](#-introduction)
- [Features](#-features)
- [Machine learning workflow](#-workflow)
- [Tech Stack](#-tech-stack)
- [Folder Structure](#-folder-structure)
- [Installation](#-installation)
- [Running the App](#-running-the-app)
- [Results](#-results)
- [Screenshots](#-screenshots)
- [Future Scope](#-future-scope)
- [Author](#-author)
- [License](#-license)

---

## 📖 Introduction

Customer churn occurs when a customer stops using a company's product or service. For telecom companies, retaining existing customers is far cheaper than acquiring new ones, which makes **predicting churn in advance** extremely valuable.

This project uses the **Telco Customer Churn dataset** to train Machine Learning models that predict churn based on customer demographics, account information, and subscribed services. The final model is deployed through a clean, interactive Streamlit web application.

> ⚠️ This is an academic project built for an MCA Machine Learning course. It intentionally uses simple, well-understood techniques rather than advanced/production-grade tooling, to keep the project easy to explain and defend in a viva.

---

## ✨ Features

- 🧹 Clean, well-documented data preprocessing pipeline
- 📊 Interactive Exploratory Data Analysis (EDA) with charts
- 🤖 Two ML models trained and compared: Logistic Regression & Random Forest
- 📈 Full evaluation: Accuracy, Precision, Recall, F1 Score, Confusion Matrix
- ⭐ Feature importance visualization
- 🔮 Live churn prediction through a simple web form
- 🎨 Clean, emoji-friendly Streamlit UI with metric cards

---

## 🔄 Machine Learning Workflow

The project follows a complete Machine Learning pipeline:

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Data Preprocessing
6. Model Training
7. Model Evaluation
8. Model Selection
9. Model Saving
10. Streamlit Deployment

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib |
| Web App | Streamlit |
| Model Persistence | Joblib |

---

## 📁 Folder Structure

```
Customer-Churn-Prediction/
│
├── data/
│   └── Telco_customer_churn.xlsx      # Raw dataset
│
├── models/
│   ├── churn_model.pkl                # Trained ML model
│   ├── scaler.pkl                     # Fitted StandardScaler
│   ├── feature_names.json             # Encoded feature list
│   ├── model_results.csv              # Model comparison metrics
│   ├── confusion_matrix.png           # Confusion matrix plot
│   └── feature_importance.png         # Feature importance plot
│
├── src/
│   ├── preprocess.py                  # Data cleaning & encoding
│   ├── train_model.py                 # Model training & evaluation
│   └── app.py                         # Streamlit web application
│
├── docs/
│   └── Project_Documentation.docx     # Full project report
│
├── screenshots/                       # App screenshots for README
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/abhay0963/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the App

**Step 1 — Train the model (only needed once, or after changing the dataset)**

```bash
cd src
python train_model.py
```

This generates `churn_model.pkl`, `scaler.pkl`, `feature_names.json`, `model_results.csv`, and the evaluation plots inside the `models/` folder.

**Step 2 — Launch the Streamlit app**

```bash
streamlit run src/app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📈 Results

Two models were trained and compared on the same train/test split:

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Logistic Regression** ✅ | **80.62%** | 65.26% | 57.75% | 61.28% |
| Random Forest | 78.99% | – | – | – |

**Logistic Regression** was selected as the final model since it achieved the highest accuracy while remaining simple and easy to interpret — a good fit for this project's goals.

---

## 🚀 Future Scope

- Add more ML models (e.g., Support Vector Machines, Gradient Boosting)
- Hyperparameter tuning for further performance improvements
- Deploy the app on Streamlit Community Cloud
- Add customer segmentation and retention strategy suggestions
- Extend into an advanced Capstone Project featuring Agentic AI, RAG, LangGraph, MCP, FastAPI, Vector Databases, Docker, and Cloud deployment (planned as a separate, future project)

---

## 👥 Author

Built by **Abhay**, **Anshika**, **Akshit**, and **Riya** as part of an MCA Machine Learning Assignment.

Repository: [github.com/abhay0963/Customer-Churn-Prediction](https://github.com/abhay0963/Customer-Churn-Prediction)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
