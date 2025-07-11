# PBEL-Project
# 💰 Salary Prediction Using Machine Learning

An interactive and user-friendly Machine Learning project that predicts salary based on inputs such as gender, education level, job title, age, and years of experience. Built with Python, Scikit-learn, and Streamlit.

---

## 📌 Project Overview

This project aims to estimate the salary of software professionals using machine learning algorithms trained on a structured salary dataset. The web interface allows users to input their details and receive salary predictions in INR, along with options to visualize trends and download results.

---

## 🚀 Features

- Predicts salary using **Random Forest Regressor**
- Converts predictions to **INR**
- Interactive **Streamlit web app**
- Custom **charts and graphs**:
  - Salary vs. Experience (Line Chart)
  - Average Salary by Job Title (Bar Chart)
  - Gender Distribution (Pie Chart)
- Option to **download predictions as CSV**
- Clean, dark-themed, and responsive UI with icons and styling

---

## 🧠 Algorithms & Tools

- **Model**: RandomForestRegressor (`scikit-learn`)
- **Preprocessing**: OneHotEncoder, ColumnTransformer
- **Visualization**: `matplotlib`, `Streamlit charts`
- **Frontend**: Streamlit with CSS tweaks for sidebar and layout

---

## 🧹 Data Preprocessing

- Removed null entries from dataset
- One-hot encoded categorical columns: `Gender`, `Education Level`, `Job Title`
- Retained numerical columns: `Age`, `Years of Experience`
- Target column: `Salary` (converted to INR)
- Train/test split: 80/20

---

## 📊 Dataset

- Filename: `Salary Data.csv`
- Sample columns:
  - Age
  - Gender
  - Education Level
  - Job Title
  - Years of Experience
  - Salary

---

## 📈 Model Performance

- 📌 Metrics (example):
  - R² Score: 0.88
  - MAE: 10178.78
  - RMSE: 14958.75
- Trained using `Pipeline` to include preprocessing and model steps

---

## 🔮 Future Scope

- Add location and company-specific features
- Resume parsing via NLP
- Cloud deployment (Streamlit Cloud, AWS)
- Live currency conversion
- Model explainability using SHAP or LIME

---

## 🛠️ How to Run Locally

```bash
# Step 1: Clone the repo
git clone https://github.com/your-username/salary-predictor.git
cd salary-predictor

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the app
streamlit run app.py
