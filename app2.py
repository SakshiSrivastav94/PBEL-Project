import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

# Load model and feature metadata
model = joblib.load("rf_salary_model_2.pkl")
feature_columns = joblib.load("rf_features_2.pkl")
encoder = joblib.load("encoder_2.pkl")
df = pd.read_csv("Salary Data.csv")

# Set page config
st.set_page_config(page_title="Employee Salary Prediction", layout="wide", page_icon="💼")

# Sidebar Design
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135755.png", width=120)
    st.title("💰 Salary Analysis")
    st.markdown("""
    This page allows you to **Explore** salaries and also make a **Salary Prediction** based on:
    - Gender ⚥  
    - Education 🎓  
    - Age 📆  
    - Job Title 💼  
    - Experience 📈  

    """)
    mode = st.selectbox("Choose Mode", ["Predict", "Explore"])

# Main title
st.markdown("<h1 style='color:white;'>Empolyee Salary Prediction</h1>", unsafe_allow_html=True)

if mode == "Predict":
    st.subheader("📝 We need some information to predict the salary")

    gender = st.selectbox("Gender", ["-- Select --", "Male", "Female"])
    education = st.selectbox("Education Level", ["-- Select --", "Bachelor's", "Master's", "PhD"])
    job_title = st.selectbox("Job Title", ["-- Select --", "Software Engineer", "Data Scientist", "Web Developer", "Marketing Analyst", "Product Manager", "Sales Manager", "Marketing Coordinator"])
 
    age = st.slider("Age", 0, 65, value=0)
    experience = st.slider("Years of Experience", 0, 50, value=0)

    if st.button("🔮 Predict Salary"):
        # # 👇 Step 1: Create user input DataFrame
        input_df = pd.DataFrame([{
            "Gender": gender,
            "Education Level": education,
            "Job Title": job_title,
            "Age": age,
            "Years of Experience": experience
        }])
        
        # 👇 Step 2: Encode categorical columns using the saved encoder
        encoded_input = encoder.transform(input_df[["Gender", "Education Level", "Job Title"]])
        encoded_input_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out())

        # 👇 Step 3: Align encoded input with training feature order
        final_input = pd.DataFrame(columns= feature_columns)
        final_input.loc[0] = 0
        final_input[encoded_input_df.columns] = encoded_input_df.loc[0]

        # predict salary
        prediction = model.predict(final_input)[0]
        st.success(f"💸 Predicted Salary: **₹{prediction:,.2f}**")

        # Download prediction
        result_df = input_df.copy()
        result_df["Predicted Salary"] = prediction
        csv = result_df.to_csv(index=False).encode("utf-8")
        filename = f"salary_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        st.download_button("📥 Download Result as CSV", data=csv, file_name=filename, mime="text/csv")

else:
    st.subheader("📊 Explore Employees Salaries")
    st.markdown("####  Survey (Sample Visuals)")

    # bar chart 
    st.markdown("#### 🕧 Average Salary by Job Title")
    avg_salary_by_job = df.groupby("Job Title")["Salary"].mean().sort_values(ascending=False)
    st.bar_chart(avg_salary_by_job)

    # line chart
    st.markdown("#### 📊 Salary by Experience")
    df_sorted = df.sort_values("Years of Experience")
    st.line_chart(df_sorted.set_index("Years of Experience")["Salary"])


    # pie chart
    st.markdown("#### 🛏️ Gender Distribution")
    gender_counts = df["Gender"].value_counts()
    fig, ax = plt.subplots()
    ax.pie(gender_counts, labels=gender_counts.index, explode=[0,0.1],autopct="%1.1f%%")
    ax.axis("equal")
    st.pyplot(fig)