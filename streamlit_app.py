import streamlit as st
import numpy as np
import joblib

# Load trained model and scaler
model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("💼 Employee Attrition Risk Predictor")
st.write("Enter employee details below to predict if they are likely to leave the company.")

# === Department Mapping ===
department_map = {
    'accounting': 0,
    'hr': 1,
    'IT': 2,
    'management': 3,
    'marketing': 4,
    'product_mng': 5,
    'RandD': 6,
    'sales': 7,
    'support': 8,
    'technical': 9
}
department_names = list(department_map.keys())
selected_department_name = st.selectbox("Department", department_names)
department = department_map[selected_department_name]

# === Salary Mapping ===
salary_map = {
    'Low': 0,
    'Medium': 1,
    'High': 2
}
salary_names = list(salary_map.keys())
selected_salary_name = st.selectbox("Salary Level", salary_names)
salary = salary_map[selected_salary_name]

# Other Inputs
satisfaction_level = st.slider("Satisfaction Level", 0.0, 1.0, 0.5)
last_evaluation = st.slider("Last Evaluation Score", 0.0, 1.0, 0.5)
number_project = st.number_input("Number of Projects", 1, 10, 3)
average_monthly_hours = st.number_input("Average Monthly Hours", 80, 310, 160)
time_spend_company = st.number_input("Years at Company", 1, 10, 3)
work_accident = st.selectbox("Had Work Accident?", ["No", "Yes"])
promotion_last_5years = st.selectbox("Got Promotion in Last 5 Years?", ["No", "Yes"])

# Convert Yes/No to binary
work_accident = 1 if work_accident == "Yes" else 0
promotion_last_5years = 1 if promotion_last_5years == "Yes" else 0

# Prepare input for prediction
user_input = np.array([[
    satisfaction_level,
    last_evaluation,
    number_project,
    average_monthly_hours,
    time_spend_company,
    work_accident,
    promotion_last_5years,
    department,
    salary
]])

# Scale input
scaled_input = scaler.transform(user_input)

# Predict
if st.button("🔍 Predict Attrition Risk"):
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][1]

    if prediction == 1:
        st.error(f"❌ Employee is likely to leave. (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Employee is likely to stay. (Confidence: {1 - probability:.2f})")

    st.caption("🔎 Note: This prediction is based on historical data patterns and may not reflect real-world situations with certainty.")
