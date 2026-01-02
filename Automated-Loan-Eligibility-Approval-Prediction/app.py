import streamlit as st
import joblib
import pandas as pd

st.title("💰 Loan Approval Prediction")
st.write("Fill in the details below to predict loan approval status")

# Load model and encoder
model = joblib.load("random_forest_loan_model.pkl")
le = joblib.load("label_encoder.pkl")
feature_names = joblib.load("feature_names.pkl")

# Create input form
st.header("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    no_of_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    income_annum = st.number_input("Annual Income (₹)", min_value=0, value=500000)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=1000000)

with col2:
    loan_term = st.number_input("Loan Term (years)", min_value=1, max_value=30, value=10)
    cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, value=700)
    residential_assets_value = st.number_input("Residential Assets Value (₹)", min_value=0, value=500000)
    commercial_assets_value = st.number_input("Commercial Assets Value (₹)", min_value=0, value=0)
    luxury_assets_value = st.number_input("Luxury Assets Value (₹)", min_value=0, value=0)

bank_asset_value = st.number_input("Bank Asset Value (₹)", min_value=0, value=100000)

# Predict button
if st.button("Predict Loan Status", type="primary"):
    # Encode categorical variables
    education_encoded = 1 if education == "Graduate" else 0
    self_employed_encoded = 1 if self_employed == "Yes" else 0
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'no_of_dependents': [no_of_dependents],
        'education': [education_encoded],
        'self_employed': [self_employed_encoded],
        'income_annum': [income_annum],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'cibil_score': [cibil_score],
        'residential_assets_value': [residential_assets_value],
        'commercial_assets_value': [commercial_assets_value],
        'luxury_assets_value': [luxury_assets_value],
        'bank_asset_value': [bank_asset_value]
    })
    
    # Reorder columns to match training data
    input_data = input_data[feature_names]
    
    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data).max()
    
    # Display result
    st.markdown("---")
    if prediction == 1:
        st.success("✅ Loan Approved!")
        st.balloons()
    else:
        st.error("❌ Loan Rejected")
    
    st.metric("Confidence", f"{probability*100:.1f}%")