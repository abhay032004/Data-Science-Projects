import streamlit as st
import pickle

st.title("Amazon Product Review Sentiment Analysis")
st.write("Enter a product review to predict its sentiment (Positive/Negative).")

# Load model and vectorizer
with open("logistic_regression_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# Input
user_review = st.text_area("Enter your review here:")

# Predict
if st.button("Predict Sentiment"):
    if user_review.strip():
        review_vectorized = vectorizer.transform([user_review])
        prediction = model.predict(review_vectorized)[0]
        confidence = model.predict_proba(review_vectorized).max()

        if prediction == 1:
            st.success(f"Positive Review 😊 (Confidence: {confidence:.2f})")
        else:
            st.error(f"Negative Review 😞 (Confidence: {confidence:.2f})")
    else:
        st.warning("Please enter a review before predicting.")