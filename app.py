import streamlit as st
import pickle

# Load model & vectorizer
model = pickle.load(open("service_classifier.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# App Config
st.set_page_config(page_title="Smart Service Classifier", page_icon="🤖", layout="centered")

st.title("🤖 Smart Service Classifier")
st.write("Enter your inquiry below and get the predicted service category with confidence score.")

# User Input
user_input = st.text_area("✍️ Type your inquiry here:", "")

if st.button("🔍 Classify"):
    if user_input.strip() != "":
        # Transform input
        vectorized_input = vectorizer.transform([user_input])
        
        # Predict
        pred = model.predict(vectorized_input)[0]
        proba = model.predict_proba(vectorized_input).max() * 100
        
        # Show results
        st.success(f"**Predicted Service:** {pred}")
        st.info(f"**Confidence Score:** {proba:.2f}%")
    else:
        st.warning("⚠️ Please enter an inquiry before classifying.")
