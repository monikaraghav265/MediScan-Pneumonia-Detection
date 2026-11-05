import streamlit as st
import numpy as np
import cv2
import joblib
import os
from PIL import Image
from fpdf import FPDF

# Load the model
model = joblib.load("pneumonia_model.pkl")

st.set_page_config(page_title="Pneumonia Detector AI App", layout="centered")
st.title("🩺 Chest X-ray Pneumonia Detection")
st.write("Upload a chest X-ray and get a quick prediction. This app also gives PDF reports and symptom check!")

# File uploader
uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    # Convert image for model
    img_array = np.array(image.convert('L'))
    img_resized = cv2.resize(img_array, (64, 64))
    img_flatten = img_resized.flatten().reshape(1, -1)

    prediction = model.predict(img_flatten)[0]
    labels = {
    0: "Normal",
    1: "Bacterial Pneumonia",
    2: "Viral Pneumonia"
    }

    result_label = labels[int(prediction)]
    st.subheader(f"🩻 Prediction Result: {result_label}")

    # Symptom Checker
    with st.expander("🤒 Symptom Checker"):
        fever = st.checkbox("Fever")
        cough = st.checkbox("Cough")
        breath = st.checkbox("Shortness of breath")
        fatigue = st.checkbox("Fatigue")

        if st.button("Check Risk"):
            symptoms = [fever, cough, breath, fatigue]
            risk_score = sum(symptoms)
            if risk_score >= 3:
                st.error("⚠️ High risk. Please consult a doctor immediately.")
            elif risk_score == 2:
                st.warning("🟠 Moderate risk. Monitor your health closely.")
            else:
                st.success("🟢 Low risk. Stay healthy!")

    # PDF Report
    if st.button("📄 Download Report as PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=14)
        pdf.cell(200, 10, txt="Chest X-ray Report", ln=True, align='C')
        pdf.cell(200, 10, txt=f"Result: {result_label}", ln=True)
        pdf.cell(200, 10, txt="Symptoms Checked:", ln=True)
        for symptom, state in zip(["Fever", "Cough", "Shortness of breath", "Fatigue"], [fever, cough, breath, fatigue]):
            status = "Yes" if state else "No"
            pdf.cell(200, 10, txt=f"- {symptom}: {status}", ln=True)

        pdf_path = "report.pdf"
        pdf.output(pdf_path)
        with open(pdf_path, "rb") as file:
            st.download_button("⬇️ Download Report", data=file, file_name="xray_report.pdf")

