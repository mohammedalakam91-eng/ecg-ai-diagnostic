import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="ECG AI Diagnostic System", layout="wide")
st.title("🩺 Dual-Mode AI Platform for ECG Diagnosis")

# تحميل الموديلات
@st.cache_resource
def load_models():
    img_model = tf.keras.models.load_model("best_ecg_model_99.keras")
    csv_model = tf.keras.models.load_model("ecg_csv_model_97.keras")
    return img_model, csv_model

try:
    img_model, csv_model = load_models()
    st.success("Models loaded successfully!")
except Exception as e:
    st.error(f"Error loading models: {e}")

tab1, tab2 = st.tabs(["🖼️ Image-Based Diagnosis Mode", "📊 Signal-Based Diagnosis Mode (1D CSV)"])

with tab1:
    st.header("Image Classification")
    uploaded_img = st.file_uploader("Upload ECG Image", type=["png", "jpg", "jpeg"])
    if uploaded_img and st.button("Run Image Diagnosis"):
        image = Image.open(uploaded_img)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # تنفيذ التنبؤ
        st.write("Processing diagnosis...")

with tab2:
    st.header("CSV Signal Classification")
    uploaded_csv = st.file_uploader("Upload ECG CSV File", type=["csv"])
    if uploaded_csv and st.button("Run CSV Diagnosis"):
        df = pd.read_csv(uploaded_csv)
        st.line_chart(df.iloc[0])
        st.write("Processing signal classification...")
