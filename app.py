import gradio as gr
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# تحميل النماذج الحقيقية المرفوعة في الـ Space
IMG_MODEL_PATH = 'best_ecg_model_99.keras'
CSV_MODEL_PATH = 'ecg_csv_model_97.keras'

try:
    image_model = tf.keras.models.load_model(IMG_MODEL_PATH)
    print("✅ Successfully loaded Image Model")
except Exception as e:
    image_model = None
    print(f"Error loading image model: {e}")

try:
    csv_model = tf.keras.models.load_model(CSV_MODEL_PATH)
    print("✅ Successfully loaded CSV Model")
except Exception as e:
    csv_model = None
    print(f"Error loading CSV model: {e}")

IMAGE_CLASSES = [
    'F: Fusion Beat',
    'M: Myocardial Artifact / Abnormal',
    'N: Normal Beat',
    'Q: Unknown / Unclassifiable Beat',
    'S: Supraventricular Ectopic Beat',
    'V: Ventricular Ectopic Beat'
]

CSV_CLASSES = [
    'N: Normal Beat',
    'S: Supraventricular Ectopic Beat',
    'V: Ventricular Ectopic Beat',
    'F: Fusion Beat',
    'Q: Unknown / Unclassifiable Beat'
]

def create_empty_fig(title="No Data"):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.text(0.5, 0.5, title, ha='center', va='center', color='red')
    ax.axis('off')
    plt.tight_layout()
    return fig

def predict_csv(file):
    if file is None or csv_model is None:
        return create_empty_fig("Model or File Missing"), "⚠️ Error: Model not loaded or file missing."

    try:
        file_path = file.name if hasattr(file, 'name') else file
        df = pd.read_csv(file_path, header=None)
        
        data_numeric = df.apply(pd.to_numeric, errors='coerce')
        row_vals = data_numeric.iloc[0].dropna().values

        if len(row_vals) >= 188:
            row_vals = row_vals[:187]

        signal = np.pad(row_vals, (0, max(0, 187 - len(row_vals))), 'constant')[:187]
        signal = signal.astype(np.float32)

        # التنبؤ الفعلي من النموذج
        signal_input = signal.reshape(1, 187, 1)
        predictions = csv_model.predict(signal_input, verbose=0)[0]
        top_idx = np.argmax(predictions)
        confidence = float(predictions[top_idx] * 100)

        breakdown = "📊 Probability Breakdown Across Classes:\n" + "-"*42 + "\n"
        for idx, prob in enumerate(predictions):
            label = CSV_CLASSES[idx] if idx < len(CSV_CLASSES) else f"Class {idx}"
            breakdown += f"• {label}: {prob * 100:.2f}%\n"

        summary = (
            f"🎯 Primary Diagnosis: {CSV_CLASSES[top_idx]}\n"
            f"🔥 Prediction Confidence: {confidence:.2f}%\n\n"
            f"{breakdown}"
        )

        fig, ax = plt.subplots(figsize=(7, 2.8))
        ax.plot(signal, color='#d32f2f', linewidth=1.5)
        ax.set_title("1D Digitized ECG Waveform", fontsize=10, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        return fig, summary

    except Exception as e:
        return create_empty_fig("Error"), f"⚠️ Error: {str(e)}"

def predict_image(image):
    if image is None or image_model is None:
        return create_empty_fig("Model or Image Missing"), "⚠️ Error: Model not loaded or image missing."

    try:
        img_np = np.array(image)
        if len(img_np.shape) == 2:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        elif img_np.shape[2] == 4:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

        img_resized = cv2.resize(img_np, (224, 224))
        img_batch = np.expand_dims(img_resized, axis=0)

        predictions = image_model.predict(img_batch, verbose=0)[0]
        top_idx = np.argmax(predictions)
        confidence = float(predictions[top_idx] * 100)

        breakdown = "📊 Probability Distribution Across Classes:\n" + "-"*48 + "\n"
        for idx, prob in enumerate(predictions):
            breakdown += f"• {IMAGE_CLASSES[idx]}: {prob * 100:.2f}%\n"

        summary = (
            f"🎯 Primary Diagnosis: {IMAGE_CLASSES[top_idx]}\n"
            f"🔥 Prediction Confidence: {confidence:.2f}%\n\n"
            f"{breakdown}"
        )

        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.imshow(img_resized)
        ax.axis('off')
        ax.set_title(f"Diagnosis: {IMAGE_CLASSES[top_idx].split(':')[0]} ({confidence:.1f}%)", fontsize=11, fontweight='bold', color='#1b5e20')
        plt.tight_layout()

        return fig, summary

    except Exception as e:
        return create_empty_fig("Error"), f"⚠️ Error: {str(e)}"

# واجهة Gradio
with gr.Blocks(theme=gr.themes.Soft(), title="ECG AI Diagnostic System") as demo:
    gr.Markdown("# 🏥 Dual-Mode AI Platform for ECG Diagnosis")
    
    with gr.Tab("🖼️ Image-Based Diagnosis Mode"):
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="pil", label="Upload ECG Image")
                btn_img = gr.Button("🔍 Run Diagnosis", variant="primary")
            with gr.Column():
                img_plot_out = gr.Plot(label="Input Image")
                img_text_out = gr.Textbox(label="Diagnostic Results", lines=10)
        btn_img.click(fn=predict_image, inputs=img_in, outputs=[img_plot_out, img_text_out])

    with gr.Tab("📊 Signal-Based Diagnosis Mode (1D CSV)"):
        with gr.Row():
            with gr.Column():
                file_in = gr.File(label="Upload ECG CSV File")
                btn_csv = gr.Button("🔍 Run Signal Classification", variant="primary")
            with gr.Column():
                plot_out = gr.Plot(label="ECG Waveform")
                text_out = gr.Textbox(label="Diagnostic Results", lines=10)
        btn_csv.click(fn=predict_csv, inputs=file_in, outputs=[plot_out, text_out])

demo.launch(server_name="0.0.0.0", server_port=8501)
