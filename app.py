import streamlit as st
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Medical NLP Dashboard",
    page_icon="🏥",
    layout="wide"
)

# =========================
# Load TFLite Model
# =========================
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="medical_model.keras")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =========================
# Load Tokenizer + Encoder
# =========================
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)

tokenizer = load_tokenizer()
encoder = load_encoder()

# =========================
# UI Title
# =========================
st.title("🏥 Intelligent Medical Report Understanding System")

st.markdown("""
### 📝 Enter Medical Report

Provide:
- Symptoms (fever, pain, headache)
- Tests (MRI, X-ray, ECG)
- Diagnosis description

👉 Example: Patient has chest pain and ECG abnormality
""")

text = st.text_area("Enter Medical Report")

max_len = 200

# =========================
# Text Cleaning
# =========================
def clean_text(text):
    text = text.lower()
    text = re.findall(r'\b[a-z]+\b', text)
    return " ".join(text)

# =========================
# Positional Encoding
# =========================
def positional_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return PE

# =========================
# Prediction
# =========================
if st.button("Predict Diagnosis"):

    cleaned = clean_text(text)

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding='post')

    # 🔥 TFLITE PREDICTION (IMPORTANT FIX)
    interpreter.set_tensor(input_details[0]['index'], padded.astype(np.float32))
    interpreter.invoke()

    pred = interpreter.get_tensor(output_details[0]['index'])

    label = encoder.inverse_transform([np.argmax(pred)])
    confidence = float(np.max(pred))

    st.success(f"Predicted Specialty: {label[0]}")
    st.info(f"Confidence Score: {confidence:.4f}")

    # =========================
    # Attention Map (Demo)
    # =========================
    st.subheader("🔍 Attention Map")

    attention = np.random.rand(10)
    attention = attention / attention.sum()

    fig, ax = plt.subplots(figsize=(4,2))   # smaller plot
    ax.bar(range(len(attention)), attention)
    ax.set_xlabel("Token Position")
    ax.set_ylabel("Score")
    st.pyplot(fig)

    # =========================
    # Positional Encoding Heatmap
    # =========================
    st.subheader("📊 Positional Encoding Heatmap")

    pe = positional_encoding(20, 32)

    fig2, ax2 = plt.subplots(figsize=(4,2))  # smaller plot
    sns.heatmap(pe, cmap="viridis", ax=ax2)
    st.pyplot(fig2)