import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
from PIL import Image
from io import StringIO

# ========== Configuration ==========
st.set_page_config(page_title="Garbage Classifier", layout="centered")

# ========== Load Model ==========
@st.cache_resource
def load_model() -> tf.keras.Model:
    """Loads and caches the trained Keras model."""
    return tf.keras.models.load_model("my_model.keras")

model = load_model()

# ========== Class Names ==========
class_names = [
    'battery', 'biological', 'cardboard', 'clothes', 'glass',
    'metal', 'paper', 'plastic', 'shoes', 'trash'
]

# ========== Sidebar ==========
st.sidebar.title("⚙️ Options")
st.sidebar.markdown("""
Upload an image for classification.

- Formats: JPG, PNG, JPEG  
- Images will be resized to 224×224 before prediction.
""")

if st.sidebar.button("Show Model Summary"):
    buffer = StringIO()
    model.summary(print_fn=lambda line: buffer.write(line + "\n"))
    st.sidebar.subheader("📋 Model Architecture")
    st.sidebar.text(buffer.getvalue())

# ========== Main Interface ==========
st.title("🧠 Garbage Classifier App")
st.markdown("Upload an image and the model will predict the type of garbage.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

# ========== Preprocessing ==========
def preprocess_image(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """Resizes and normalizes the image."""
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    return np.expand_dims(img_array, axis=0)

# ========== Prediction ==========
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Preprocess & predict
    input_tensor = preprocess_image(image)
    predictions = model.predict(input_tensor)[0]

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = predictions[predicted_index] * 100

    st.markdown(f"### 🎯 Prediction: `{predicted_class}`")
    st.markdown(f"**Confidence:** `{confidence:.2f}%`")

    # Show all class probabilities
    st.subheader("📊 Class Probabilities")
    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability": [f"{p:.4f}" for p in predictions]
    })
    st.dataframe(prob_df, use_container_width=True)
