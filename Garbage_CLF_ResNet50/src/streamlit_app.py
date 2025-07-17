import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from io import StringIO



# ========== Load Model ========== 
MODEL_PATH = os.path.join("src", "models", "model.h5") # This is the current correct path

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

model = tf.keras.models.load_model(MODEL_PATH)

# ========== Class Labels ==========
# Replace with your actual class names if needed
class_names = ["battery", "biological", "cardboard", "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"]

# ========== Sidebar ==========
st.sidebar.title("⚙️ Options")
st.sidebar.markdown("""
Upload an image for classification.
- Supported formats: JPG, PNG, JPEG
- Input image will be resized to 224x224
""")

if st.sidebar.button("Show Model Summary"):
    string_io = StringIO()
    model.summary(print_fn=lambda x: string_io.write(x + "\n"))
    summary_string = string_io.getvalue()
    st.sidebar.subheader("📋 Model Architecture")
    st.sidebar.text(summary_string)

# ========== Main Title ==========
st.title("🧠 Garbage Classifier App")
st.markdown("Upload an image, and the model will predict the class.")

# ========== File Upload ==========
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# ========== Preprocessing ==========
IMG_SIZE = (224, 224)
def preprocess_image(img):
    return preprocess_input(np.expand_dims(img_to_array(img.convert("RGB").resize(IMG_SIZE)), axis=0))

# ========== Inference ==========
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # Show uploaded image
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    # Preprocess and predict
    processed = preprocess_image(image)
    prediction = model.predict(processed)[0]
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = prediction[predicted_index] * 100

    # Show result
    st.markdown(f"### 🎯 Prediction: `{predicted_class}`")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # Display all class probabilities in a table
    st.subheader("📊 Class Probabilities")
    prob_dict = {class_names[i]: float(f"{prediction[i]:.4f}") for i in range(len(class_names))}
    st.dataframe(prob_dict, use_container_width=True)