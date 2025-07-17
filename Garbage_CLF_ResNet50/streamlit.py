import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

IMG_SIZE = (128, 128)  # Match the input size your model was trained on

# Load the trained model and cache it
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')  # Use your .h5 model here
    return model

model = load_model()

# Class names used during training (adjust as needed)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Prediction function
def predict_image(image, model):
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)[0]
    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))
    return pred_class, confidence, {label: float(score) for label, score in zip(class_names, preds)}

# Streamlit interface
st.title("🗑️ Garbage Classification with ResNet50")
st.write("Upload an image of garbage to classify it into one of the categories:")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        with st.spinner('Classifying...'):
            pred_class, confidence, all_preds = predict_image(image, model)

        st.success(f"Prediction: **{pred_class}** with confidence {confidence*100:.2f}%")

        st.write("Class probabilities:")
        for cls, score in all_preds.items():
            st.write(f"- {cls}: {score*100:.2f}%")
