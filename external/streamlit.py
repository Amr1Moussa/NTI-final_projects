import external.streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

IMG_SIZE = (224, 224)  # Input image size for ResNet50

# Class names used during training
class_names = [
    'battery',
    'biological',
    'cardboard',
    'clothes',
    'glass',
    'metal',
    'paper',
    'plastic',
    'shoes',
    'trash'
]

# Load the trained model and cache it
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('Garbage_Clf/my_model.keras')
        return model
    except Exception as e:
        st.error("❌ Failed to load model. Make sure the path is correct.")
        raise e

model = load_model()

# Prediction function
def predict_image(image, model):
    img = image.convert("RGB").resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds = model.predict(img_array)[0]
    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))
    all_preds = {label: float(score) for label, score in zip(class_names, preds)}
    return pred_class, confidence, all_preds

# Streamlit UI
st.set_page_config(page_title="Garbage Classification", page_icon="🗑")
st.title("🗑 Garbage Classification with ResNet50")
st.markdown("Upload an image of garbage to classify it into one of the categories:")

uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('🔍 Predict'):
        with st.spinner('🔎 Classifying...'):
            try:
                pred_class, confidence, all_preds = predict_image(image, model)
                st.success(f"✅ Prediction: **{pred_class}** with confidence **{confidence*100:.2f}%**")

                st.markdown("### 📊 Class Probabilities:")
                for cls, score in all_preds.items():
                    st.write(f"- **{cls}**: {score*100:.2f}%")

                # Optional: Plot bar chart
                fig, ax = plt.subplots()
                ax.barh(list(all_preds.keys()), list(all_preds.values()), color='skyblue')
                ax.set_xlim(0, 1)
                ax.set_xlabel("Probability")
                st.pyplot(fig)

            except Exception as e:
                st.error("⚠️ Error during prediction. Please try another image.")
                st.exception(e)
