import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import gradio as gr
import os

# Constants
IMG_SIZE = (128, 128)
MODEL_PATH = "model.h5"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class names
train_dir = "C:/Users/Acer/PycharmProjects/pythonProject11/garbage_data/train"
class_names = sorted(os.listdir(train_dir))

# Prediction function
def predict_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    preds = model.predict(img_array)[0]
    return {label: float(score) for label, score in zip(class_names, preds)}

# Gradio Interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🗑️ Garbage Classifier",
    description="Upload a garbage image to classify it into: cardboard, glass, metal, paper, plastic, or trash."
)

interface.launch()
