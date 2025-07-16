from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image
import io
import json

app = Flask(__name__)

# Load model
model = load_model('model.h5')

# Define the class labels (must match the training labels order!)
class_labels = [
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
# Image input size for ResNet50
IMG_SIZE = (224, 224)

# Preprocessing function
def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = preprocess_input(img_array)  # match ResNet50 preprocessing
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img_array = preprocess(file.read())
    preds = model.predict(img_array)[0]

    predicted_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    print("=== DEBUG ===")
    print("Preds shape:", preds.shape)
    print("Preds:", preds)
    print("Predicted index:", predicted_idx)
    print("Number of class_labels:", len(class_labels))
    print("Class labels:", class_labels)

    try:
        predicted_label = class_labels[predicted_idx]
    except IndexError:
        return jsonify({
            "error": f"Model predicted index {predicted_idx}, but class_labels only has {len(class_labels)} classes."
        }), 500

    return jsonify({
        'prediction': predicted_label,
        'confidence': f"{confidence:.2f}",
        'raw_probs': {class_labels[i]: float(prob) for i, prob in enumerate(preds) if i < len(class_labels)}
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
