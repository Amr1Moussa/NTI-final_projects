import numpy as np
import gradio as gr
from PIL import Image
import joblib

# ===== Load Saved Model and Scaler =====
mlp_mnist = joblib.load("mlp_digit_model.joblib")


# ===== Prediction Function =====
def predict_digit(image):
    if image is None:
        return "Please draw a digit", {str(i): 0.0 for i in range(10)}

    if isinstance(image, dict):
        image_data = image.get('image') or image.get('composite')
        if image_data is None:
            return "Invalid image format", {str(i): 0.0 for i in range(10)}
    else:
        image_data = image

    if isinstance(image_data, np.ndarray):
        image_pil = Image.fromarray(image_data).convert("L")
    elif isinstance(image_data, Image.Image):
        image_pil = image_data.convert("L")
    else:
        return "Unsupported image format", {str(i): 0.0 for i in range(10)}

    # Resize to 8x8 for load_digits compatibility
    image_pil = image_pil.resize((28, 28))
    img_array = 255 - np.array(image_pil)  # Invert
    img_array = img_array.astype("float32") / 255.0
    img_flat = img_array.flatten().reshape(1, -1)

    # Predict
    probs = mlp_mnist.predict_proba(img_flat)[0]
    pred = int(np.argmax(probs))
    prob_dict = {str(i): float(f"{probs[i]:.3f}") for i in range(10)}

    return f"Predicted Digit: {pred}", prob_dict

# ===== Gradio Interface =====
def create_gradio_interface():
    return gr.Interface(
        fn=predict_digit,
        inputs=gr.Sketchpad(canvas_size=(280, 280), label="✏️ Draw a digit (0–9)"),
        outputs=[
            gr.Textbox(label="Prediction"),
            gr.Label(num_top_classes=10, label="Class Probabilities")
        ],
        title="Digit Classifier",
        description="Draw a digit to see the prediction and probabilities.",
        theme=gr.themes.Soft()
    )

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(share=True, debug=True, server_name="127.0.0.1")
