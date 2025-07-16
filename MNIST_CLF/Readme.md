# 🧠 Digit Classifier with MLP & Gradio

An interactive digit recognition app that lets you draw digits (0–9) on a canvas, and predicts the digit using a trained Multi-Layer Perceptron (MLP) model. Built with Python, scikit-learn, and Gradio.

---

## 📌 Project Overview

This project demonstrates handwritten digit classification using the MNIST dataset. An MLP classifier is trained and deployed with a simple Gradio interface for real-time inference via a sketchpad input.

---

## 🧠 Model Details

- **Model Type:** `MLPClassifier` from scikit-learn
- **Input:** Flattened 28x28 grayscale image (784 features)
- **Architecture:**
  - Hidden Layers: `(256, 128, 64, 32)`
  - Activation: `ReLU`
  - Solver: `Adam`
  - Max Iterations: `100`
  - Random State: `42`

### Training Code

```python
from sklearn.neural_network import MLPClassifier

mlp_mnist = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=100,
    random_state=42
)

print("Training MLP on MNIST dataset...")
mlp_mnist.fit(X_train_flat, y_train)
````

---

## 📊 Model Evaluation

* **Test Accuracy:** `98.31%`

### 📄 Classification Report

| Class | Precision | Recall | F1-Score | Support |
| ----- | --------- | ------ | -------- | ------- |
| 0     | 0.98      | 0.99   | 0.99     | 980     |
| 1     | 0.99      | 1.00   | 0.99     | 1135    |
| 2     | 0.98      | 0.98   | 0.98     | 1032    |
| 3     | 0.99      | 0.98   | 0.98     | 1010    |
| 4     | 0.97      | 0.99   | 0.98     | 982     |
| 5     | 0.99      | 0.98   | 0.98     | 892     |
| 6     | 0.99      | 0.98   | 0.99     | 958     |
| 7     | 0.98      | 0.98   | 0.98     | 1028    |
| 8     | 0.98      | 0.98   | 0.98     | 974     |
| 9     | 0.98      | 0.98   | 0.98     | 1009    |

---

## 🖼️ Gradio Interface

<p align="center">
  <img src="https://user-images.githubusercontent.com/your-username/sketchpad-demo.gif" width="400">
</p>

* Input: Draw a digit on a 280x280 canvas.
* Output:

  * **Prediction**: Top predicted digit
  * **Probabilities**: Confidence scores for all classes (0–9)

### Launch App

```bash
python app.py
```

Or, if deployed with sharing enabled:

```bash
demo.launch(share=True)
```

---

## 🛠️ Requirements

```bash
pip install numpy gradio pillow scikit-learn joblib
```

---

## 📁 File Structure

```
.
├── app.py                 # Gradio app with prediction logic
├── mlp_digit_model.joblib # Trained MLP model
└── README.md              # Project overview and usage
```

---

## ✨ Credits

* MNIST dataset from \[scikit-learn or tensorflow\.keras.datasets]
* UI built with [Gradio](https://gradio.app/)
* Model trained with [scikit-learn](https://scikit-learn.org)

---

## 📌 Future Improvements

* Switch to CNN for better generalization
* Add support for uploading images
* Integrate training pipeline into the app

```


