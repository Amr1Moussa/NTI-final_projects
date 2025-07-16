# ♻️ Waste Category Image Classifier (Streamlit + Keras)

An interactive web app for classifying waste images into **10 distinct categories** using a custom-trained Convolutional Neural Network (CNN). Built with **TensorFlow/Keras** and deployed via **Streamlit**, this app demonstrates the power of deep learning in sustainability-oriented image classification.

---

## 🚀 Demo

Upload an image and instantly get a class prediction with confidence and full class-wise probabilities.

![demo-screenshot](assets/demo.png)

---

## 🧠 Model Summary

- **Architecture**: Custom CNN
- **Input Shape**: 224×224 RGB
- **Layers**:
  - Convolutional layers with BatchNorm and ReLU
  - MaxPooling
  - Global Average Pooling
  - Dense layers with Dropout
- **Optimizer**: Adam (`lr=1e-5`)
- **Loss**: Categorical Crossentropy
- **Final Layer**: Softmax (10 classes)

---

## 📂 Dataset

The dataset consists of images from 10 waste-related categories:

| Class         | Count |
|---------------|-------|
| battery       | 944   |
| biological    | 997   |
| cardboard     | 1825  |
| clothes       | 5327  |
| glass         | 3061  |
| metal         | 1020  |
| paper         | 1680  |
| plastic       | 1984  |
| shoes         | 1977  |
| trash         | 947   |

### 🔀 Data Split

| Split     | Images |
|-----------|--------|
| Training  | 15,806 |
| Validation| 1,976  |
| Testing   | 1,980  |

### 🧼 Preprocessing

- All images resized to `224×224`
- Normalized using: `rescale=1./255`

---

## 🏗️ Model Architecture (Keras Sequential)

```python
Input(shape=(224, 224, 3)) ➜
Conv2D(32) ➜ BN ➜ ReLU ➜
Conv2D(64) ➜ BN ➜ ReLU ➜ MaxPool(4×4) ➜
Conv2D(128) ➜ BN ➜ ReLU ➜
Conv2D(128) ➜ BN ➜ ReLU ➜ MaxPool(2×2) ➜
Conv2D(256) ➜ BN ➜ ReLU ➜ MaxPool(2×2) ➜
Conv2D(512) ➜ BN ➜ ReLU ➜ MaxPool(2×2) ➜
GlobalAveragePooling ➜
Dense(256) ➜ Dropout(0.4) ➜
Dense(10, activation='softmax')
````

---

## 🖥️ Web App Features

* ✅ Upload image and classify in real-time
* 📈 Display confidence score
* 📊 Class probabilities table/bar chart
* 🧠 Show model architecture summary in sidebar

---

## 📦 Installation & Run Locally

1. **Clone the repo**

```bash
git clone https://github.com/Amr1Moussa/NTI-final_projects.git
cd NTI-final_projects
```

2. **(Optional)** Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the app**

```bash
streamlit run app.py
```

---

## 📄 Requirements

See `requirements.txt`:

```txt
streamlit
tensorflow
numpy
pillow
pandas
matplotlib
```

## 📃 License

MIT License — use freely, contribute openly, credit kindly.

````
