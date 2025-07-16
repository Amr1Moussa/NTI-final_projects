# Garbage Classification Using ResNet50

This project implements an image classification model using **Transfer Learning with ResNet50** to categorize images of garbage into multiple classes. The model is trained on the ["Garbage Classification V2"](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2) dataset and built with TensorFlow and Keras.

---

## 🚀 Demo

Upload an image and instantly get a class prediction with confidence and full class-wise probabilities.

![demo-screenshot](https://github.com/user-attachments/assets/b10da05a-33e3-46b1-9468-bc0b88823414)

## 📁 Dataset

* **Source**: [Kaggle - Garbage Classification V2](https://www.kaggle.com/datasets/sumn2u/garbage-classification-v2)
* **Classes**: Mixed garbage categories such as cardboard, glass, metal, paper, plastic, trash, etc.
* **Format**: Folder-based classification (i.e., one subfolder per class)

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

---

### 🔀 Data Split

| Split     | Images |
|-----------|--------|
| Training  | 13,833 |
| Validation| 3,952  |
| Testing   | 1,976  |

---

### 🧼 Preprocessing

- ```tensorflow.keras.applications.resnet50.preprocess_input```

---

## 🧠 Model Architecture
```
Input Image (224x224x3)
        ↓
ResNet50 Base (Frozen)
        ↓
GlobalAveragePooling2D()
        ↓
Dense(256, activation='relu')
        ↓
Dropout(0.5)
        ↓
Dense(num_classes, activation='softmax')
```
---

## 🧪 Training Configuration

* **Image Size**: 224x224
* **Batch Size**: 32
* **Epochs**: Customizable (default usually 10–25)
* **Optimizer**: Adam
* **Loss Function**: Categorical Crossentropy
* **Metrics**: Accuracy

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

---

## 📃 License

MIT License — use freely, contribute openly, credit kindly.