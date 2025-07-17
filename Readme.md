# NTI Final Projects

A collection of end-to-end classification projects built during the NTI program. Each subfolder contains a Jupyter notebook showcasing a different model and dataset.

---

## 📂 Project Structure

```
NTI-final_projects/
├── MNIST_CLF/ # Handwritten digit classification
│ └── MNIST_CLF.ipynb
├── Garbage_Clf_CNN/ # Garbage classification using a custom CNN
│ └── Garbage_Clf_CNN.ipynb
└── Garbage_CLF_ResNet50/ # Garbage classification using transfer learning (ResNet50)
└── Garbage_CLF_ResNet50.ipynb
```
---

## 🧠 Project Overviews

### 1. MNIST_CLF ([Deployed on Hugging Face](https://huggingface.co/spaces/amr-moussa/MNIST_CLF2))
- **Task**: Classify handwritten digits (0–9) using the MNIST dataset.
- **Model**: A simple convolutional neural network (CNN) achieving ~99% test accuracy.
- **Notebook Includes**: Data loading, normalization, model architecture, training plot, evaluation, and sample predictions.

### 2. Garbage_Clf_CNN
- **Task**: Classify garbage images into categories (e.g., plastic, paper, glass) using a custom CNN.
- **Model**: A multi-layer CNN with data preprocessing and augmentation.
- **Notebook Includes**: Data loading, augmentation setup, model training, accuracy/loss curves, evaluation metrics.

### 3. Garbage_CLF_ResNet50 ([Deployed on Hugging Face](https://huggingface.co/spaces/EmanHussein/ResNet50_Garbage_Classifier))
- **Task**: Same garbage classification, enhanced with transfer learning.
- **Model**: Pre-trained ResNet50 backbone plus custom top layers.
- **Notebook Includes**: Fine-tuning the ResNet50 base, data augmentation, training history, model evaluation, and sample predictions.

---

## 🛠️ Technologies Used

- **Python** 🐍
- **TensorFlow / Keras** – Deep learning framework
- **scikit-learn** – Image handling and preprocessing
- **NumPy, Pandas** – Data manipulation
- **Matplotlib** – Visualization
- **Jupyter Notebook** – Interactive experimentation
- **PI** – Image handling and preprocessing

---
## 🧬 Clone and use the repo:
   ```
   bash
   git clone https://github.com/Amr1Moussa/NTI-final_projects.git
   ```
---
## 📄 License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute this code, with attribution.
