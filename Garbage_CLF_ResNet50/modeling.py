import shutil
from sklearn.model_selection import train_test_split

# Set paths
ORIGINAL_DATASET = "/content/garbage-classification-v2/garbage-dataset"
BASE_DIR = "garbage_data"  # Output directory

# Create folders
for split in ["train", "val", "test"]:
    for category in os.listdir(ORIGINAL_DATASET):
        os.makedirs(os.path.join(BASE_DIR, split, category), exist_ok=True)

# Split data
for category in os.listdir(ORIGINAL_DATASET):
    category_path = os.path.join(ORIGINAL_DATASET, category)
    images = os.listdir(category_path)

    train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)  # 0.33 of 0.3 = 10%

    for img_name in train_imgs:
        shutil.copy(os.path.join(category_path, img_name), os.path.join(BASE_DIR, "train", category, img_name))

    for img_name in val_imgs:
        shutil.copy(os.path.join(category_path, img_name), os.path.join(BASE_DIR, "val", category, img_name))

    for img_name in test_imgs:
        shutil.copy(os.path.join(category_path, img_name), os.path.join(BASE_DIR, "test", category, img_name))

print("✅ Done splitting dataset!")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

# Set dataset directories
BASE_DIR = "C:/Users/Acer/PycharmProjects/pythonProject11/garbage_data"

train_dir = os.path.join(BASE_DIR, 'train')
val_dir = os.path.join(BASE_DIR, 'val')
test_dir = os.path.join(BASE_DIR, 'test')

# Image size and batch size
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Image data generators with normalization
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen   = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_data = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_data   = val_datagen.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
test_data  = test_datagen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_data, epochs=10, validation_data=val_data)

# ✅ Save model after training
model.save("model.h5")
# Evaluate on test data
test_loss, test_acc = model.evaluate(test_data)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")

# Classification report
y_true = test_data.classes
y_pred = np.argmax(model.predict(test_data), axis=1)
target_names = list(test_data.class_indices.keys())

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# Plot training & validation accuracy and loss
plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Confusion matrix plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()