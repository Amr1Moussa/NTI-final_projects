import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import shutil
from sklearn.model_selection import train_test_split

import shutil
from sklearn.model_selection import train_test_split

# Set paths
ORIGINAL_DATASET = "C:/Users/Acer/Desktop/garbage-dataset"
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
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)

    for img_name in train_imgs:
        shutil.copy(os.path.join(category_path, img_name), os.path.join(BASE_DIR, "train", category, img_name))

    for img_name in val_imgs:
        shutil.copy(os.path.join(category_path, img_name), os.path.join(BASE_DIR, "val", category, img_name))

    for img_name in test_imgs:
        shutil.copy(os.path.join(category_path, img_name), os.path.join(BASE_DIR, "test", category, img_name))

print("Done splitting dataset!")

import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_counts(data_dir, output_path):
    # Get class folders
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    # Count number of images per class
    counts = [len(os.listdir(os.path.join(data_dir, cls))) for cls in classes]

    # Prepare dataframe (optional)
    df = pd.DataFrame({"class": classes, "count": counts}).sort_values("count", ascending=False)

    # Plot inside the function
    plt.figure(figsize=(10, 5))
    plt.bar(df['class'], df['count'], color='skyblue')
    plt.title("Image Count per Class")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)

    # Ensure output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")


plot_counts(r'C:/Users/Acer/Desktop/garbage-dataset', 'plots/countplot.png')

train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

# Path to the glass folder
folder_path = "C:/Users/Acer/Desktop/garbage-dataset/glass"

# Get the first image from the folder
first_img_name = os.listdir(folder_path)[0]  # e.g., "glass1.jpg"
img_path = os.path.join(folder_path, first_img_name)

# Load and convert image to array
img = load_img(img_path)
x = img_to_array(img)
x = x.reshape((1,) + x.shape)  # Add batch dimension

# Create output folder for augmented images
output_folder = "preview"
os.makedirs(output_folder, exist_ok=True)

# Create data generator with augmentations
train_gen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# Generate and save 20 augmented images
i = 0
for batch in train_gen.flow(x, batch_size=1, save_to_dir=output_folder, save_prefix='glass', save_format='jpeg'):
    i += 1
    if i >= 20:
        break

print("✅ Augmented images saved to 'preview/' folder")


# Display first 9 augmented images

# List all JPEG images in preview folder
augmented_images = [f for f in os.listdir(output_folder) if f.endswith(".jpeg")]
augmented_images.sort()  # Sort for consistency

# Plot the first 9
plt.figure(figsize=(8, 6), dpi=120)
for i in range(9):
    img_path = os.path.join(output_folder, augmented_images[i])
    img = load_img(img_path)
    plt.subplot(3, 3, i + 1)
    plt.imshow(img)
    plt.axis('off')

plt.tight_layout()
plt.savefig('preview.png')
plt.show()


val_test_gen = ImageDataGenerator(rescale=1./255)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load the data
train_data = train_gen.flow_from_directory(
    "garbage_data/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_test_gen.flow_from_directory(
    "garbage_data/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = val_test_gen.flow_from_directory(
    "garbage_data/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Important for consistent test evaluation
)

class_names = list(train_data.class_indices.keys())

# Save class names to a file
import json
with open("class_names.json", "w") as f:
    json.dump(class_names, f)