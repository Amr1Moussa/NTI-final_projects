import os
import shutil
from sklearn.model_selection import train_test_split

# Set paths
try:
    ORIGINAL_DATASET = input("Enter the path to the original dataset: ")  # Path to the original dataset
    if not ORIGINAL_DATASET:
        raise ValueError("Please provide the path to the original dataset.")
    BASE_DIR = "../data/preprocessed_data"  # Output directory

    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
except Exception as e:
    print(f"Error: {e}")
    exit(1)

# Check if the original dataset exists
if not os.path.exists(ORIGINAL_DATASET):
    print(f"Error: The original dataset path '{ORIGINAL_DATASET}' does not exist.")
    exit(1)

try:
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

    print("Done splitting dataset!")
except Exception as e:
    print(f"An error occurred while splitting the dataset: {e}, please check the paths and try again.")
    exit(1)