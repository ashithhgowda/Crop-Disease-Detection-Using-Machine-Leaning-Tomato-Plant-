import os
import shutil
import random

# Paths
source_dir = 'dataset/'  # Path where your dataset folders are located
target_dir = 'dataset_split/'  # Path for split dataset

# Categories (subfolders)
categories = ['Tomato___Early_blight', 'Tomato___Healthy', 'Tomato___Late_blight',
              'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite']

# Create train, validation, and test folders
for split in ['train', 'validation', 'test']:
    for category in categories:
        os.makedirs(os.path.join(target_dir, split, category), exist_ok=True)

# Split data (70% train, 20% validation, 10% test)
for category in categories:
    category_path = os.path.join(source_dir, category)
    
    # Debugging: Check if the category folder exists
    if not os.path.exists(category_path):
        print(f"Warning: Folder {category_path} does not exist!")
        continue

    # Filter valid image files only
    files = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

    # Debugging: Check if the folder contains files
    if not files:
        print(f"Warning: No files found in {category_path}!")
        continue

    # Shuffle and split
    random.shuffle(files)
    train_split = int(0.7 * len(files))
    val_split = int(0.9 * len(files))

    for i, file in enumerate(files):
        src = os.path.join(category_path, file)
        if i < train_split:
            dest = os.path.join(target_dir, 'train', category, file)
        elif i < val_split:
            dest = os.path.join(target_dir, 'validation', category, file)
        else:
            dest = os.path.join(target_dir, 'test', category, file)

        # Copy files
        shutil.copy(src, dest)

    print(f"Category '{category}' successfully split into train, validation, and test sets.")

print("Dataset split into train, validation, and test sets!")
