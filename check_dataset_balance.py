import os

# Paths to dataset directories
train_dir = 'dataset_split/train'
val_dir = 'dataset_split/validation'
test_dir = 'dataset_split/test'

def count_images_per_class(data_dir):
    class_counts = {}
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))
    return class_counts

# Count for each dataset
print("Training Dataset Distribution:")
print(count_images_per_class(train_dir))

print("\nValidation Dataset Distribution:")
print(count_images_per_class(val_dir))

print("\nTest Dataset Distribution:")
print(count_images_per_class(test_dir))
