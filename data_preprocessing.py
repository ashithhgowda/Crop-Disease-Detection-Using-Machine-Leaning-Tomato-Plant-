from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths to train, validation, and test directories
train_dir = 'dataset_split/train'
val_dir = 'dataset_split/validation'
test_dir = 'dataset_split/test'

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255,          # Normalize pixel values
    rotation_range=20,        # Randomly rotate images
    width_shift_range=0.2,    # Horizontal shift
    height_shift_range=0.2,   # Vertical shift
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,           # Random zoom
    horizontal_flip=True,     # Random horizontal flip
    vertical_flip=True,
    shear_range=0.15,
    channel_shift_range=0.1
)

# Validation and test data generators (just normalization)
val_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Generate batches of image data
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),  # Resize images to 128x128
    batch_size=32,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Important for prediction: Keep order
)

print("Data preprocessing complete!")

# Export variables
__all__ = ['train_data', 'val_data', 'test_data']
