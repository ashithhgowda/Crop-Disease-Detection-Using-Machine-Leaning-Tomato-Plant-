import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Step 1: Define paths
model_path = 'tomato_disease_model.h5'  # Path to the trained model
image_folder = 'images'  # Folder containing images for prediction
target_size = (128, 128)  # Image size used during model training

# Step 2: Load the trained model
try:
    model = load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Step 3: Load class labels from the model's training data
class_labels = [
    'Tomato___Early_blight', 
    'Tomato___Healthy', 
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 
    'Tomato___Spider_mites Two-spotted_spider_mite'
]
print(f"Class labels: {class_labels}")

# Step 4: Preprocess and predict for each image
def preprocess_image(image_path, target_size):
    try:
        img = load_img(image_path, target_size=target_size)  # Load and resize image
        img_array = img_to_array(img)  # Convert to numpy array
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

print("Starting predictions...\n")
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    
    # Skip non-image files
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        print(f"Skipping non-image file: {image_name}")
        continue
    
    # Preprocess image
    image = preprocess_image(image_path, target_size)
    if image is None:
        continue  # Skip if preprocessing failed

    # Predict class probabilities
    try:
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)  # Get the index of the highest probability
        predicted_label = class_labels[predicted_class]  # Get the class label
        confidence = prediction[0][predicted_class]  # Get confidence score for the prediction

        print(f"Image: {image_name} -> Predicted Class: {predicted_label} (Confidence: {confidence:.2f})")
    except Exception as e:
        print(f"Error predicting image {image_name}: {e}")

