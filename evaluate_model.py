import numpy as np
from tensorflow.keras.models import load_model
from data_preprocessing import test_data

# Step 1: Load the trained model
model = load_model('tomato_disease_model.h5')
print("Model loaded successfully!")

# Step 2: Evaluate the model on test data
print("Evaluating model on test data...")
loss, accuracy = model.evaluate(test_data)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")
