import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parameters

data_dir = r'E:\E2CR\output\R. 317 (3)\text_rows'
image_size = (128, 128)

csv_file = r'E:\E2CR\output\R. 317 (3)\text_rows\break_points.csv'
max_sequence_length = 13  # Adjust this value based on your data

# Load CSV with error handling
df = pd.read_csv(csv_file, on_bad_lines='skip')

# Prepare data
def load_data():
    images = []
    x_values = []
    for index, row in df.iterrows():
        img_name = row['file_name']
        if isinstance(img_name, str):  # Ensure the file_name is a string
            img_path = os.path.join(data_dir, img_name)
            if os.path.exists(img_path):
                print(f"Loading image: {img_path}")  # Debugging print
                image = cv2.imread(img_path)
                image = cv2.resize(image, image_size)
                image = image / 255.0  # Normalize image
                images.append(image)
                
                try:
                    # Ensure break_points is a string and strip quotes
                    break_points = row['break_points'].strip('"')
                    # Handle spaces after commas
                    x_value = [int(x.strip()) for x in break_points.split(',')]
                    x_values.append(x_value)
                except ValueError as e:
                    print(f"Error parsing x values for {img_name}: {e}")
            else:
                print(f"File does not exist: {img_path}")  # Debugging print
    
    images = np.array(images)
    x_values = pad_sequences(x_values, maxlen=max_sequence_length, padding='post', truncating='post')
    
    return images, x_values

# Load data and check
images, x_values = load_data()
print(f"Total images loaded: {len(images)}")
print(f"Total x_values loaded: {len(x_values)}")

if len(images) == 0 or len(x_values) == 0:
    raise ValueError("No data loaded. Please check the CSV file and image paths.")

# Split data
split_index = int(0.8 * len(images))
train_images, test_images = images[:split_index], images[split_index:]
train_x_values, test_x_values = x_values[:split_index], x_values[split_index:]

# Normalize x values
image_width = image_size[0]
train_x_values = train_x_values / image_width
test_x_values = test_x_values / image_width

# Check split sizes
print(f"Training images: {len(train_images)}, Training x_values: {len(train_x_values)}")
print(f"Testing images: {len(test_images)}, Testing x_values: {len(test_x_values)}")

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(max_sequence_length)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
if len(train_images) > 0:
    model.fit(train_images, train_x_values, epochs=100, validation_split=0.2)
else:
    raise ValueError("Training data contains 0 samples. Please provide more data.")

# Evaluate the model
loss, mae = model.evaluate(test_images, test_x_values)
print(f'Test Mean Absolute Error: {mae}')

# Predict x values for new images (example)
def preprocess_new_images(image_paths):
    new_images = []
    for img_path in image_paths:
        image = cv2.imread(img_path)
        image = cv2.resize(image, image_size)
        image = image / 255.0  # Normalize image
        new_images.append(image)
    
    return np.array(new_images)

# Example usage with new images
new_image_paths = [r'E:\E2CR\output\R. 317 (3)\text_rows\R. 317 (3)_001.jpg']  # Add your new images here
new_images = preprocess_new_images(new_image_paths)

# Predict x values
predicted_x_values = model.predict(new_images)

# Denormalize x values
predicted_x_values = predicted_x_values * image_width

print(predicted_x_values)