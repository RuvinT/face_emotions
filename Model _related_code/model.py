#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 19:13:20 2024

@author: ruvinjagoda
"""

import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from keras.callbacks import EarlyStopping
import tf2onnx

def preprocess_image(image, target_size, resize_image=True, blur_sigma=None, to_gray=True):
    # Apply Gaussian blur if sigma value is provided
    if blur_sigma is not None:
        image_blurred = image.filter(ImageFilter.GaussianBlur(blur_sigma))
    else:
        image_blurred = image
    
    if resize_image:
        # Resize image
        resized_image = image_blurred.resize(target_size, Image.LANCZOS)
    else:
        resized_image = image_blurred
    
    if to_gray:
        # Convert image to grayscale
        grayscale_image = resized_image.convert('L')
    else:
        grayscale_image = resized_image
    
    # Convert PIL image to numpy array and normalize
    image_processed = np.array(grayscale_image, dtype=np.float32) / 255.0
    
    return image_processed

BASE_DIR = './FaceExpressions/dataset'

# Define preprocessing parameters
target_size = (64, 64)
threshold = 100
blur_sigma = 1.5

# Initialize an empty list to store preprocessed images and their corresponding emotions
data = {'preprocessed_image': [], 'emotion': []}

# Iterate over each folder in the dataset directory
for folder_name in os.listdir(BASE_DIR):
    folder_path = os.path.join(BASE_DIR, folder_name)
    
    # Check if the path is a directory
    if os.path.isdir(folder_path):
        # Extract emotion label from folder name
        emotion = folder_name
        
        # Iterate over each image file in the folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            
            # Load image
            image = Image.open(image_path)
            
            # Preprocess image
            preprocessed_image = preprocess_image(image, target_size,to_gray=False)
            
            # Append the preprocessed image and emotion to the data dictionary
            data['preprocessed_image'].append(preprocessed_image)
            data['emotion'].append(emotion)

# Create a DataFrame from the data dictionary
df_preprocessed = pd.DataFrame(data)

# Display the DataFrame
print(df_preprocessed)




# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

from tensorflow.keras.utils import to_categorical

# Encode labels
label_encoder = LabelEncoder()
# Prepare data for training
X = np.array(df_preprocessed['preprocessed_image'].tolist())

y = label_encoder.fit_transform(df_preprocessed['emotion'])
# Reshape X to fit CNN input shape
X = X.reshape(-1, 64, 64,3)

import joblib

# Save the LabelEncoder object
joblib.dump(label_encoder, 'label_encoder.pkl')

# Convert labels to one-hot encoding
num_classes = len(label_encoder.classes_)
y_one_hot = to_categorical(y, num_classes=num_classes)


# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_one_hot, test_size=0.2, random_state=42, shuffle=True)

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(X_train, y_train, epochs=100, batch_size=64, 
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping])
# Save the model
model.save("face_emotion_detection_model.h5")
print("save complete")



# Convert TensorFlow model to ONNX format
onnx_model = tf2onnx.convert.from_keras(model)

# Save the ONNX model to a file
output_path = 'model.onnx'
tf2onnx.utils.save_onnx_model(output_path, onnx_model)

print(f"ONNX model saved to {output_path}")

