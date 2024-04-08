#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 22:05:49 2024

@author: ruvinjagoda
"""
from flask import Flask, request, jsonify,render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
import numpy as np
from flask import Flask, request, redirect, url_for
import os
import joblib
from PIL import Image
from io import BytesIO
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

def load_custom_model(model_path):
    custom_objects = {'InputLayer': InputLayer}
    return load_model(model_path, custom_objects=custom_objects)

model = load_custom_model("face_emotion_detection_model.h5")


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return redirect(url_for('static', filename='upload_image.html'))

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is uploaded
    if 'image' not in request.files:
        return 'No image file found.', 400
    
    file = request.files['image']
    
    
    # Read the image file and preprocess it
    image = preprocess_image(file)
    
    # Perform prediction
    prediction = model.predict(image)
    
    # Load the LabelEncoder object
    label_encoder_loaded = joblib.load('label_encoder.pkl')
    
    # Decode one-hot encoded predictions back to original labels
    decoded_predictions = label_encoder_loaded.inverse_transform(np.argmax(prediction, axis=1))
    
    # Return prediction result
    return jsonify(decoded_predictions.tolist())

def preprocess_image(file):
    # Read the image file and convert to numpy array
    img = Image.open(BytesIO(file.read()))
    
    # Resize the image to match the input shape of the model
    img = img.resize((64, 64))
    
    # Convert image to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Reshape image data for model input
    img_array = img_array.reshape(-1, 64, 64, 3)
    
    return img_array

if __name__ == '__main__':
    app.run(debug=True)
