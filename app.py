import logging
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
import numpy as np
import os
import joblib
from PIL import Image
from io import BytesIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


model_file_path = 'face_emotion_detection_model.h5'

def load_custom_model(model_path):
    custom_objects = {'InputLayer': InputLayer}
    return load_model(model_path, custom_objects=custom_objects)

model = load_custom_model(model_file_path)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    logger.info('Rendering home page')
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    logger.info('Received prediction request')
    try:
        if 'image' not in request.files:
            logger.error('No image file found in request')
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

        logger.info('Prediction successful')
        # Return prediction result
        return jsonify(decoded_predictions.tolist())
    except Exception as e:
        logger.error(f"An error occurred during prediction: {e}")
        return 'Error during prediction.', 500

def preprocess_image(file):
    img = Image.open(BytesIO(file.read()))
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(-1, 64, 64, 3)
    return img_array

@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

if __name__ == '__main__':
    app.run()
