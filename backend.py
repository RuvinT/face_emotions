#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 22:22:20 2024

@author: ruvinjagoda
"""

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)
#model = load_model("emotion_detection_model_final.h5")

@app.route('/')
def home():
    return 'flask app'
    
'''   
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict(data['image'])
    return jsonify(prediction.tolist())
'''
if __name__ == '__main__':
    app.run(debug=True)
