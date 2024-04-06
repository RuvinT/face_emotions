#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 22:22:20 2024

@author: ruvinjagoda
"""

from flask import Flask




app = Flask(__name__)
#model = load_model("emotion_detection_model_final.h5")

@app.route('/')
def home():
    return 'flask app'
    
if __name__ == '__main__':
    app.run()
