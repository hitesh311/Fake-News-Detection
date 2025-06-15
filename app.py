import os
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
import json

app = Flask(__name__)

# Load the model and tokenizer
model_path = "./model_save"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

def predict_news(text):
    """Predict if the news is fake or real"""
    # Tokenize the input
    inputs = tokenizer(text, 
                      max_length=128, 
                      padding='max_length', 
                      truncation=True, 
                      return_tensors="pt")
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    probs = softmax(outputs.logits, dim=1)[0]
    fake_prob = probs[0].item()
    real_prob = probs[1].item()
    
    # Get predicted label
    pred_label = "Fake" if fake_prob > real_prob else "Real"
    
    return {
        'prediction': pred_label,
        'confidence': {
            'Fake': float(fake_prob * 100),
            'Real': float(real_prob * 100)
        },
        'explanation': generate_explanation(text, pred_label, max(fake_prob, real_prob))
    }

def generate_explanation(text, prediction, confidence):
    """Generate a simple explanation for the prediction"""
    confidence_pct = round(confidence * 100, 2)
    if prediction == "Fake":
        return f"The model is {confidence_pct}% confident this is fake news. It may contain sensational language, lack credible sources, or show patterns common in misinformation."
    else:
        return f"The model is {confidence_pct}% confident this is real news. The content appears to be factual and well-sourced."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'text' not in request.form:
        return jsonify({'error': 'No text provided'}), 400
    
    text = request.form['text']
    if not text.strip():
        return jsonify({'error': 'Text cannot be empty'}), 400
    
    try:
        result = predict_news(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    if not text.strip():
        return jsonify({'error': 'Text cannot be empty'}), 400
    
    try:
        result = predict_news(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5000)
