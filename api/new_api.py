import pandas as pd
import re
import nltk
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from flask import Flask, request, jsonify
import joblib
import os
from nltk.corpus import stopwords
import logging

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt')  # for word tokenization

app = Flask(__name__)

# Load configuration
MODEL_PATH = 'final_trained_bert_model'
TOKENIZER_PATH = MODEL_PATH
LABEL_ENCODER_PATH = os.path.join(os.getcwd(), 'category_encoder.joblib')

# Load the trained model and label encoder
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(label_encoder.classes_))
model.to(device)

stop_words = set(stopwords.words('portuguese'))

# Set up logging
logging.basicConfig(level=logging.INFO)

# Preprocess text function
def preprocess_text(text):
    """Preprocess the input text by removing non-alphanumeric characters and stop words."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

@app.route('/predict', methods=['POST'])
def predict():
    """Predict the category of a product based on its name."""
    data = request.json
    product_name = data.get('product_name')
    
    if not product_name:
        logging.error('No product name provided')
        return jsonify({'error': 'No product name provided'}), 400
    
    # Preprocess the input text
    processed_text = preprocess_text(product_name)
    
    # Tokenize and convert to tensor
    encoding = tokenizer(processed_text, truncation=True, padding='max_length', max_length=64, return_tensors='pt')
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Make prediction
    with torch.no_grad():
        model.eval()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        _, predicted_idx = torch.max(logits, 1)
        predicted_label = label_encoder.inverse_transform(predicted_idx.cpu().numpy())[0]
    
    logging.info(f'Predicted category for "{product_name}": {predicted_label}')
    return jsonify({'predicted_category': predicted_label})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)