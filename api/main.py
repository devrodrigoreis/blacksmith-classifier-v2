import os
import re
import logging
import time
from typing import Dict, Any
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import uvicorn
from sklearn.preprocessing import LabelEncoder
import yaml
from contextlib import asynccontextmanager

# Define TextClassifier (matches fallback training script)
class TextClassifier(torch.nn.Module):
    def __init__(self, num_classes, pretrained_model="neuralmind/bert-base-portuguese-cased", dropout_rate=0.3):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(self.bert.config.hidden_size, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 1),
            torch.nn.Softmax(dim=1)
        )
        self.linear = torch.nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        attention_weights = self.attention(hidden_states).transpose(-1, -2)
        weighted_output = torch.bmm(attention_weights, hidden_states).squeeze(1)
        x = self.dropout(weighted_output)
        logits = self.linear(x)
        return logits

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api.log")
    ]
)
logger = logging.getLogger("product-classifier-api")

# Global variables
bert_model = None
bert_tokenizer = None
bert_label_encoder = None
fallback_model = None
fallback_tokenizer = None
fallback_label_encoder = None
category_mapping = None
device = None

def load_models():
    """Load all models and resources"""
    global bert_model, bert_tokenizer, bert_label_encoder
    global fallback_model, fallback_tokenizer, fallback_label_encoder
    global category_mapping, device
    
    config_path = 'api/config.yaml'
    if not os.path.exists(config_path):
        config_path = 'config.yaml'
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Load BERT model and tokenizer
        model_dir = config.get('api', {}).get('model_path', 'models')
        bert_model_path = os.path.join(model_dir, 'bert_model_double.bin')
        pretrained_model_name = config.get('models', {}).get('bert', {}).get('model_name', 'neuralmind/bert-base-portuguese-cased')
        
        # Load BERT tokenizer (use pretrained model name since model_dir might not have tokenizer files)
        logger.info(f"Loading BERT tokenizer from pretrained model: {pretrained_model_name}")
        bert_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)
        
        logger.info(f"Loading BERT model state_dict from {bert_model_path}")
        if os.path.exists(bert_model_path):
            # Load label encoder to get num_classes
            encoder_path = config.get('api', {}).get('label_encoder_path', 'category_encoder.joblib')
            if os.path.exists(encoder_path):
                bert_label_encoder = joblib.load(encoder_path)
                logger.info(f"Loaded label encoder from {encoder_path}")
            else:
                raise FileNotFoundError(f"Label encoder not found at {encoder_path}")
            
            num_classes = len(bert_label_encoder.classes_)
            bert_model = BertForSequenceClassification.from_pretrained(pretrained_model_name, num_labels=num_classes)
            state_dict = torch.load(bert_model_path, map_location=device)
            bert_model.load_state_dict(state_dict)
            bert_model.to(device)
            bert_model.eval()
            logger.info("BERT model loaded successfully")

            #torch.save(bert_model.state_dict(), 'converted_bert_model.bin')

        else:
            raise FileNotFoundError(f"BERT model .bin file not found at {bert_model_path}")

        # Load fallback model (state_dict format)
        logger.info("Loading fallback model...")
        fallback_model_path = config.get('models', {}).get('fallback', {}).get('model_path', 'models/fallback_model.bin')
        fallback_tokenizer_path = config.get('models', {}).get('fallback', {}).get('tokenizer_path', 'fallback_tokenizer.joblib')
        fallback_label_encoder_path = config.get('models', {}).get('fallback', {}).get('label_encoder_path', 'fallback_label_encoder.joblib')

        if os.path.exists(fallback_model_path):
            # Load fallback label encoder to determine num_classes
            if os.path.exists(fallback_label_encoder_path):
                fallback_label_encoder = joblib.load(fallback_label_encoder_path)
                logger.info(f"Loaded fallback label encoder from {fallback_label_encoder_path}")
            else:
                logger.warning(f"Fallback label encoder not found at {fallback_label_encoder_path}. Using BERT label encoder.")
                fallback_label_encoder = bert_label_encoder

            num_classes = len(fallback_label_encoder.classes_)
            logger.info(f"Number of classes for fallback model: {num_classes}")
            fallback_model = TextClassifier(num_classes=num_classes, pretrained_model=pretrained_model_name)
            state_dict = torch.load(fallback_model_path, map_location=device)
            fallback_model.load_state_dict(state_dict)
            fallback_model.to(device)
            fallback_model.eval()

            # Load fallback tokenizer (check if it's a joblib file or a pretrained model directory)
            if os.path.exists(fallback_tokenizer_path) and fallback_tokenizer_path.endswith('.joblib'):
                fallback_tokenizer = joblib.load(fallback_tokenizer_path)
                logger.info(f"Loaded fallback tokenizer from {fallback_tokenizer_path}")
            else:
                logger.info(f"Loading fallback tokenizer from pretrained model: {pretrained_model_name}")
                fallback_tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

            logger.info("Fallback model loaded successfully")
        else:
            logger.warning(f"Fallback model not found at {fallback_model_path}. Skipping fallback model loading.")
            fallback_model = None
            fallback_tokenizer = None
            fallback_label_encoder = None

        # Load category mapping
        logger.info("Loading category mapping...")
        categories_file = config.get('data', {}).get('categories_file', 'data/categories.csv')
        if os.path.exists(categories_file):
            categories_df = pd.read_csv(categories_file)
            if 'category_name' not in categories_df.columns or 'category_id' not in categories_df.columns:
                raise ValueError(f"Categories file {categories_file} must contain 'category_name' and 'category_id' columns")
            category_mapping = dict(zip(categories_df['category_name'], categories_df['category_id']))
            logger.info(f"Category mapping loaded with {len(category_mapping)} categories")
        else:
            logger.warning(f"Categories file not found at {categories_file}. Category IDs will be N/A.")
            category_mapping = {}
    
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    try:
        load_models()
        yield
    except Exception as e:
        logger.error(f"Failed to load models at startup: {str(e)}")
        raise
    finally:
        # Safely clean up resources
        global bert_model, fallback_model
        if bert_model is not None:
            logger.info("Cleaning up BERT model")
            del bert_model
            bert_model = None
        if fallback_model is not None:
            logger.info("Cleaning up fallback model")
            del fallback_model
            fallback_model = None
        torch.cuda.empty_cache()

app = FastAPI(
    title="Product Category Classification API",
    description="API for classifying product names into categories using BERT and fallback models",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProductRequest(BaseModel):
    product_name: str

class PredictionResponse(BaseModel):
    bert_prediction: Dict[str, Any]
    fallback_prediction: Dict[str, Any]
    recommended_prediction: Dict[str, Any]

def preprocess_text(text):
    """Preprocess text for inference"""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def predict_with_bert_model(product_name: str) -> Dict[str, Any]:
    """Make prediction using BERT model"""
    try:
        if bert_model is None or bert_tokenizer is None or bert_label_encoder is None:
            raise ValueError("BERT model, tokenizer, or label encoder not loaded")
        
        processed_text = preprocess_text(product_name)
        inputs = bert_tokenizer(processed_text, truncation=True, padding=True, 
                                max_length=64, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
        
        predicted_class_idx = predicted_class.item()
        confidence_score = confidence.item()
        category_name = bert_label_encoder.inverse_transform([predicted_class_idx])[0]
        category_id = category_mapping.get(category_name, "N/A")
        
        return {
            "category_name": category_name,
            "category_id": category_id,
            "confidence": confidence_score,
            "model": "bert"
        }
    
    except Exception as e:
        logger.error(f"Error predicting with BERT model: {str(e)}")
        return {
            "category_name": "Error",
            "category_id": "N/A",
            "confidence": 0.0,
            "model": "bert",
            "error": str(e)
        }

def predict_with_fallback_model(product_name: str) -> Dict[str, Any]:
    """Make prediction using fallback model"""
    if fallback_model is None or fallback_tokenizer is None or fallback_label_encoder is None:
        return {
            "category_name": "N/A",
            "category_id": "N/A",
            "confidence": 0.0,
            "model": "fallback",
            "error": "Fallback model, tokenizer, or label encoder not loaded"
        }
    
    try:
        processed_text = preprocess_text(product_name)
        encoding = fallback_tokenizer(processed_text, truncation=True, padding=True, 
                                     max_length=128, return_tensors='pt')
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = fallback_model(input_ids, attention_mask)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)
        
        predicted_category = fallback_label_encoder.inverse_transform([predicted.item()])[0]
        category_id = category_mapping.get(predicted_category, "N/A")
        
        return {
            "category_name": predicted_category,
            "category_id": category_id,
            "confidence": confidence.item(),
            "model": "fallback"
        }
    
    except Exception as e:
        logger.error(f"Error predicting with fallback model: {str(e)}")
        return {
            "category_name": "Error",
            "category_id": "N/A",
            "confidence": 0.0,
            "model": "fallback",
            "error": str(e)
        }

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Middleware to track request processing time"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
def get_root():
    """Root endpoint"""
    return {"message": "Product Classification API is running. Use /predict endpoint for predictions."}

@app.get("/health")
def health_check():
    """Health check endpoint"""
    if bert_model is None:
        raise HTTPException(status_code=503, detail="BERT model not loaded")
    return {"status": "healthy", "models_loaded": {"bert": True, "fallback": fallback_model is not None}}

@app.post("/predict", response_model=PredictionResponse)
def predict_category(request: ProductRequest):
    """Predict product category based on product name"""
    if not request.product_name:
        raise HTTPException(status_code=400, detail="Product name cannot be empty")
    
    logger.info(f"Processing prediction request for: {request.product_name}")
    
    bert_prediction = predict_with_bert_model(request.product_name)
    fallback_prediction = predict_with_fallback_model(request.product_name)
    
    # Use BERT prediction if confidence >= threshold, otherwise fallback
    confidence_threshold = 0.7  # Match the config value
    recommended_prediction = bert_prediction.copy() if bert_prediction.get("confidence", 0) >= confidence_threshold else fallback_prediction.copy()
    recommended_prediction["recommended"] = True
    
    return {
        "bert_prediction": bert_prediction,
        "fallback_prediction": fallback_prediction,
        "recommended_prediction": recommended_prediction
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)