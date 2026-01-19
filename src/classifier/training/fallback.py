
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import joblib
import gc

from ..config import config
from ..data.processing import load_and_preprocess_data
from ..data.dataset import ProductDataset
from ..models.custom import CustomBertClassifier
from ..utils.system import check_gpu
from transformers import BertTokenizer

class FallbackTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self):
        check_gpu()
        
        # Load Data
        logging_file = config.training.get('products_file', 'data/products.csv') # Fallback default
        # Actually usage: config.data_config['products_file'] ??
        # The original fallback_training main used argparse, defaults to data/products.csv
        
        products_file = config.data_config.get('products_file', 'data/products.csv')
        
        data = load_and_preprocess_data(products_file)
        if len(data) == 0:
            print("No data found.")
            return

        # Oversample
        ros = RandomOverSampler(random_state=42, sampling_strategy='not majority')
        X_resampled, y_resampled = ros.fit_resample(data[['product_name']], data['full_category'])
        
        # Encode
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_resampled)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            X_resampled['product_name'].values,
            y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )
        
        # Tokenizer
        tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
        
        # Datasets
        train_dataset = ProductDataset(train_texts.tolist(), train_labels, tokenizer)
        val_dataset = ProductDataset(val_texts.tolist(), val_labels, tokenizer)
        
        # Loaders
        batch_size = config.training.get('per_device_train_batch_size', 32)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Model
        num_classes = len(label_encoder.classes_)
        model = CustomBertClassifier(num_classes=num_classes).to(self.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=config.training.get('learning_rate', 2e-5))
        criterion = nn.CrossEntropyLoss()
        scaler = torch.cuda.amp.GradScaler()
        
        epochs = config.training.get('epochs', 3)
        best_val_f1 = 0
        
        for epoch in trange(epochs, desc="Epochs"):
            model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc="Training", leave=False):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()

            # Validation
            model.eval()
            predictions, true_labels = [], []
            val_loss = 0
            with torch.no_grad():
                 for batch in tqdm(val_loader, desc="Validation", leave=False):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    with torch.cuda.amp.autocast():
                        outputs = model(input_ids, attention_mask)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, dim=1)
                    predictions.extend(preds.cpu().tolist())
                    true_labels.extend(labels.cpu().tolist())
            
            val_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=1)
            print(f"Epoch {epoch+1} | F1: {val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), 'best_fallback_model.bin')
        
        # Final Save
        torch.save(model.state_dict(), 'fine_tuned_model.bin')
        joblib.dump(label_encoder, 'fallback_label_encoder.joblib')
        joblib.dump(tokenizer, 'fallback_tokenizer.joblib')

def train():
    trainer = FallbackTrainer()
    trainer.train()
