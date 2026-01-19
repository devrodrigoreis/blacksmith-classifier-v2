import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import re
import gc
import joblib
from transformers import BertTokenizer, BertModel
from tqdm import tqdm, trange
import os

class ProductDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

class TextClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model="neuralmind/bert-base-portuguese-cased", dropout_rate=0.3):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.Tanh(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        attention_weights = self.attention(hidden_states).transpose(-1, -2)
        weighted_output = torch.bmm(attention_weights, hidden_states).squeeze(1)
        x = self.dropout(weighted_output)
        logits = self.linear(x)
        return logits

def load_and_preprocess_data(products_file, categories_file, min_samples=50):
    try:
        products_data = pd.read_csv(products_file)
        categories_data = pd.read_csv(categories_file)
        print(f"Loaded {len(products_data)} products, {len(categories_data)} categories.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()

    # Clean product names
    products_data['product_name'] = products_data['product_name'].str.lower()
    products_data['product_name'] = products_data['product_name'].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
    
    # Extract full category path
    products_data['full_category'] = products_data['product_category_tree'].apply(
        lambda x: x.strip() if isinstance(x, str) else ''
    )
    products_data = products_data.dropna(subset=['product_name', 'full_category'])
    print(f"After cleaning: {len(products_data)} rows.")

    # Reduce small classes
    #class_counts = products_data['full_category'].value_counts()
    #small_classes = class_counts[class_counts < min_samples].index
    #products_data['full_category'] = products_data['full_category'].replace(small_classes, 'Other')
    #print(f"Reduced classes from {len(class_counts)} to {products_data['full_category'].nunique()}.")

    return products_data, categories_data

def train_fallback_model(products_file, categories_file, batch_size=32, epochs=3, learning_rate=2e-5, min_samples=50, max_length=64):
    print("Loading and preprocessing data...")
    products_data, categories_data = load_and_preprocess_data(products_file, categories_file, min_samples)
    
    if len(products_data) == 0:
        print("Error: No valid product data.")
        return

    # Oversample to balance classes
    ros = RandomOverSampler(random_state=42, sampling_strategy='not majority')
    resample_result = ros.fit_resample(products_data[['product_name']], products_data['full_category'])
    X_resampled, y_resampled = resample_result[0], resample_result[1]
    products_data = pd.DataFrame({'product_name': X_resampled['product_name'], 'full_category': y_resampled})
    print(f"After oversampling: {len(products_data)} rows.")

    # Encode categories
    label_encoder = LabelEncoder()
    products_data['encoded_category'] = label_encoder.fit_transform(products_data['full_category'])
    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        products_data['product_name'].values,
        products_data['encoded_category'].values,
        test_size=0.2,
        random_state=42,
        stratify=np.array(products_data['encoded_category'].values)
    )
    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}")

    # Verify train_texts
    if not train_texts.size or any(not isinstance(t, str) for t in train_texts):
        print("Error: train_texts is empty or contains invalid entries.")
        print(f"Sample train_texts: {train_texts[:5]}")
        return

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    print("Tokenizer initialized.")

    # Tokenize with error handling and logging
    try:
        print(f"Starting tokenization of {len(train_texts)} training samples...")
        train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=max_length)
        print(f"Train tokenization complete. Keys: {list(train_encodings.keys())}")
        
        print(f"Starting tokenization of {len(val_texts)} validation samples...")
        val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=max_length)
        
        print("Validation tokenization complete.")
    except Exception as e:
        print(f"Error during tokenization: {e}")
        return

    # Create datasets
    print("Preparing datasets...")
    train_dataset = ProductDataset(train_encodings, train_labels)
    val_dataset = ProductDataset(val_encodings, val_labels)
    print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    print("Data loaders created.")

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TextClassifier(num_classes=num_classes, pretrained_model='neuralmind/bert-base-portuguese-cased').to(device)
    print(f"Model initialized on {device}.")

    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    print(f"Training for {epochs} epochs...")
    best_val_f1 = 0
    patience = 3
    no_improve = 0

    for epoch in trange(epochs, desc="Epochs"):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc="Training", leave=False):
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # Evaluation
        model.eval()
        predictions, true_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                true_labels.extend(labels.cpu().tolist())

        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(true_labels, predictions, average='weighted', zero_division=1)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_fallback_model.bin')
            print("New best model saved!")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Save final model and encoders
    torch.save(model.state_dict(), 'fine_tuned_model.bin')
    joblib.dump(label_encoder, 'fallback_label_encoder.joblib')
    joblib.dump(tokenizer, 'fallback_tokenizer.joblib')
    print("Model and encoders saved as fine_tuned_model.bin, etc.")

    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train a BERT-based product classifier')
    parser.add_argument('--products', type=str, default='data/products.csv')
    parser.add_argument('--categories', type=str, default='data/categories.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--min_samples', type=int, default=50)
    parser.add_argument('--max_length', type=int, default=64)
    args = parser.parse_args()

    products_path = os.path.abspath(args.products)
    categories_path = os.path.abspath(args.categories)

    if not os.path.exists(products_path) or not os.path.exists(categories_path):
        print(f"Error: File(s) not found: {products_path}, {categories_path}")
        exit(1)

    train_fallback_model(
        products_path,
        categories_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        min_samples=args.min_samples,
        max_length=args.max_length
    )