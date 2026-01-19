import psutil
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import EarlyStoppingCallback
import joblib
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import gc
from torch.utils.data import Dataset as TorchDataset
import sys
import threading
import time
import os
import multiprocessing
import logging
import yaml

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set environment variable for CUDA memory configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Memory monitoring configuration
MAX_MEMORY_PERCENT = config.get('memory', {}).get('max_memory_percent', 98.0)
MEMORY_CHECK_INTERVAL = config.get('memory', {}).get('memory_check_interval', 10)

# Global variable to control execution
running = True

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def memory_monitor():
    process = psutil.Process()
    while running:
        mem_percent = process.memory_percent()
        if mem_percent > MAX_MEMORY_PERCENT:
            logging.error(f"Memory usage exceeded {MAX_MEMORY_PERCENT}% ({mem_percent:.2f}%). Terminating process.")
            sys.exit(1)
        time.sleep(MEMORY_CHECK_INTERVAL)

# Start memory monitoring thread
monitor_thread = threading.Thread(target=memory_monitor)
monitor_thread.start()

# Download necessary NLTK resources
nltk.download('stopwords', quiet=True)

# Define stop words
stop_words = set(stopwords.words('portuguese'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

class ProductDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Batch tokenize
        encodings = tokenizer.batch_encode_plus(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def compute_metrics(p):
    with torch.no_grad():
        predictions = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=1)
        acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def preprocess_text_parallel(data_chunk):
    return data_chunk['product_name'].apply(preprocess_text)

def load_and_preprocess_data(file_path, chunk_size=10000):
    preprocessed_path = file_path.replace('.csv', '.parquet')
    if os.path.exists(preprocessed_path):
        data = pd.read_parquet(preprocessed_path)
        total_lines = len(data)
        logging.info(f"Loaded preprocessed data from {preprocessed_path}")
    else:
        total_lines = 0
        chunks = []
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, dtype={'product_name': 'string', 'product_category_tree': 'string'})):
                chunk['full_category'] = chunk['product_category_tree'].apply(lambda x: x.replace('[', '').replace(']', '') if isinstance(x, str) else '')
                chunk['product_name'] = pool.apply_async(preprocess_text_parallel, (chunk,)).get()
                chunks.append(chunk[['product_name', 'full_category']])
                total_lines += len(chunk)
                logging.info(f"Chunk {i + 1} processed, total lines: {total_lines}")
        data = pd.concat(chunks)
        data.to_parquet(preprocessed_path)
        logging.info(f"Saved preprocessed data to {preprocessed_path}")
    return data, total_lines

def filter_small_classes(data, min_samples=10):
    class_counts = data['full_category'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    return data[data['full_category'].isin(valid_classes)]

def main():
    global running

    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9, device=0)
        logging.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("CUDA not available, using CPU")

    # Load and preprocess data
    logging.info("Loading and preprocessing data...")
    # Load train and eval data from unique product_name splits with details
    train_data = pd.read_csv('data/products_train_unique.csv')
    eval_data = pd.read_csv('data/products_eval_unique.csv')
    details_df = pd.read_csv('data/products_with_descriptions.csv')

    # Merge details into train and eval sets
    train_data = train_data.merge(details_df[['pid', 'details']], on='pid', how='left')
    eval_data = eval_data.merge(details_df[['pid', 'details']], on='pid', how='left')

    # Fill missing details with empty string
    train_data['details'] = train_data['details'].fillna('')
    eval_data['details'] = eval_data['details'].fillna('')

    # Combine product_name and details for input
    train_data['input_text'] = train_data['product_name'].astype(str) + ' [SEP] ' + train_data['details'].astype(str)
    eval_data['input_text'] = eval_data['product_name'].astype(str) + ' [SEP] ' + eval_data['details'].astype(str)

    # Encode labels using the same label encoder
    label_encoder = LabelEncoder()
    all_categories = pd.concat([train_data['product_category_tree'], eval_data['product_category_tree']])
    label_encoder.fit(all_categories)
    train_data['full_category'] = label_encoder.transform(train_data['product_category_tree'])
    eval_data['full_category'] = label_encoder.transform(eval_data['product_category_tree'])

    # Preprocess input_text
    train_data['input_text'] = train_data['input_text'].apply(preprocess_text)
    eval_data['input_text'] = eval_data['input_text'].apply(preprocess_text)

    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(config['training']['model_name'])

    # Create datasets using input_text
    train_dataset = ProductDataset(
        train_data['input_text'].tolist(),
        train_data['full_category'].tolist(),
        tokenizer,
        max_length=32
    )
    eval_dataset = ProductDataset(
        eval_data['input_text'].tolist(),
        eval_data['full_category'].tolist(),
        tokenizer,
        max_length=32
    )
    logging.info(f"Train set size: {len(train_data)}, Eval set size: {len(eval_data)}")

    # Define model initialization
    def model_init():
        model = BertForSequenceClassification.from_pretrained(
            config['training']['model_name'],
            num_labels=len(label_encoder.classes_)
        )
        model.gradient_checkpointing_enable()
        return model

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=15,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        warmup_steps=config.get('training', {}).get('warmup_steps', 500),
        weight_decay=config.get('training', {}).get('weight_decay', 0.01),
        logging_dir=config['training']['logging_dir'],
        logging_steps=config.get('training', {}).get('logging_steps', 100),
        eval_strategy=config.get('training', {}).get('eval_strategy', 'steps'),
        eval_steps=config.get('training', {}).get('eval_steps', 500),
        save_strategy=config.get('training', {}).get('save_strategy', 'steps'),
        save_steps=config.get('training', {}).get('save_steps', 500),
        save_total_limit=config.get('training', {}).get('save_total_limit', 2),
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        fp16=True,
        gradient_accumulation_steps=4,
        dataloader_num_workers=multiprocessing.cpu_count(),
        report_to='none'  # Disable wandb/tensorboard unless configured
    )

    # Initialize trainer with early stopping
    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        compute_metrics=compute_metrics
    )

    # Train model
    logging.info("Starting model training...")
    train_result = trainer.train()
    logging.info("Training completed")

    # Evaluate on validation set and print metrics
    eval_metrics = trainer.evaluate()
    logging.info(f"Validation metrics: {eval_metrics}")
    print("Validation metrics:")
    for k, v in eval_metrics.items():
        print(f"{k}: {v}")

    # Clear GPU memory
    torch.cuda.empty_cache()
    gc.collect()

    # Save model and artifacts
    model_path = os.path.join(os.getcwd(), "bert_model")
    bin_save_path = os.path.join(os.getcwd(), "bert_model_double.bin")
    label_encoder_path = os.path.join(os.getcwd(), "bert_label_encoder.joblib")
    
    logging.info(f"Saving model to {model_path} and {bin_save_path}...")
    trainer.save_model(model_path)
    if trainer.model is not None:
        torch.save(trainer.model.state_dict(), bin_save_path)
        
        # Also save to models directory for API usage
        models_dir = os.path.join(os.getcwd(), "models")
        os.makedirs(models_dir, exist_ok=True)
        api_bin_path = os.path.join(models_dir, "bert_model_double.bin")
        torch.save(trainer.model.state_dict(), api_bin_path)
        logging.info(f"Saved model state dict to API path: {api_bin_path}")
    else:
        logging.error("trainer.model is None. Model state_dict not saved.")
    tokenizer.save_pretrained(model_path)
    
    # Save label encoder for continuous learning compatibility
    joblib.dump(label_encoder, label_encoder_path)
    # Also save to original path for backward compatibility
    if 'api' in config and 'label_encoder_path' in config['api']:
        joblib.dump(label_encoder, os.path.join(os.getcwd(), config['api']['label_encoder_path']))
    
    logging.info("Model and artifacts saved successfully")
    logging.info(f"Label encoder saved to: {label_encoder_path}")
    
    # Save training metadata for continuous learning
    metadata = {
        'training_completed': train_result.metrics if hasattr(train_result, 'metrics') else {},
        'validation_metrics': eval_metrics,
        'num_labels': len(label_encoder.classes_),
        'categories': label_encoder.classes_.tolist(),
        'model_architecture': 'bert-base-uncased',
        'training_timestamp': pd.Timestamp.now().isoformat()
    }
    
    metadata_path = os.path.join(model_path, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        import json
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Training metadata saved to: {metadata_path}")

    # Stop memory monitor
    running = False
    monitor_thread.join()

if __name__ == "__main__":
    logging.info("Starting BERT model training...")
    try:
        main()
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        running = False
        monitor_thread.join()
        raise