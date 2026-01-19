import psutil
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
import joblib
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
import yaml
import logging
from nlpaug.augmenter.word import SynonymAug

# Load configuration
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Set environment variable for CUDA memory configuration
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Memory monitoring configuration
MAX_MEMORY_PERCENT = config['memory']['max_memory_percent']
MEMORY_CHECK_INTERVAL = config['memory']['memory_check_interval']

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
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('omw-1.4')
# Define stop words
stop_words = set(stopwords.words('portuguese'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Data augmentation for small classes
def augment_text(text, aug):
    if not isinstance(text, str) or text.strip() == "":
        return text
    return aug.augment(text)[0]

def augment_small_classes(data, aug, min_samples=50):  # Fixed: Moved 'aug' before 'min_samples'
    class_counts = data['full_category'].value_counts()
    small_classes = class_counts[class_counts < min_samples].index
    augmented_data = []
    
    for category in small_classes:
        class_data = data[data['full_category'] == category]
        num_samples = len(class_data)
        if num_samples == 0:
            continue
        num_to_generate = min_samples - num_samples
        if num_to_generate <= 0:
            continue
        for _ in range(num_to_generate):
            sample = class_data.sample(n=1)
            augmented_text = augment_text(sample['product_name'].iloc[0], aug)
            augmented_data.append({'product_name': augmented_text, 'full_category': category})
    
    if augmented_data:
        augmented_df = pd.DataFrame(augmented_data)
        data = pd.concat([data, augmented_df], ignore_index=True)
        logging.info(f"Augmented {len(augmented_data)} samples for small classes.")
    return data

class ProductDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(p):
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

def compute_class_weights(labels, num_classes):
    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts = np.maximum(class_counts, 1)  # Avoid division by zero
    total_samples = len(labels)
    weights = total_samples / (num_classes * class_counts)
    weights = np.clip(weights, 0.1, 10.0)  # Avoid extreme weights
    return torch.tensor(weights, dtype=torch.float)

def load_and_preprocess_data(file_path, chunk_size=10000):
    total_lines = 0
    chunks = []
    for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size, dtype={'product_name': 'string', 'product_category_tree': 'string'})):
        chunk['full_category'] = chunk['product_category_tree'].apply(lambda x: x.replace('[', '').replace(']', '') if isinstance(x, str) else '')
        chunk['product_name'] = chunk['product_name'].apply(preprocess_text)
        chunks.append(chunk[['product_name', 'full_category']])
        total_lines += len(chunk)
        logging.info(f"Chunk {i + 1} processed, total lines: {total_lines}")
    return pd.concat(chunks), total_lines

def filter_small_classes(data, min_samples=1):
    class_counts = data['full_category'].value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index
    return data[data['full_category'].isin(valid_classes)]

def main():
    global running

    logging.info("Loading and preprocessing data...")
    data, total_lines = load_and_preprocess_data(config['data']['products_file'])
    logging.info(f"Data loaded and preprocessed. Total lines: {total_lines}")

    # Load all categories from categories.csv to ensure label encoder includes them
    categories_df = pd.read_csv(config['data']['categories_file'])
    all_categories = categories_df['category_name'].unique()
    logging.info(f"Loaded {len(all_categories)} categories from categories.csv")

    # Filter out classes with 0 samples in the data
    data = filter_small_classes(data, min_samples=1)

    # Augment small classes
    aug = SynonymAug(aug_src='wordnet', lang='por')
    data = augment_small_classes(data, aug, min_samples=50)  # Updated call to match new signature

    # Create label encoder with all categories
    label_encoder = LabelEncoder()
    label_encoder.fit(all_categories)  # Fit on all categories from categories.csv
    logging.info(f"Label encoder initialized with {len(label_encoder.classes_)} classes")

    # Map data categories to encoded labels
    data['full_category'] = data['full_category'].map(lambda x: x if x in label_encoder.classes_ else 'Other')
    data['full_category'] = label_encoder.transform(data['full_category'])

    # Compute class weights
    class_weights = compute_class_weights(data['full_category'].values, len(label_encoder.classes_))
    logging.info(f"Computed class weights for {len(class_weights)} classes.")

    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

    def model_init():
        model = BertForSequenceClassification.from_pretrained(
            'neuralmind/bert-base-portuguese-cased',
            num_labels=len(label_encoder.classes_)
        )
        model.gradient_checkpointing_enable()
        return model

    # Custom Trainer to apply class weights
    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    training_args = TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=config['training']['weight_decay'],
        logging_dir=config['training']['logging_dir'],
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_f1',
        fp16=True,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        learning_rate=2e-5,
    )

    # Single chunk
    chunk_size = total_lines
    current_chunk_start = 0
    total_chunks = 1
    models = []

    while current_chunk_start < total_lines:
        current_chunk_end = min(current_chunk_start + chunk_size, total_lines)
        chunk_data = data.iloc[current_chunk_start:current_chunk_end]
        
        if chunk_data['full_category'].nunique() < 2:
            logging.warning("Insufficient classes in data. Exiting.")
            break

        train_data, eval_data = train_test_split(chunk_data, test_size=0.2, random_state=42, stratify=chunk_data['full_category'])

        train_dataset = ProductDataset(train_data['product_name'].tolist(), train_data['full_category'].tolist(), tokenizer, max_length=64)
        eval_dataset = ProductDataset(eval_data['product_name'].tolist(), eval_data['full_category'].tolist(), tokenizer, max_length=64)

        trainer = WeightedTrainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            compute_metrics=compute_metrics
        )

        logging.info(f"Training on chunk {current_chunk_start // chunk_size + 1} of {total_chunks}")
        trainer.train()

        logging.info(f"GPU Memory Summary:\n{torch.cuda.memory_summary()}")

        model_path = os.path.join('./models', f'model_chunk_{current_chunk_start // chunk_size + 1}')
        trainer.save_model(model_path)
        models.append(trainer.model)
        
        torch.cuda.empty_cache()
        gc.collect()

        current_chunk_start = current_chunk_end

    # Single chunk, no averaging needed
    if len(models) == 1:
        final_model = models[0]
        logging.info("Single chunk trained, skipping model averaging.")
    else:
        logging.info("Averaging model weights from all chunks...")
        final_model = average_model_weights(models, label_encoder)

    # Save the final model
    model_save_path = os.path.join(os.getcwd(), config['api']['model_path'])
    logging.info(f"Saving the final trained model to {model_save_path}...")
    
    final_model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    joblib.dump(label_encoder, os.path.join(os.getcwd(), config['api']['label_encoder_path']))
    
    logging.info("Final BERT model saved successfully.")

    running = False
    monitor_thread.join()

def average_model_weights(models, label_encoder):
    model_state_dicts = [model.state_dict() for model in models]
    avg_state_dict = model_state_dicts[0]

    for key in avg_state_dict:
        avg_state_dict[key] = torch.stack([model_state_dict[key].float() for model_state_dict in model_state_dicts]).mean(dim=0)

    averaged_model = BertForSequenceClassification.from_pretrained(
        'neuralmind/bert-base-portuguese-cased',
        num_labels=len(label_encoder.classes_)
    )
    averaged_model.load_state_dict(avg_state_dict)
    return averaged_model

if __name__ == "__main__":
    logging.info("Starting BERT model training...")
    main()