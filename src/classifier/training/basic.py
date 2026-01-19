
import os
import pandas as pd
import torch
import logging
import gc
import multiprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import joblib

from ..config import config
from ..data.dataset import ProductDataset
from ..utils.text import preprocess_text
from ..utils.memory import MemoryMonitor
from ..utils.system import check_gpu

def compute_metrics(p):
    with torch.no_grad():
        predictions = np.argmax(p.predictions, axis=1)
        labels = p.label_ids
        # zero_division=1 to match original behavior
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=1)
        acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

class StandardTrainer:
    def __init__(self):
        self.tokenizer = None
        self.label_encoder = None
    
    def load_data_advanced(self):
        """
        Replicates the advanced loading logic from trainers/trainer2.py
        (merging details, using explicit train/eval splits)
        """
        # TODO: Move these paths to config.yaml
        train_path = 'data/products_train_unique.csv'
        eval_path = 'data/products_eval_unique.csv'
        details_path = 'data/products_with_descriptions.csv'
        
        logging.info("Loading data from explicit splits...")
        
        if not (os.path.exists(train_path) and os.path.exists(eval_path)):
             logging.warning("Explicit split files not found. Falling back to simple config load.")
             return None, None

        train_data = pd.read_csv(train_path)
        eval_data = pd.read_csv(eval_path)
        
        if os.path.exists(details_path):
            details_df = pd.read_csv(details_path)
            # Merge details
            train_data = train_data.merge(details_df[['pid', 'details']], on='pid', how='left')
            eval_data = eval_data.merge(details_df[['pid', 'details']], on='pid', how='left')
            train_data['details'] = train_data['details'].fillna('')
            eval_data['details'] = eval_data['details'].fillna('')
            
            # Combine
            train_data['input_text'] = train_data['product_name'] + ' [SEP] ' + train_data['details']
            eval_data['input_text'] = eval_data['product_name'] + ' [SEP] ' + eval_data['details']
        else:
             train_data['input_text'] = train_data['product_name']
             eval_data['input_text'] = eval_data['product_name']
             
        return train_data, eval_data

    def train(self):
        check_gpu()
        
        # Start memory monitor
        monitor = MemoryMonitor(
            max_memory_percent=config.memory.get('max_memory_percent', 90),
            check_interval=config.memory.get('memory_check_interval', 10)
        )
        monitor.start()
        
        try:
            # Data Loading
            train_data, eval_data = self.load_data_advanced()
            
            if train_data is None:
                # Fallback to simple load
                from ..data.processing import load_and_preprocess_data, balance_classes
                raw_data = load_and_preprocess_data(config.data_config['products_file'])
                
                # Encoder
                self.label_encoder = LabelEncoder()
                raw_data['full_category'] = self.label_encoder.fit_transform(raw_data['full_category'])
                
                # Balance
                data_balanced = balance_classes(raw_data)
                
                # Split
                train_data, eval_data = train_test_split(
                    data_balanced, test_size=0.2, random_state=42, stratify=data_balanced['full_category']
                )
                train_data['input_text'] = train_data['product_name']
                eval_data['input_text'] = eval_data['product_name']
            else:
                # Advanced load uses 'product_category_tree' typically?
                # trainers/trainer2.py used 'product_category_tree' for fitting encoder
                self.label_encoder = LabelEncoder()
                all_cats = pd.concat([train_data['product_category_tree'], eval_data['product_category_tree']])
                self.label_encoder.fit(all_cats)
                
                train_data['full_category'] = self.label_encoder.transform(train_data['product_category_tree'])
                eval_data['full_category'] = self.label_encoder.transform(eval_data['product_category_tree'])

            # Preprocess text (clean)
            train_data['input_text'] = train_data['input_text'].apply(preprocess_text)
            eval_data['input_text'] = eval_data['input_text'].apply(preprocess_text)

            # Tokenizer
            model_name = config.training['model_name']
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            
            # Datasets
            # Use max_length from config (default 256 for optimization)
            max_len = config.models.get('bert', {}).get('max_length', 256)
            
            train_dataset = ProductDataset(
                train_data['input_text'].tolist(),
                train_data['full_category'].tolist(),
                self.tokenizer,
                max_length=max_len
            )
            eval_dataset = ProductDataset(
                eval_data['input_text'].tolist(),
                eval_data['full_category'].tolist(),
                self.tokenizer,
                max_length=max_len
            )
            
            # Model Init
            def model_init():
                model = BertForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=len(self.label_encoder.classes_)
                )
                if torch.cuda.is_available():
                    model.gradient_checkpointing_enable()
                return model
            
            # Args
            training_args = TrainingArguments(
                output_dir=config.training.get('output_dir', './results'),
                num_train_epochs=config.training.get('num_train_epochs', 3),
                per_device_train_batch_size=config.training.get('per_device_train_batch_size', 16),
                per_device_eval_batch_size=config.training.get('per_device_eval_batch_size', 32),
                warmup_steps=config.training.get('warmup_steps', 500),
                weight_decay=config.training.get('weight_decay', 0.01),
                logging_dir=config.training.get('logging_dir', './logs'),
                logging_steps=config.training.get('logging_steps', 100),
                eval_strategy=config.training.get('eval_strategy', 'steps'),
                eval_steps=config.training.get('eval_steps', 500),
                save_strategy=config.training.get('save_strategy', 'steps'),
                save_steps=config.training.get('save_steps', 500),
                save_total_limit=config.training.get('save_total_limit', 2),
                load_best_model_at_end=True,
                metric_for_best_model='f1',
                fp16=config.training.get('fp16', False),
                bf16=config.training.get('bf16', False), # Support for BF16 (Ampere/Ada)
                dataloader_num_workers=config.training.get('dataloader_num_workers', 4),
                report_to='none'
            )
            
            trainer = Trainer(
                model_init=model_init,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
                compute_metrics=compute_metrics
            )
            
            logging.info("Starting training...")
            trainer.train()
            
            logging.info("Evaluating...")
            metrics = trainer.evaluate()
            logging.info(f"Metrics: {metrics}")
            
            # Save
            self.save(trainer)
            
        finally:
            monitor.stop()

    def save(self, trainer):
        output_dir = "bert_model"
        logging.info(f"Saving model to {output_dir}")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save encoder
        joblib.dump(self.label_encoder, "bert_label_encoder.joblib")
        logging.info("Artifacts saved.")

def train():
    trainer = StandardTrainer()
    trainer.train()
