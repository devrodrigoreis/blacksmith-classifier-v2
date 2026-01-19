"""
Data pipeline for continuous learning with replay buffer and preprocessing
"""
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
import logging
from datetime import datetime
import joblib
import re
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer
import random
from pathlib import Path

class ReplayBuffer:
    """Maintains a buffer of old samples to prevent catastrophic forgetting"""
    
    def __init__(self, config, max_size: int = None):
        self.config = config
        self.max_size = max_size or config.replay_buffer_size
        self.buffer_path = config.replay_buffer_path
        self.logger = logging.getLogger(__name__)
        
        # Load existing buffer if it exists
        if os.path.exists(self.buffer_path):
            self.data = pd.read_parquet(self.buffer_path)
            self.logger.info(f"Loaded replay buffer with {len(self.data)} samples")
        else:
            self.data = pd.DataFrame(columns=['input_text', 'product_category_tree', 'timestamp'])
            self.logger.info("Initialized empty replay buffer")
    
    def add_samples(self, new_data: pd.DataFrame):
        """Add new samples to the replay buffer"""
        # Add timestamp
        new_data = new_data.copy()
        new_data['timestamp'] = datetime.now().isoformat()
        
        # Append to buffer
        self.data = pd.concat([self.data, new_data], ignore_index=True)
        
        # Maintain buffer size limit using reservoir sampling for diversity
        if len(self.data) > self.max_size:
            self.data = self._reservoir_sample(self.data, self.max_size)
        
        # Save updated buffer
        self.save_buffer()
        self.logger.info(f"Added {len(new_data)} samples to replay buffer. Current size: {len(self.data)}")
    
    def _reservoir_sample(self, data: pd.DataFrame, k: int) -> pd.DataFrame:
        """Reservoir sampling to maintain diversity in replay buffer"""
        if len(data) <= k:
            return data
        
        # Group by category to ensure representation
        categories = data['product_category_tree'].unique()
        samples_per_category = max(1, k // len(categories))
        
        sampled_data = []
        remaining_slots = k
        
        for category in categories:
            category_data = data[data['product_category_tree'] == category]
            n_samples = min(samples_per_category, len(category_data), remaining_slots)
            
            if n_samples > 0:
                sampled = category_data.sample(n=n_samples, random_state=42)
                sampled_data.append(sampled)
                remaining_slots -= n_samples
        
        # Fill remaining slots randomly
        if remaining_slots > 0:
            already_sampled = pd.concat(sampled_data) if sampled_data else pd.DataFrame()
            remaining_data = data[~data.index.isin(already_sampled.index)]
            
            if len(remaining_data) > 0:
                additional = remaining_data.sample(
                    n=min(remaining_slots, len(remaining_data)), 
                    random_state=42
                )
                sampled_data.append(additional)
        
        return pd.concat(sampled_data, ignore_index=True) if sampled_data else pd.DataFrame()
    
    def get_replay_samples(self, n_samples: int) -> pd.DataFrame:
        """Get random samples from the replay buffer"""
        if len(self.data) == 0:
            return pd.DataFrame(columns=['input_text', 'product_category_tree'])
        
        n_samples = min(n_samples, len(self.data))
        return self.data.sample(n=n_samples, random_state=None)[['input_text', 'product_category_tree']]
    
    def save_buffer(self):
        """Save the replay buffer to disk"""
        os.makedirs(os.path.dirname(self.buffer_path), exist_ok=True)
        self.data.to_parquet(self.buffer_path, index=False)
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get the distribution of categories in the replay buffer"""
        return self.data['product_category_tree'].value_counts().to_dict()

class IncrementalDataProcessor:
    """Processes new data for incremental training"""
    
    def __init__(self, config, tokenizer: BertTokenizer, label_encoder: LabelEncoder):
        self.config = config
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.logger = logging.getLogger(__name__)
        
        # Ensure incremental data directory exists
        os.makedirs(config.incremental_data_dir, exist_ok=True)
    
    def preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing with category-indicative keyword extraction"""
        if not isinstance(text, str):
            return ""
        
        # Basic cleaning
        text = text.lower()
        
        # Normalize accents (simple approach)
        text = self._normalize_accents(text)
        
        # Remove extra punctuation but keep important separators
        text = re.sub(r'[^\w\s\-/]', ' ', text)
        
        # Extract category-indicative keywords
        keywords = self._extract_category_keywords(text)
        
        # Add keywords as features (prefix with special token)
        if keywords:
            text = f"{text} [KEYWORDS] {' '.join(keywords)}"
        
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def _normalize_accents(self, text: str) -> str:
        """Simple accent normalization"""
        accent_map = {
            'á': 'a', 'à': 'a', 'ã': 'a', 'â': 'a',
            'é': 'e', 'ê': 'e',
            'í': 'i',
            'ó': 'o', 'ô': 'o', 'õ': 'o',
            'ú': 'u', 'ü': 'u',
            'ç': 'c'
        }
        
        for accented, normal in accent_map.items():
            text = text.replace(accented, normal)
        
        return text
    
    def _extract_category_keywords(self, text: str) -> List[str]:
        """Extract category-indicative keywords"""
        # Define category keyword mappings
        category_keywords = {
            'clothing': ['roupa', 'camisa', 'calça', 'vestido', 'sapato', 'tênis', 'blusa', 'shorts'],
            'electronics': ['eletrônico', 'smartphone', 'tv', 'computador', 'tablet', 'fone', 'carregador'],
            'automotive': ['carro', 'auto', 'peça', 'motor', 'pneu', 'óleo', 'filtro'],
            'home': ['casa', 'cozinha', 'banheiro', 'quarto', 'decoração', 'móvel'],
            'beauty': ['beleza', 'cosmético', 'perfume', 'maquiagem', 'cabelo', 'pele'],
            'sports': ['esporte', 'fitness', 'academia', 'bola', 'exercício'],
            'books': ['livro', 'revista', 'educação', 'literatura'],
            'toys': ['brinquedo', 'criança', 'bebê', 'infantil'],
            'health': ['saúde', 'medicamento', 'vitamina', 'suplemento'],
            'garden': ['jardim', 'planta', 'semente', 'fertilizante', 'vaso']
        }
        
        found_keywords = []
        words = text.lower().split()
        
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in words or any(keyword in word for word in words):
                    found_keywords.append(category)
                    break
        
        return found_keywords
    
    def process_new_data(self, data_path: str) -> pd.DataFrame:
        """Process new incremental data"""
        self.logger.info(f"Processing new data from {data_path}")
        
        # Load data
        if data_path.endswith('.csv'):
            new_data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            new_data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Ensure required columns exist
        required_columns = ['product_name', 'product_category_tree']
        missing_columns = [col for col in required_columns if col not in new_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Handle details column if present
        if 'details' in new_data.columns:
            new_data['details'] = new_data['details'].fillna('')
            new_data['input_text'] = new_data['product_name'] + ' [SEP] ' + new_data['details']
        else:
            new_data['input_text'] = new_data['product_name']
        
        # Preprocess text
        new_data['input_text'] = new_data['input_text'].apply(self.preprocess_text)
        
        # Handle new categories
        new_data = self._handle_new_categories(new_data)
        
        # Save processed data with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_path = os.path.join(
            self.config.incremental_data_dir, 
            f"processed_{timestamp}.parquet"
        )
        new_data.to_parquet(processed_path, index=False)
        
        self.logger.info(f"Processed {len(new_data)} samples. Saved to {processed_path}")
        return new_data
    
    def _handle_new_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle new categories by expanding the label encoder"""
        existing_categories = set(self.label_encoder.classes_)
        new_categories = set(data['product_category_tree'].unique()) - existing_categories
        
        if new_categories:
            self.logger.info(f"Found {len(new_categories)} new categories: {new_categories}")
            
            # Expand label encoder
            all_categories = list(existing_categories) + list(new_categories)
            new_label_encoder = LabelEncoder()
            new_label_encoder.fit(all_categories)
            
            # Update the label encoder
            self.label_encoder.classes_ = new_label_encoder.classes_
            
            self.logger.info(f"Label encoder expanded to {len(self.label_encoder.classes_)} categories")
        
        return data

class ContinuousLearningDataset(Dataset):
    """Dataset for continuous learning with replay samples"""
    
    def __init__(self, new_data: pd.DataFrame, replay_data: pd.DataFrame, 
                 tokenizer: BertTokenizer, label_encoder: LabelEncoder, 
                 config, max_length: int = 128):
        self.config = config
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.logger = logging.getLogger(__name__)
        
        # Combine new and replay data
        self.data = self._combine_data(new_data, replay_data)
        
        # Encode labels
        self.labels = label_encoder.transform(self.data['product_category_tree'])
        
        # Tokenize all texts at once for efficiency
        self.encodings = tokenizer.batch_encode_plus(
            self.data['input_text'].tolist(),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        self.logger.info(f"Dataset created with {len(self.data)} samples "
                        f"(new: {len(new_data)}, replay: {len(replay_data)})")
    
    def _combine_data(self, new_data: pd.DataFrame, replay_data: pd.DataFrame) -> pd.DataFrame:
        """Combine new and replay data with proper balancing"""
        if len(replay_data) == 0:
            return new_data.copy()
        
        if len(new_data) == 0:
            return replay_data.copy()
        
        # Calculate target replay ratio
        target_replay_size = int(len(new_data) * self.config.replay_ratio / (1 - self.config.replay_ratio))
        actual_replay_size = min(target_replay_size, len(replay_data))
        
        # Sample replay data if needed
        if actual_replay_size < len(replay_data):
            replay_sample = replay_data.sample(n=actual_replay_size, random_state=42)
        else:
            replay_sample = replay_data
        
        # Combine data
        combined = pd.concat([new_data, replay_sample], ignore_index=True)
        
        # Shuffle
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return combined
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }
    
    def get_category_distribution(self) -> Dict[str, int]:
        """Get the distribution of categories in the dataset"""
        return self.data['product_category_tree'].value_counts().to_dict()

class DataPipeline:
    """Main data pipeline for continuous learning"""
    
    def __init__(self, config, tokenizer: BertTokenizer, label_encoder: LabelEncoder):
        self.config = config
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.replay_buffer = ReplayBuffer(config)
        self.processor = IncrementalDataProcessor(config, tokenizer, label_encoder)
    
    def prepare_incremental_training_data(self, new_data_path: str) -> Tuple[DataLoader, DataLoader]:
        """Prepare data loaders for incremental training"""
        # Process new data
        new_data = self.processor.process_new_data(new_data_path)
        
        # Get replay samples
        replay_size = int(len(new_data) * self.config.replay_ratio / (1 - self.config.replay_ratio))
        replay_data = self.replay_buffer.get_replay_samples(replay_size)
        
        # Split new data into train and validation
        val_size = max(1, int(0.2 * len(new_data)))
        new_train = new_data[:-val_size] if val_size < len(new_data) else new_data
        new_val = new_data[-val_size:] if val_size < len(new_data) else new_data.head(1)
        
        # Create datasets
        train_dataset = ContinuousLearningDataset(
            new_train, replay_data, self.tokenizer, self.label_encoder, self.config
        )
        
        val_dataset = ContinuousLearningDataset(
            new_val, pd.DataFrame(columns=['input_text', 'product_category_tree']), 
            self.tokenizer, self.label_encoder, self.config
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Add new data to replay buffer for future training
        self.replay_buffer.add_samples(new_data[['input_text', 'product_category_tree']])
        
        self.logger.info(f"Training data prepared: {len(train_dataset)} train, {len(val_dataset)} val")
        
        return train_loader, val_loader
    
    def create_validation_loader(self, data_path: str) -> DataLoader:
        """Create validation loader from existing data"""
        if data_path.endswith('.csv'):
            val_data = pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            val_data = pd.read_parquet(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Preprocess if needed
        if 'input_text' not in val_data.columns:
            if 'details' in val_data.columns:
                val_data['details'] = val_data['details'].fillna('')
                val_data['input_text'] = val_data['product_name'] + ' [SEP] ' + val_data['details']
            else:
                val_data['input_text'] = val_data['product_name']
        
        val_data['input_text'] = val_data['input_text'].apply(self.processor.preprocess_text)
        
        # Create dataset
        val_dataset = ContinuousLearningDataset(
            val_data, pd.DataFrame(columns=['input_text', 'product_category_tree']),
            self.tokenizer, self.label_encoder, self.config
        )
        
        return DataLoader(
            val_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
