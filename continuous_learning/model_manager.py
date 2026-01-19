"""
Model management for continuous learning including checkpointing, versioning, and knowledge distillation
"""
import os
import shutil
import torch
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from pathlib import Path
import json

class ModelManager:
    """Manages model loading, saving, versioning, and knowledge distillation"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.current_version = self._get_latest_version()
        
        # Ensure directories exist
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.model_registry_dir, exist_ok=True)
        
        # Handle teacher model path - ensure it's a directory path
        teacher_model_dir = self.config.teacher_model_path
        if not os.path.dirname(teacher_model_dir):
            # If it's just a filename, treat it as a directory
            teacher_model_dir = teacher_model_dir
        else:
            teacher_model_dir = os.path.dirname(teacher_model_dir)
        
        if teacher_model_dir:
            os.makedirs(teacher_model_dir, exist_ok=True)
        
    def _get_latest_version(self) -> str:
        """Get the latest model version"""
        if not os.path.exists(self.config.model_registry_dir):
            return f"{self.config.model_version_prefix}1.0"
            
        versions = []
        for item in os.listdir(self.config.model_registry_dir):
            if item.startswith(self.config.model_version_prefix):
                try:
                    version_num = float(item.replace(self.config.model_version_prefix, ''))
                    versions.append(version_num)
                except ValueError:
                    continue
                    
        if not versions:
            return f"{self.config.model_version_prefix}1.0"
            
        latest = max(versions)
        return f"{self.config.model_version_prefix}{latest + 0.1:.1f}"
    
    def load_base_model(self, model_path: Optional[str] = None) -> Tuple[BertForSequenceClassification, BertTokenizer, LabelEncoder]:
        """Load the base BERT model, tokenizer, and label encoder"""
        if model_path is None:
            model_path = self.config.base_model_path
            
        self.logger.info(f"Loading base model from {model_path}")
        
        try:
            # Load model and tokenizer
            model = BertForSequenceClassification.from_pretrained(model_path)
            tokenizer = BertTokenizer.from_pretrained(model_path)
            
            # Load label encoder
            label_encoder_path = "bert_label_encoder.joblib"
            if os.path.exists(label_encoder_path):
                label_encoder = joblib.load(label_encoder_path)
            else:
                # Fallback to correct path
                label_encoder = joblib.load("trainers/models/bert_label_encoder.joblib")
                
            self.logger.info(f"Model loaded successfully with {model.config.num_labels} labels")
            return model, tokenizer, label_encoder
            
        except Exception as e:
            self.logger.error(f"Failed to load base model: {e}")
            raise
    
    def create_teacher_model(self, model: BertForSequenceClassification):
        """Create and save a frozen copy of the model for knowledge distillation"""
        self.logger.info("Creating teacher model for knowledge distillation")
        
        # Create a deep copy of the model
        teacher_model = BertForSequenceClassification.from_pretrained(
            self.config.base_model_path,
            num_labels=model.config.num_labels
        )
        teacher_model.load_state_dict(model.state_dict())
        
        # Freeze all parameters
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.eval()
        
        # Save teacher model
        teacher_path = self.config.teacher_model_path
        os.makedirs(teacher_path, exist_ok=True)
        teacher_model.save_pretrained(teacher_path)
        
        self.logger.info(f"Teacher model saved to {teacher_path}")
        return teacher_model
    
    def load_teacher_model(self) -> Optional[BertForSequenceClassification]:
        """Load the teacher model for knowledge distillation"""
        if not os.path.exists(self.config.teacher_model_path):
            self.logger.warning("Teacher model not found")
            return None
            
        try:
            teacher_model = BertForSequenceClassification.from_pretrained(self.config.teacher_model_path)
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model.eval()
            
            self.logger.info("Teacher model loaded successfully")
            return teacher_model
            
        except Exception as e:
            self.logger.error(f"Failed to load teacher model: {e}")
            return None
    
    def save_checkpoint(self, model: BertForSequenceClassification, tokenizer: BertTokenizer, 
                       label_encoder: LabelEncoder, epoch: int, metrics: Dict, 
                       optimizer_state: Optional[Dict] = None):
        """Save a training checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}"
        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_name)
        
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        
        # Save label encoder
        joblib.dump(label_encoder, os.path.join(checkpoint_path, "label_encoder.joblib"))
        
        # Save optimizer state if provided
        if optimizer_state:
            torch.save(optimizer_state, os.path.join(checkpoint_path, "optimizer.pt"))
        
        # Save metrics and metadata
        metadata = {
            'epoch': epoch,
            'timestamp': timestamp,
            'metrics': metrics,
            'num_labels': len(label_encoder.classes_),
            'classes': label_encoder.classes_.tolist()
        }
        
        with open(os.path.join(checkpoint_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str) -> Tuple[BertForSequenceClassification, BertTokenizer, LabelEncoder, Dict]:
        """Load a training checkpoint"""
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            # Load model and tokenizer
            model = BertForSequenceClassification.from_pretrained(checkpoint_path)
            tokenizer = BertTokenizer.from_pretrained(checkpoint_path)
            
            # Load label encoder
            label_encoder = joblib.load(os.path.join(checkpoint_path, "label_encoder.joblib"))
            
            # Load metadata
            with open(os.path.join(checkpoint_path, "metadata.json"), 'r') as f:
                metadata = json.load(f)
            
            self.logger.info(f"Checkpoint loaded successfully from epoch {metadata['epoch']}")
            return model, tokenizer, label_encoder, metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint"""
        if not os.path.exists(self.config.checkpoint_dir):
            return None
            
        checkpoints = [d for d in os.listdir(self.config.checkpoint_dir) 
                      if d.startswith("checkpoint_") and 
                      os.path.isdir(os.path.join(self.config.checkpoint_dir, d))]
        
        if not checkpoints:
            return None
            
        # Sort by creation time
        checkpoints.sort(key=lambda x: os.path.getctime(
            os.path.join(self.config.checkpoint_dir, x)
        ), reverse=True)
        
        return os.path.join(self.config.checkpoint_dir, checkpoints[0])
    
    def register_model_version(self, model: BertForSequenceClassification, 
                             tokenizer: BertTokenizer, label_encoder: LabelEncoder, 
                             metrics: Dict) -> str:
        """Register a new model version in the model registry"""
        version_path = os.path.join(self.config.model_registry_dir, self.current_version)
        os.makedirs(version_path, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(version_path)
        tokenizer.save_pretrained(version_path)
        
        # Save label encoder
        joblib.dump(label_encoder, os.path.join(version_path, "label_encoder.joblib"))
        
        # Save version metadata
        metadata = {
            'version': self.current_version,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'num_labels': len(label_encoder.classes_),
            'classes': label_encoder.classes_.tolist()
        }
        
        with open(os.path.join(version_path, "version_info.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model version {self.current_version} registered")
        
        # Clean up old versions
        self._cleanup_old_versions()
        
        # Update current version for next time
        current_num = float(self.current_version.replace(self.config.model_version_prefix, ''))
        self.current_version = f"{self.config.model_version_prefix}{current_num + 0.1:.1f}"
        
        return version_path
    
    def _cleanup_old_versions(self):
        """Remove old model versions, keeping only the specified number"""
        if not os.path.exists(self.config.model_registry_dir):
            return
            
        versions = []
        for item in os.listdir(self.config.model_registry_dir):
            if item.startswith(self.config.model_version_prefix):
                version_path = os.path.join(self.config.model_registry_dir, item)
                if os.path.isdir(version_path):
                    try:
                        version_num = float(item.replace(self.config.model_version_prefix, ''))
                        versions.append((version_num, version_path))
                    except ValueError:
                        continue
        
        # Sort by version number and keep only the latest ones
        versions.sort(key=lambda x: x[0], reverse=True)
        
        for i, (version_num, version_path) in enumerate(versions):
            if i >= self.config.backup_models_count:
                self.logger.info(f"Removing old model version: {version_path}")
                shutil.rmtree(version_path)
    
    def rollback_to_checkpoint(self, checkpoint_path: str):
        """Rollback to a specific checkpoint"""
        self.logger.info(f"Rolling back to checkpoint: {checkpoint_path}")
        
        # Copy checkpoint to base model path
        if os.path.exists(self.config.base_model_path):
            backup_path = f"{self.config.base_model_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.move(self.config.base_model_path, backup_path)
            self.logger.info(f"Current model backed up to: {backup_path}")
        
        shutil.copytree(checkpoint_path, self.config.base_model_path)
        self.logger.info("Rollback completed")

class EWCRegularizer:
    """Elastic Weight Consolidation for preventing catastrophic forgetting"""
    
    def __init__(self, model: BertForSequenceClassification, dataset, config):
        self.model = model
        self.config = config
        self.fisher_information = {}
        self.optimal_params = {}
        self.logger = logging.getLogger(__name__)
        
        # Compute Fisher Information Matrix
        self._compute_fisher_information(dataset)
    
    def _compute_fisher_information(self, dataset):
        """Compute Fisher Information Matrix for important parameters"""
        self.logger.info("Computing Fisher Information Matrix...")
        
        self.model.eval()
        
        # Initialize Fisher information storage
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.fisher_information[name] = torch.zeros_like(param)
                self.optimal_params[name] = param.data.clone()
        
        # Compute Fisher information using validation data
        for batch in dataset:
            self.model.zero_grad()
            
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs.loss
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_information[name] += param.grad.data ** 2
        
        # Normalize by dataset size
        dataset_size = len(dataset)
        for name in self.fisher_information:
            self.fisher_information[name] /= dataset_size
        
        self.logger.info("Fisher Information Matrix computed")
    
    def get_ewc_loss(self) -> torch.Tensor:
        """Compute EWC regularization loss"""
        ewc_loss = 0
        for name, param in self.model.named_parameters():
            if name in self.fisher_information:
                ewc_loss += (self.fisher_information[name] * 
                           (param - self.optimal_params[name]) ** 2).sum()
        
        return self.config.ewc_lambda * ewc_loss
