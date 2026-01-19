"""
Continuous learning trainer with knowledge distillation and regularization
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import os

class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss combining hard and soft targets"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        self.alpha = config.knowledge_distillation_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                hard_targets: torch.Tensor) -> torch.Tensor:
        """
        Compute knowledge distillation loss
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            hard_targets: Ground truth labels
        """
        # Hard target loss (standard cross-entropy)
        hard_loss = self.ce_loss(student_logits, hard_targets)
        
        # Soft target loss (knowledge distillation)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_loss(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss
        
        return total_loss, hard_loss, soft_loss

class ContinuousLearningTrainer:
    """Main trainer for continuous learning with multiple strategies"""
    
    def __init__(self, config, model_manager, data_pipeline):
        self.config = config
        self.model_manager = model_manager
        self.data_pipeline = data_pipeline
        self.logger = logging.getLogger(__name__)
        
        # Training components
        self.model = None
        self.teacher_model = None
        self.optimizer = None
        self.scheduler = None
        self.ewc_regularizer = None
        
        # Metrics tracking
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'forgetting_metrics': []
        }
    
    def setup_training(self, model: BertForSequenceClassification, 
                      use_knowledge_distillation: bool = True,
                      use_ewc: bool = False,
                      old_val_loader=None):
        """Setup training components"""
        self.model = model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        
        # Setup teacher model for knowledge distillation
        if use_knowledge_distillation:
            self.teacher_model = self.model_manager.load_teacher_model()
            if self.teacher_model is None:
                # Create teacher model from current model
                self.teacher_model = self.model_manager.create_teacher_model(model)
            self.teacher_model.to(device)
            self.kd_loss = KnowledgeDistillationLoss(self.config)
        
        # Setup EWC regularization
        if use_ewc and old_val_loader is not None:
            from .model_manager import EWCRegularizer
            self.ewc_regularizer = EWCRegularizer(model, old_val_loader, self.config)
        
        # Setup optimizer and scheduler
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.l2_reg_weight,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        
        self.logger.info("Training setup completed")
    
    def train_incremental(self, train_loader, val_loader, old_val_loader=None) -> Dict:
        """
        Train the model incrementally with the specified strategy
        
        Args:
            train_loader: DataLoader for new training data (includes replay samples)
            val_loader: DataLoader for new validation data
            old_val_loader: DataLoader for old validation data (to check forgetting)
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Calculate total training steps
        total_steps = len(train_loader) * self.config.max_epochs
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        best_val_f1 = 0.0
        best_model_state = None
        patience_counter = 0
        max_patience = 3
        
        self.logger.info(f"Starting incremental training for {self.config.max_epochs} epochs")
        
        for epoch in range(self.config.max_epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader, device, epoch)
            
            # Validation phase
            val_metrics = self._evaluate(val_loader, device, "New Data Validation")
            
            # Check for forgetting on old data
            forgetting_metrics = {}
            if old_val_loader is not None:
                old_metrics = self._evaluate(old_val_loader, device, "Old Data Validation")
                forgetting_metrics = old_metrics
            
            # Log metrics
            self.training_history['epochs'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_metrics.get('loss', 0))
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            self.training_history['forgetting_metrics'].append(forgetting_metrics)
            
            self.logger.info(
                f"Epoch {epoch + 1}/{self.config.max_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            if forgetting_metrics:
                self.logger.info(
                    f"Old Data - Acc: {forgetting_metrics['accuracy']:.4f}, "
                    f"F1: {forgetting_metrics['f1']:.4f}"
                )
            
            # Early stopping and model selection
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                
                # Save checkpoint
                checkpoint_path = self.model_manager.save_checkpoint(
                    self.model, 
                    self.data_pipeline.tokenizer,
                    self.data_pipeline.label_encoder,
                    epoch,
                    {**val_metrics, **forgetting_metrics},
                    self.optimizer.state_dict()
                )
                
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        final_metrics = self._evaluate(val_loader, device, "Final Validation")
        final_forgetting = {}
        if old_val_loader is not None:
            final_forgetting = self._evaluate(old_val_loader, device, "Final Old Data Check")
        
        # Check if training was successful
        training_successful = self._check_training_success(final_metrics, final_forgetting)
        
        results = {
            'final_metrics': final_metrics,
            'forgetting_metrics': final_forgetting,
            'training_successful': training_successful,
            'best_f1': best_val_f1,
            'training_history': self.training_history
        }
        
        self.logger.info(f"Incremental training completed. Success: {training_successful}")
        
        return results
    
    def _train_epoch(self, train_loader, device, epoch) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Add knowledge distillation loss if teacher model is available
            if self.teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                kd_loss, hard_loss, soft_loss = self.kd_loss(
                    outputs.logits, teacher_outputs.logits, labels
                )
                loss = kd_loss
            
            # Add EWC regularization if available
            if self.ewc_regularizer is not None:
                ewc_loss = self.ewc_regularizer.get_ewc_loss()
                loss += ewc_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % 50 == 0:
                self.logger.debug(
                    f"Epoch {epoch + 1}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _evaluate(self, data_loader, device, desc="Evaluation") -> Dict:
        """Evaluate the model on a dataset"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=1
        )
        
        metrics = {
            'loss': total_loss / len(data_loader) if len(data_loader) > 0 else 0,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_samples': len(all_labels)
        }
        
        self.logger.debug(f"{desc} - Acc: {accuracy:.4f}, F1: {f1:.4f}")
        
        return metrics
    
    def _check_training_success(self, val_metrics: Dict, forgetting_metrics: Dict) -> bool:
        """Check if training was successful based on thresholds"""
        # Check minimum accuracy threshold
        if val_metrics['accuracy'] < self.config.min_accuracy_threshold:
            self.logger.warning(
                f"Validation accuracy {val_metrics['accuracy']:.4f} below threshold "
                f"{self.config.min_accuracy_threshold}"
            )
            return False
        
        # Check forgetting threshold if old data metrics are available
        if forgetting_metrics and 'accuracy' in forgetting_metrics:
            # This is a simplified check - in practice, you'd compare with baseline metrics
            if forgetting_metrics['accuracy'] < 0.7:  # Simplified threshold
                self.logger.warning(
                    f"Old data accuracy {forgetting_metrics['accuracy']:.4f} indicates forgetting"
                )
                return False
        
        return True
    
    def generate_training_report(self, results: Dict, output_dir: str):
        """Generate a comprehensive training report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot training history
        self._plot_training_history(results['training_history'], output_dir)
        
        # Generate confusion matrix
        if 'confusion_matrix' in results:
            self._plot_confusion_matrix(results['confusion_matrix'], output_dir)
        
        # Save detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'final_metrics': results['final_metrics'],
            'forgetting_metrics': results['forgetting_metrics'],
            'training_successful': results['training_successful'],
            'config': self.config.to_dict(),
            'training_history': results['training_history']
        }
        
        report_path = os.path.join(output_dir, 'training_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Training report saved to {output_dir}")
    
    def _plot_training_history(self, history: Dict, output_dir: str):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = history['epochs']
        
        # Training loss
        axes[0, 0].plot(epochs, history['train_loss'], label='Train Loss')
        axes[0, 0].plot(epochs, history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(epochs, history['val_accuracy'], label='Validation Accuracy')
        axes[0, 1].set_title('Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # F1 Score
        axes[1, 0].plot(epochs, history['val_f1'], label='Validation F1')
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        
        # Forgetting metrics
        if history['forgetting_metrics'] and any(history['forgetting_metrics']):
            forgetting_acc = [m.get('accuracy', 0) for m in history['forgetting_metrics']]
            axes[1, 1].plot(epochs, forgetting_acc, label='Old Data Accuracy')
            axes[1, 1].set_title('Forgetting Check (Old Data Accuracy)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, cm: np.ndarray, output_dir: str):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

class ContinuousLearningStrategies:
    """Different strategies for continuous learning"""
    
    @staticmethod
    def replay_with_finetuning(trainer: ContinuousLearningTrainer, 
                              train_loader, val_loader, old_val_loader=None):
        """Replay + Fine-tuning strategy"""
        trainer.logger.info("Using Replay + Fine-tuning strategy")
        trainer.setup_training(
            trainer.model, 
            use_knowledge_distillation=False, 
            use_ewc=False
        )
        return trainer.train_incremental(train_loader, val_loader, old_val_loader)
    
    @staticmethod
    def knowledge_distillation(trainer: ContinuousLearningTrainer,
                              train_loader, val_loader, old_val_loader=None):
        """Knowledge distillation strategy"""
        trainer.logger.info("Using Knowledge Distillation strategy")
        trainer.setup_training(
            trainer.model,
            use_knowledge_distillation=True,
            use_ewc=False
        )
        return trainer.train_incremental(train_loader, val_loader, old_val_loader)
    
    @staticmethod
    def ewc_regularization(trainer: ContinuousLearningTrainer,
                          train_loader, val_loader, old_val_loader):
        """Elastic Weight Consolidation strategy"""
        trainer.logger.info("Using EWC Regularization strategy")
        trainer.setup_training(
            trainer.model,
            use_knowledge_distillation=False,
            use_ewc=True,
            old_val_loader=old_val_loader
        )
        return trainer.train_incremental(train_loader, val_loader, old_val_loader)
    
    @staticmethod
    def hybrid_approach(trainer: ContinuousLearningTrainer,
                       train_loader, val_loader, old_val_loader):
        """Hybrid approach combining multiple strategies"""
        trainer.logger.info("Using Hybrid approach (KD + EWC)")
        trainer.setup_training(
            trainer.model,
            use_knowledge_distillation=True,
            use_ewc=True,
            old_val_loader=old_val_loader
        )
        return trainer.train_incremental(train_loader, val_loader, old_val_loader)
