"""
Main continuous learning pipeline orchestrator
"""
import os
import logging
import torch
from datetime import datetime
from typing import Dict, Optional, List
import yaml
import json
from pathlib import Path

from .config import ContinuousLearningConfig
from .model_manager import ModelManager
from .data_pipeline import DataPipeline
from .trainer import ContinuousLearningTrainer, ContinuousLearningStrategies

class ContinuousLearningPipeline:
    """Main pipeline orchestrator for continuous learning"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the continuous learning pipeline"""
        # Load configuration
        self.config = ContinuousLearningConfig.from_yaml(config_path)
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.model_manager = ModelManager(self.config)
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.data_pipeline = None
        self.trainer = None
        
        # Pipeline state
        self.is_initialized = False
        
        self.logger.info("Continuous Learning Pipeline initialized")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"continuous_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def initialize(self):
        """Initialize the pipeline with base model and components"""
        try:
            # Load base model
            self.model, self.tokenizer, self.label_encoder = self.model_manager.load_base_model()
            
            # Initialize data pipeline
            self.data_pipeline = DataPipeline(self.config, self.tokenizer, self.label_encoder)
            
            # Initialize trainer
            self.trainer = ContinuousLearningTrainer(self.config, self.model_manager, self.data_pipeline)
            
            self.is_initialized = True
            self.logger.info("Pipeline initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {e}")
            raise
    
    def train_incremental(self, new_data_path: str, 
                         strategy: str = "knowledge_distillation",
                         old_validation_data_path: Optional[str] = None) -> Dict:
        """
        Perform incremental training on new data
        
        Args:
            new_data_path: Path to new training data
            strategy: Training strategy ('replay', 'knowledge_distillation', 'ewc', 'hybrid')
            old_validation_data_path: Path to old validation data for forgetting check
        
        Returns:
            Dictionary with training results
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        self.logger.info(f"Starting incremental training with strategy: {strategy}")
        self.logger.info(f"New data: {new_data_path}")
        
        try:
            # Prepare training data
            train_loader, val_loader = self.data_pipeline.prepare_incremental_training_data(new_data_path)
            
            # Prepare old validation data if provided
            old_val_loader = None
            if old_validation_data_path and os.path.exists(old_validation_data_path):
                old_val_loader = self.data_pipeline.create_validation_loader(old_validation_data_path)
                self.logger.info(f"Old validation data loaded: {old_validation_data_path}")
            
            # Update model for new categories if needed
            self._update_model_for_new_categories()
            
            # Set model for trainer
            self.trainer.model = self.model
            
            # Select and execute training strategy
            strategy_methods = {
                'replay': ContinuousLearningStrategies.replay_with_finetuning,
                'knowledge_distillation': ContinuousLearningStrategies.knowledge_distillation,
                'ewc': ContinuousLearningStrategies.ewc_regularization,
                'hybrid': ContinuousLearningStrategies.hybrid_approach
            }
            
            if strategy not in strategy_methods:
                raise ValueError(f"Unknown strategy: {strategy}. Available: {list(strategy_methods.keys())}")
            
            # Execute training
            results = strategy_methods[strategy](self.trainer, train_loader, val_loader, old_val_loader)
            
            # Post-training evaluation and decision
            training_decision = self._evaluate_training_results(results)
            
            if training_decision['accept_update']:
                # Register new model version
                model_version_path = self.model_manager.register_model_version(
                    self.model, self.tokenizer, self.label_encoder, results['final_metrics']
                )
                results['model_version_path'] = model_version_path
                self.logger.info(f"Model update accepted and registered: {model_version_path}")
            else:
                # Rollback to previous checkpoint
                latest_checkpoint = self.model_manager.get_latest_checkpoint()
                if latest_checkpoint:
                    self.model_manager.rollback_to_checkpoint(latest_checkpoint)
                    self.logger.warning("Model update rejected. Rolled back to previous checkpoint.")
                results['rollback'] = True
            
            # Generate training report
            report_dir = f"reports/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.trainer.generate_training_report(results, report_dir)
            results['report_path'] = report_dir
            
            # Add training decision to results
            results.update(training_decision)
            
            self.logger.info("Incremental training completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Incremental training failed: {e}")
            raise
    
    def _update_model_for_new_categories(self):
        """Update model architecture if new categories were added"""
        current_num_labels = self.model.config.num_labels
        required_num_labels = len(self.label_encoder.classes_)
        
        if required_num_labels > current_num_labels:
            self.logger.info(f"Expanding model from {current_num_labels} to {required_num_labels} labels")
            
            # Create new model with expanded classification head
            from transformers import BertForSequenceClassification
            new_model = BertForSequenceClassification.from_pretrained(
                self.config.base_model_path,
                num_labels=required_num_labels
            )
            
            # Copy weights from old model
            new_model.bert.load_state_dict(self.model.bert.state_dict())
            
            # Initialize new classification head weights
            if current_num_labels > 0:
                # Copy existing classifier weights
                new_model.classifier.weight.data[:current_num_labels] = self.model.classifier.weight.data
                new_model.classifier.bias.data[:current_num_labels] = self.model.classifier.bias.data
                
                # Initialize new weights with small random values
                torch.nn.init.normal_(new_model.classifier.weight.data[current_num_labels:], std=0.02)
                torch.nn.init.zeros_(new_model.classifier.bias.data[current_num_labels:])
            
            self.model = new_model
            self.logger.info("Model successfully expanded for new categories")
    
    def _evaluate_training_results(self, results: Dict) -> Dict:
        """Evaluate training results and decide whether to accept the update"""
        decision = {
            'accept_update': True,
            'reasons': []
        }
        
        # Check accuracy threshold
        if results['final_metrics']['accuracy'] < self.config.min_accuracy_threshold:
            decision['accept_update'] = False
            decision['reasons'].append(f"Accuracy {results['final_metrics']['accuracy']:.4f} below threshold {self.config.min_accuracy_threshold}")
        
        # Check forgetting threshold
        if results['forgetting_metrics'] and 'accuracy' in results['forgetting_metrics']:
            old_accuracy = results['forgetting_metrics']['accuracy']
            # This is a simplified check - in practice, compare with baseline
            if old_accuracy < 0.7:  # Simplified threshold
                decision['accept_update'] = False
                decision['reasons'].append(f"Catastrophic forgetting detected: old data accuracy {old_accuracy:.4f}")
        
        # Check if training was successful according to trainer
        if not results.get('training_successful', True):
            decision['accept_update'] = False
            decision['reasons'].append("Training marked as unsuccessful by trainer")
        
        return decision
    
    def predict(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Make predictions on new texts
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
        
        Returns:
            List of prediction dictionaries
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.model.eval()
        
        predictions = []
        
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Preprocess texts
            processed_texts = [self.data_pipeline.processor.preprocess_text(text) for text in batch_texts]
            
            # Tokenize
            encodings = self.tokenizer.batch_encode_plus(
                processed_texts,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                predicted_classes = torch.argmax(outputs.logits, dim=-1)
            
            # Convert to results
            for j, (text, pred_class, probs) in enumerate(zip(batch_texts, predicted_classes, probabilities)):
                predicted_label = self.label_encoder.inverse_transform([pred_class.item()])[0]
                confidence = probs[pred_class].item()
                
                # Get top 3 predictions
                top_k = min(3, len(self.label_encoder.classes_))
                top_indices = torch.topk(probs, top_k).indices
                top_predictions = [
                    {
                        'label': self.label_encoder.inverse_transform([idx.item()])[0],
                        'confidence': probs[idx].item()
                    }
                    for idx in top_indices
                ]
                
                predictions.append({
                    'text': text,
                    'predicted_label': predicted_label,
                    'confidence': confidence,
                    'top_predictions': top_predictions
                })
        
        return predictions
    
    def get_model_info(self) -> Dict:
        """Get information about the current model"""
        if not self.is_initialized:
            return {'error': 'Pipeline not initialized'}
        
        return {
            'num_labels': len(self.label_encoder.classes_),
            'categories': self.label_encoder.classes_.tolist(),
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'replay_buffer_size': len(self.data_pipeline.replay_buffer.data),
            'replay_buffer_categories': self.data_pipeline.replay_buffer.get_category_distribution()
        }
    
    def watch_for_new_data(self, watch_directory: str, check_interval: int = 300):
        """
        Watch for new data files and automatically trigger training
        
        Args:
            watch_directory: Directory to watch for new data files
            check_interval: Check interval in seconds
        """
        import time
        import glob
        
        self.logger.info(f"Starting data watcher for directory: {watch_directory}")
        processed_files = set()
        
        while True:
            try:
                # Look for new CSV/Parquet files
                pattern = os.path.join(watch_directory, "*.{csv,parquet}")
                data_files = set(glob.glob(pattern))
                
                new_files = data_files - processed_files
                
                for file_path in new_files:
                    self.logger.info(f"New data file detected: {file_path}")
                    
                    try:
                        # Trigger incremental training
                        results = self.train_incremental(
                            file_path, 
                            strategy="knowledge_distillation"
                        )
                        
                        if results.get('accept_update', False):
                            self.logger.info(f"Successfully processed {file_path}")
                        else:
                            self.logger.warning(f"Training update rejected for {file_path}")
                        
                        processed_files.add(file_path)
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process {file_path}: {e}")
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                self.logger.info("Data watcher stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Error in data watcher: {e}")
                time.sleep(check_interval)
    
    def export_model_for_deployment(self, output_path: str):
        """Export the current model for deployment"""
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save label encoder
        import joblib
        joblib.dump(self.label_encoder, os.path.join(output_path, "label_encoder.joblib"))
        
        # Save deployment metadata
        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'num_labels': len(self.label_encoder.classes_),
            'categories': self.label_encoder.classes_.tolist(),
            'model_config': self.config.to_dict()
        }
        
        with open(os.path.join(output_path, "deployment_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Model exported for deployment to: {output_path}")
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Clean up old checkpoints to save space"""
        checkpoint_dir = self.config.checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            return
        
        checkpoints = []
        for item in os.listdir(checkpoint_dir):
            checkpoint_path = os.path.join(checkpoint_dir, item)
            if os.path.isdir(checkpoint_path) and item.startswith("checkpoint_"):
                checkpoints.append((os.path.getctime(checkpoint_path), checkpoint_path))
        
        # Sort by creation time and keep only the latest ones
        checkpoints.sort(reverse=True)
        
        for i, (_, checkpoint_path) in enumerate(checkpoints):
            if i >= keep_last_n:
                import shutil
                shutil.rmtree(checkpoint_path)
                self.logger.info(f"Removed old checkpoint: {checkpoint_path}")
        
        self.logger.info(f"Checkpoint cleanup completed. Kept {min(len(checkpoints), keep_last_n)} checkpoints.")

def create_continuous_learning_config():
    """Create a default continuous learning configuration file"""
    config = ContinuousLearningConfig()
    config.save_to_yaml("continuous_learning_config.yaml")
    print("Default continuous learning configuration saved to continuous_learning_config.yaml")
    return config
