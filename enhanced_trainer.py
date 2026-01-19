"""
Wrapper to integrate continuous learning with your existing trainer2.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from continuous_learning.pipeline import ContinuousLearningPipeline
from continuous_learning.config import ContinuousLearningConfig
import logging

class EnhancedTrainer:
    """Enhanced version of your trainer with continuous learning capabilities"""
    
    def __init__(self, config_path="continuous_learning_config.yaml"):
        self.pipeline = ContinuousLearningPipeline(config_path)
        self.logger = logging.getLogger(__name__)
    
    def train_from_scratch(self):
        """Run your original training script for initial model"""
        print("Running initial training...")
        
        # Import and run your original trainer
        try:
            from trainers.trainer2 import main as original_main
            original_main()
            print("✓ Initial training completed")
        except Exception as e:
            print(f"✗ Initial training failed: {e}")
            raise
    
    def setup_continuous_learning(self):
        """Setup continuous learning after initial training"""
        print("Setting up continuous learning...")
        
        try:
            self.pipeline.initialize()
            print("✓ Continuous learning initialized")
            return True
        except Exception as e:
            print(f"✗ Continuous learning setup failed: {e}")
            return False
    
    def train_incremental(self, new_data_path, strategy="knowledge_distillation"):
        """Train on new data incrementally"""
        if not self.pipeline.is_initialized:
            if not self.setup_continuous_learning():
                return None
        
        return self.pipeline.train_incremental(
            new_data_path=new_data_path,
            strategy=strategy,
            old_validation_data_path="data/products_eval_unique.csv"
        )
    
    def predict(self, texts, batch_size=32):
        """Make predictions using the current model"""
        if not self.pipeline.is_initialized:
            self.setup_continuous_learning()
        
        return self.pipeline.predict(texts, batch_size)
    
    def get_model_info(self):
        """Get model information"""
        if not self.pipeline.is_initialized:
            self.setup_continuous_learning()
        
        return self.pipeline.get_model_info()

if __name__ == "__main__":
    trainer = EnhancedTrainer()
    
    # Run initial training if no model exists
    if not os.path.exists("bert_model"):
        trainer.train_from_scratch()
    
    # Setup continuous learning
    trainer.setup_continuous_learning()
    
    print("Enhanced trainer ready for continuous learning!")
