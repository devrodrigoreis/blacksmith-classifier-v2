"""
Configuration settings for continuous learning
"""
import os
from dataclasses import dataclass
from typing import Dict, Any, List
import yaml

@dataclass
class ContinuousLearningConfig:
    """Configuration for continuous learning pipeline"""
    
    # Model paths
    base_model_path: str = "bert_model"
    teacher_model_path: str = "teacher_model"
    checkpoint_dir: str = "checkpoints"
    model_registry_dir: str = "model_registry"
    
    # Data paths
    replay_buffer_path: str = "replay_buffer.parquet"
    incremental_data_dir: str = "datasets/incremental"
    
    # Training parameters
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    max_epochs: int = 3
    gradient_accumulation_steps: int = 4
    
    # Continuous learning parameters
    replay_buffer_size: int = 1000
    replay_ratio: float = 0.3  # Ratio of old samples in each batch
    knowledge_distillation_weight: float = 0.5
    temperature: float = 3.0  # For knowledge distillation
    
    # Regularization parameters
    ewc_lambda: float = 1000.0  # Elastic Weight Consolidation penalty
    l2_reg_weight: float = 0.01
    
    # Evaluation thresholds
    min_accuracy_threshold: float = 0.80
    max_forgetting_threshold: float = 0.05  # Max allowed drop in old category performance
    
    # Memory management
    max_memory_percent: float = 90.0
    memory_check_interval: int = 10
    
    # API settings
    model_version_prefix: str = "bert_v"
    backup_models_count: int = 3
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'ContinuousLearningConfig':
        """Load configuration from YAML file"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                return cls(**config_dict.get('continuous_learning', {}))
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'continuous_learning': {
                field.name: getattr(self, field.name) 
                for field in self.__dataclass_fields__.values()
            }
        }
    
    def save_to_yaml(self, config_path: str):
        """Save configuration to YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
