"""
Setup script to integrate continuous learning with your existing BERT trainer
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install additional requirements for continuous learning"""
    print("Installing continuous learning requirements...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "continuous_learning_requirements.txt"
        ])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install requirements: {e}")
        return False
    
    return True

def setup_directories():
    """Create necessary directories for continuous learning"""
    print("Setting up directories...")
    
    directories = [
        "checkpoints",
        "model_registry", 
        "datasets/incremental",
        "datasets/incoming",
        "logs",
        "reports",
        "training_results",
        "batch_predictions",
        "teacher_model"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def create_config():
    """Create initial configuration file"""
    print("Creating configuration file...")
    
    from continuous_learning.config import ContinuousLearningConfig
    
    config = ContinuousLearningConfig()
    config.save_to_yaml("continuous_learning_config.yaml")
    print("✓ Configuration file created: continuous_learning_config.yaml")

def backup_existing_model():
    """Backup existing model before migration"""
    print("Backing up existing model...")
    
    model_paths = ["bert_model", "bert_model_double.bin", "bert_label_encoder.joblib"]
    backup_dir = "model_backup"
    
    os.makedirs(backup_dir, exist_ok=True)
    
    for path in model_paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.copytree(path, os.path.join(backup_dir, path), dirs_exist_ok=True)
            else:
                shutil.copy2(path, backup_dir)
            print(f"✓ Backed up: {path}")

def update_existing_trainer():
    """Update your existing trainer to be compatible with continuous learning"""
    print("Creating continuous learning wrapper for existing trainer...")
    
    wrapper_code = '''"""
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
'''
    
    with open("enhanced_trainer.py", "w", encoding="utf-8") as f:
        f.write(wrapper_code)
    
    print("✓ Created enhanced_trainer.py")

def create_api_starter():
    """Create API starter script"""
    print("Creating API starter script...")
    
    api_starter = '''#!/usr/bin/env python
"""
Starter script for the continuous learning API
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from continuous_learning.api import app
import uvicorn

if __name__ == "__main__":
    print("Starting Continuous Learning API...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )
'''
    
    with open("start_api.py", "w", encoding="utf-8") as f:
        f.write(api_starter)
    
    print("✓ Created start_api.py")

def create_cli_wrapper():
    """Create CLI wrapper script"""
    print("Creating CLI wrapper...")
    
    cli_wrapper = '''#!/usr/bin/env python
"""
CLI wrapper for continuous learning
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from continuous_learning.cli import main

if __name__ == "__main__":
    main()
'''
    
    with open("cl_cli.py", "w", encoding="utf-8") as f:
        f.write(cli_wrapper)
    
    print("✓ Created cl_cli.py")

def create_docker_files():
    """Create Docker files for deployment"""
    print("Creating Docker files...")
    
    # Dockerfile for continuous learning API
    dockerfile_content = '''FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt continuous_learning_requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r continuous_learning_requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p checkpoints model_registry datasets/incremental logs reports

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start API
CMD ["python", "start_api.py"]
'''
    
    with open("Dockerfile.continuous_learning", "w") as f:
        f.write(dockerfile_content)
    
    # Docker Compose for full stack
    docker_compose_content = '''version: '3.8'

services:
  continuous-learning-api:
    build:
      context: .
      dockerfile: Dockerfile.continuous_learning
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data:ro
      - ./checkpoints:/app/checkpoints
      - ./model_registry:/app/model_registry
      - ./datasets:/app/datasets
      - ./logs:/app/logs
      - ./reports:/app/reports
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped

  # Optional: Add monitoring services
  # prometheus:
  #   image: prom/prometheus
  #   ports:
  #     - "9090:9090"
  #   volumes:
  #     - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  # grafana:
  #   image: grafana/grafana
  #   ports:
  #     - "3000:3000"
  #   environment:
  #     - GF_SECURITY_ADMIN_PASSWORD=admin
'''
    
    with open("docker-compose.continuous_learning.yml", "w") as f:
        f.write(docker_compose_content)
    
    print("✓ Created Docker files")

def create_usage_examples():
    """Create usage examples and documentation"""
    print("Creating usage examples...")
    
    usage_doc = '''# Continuous Learning for BERT Product Classification

## Quick Start

### 1. Initial Setup
```bash
python setup_continuous_learning.py
```

### 2. Start API Server
```bash
python start_api.py
```

### 3. Train Incrementally
```bash
python cl_cli.py train new_data.csv --strategy knowledge_distillation
```

### 4. Make Predictions
```bash
python cl_cli.py predict --text "iPhone 14 Pro Max"
```

## API Usage

### Start API
```bash
python start_api.py
```

Visit http://localhost:8000/docs for interactive API documentation.

### Example API Calls

#### Predict
```bash
curl -X POST "http://localhost:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"texts": ["Samsung Galaxy S23"]}'
```

#### Incremental Training
```bash
curl -X POST "http://localhost:8000/train/incremental" \\
     -F "file=@new_products.csv" \\
     -F "strategy=knowledge_distillation"
```

## Programmatic Usage

```python
from continuous_learning.pipeline import ContinuousLearningPipeline

# Initialize pipeline
pipeline = ContinuousLearningPipeline()
pipeline.initialize()

# Train incrementally
results = pipeline.train_incremental(
    new_data_path="new_products.csv",
    strategy="knowledge_distillation"
)

# Make predictions
predictions = pipeline.predict(["New product description"])
```

## Continuous Learning Strategies

1. **Replay + Fine-tuning**: Train on new data + old samples
2. **Knowledge Distillation**: Use teacher model to prevent forgetting
3. **EWC (Elastic Weight Consolidation)**: Regularize important parameters
4. **Hybrid**: Combine multiple strategies

## Configuration

Edit `continuous_learning_config.yaml` to customize:
- Batch sizes and learning rates
- Replay buffer size and ratio
- Knowledge distillation parameters
- Evaluation thresholds

## Monitoring

- Logs: `logs/` directory
- Training reports: `reports/` directory
- Model versions: `model_registry/` directory
- Checkpoints: `checkpoints/` directory

## Deployment

### Docker
```bash
docker-compose -f docker-compose.continuous_learning.yml up
```

### Production
1. Use `pipeline.export_model_for_deployment()` to create deployment packages
2. Monitor model performance over time
3. Schedule regular full retraining (weekly/monthly)
'''
    
    with open("CONTINUOUS_LEARNING_README.md", "w") as f:
        f.write(usage_doc)
    
    print("✓ Created CONTINUOUS_LEARNING_README.md")

def run_tests():
    """Run basic tests to verify setup"""
    print("Running setup verification tests...")
    
    try:
        # Test imports
        from continuous_learning.config import ContinuousLearningConfig
        from continuous_learning.pipeline import ContinuousLearningPipeline
        print("✓ Imports working")
        
        # Test configuration
        config = ContinuousLearningConfig()
        print("✓ Configuration creation working")
        
        # Test directory structure
        required_dirs = ["checkpoints", "model_registry", "datasets/incremental"]
        for directory in required_dirs:
            if os.path.exists(directory):
                print(f"✓ Directory exists: {directory}")
            else:
                print(f"✗ Directory missing: {directory}")
                return False
        
        print("✓ Basic tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Tests failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Continuous Learning for BERT Product Classification")
    print("=" * 70)
    
    steps = [
        ("Installing requirements", install_requirements),
        ("Setting up directories", setup_directories),
        ("Creating configuration", create_config),
        ("Backing up existing model", backup_existing_model),
        ("Creating enhanced trainer", update_existing_trainer),
        ("Creating API starter", create_api_starter),
        ("Creating CLI wrapper", create_cli_wrapper),
        ("Creating Docker files", create_docker_files),
        ("Creating documentation", create_usage_examples),
        ("Running verification tests", run_tests)
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        try:
            result = step_func()
            if result:
                success_count += 1
                print(f"✅ {step_name} COMPLETED")
            else:
                print(f"❌ {step_name} FAILED")
        except Exception as e:
            print(f"❌ {step_name} FAILED: {e}")
    
    print(f"\n� Setup Results: {success_count}/{len(steps)} steps successful")
    
    if success_count == len(steps):
        print("\n✨ Next Steps:")
        print("1. Review the configuration: continuous_learning_config.yaml")
        print("2. Start the API: python start_api.py")
        print("3. Try the CLI: python cl_cli.py info")
        print("4. Read the documentation: CONTINUOUS_LEARNING_README.md")
        print("5. Run the example: python continuous_learning_example.py")
    else:
        print("\n⚠️  Some steps failed. Check the error messages above.")

if __name__ == "__main__":
    main()
