# 🚀 Continuous Learning for BERT Product Classification

This implementation provides a complete continuous learning system for your BERT product classification model, enabling it to adapt to new product data without catastrophic forgetting while maintaining production-ready performance.

## 🌟 Features

### ✅ **Complete Implementation of Your Requirements**

1. **✅ Model Loading & Checkpointing**
   - Loads existing BERT models from HuggingFace format
   - Maintains frozen teacher models for knowledge distillation
   - Comprehensive checkpoint system with rollback capabilities
   - Version management and model registry

2. **✅ Data Pipeline for Continuous Learning**
   - Accepts mini-batches of new product data
   - Versioned dataset storage with timestamps
   - Advanced preprocessing with accent normalization and category keyword extraction
   - Intelligent replay buffer with reservoir sampling for diversity

3. **✅ Incremental Training Strategies**
   - **Replay + Fine-tuning**: Combines new data with old samples
   - **Knowledge Distillation**: Uses teacher model to prevent forgetting
   - **Elastic Weight Consolidation (EWC)**: Regularizes important parameters
   - **Hybrid Approach**: Combines multiple strategies

4. **✅ Evaluation & Monitoring**
   - Comprehensive metrics tracking (accuracy, F1, per-class performance)
   - Forgetting detection on old categories
   - Early stopping and model selection
   - Detailed training reports with visualizations

5. **✅ Knowledge Distillation**
   - Temperature-scaled soft targets from teacher model
   - Weighted combination of hard and soft losses
   - Configurable distillation parameters

6. **✅ Deployment & API Integration**
   - FastAPI-based REST API for predictions and training
   - Model versioning and backward compatibility
   - Export functionality for deployment packages
   - Health checks and monitoring endpoints

7. **✅ Automation & Continuous Learning Loop**
   - File watcher for automatic training on new data
   - Scheduled batch processing capabilities
   - Background training with status tracking
   - Automated model validation and deployment

8. **✅ Scalability & Optimization**
   - Mixed precision training (FP16) support
   - GPU acceleration with CPU fallback
   - Gradient accumulation for memory efficiency
   - Dynamic category expansion without full retraining

## 🏗️ Architecture

```
continuous_learning/
├── __init__.py                 # Package initialization
├── config.py                  # Configuration management
├── model_manager.py           # Model loading, saving, versioning
├── data_pipeline.py           # Data processing and replay buffer
├── trainer.py                 # Training strategies and algorithms
├── pipeline.py                # Main orchestrator
├── api.py                     # FastAPI REST endpoints
└── cli.py                     # Command-line interface
```

## 🚀 Quick Start

### 1. Installation and Setup

```bash
# Install additional requirements
pip install -r continuous_learning_requirements.txt

# Run the setup script
python setup_continuous_learning.py
```

### 2. Initialize Configuration

```python
from continuous_learning.config import ContinuousLearningConfig

# Create default configuration
config = ContinuousLearningConfig()
config.save_to_yaml("continuous_learning_config.yaml")
```

### 3. Basic Usage

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
predictions = pipeline.predict(["iPhone 15 Pro Max"])
```

## 📚 Usage Examples

### Training Strategies

#### 1. Knowledge Distillation (Recommended)
```python
results = pipeline.train_incremental(
    new_data_path="new_products.csv",
    strategy="knowledge_distillation",
    old_validation_data_path="validation_data.csv"
)
```

#### 2. Replay + Fine-tuning
```python
results = pipeline.train_incremental(
    new_data_path="new_products.csv",
    strategy="replay"
)
```

#### 3. Elastic Weight Consolidation
```python
results = pipeline.train_incremental(
    new_data_path="new_products.csv", 
    strategy="ewc",
    old_validation_data_path="validation_data.csv"  # Required for EWC
)
```

#### 4. Hybrid Approach
```python
results = pipeline.train_incremental(
    new_data_path="new_products.csv",
    strategy="hybrid",
    old_validation_data_path="validation_data.csv"
)
```

### API Usage

#### Start API Server
```bash
python start_api.py
```

#### API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Make Predictions:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Samsung Galaxy S24 Ultra"]}'
```

**Incremental Training:**
```bash
curl -X POST "http://localhost:8000/train/incremental" \
     -F "file=@new_products.csv" \
     -F "strategy=knowledge_distillation"
```

**Model Information:**
```bash
curl http://localhost:8000/model/info
```

### CLI Usage

```bash
# Initialize configuration
python cl_cli.py init

# Train incrementally
python cl_cli.py train new_data.csv --strategy knowledge_distillation

# Make predictions
python cl_cli.py predict --text "MacBook Pro M3"

# Batch predictions
python cl_cli.py predict --file products.csv --output predictions.csv

# Model information
python cl_cli.py info

# Export for deployment
python cl_cli.py export deployed_model/

# Start API server
python cl_cli.py api --host 0.0.0.0 --port 8000

# Watch directory for new data
python cl_cli.py watch datasets/incoming --interval 300
```

## ⚙️ Configuration

Edit `continuous_learning_config.yaml`:

```yaml
continuous_learning:
  # Model paths
  base_model_path: "bert_model"
  teacher_model_path: "teacher_model"
  checkpoint_dir: "checkpoints"
  model_registry_dir: "model_registry"
  
  # Training parameters
  batch_size: 16
  eval_batch_size: 32
  learning_rate: 2.0e-05
  max_epochs: 3
  
  # Continuous learning parameters
  replay_buffer_size: 1000
  replay_ratio: 0.3
  knowledge_distillation_weight: 0.5
  temperature: 3.0
  
  # Regularization
  ewc_lambda: 1000.0
  l2_reg_weight: 0.01
  
  # Evaluation thresholds
  min_accuracy_threshold: 0.8
  max_forgetting_threshold: 0.05
```

## 🔍 Data Format

### Input Data Format
```csv
product_name,product_category_tree,details
"iPhone 15 Pro","[Electronics >> Mobile Phones >> Smartphones]","Latest iPhone with A17 Pro chip"
"Nike Air Max","[Fashion >> Shoes >> Athletic]","Comfortable running shoes"
```

### Required Columns
- `product_name`: Product title/name
- `product_category_tree`: Category hierarchy in bracket format
- `details` (optional): Additional product description

## 📊 Monitoring and Evaluation

### Training Reports
After each training session, detailed reports are generated in `reports/`:
- Training curves and metrics
- Confusion matrices
- Forgetting analysis
- Model comparison

### Model Versioning
Models are automatically versioned in `model_registry/`:
```
model_registry/
├── bert_v1.0/
├── bert_v1.1/
└── bert_v1.2/
```

### Checkpoints
Training checkpoints are saved in `checkpoints/`:
```
checkpoints/
├── checkpoint_epoch_1_20240825_143022/
├── checkpoint_epoch_2_20240825_143045/
└── checkpoint_epoch_3_20240825_143108/
```

## 🐳 Docker Deployment

### Build and Run
```bash
# Build continuous learning API
docker-compose -f docker-compose.continuous_learning.yml up --build
```

### Production Deployment
```bash
# Use production configuration
docker-compose -f docker-compose.continuous_learning.yml -f docker-compose.prod.yml up
```

## 🔧 Advanced Features

### 1. Automated Training Pipeline
```python
# Watch directory for new data files
pipeline.watch_for_new_data(
    watch_directory="datasets/incoming",
    check_interval=300  # Check every 5 minutes
)
```

### 2. Dynamic Category Expansion
The system automatically handles new product categories:
```python
# New categories are automatically detected and added
# Model architecture is expanded dynamically
# No manual intervention required
```

### 3. Replay Buffer Management
```python
# Get replay buffer statistics
stats = pipeline.data_pipeline.replay_buffer.get_category_distribution()

# Manually add samples to replay buffer
pipeline.data_pipeline.replay_buffer.add_samples(important_samples)
```

### 4. Model Export for Production
```python
# Export optimized model for deployment
pipeline.export_model_for_deployment("production_model/")

# Model includes:
# - Optimized BERT model
# - Tokenizer
# - Label encoder
# - Deployment metadata
```

## 🎯 Integration with Existing Code

Your existing `trainer2.py` has been enhanced to work with continuous learning:

### Enhanced Trainer
```python
from enhanced_trainer import EnhancedTrainer

trainer = EnhancedTrainer()

# Run initial training (your original code)
if not os.path.exists("bert_model"):
    trainer.train_from_scratch()

# Setup continuous learning
trainer.setup_continuous_learning()

# Train incrementally
results = trainer.train_incremental("new_data.csv")
```

### Backward Compatibility
- Your existing model files are automatically backed up
- Original API endpoints continue to work
- Configuration is preserved
- No breaking changes to existing code

## 📈 Performance Optimization

### Memory Management
```python
# Automatic memory monitoring
# Configurable memory thresholds
# Gradient accumulation for large models
# Mixed precision training
```

### GPU Optimization
```python
# Automatic GPU detection and usage
# CUDA memory fraction configuration
# Batch size optimization
# CPU fallback support
```

## 🔍 Troubleshooting

### Common Issues

1. **Memory Issues**
   - Reduce batch size in configuration
   - Enable gradient accumulation
   - Use CPU training for large datasets

2. **Model Loading Errors**
   - Ensure model paths are correct
   - Check file permissions
   - Verify model compatibility

3. **Training Convergence**
   - Adjust learning rate
   - Modify replay buffer size
   - Try different strategies

### Debug Mode
```bash
# Enable verbose logging
python cl_cli.py train new_data.csv --verbose

# Check logs
tail -f logs/continuous_learning_*.log
```

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Check code quality
black continuous_learning/
flake8 continuous_learning/
```

### Adding New Features
1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## 📄 License

This implementation follows the same license as your existing project.

## 🙏 Acknowledgments

- Built on top of HuggingFace Transformers
- Uses FastAPI for REST API
- Inspired by research in continuous learning and knowledge distillation

---

**Ready to start continuous learning? Run `python setup_continuous_learning.py` to get started!** 🚀
