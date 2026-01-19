# Continuous Learning for BERT Product Classification

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
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Samsung Galaxy S23"]}'
```

#### Incremental Training
```bash
curl -X POST "http://localhost:8000/train/incremental" \
     -F "file=@new_products.csv" \
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
