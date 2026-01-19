"""
Example script showing how to integrate continuous learning with your existing BERT trainer
"""
import os
import sys
import logging
import pandas as pd
from datetime import datetime

# Add the continuous learning module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from continuous_learning.pipeline import ContinuousLearningPipeline
from continuous_learning.config import ContinuousLearningConfig

def setup_continuous_learning_from_existing_model():
    """Setup continuous learning using your existing trained model"""
    
    # Create configuration
    config = ContinuousLearningConfig(
        base_model_path="bert_model",  # Your existing model path
        teacher_model_path="teacher_model",
        checkpoint_dir="checkpoints",
        model_registry_dir="model_registry",
        replay_buffer_path="replay_buffer.parquet",
        incremental_data_dir="datasets/incremental",
        
        # Training parameters
        batch_size=16,
        eval_batch_size=32,
        learning_rate=2e-5,
        max_epochs=3,
        
        # Continuous learning parameters
        replay_buffer_size=1000,
        replay_ratio=0.3,
        knowledge_distillation_weight=0.5,
        temperature=3.0,
        
        # Thresholds
        min_accuracy_threshold=0.80,
        max_forgetting_threshold=0.05
    )
    
    # Save configuration
    config.save_to_yaml("continuous_learning_config.yaml")
    print("Continuous learning configuration created")
    
    return config

def migrate_existing_model():
    """Migrate your existing model to continuous learning format"""
    
    # Initialize pipeline
    pipeline = ContinuousLearningPipeline("continuous_learning_config.yaml")
    
    try:
        pipeline.initialize()
        print("Pipeline initialized successfully")
        
        # Get model info
        info = pipeline.get_model_info()
        print(f"Model loaded with {info['num_labels']} categories")
        
        return pipeline
        
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return None

def example_incremental_training():
    """Example of how to perform incremental training"""
    
    # Setup
    config = setup_continuous_learning_from_existing_model()
    pipeline = migrate_existing_model()
    
    if pipeline is None:
        print("Failed to setup pipeline")
        return None
    
    # Create some example new data
    new_data = pd.DataFrame({
        'product_name': [
            'iPhone 13 Pro Max 256GB',
            'Samsung Galaxy S22 Ultra',
            'MacBook Pro M2 14 inch',
            'Dell XPS 13 Laptop',
            'Sony WH-1000XM4 Headphones'
        ],
        'product_category_tree': [
            '[Electronics >> Mobile Phones >> Smartphones]',
            '[Electronics >> Mobile Phones >> Smartphones]', 
            '[Electronics >> Computers >> Laptops]',
            '[Electronics >> Computers >> Laptops]',
            '[Electronics >> Audio >> Headphones]'
        ],
        'details': [
            'Latest iPhone with A15 Bionic chip, 256GB storage, Pro camera system',
            'Android smartphone with S Pen, 108MP camera, 5G connectivity',
            'Professional laptop with M2 chip, 14-inch Liquid Retina XDR display',
            'Premium ultrabook with 11th Gen Intel Core processor',
            'Wireless noise cancelling headphones with 30-hour battery life'
        ]
    })
    
    # Save example data
    os.makedirs("datasets/incremental", exist_ok=True)
    new_data_path = "datasets/incremental/example_new_products.csv"
    new_data.to_csv(new_data_path, index=False)
    print(f"Example data saved to {new_data_path}")
    
    # Perform incremental training
    print("\nStarting incremental training...")
    results = pipeline.train_incremental(
        new_data_path=new_data_path,
        strategy="knowledge_distillation",  # Use knowledge distillation
        old_validation_data_path="data/products_eval_unique.csv"  # Your existing validation data
    )
    
    # Print results
    print("\nTraining Results:")
    print(f"  Training Successful: {results.get('training_successful', False)}")
    print(f"  Update Accepted: {results.get('accept_update', False)}")
    print(f"  Final Accuracy: {results['final_metrics']['accuracy']:.4f}")
    print(f"  Final F1: {results['final_metrics']['f1']:.4f}")
    
    if results.get('forgetting_metrics'):
        print(f"  Old Data Accuracy: {results['forgetting_metrics']['accuracy']:.4f}")
    
    if results.get('model_version_path'):
        print(f"  New Model Version: {results['model_version_path']}")
    
    return pipeline, results

def example_predictions():
    """Example of making predictions with the updated model"""
    
    pipeline = ContinuousLearningPipeline("continuous_learning_config.yaml")
    pipeline.initialize()
    
    # Example texts for prediction
    test_texts = [
        "Apple Watch Series 8 GPS 45mm",
        "Nike Air Force 1 White Sneakers",
        "The Great Gatsby Novel Book",
        "Car Engine Oil 5W-30 Synthetic",
        "Organic Tomato Seeds for Garden"
    ]
    
    print("Making predictions...")
    predictions = pipeline.predict(test_texts)
    
    for pred in predictions:
        print(f"\nText: {pred['text']}")
        print(f"Predicted Category: {pred['predicted_label']}")
        print(f"Confidence: {pred['confidence']:.4f}")
        print("Top 3 Predictions:")
        for i, top_pred in enumerate(pred['top_predictions'], 1):
            print(f"  {i}. {top_pred['label']} ({top_pred['confidence']:.4f})")

def example_automated_workflow():
    """Example of automated continuous learning workflow"""
    
    pipeline = ContinuousLearningPipeline("continuous_learning_config.yaml")
    pipeline.initialize()
    
    # Setup directories
    watch_dir = "datasets/incoming"
    os.makedirs(watch_dir, exist_ok=True)
    
    print(f"Setting up automated workflow...")
    print(f"Place new data files in: {watch_dir}")
    print(f"The system will automatically process them")
    
    # In a real scenario, you would run this:
    # pipeline.watch_for_new_data(watch_dir, check_interval=300)
    
    print("Note: Call pipeline.watch_for_new_data() to start monitoring")

def compare_strategies():
    """Compare different continuous learning strategies"""
    
    pipeline = ContinuousLearningPipeline("continuous_learning_config.yaml")
    pipeline.initialize()
    
    # Prepare test data
    test_data_path = "datasets/incremental/example_new_products.csv"
    strategies = ["replay", "knowledge_distillation", "hybrid"]
    
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        
        try:
            result = pipeline.train_incremental(
                new_data_path=test_data_path,
                strategy=strategy,
                old_validation_data_path="data/products_eval_unique.csv"
            )
            
            results[strategy] = {
                'accuracy': result['final_metrics']['accuracy'],
                'f1': result['final_metrics']['f1'],
                'forgetting': result.get('forgetting_metrics', {}).get('accuracy', 0),
                'accepted': result.get('accept_update', False)
            }
            
        except Exception as e:
            print(f"  Error with {strategy}: {e}")
            results[strategy] = {'error': str(e)}
    
    # Print comparison
    print("\nStrategy Comparison:")
    print("Strategy".ljust(20) + "Accuracy".ljust(12) + "F1".ljust(12) + "Old Acc".ljust(12) + "Accepted")
    print("-" * 70)
    
    for strategy, result in results.items():
        if 'error' not in result:
            print(f"{strategy.ljust(20)}{result['accuracy']:.4f}".ljust(12) + 
                  f"{result['f1']:.4f}".ljust(12) + 
                  f"{result['forgetting']:.4f}".ljust(12) + 
                  f"{result['accepted']}")
        else:
            print(f"{strategy.ljust(20)}ERROR: {result['error']}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("Continuous Learning Integration Example")
    print("=" * 50)
    
    # Example 1: Setup and basic training
    print("\n1. Setting up continuous learning...")
    result = example_incremental_training()
    if result is None:
        print("Failed to setup continuous learning")
        return
    pipeline, results = result
    
    # Example 2: Making predictions
    print("\n2. Making predictions with updated model...")
    example_predictions()
    
    # Example 3: Show model info
    print("\n3. Model information...")
    info = pipeline.get_model_info()
    print(f"Categories: {len(info['categories'])}")
    print(f"Replay buffer size: {info['replay_buffer_size']}")
    
    # Example 4: Export model
    print("\n4. Exporting model for deployment...")
    export_path = "deployed_model"
    pipeline.export_model_for_deployment(export_path)
    print(f"Model exported to: {export_path}")
    
    print("\nContinuous learning integration complete!")
    print("\nNext steps:")
    print("1. Integrate with your API using continuous_learning/api.py")
    print("2. Use the CLI tool: python -m continuous_learning.cli")
    print("3. Set up automated monitoring with watch_for_new_data()")
    print("4. Customize the configuration for your specific needs")
