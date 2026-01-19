"""
Command line interface for continuous learning pipeline
"""
import argparse
import logging
import sys
import os
from pathlib import Path

from .pipeline import ContinuousLearningPipeline, create_continuous_learning_config
from .config import ContinuousLearningConfig

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def cmd_init(args):
    """Initialize continuous learning configuration"""
    config = create_continuous_learning_config()
    print(f"Configuration initialized: {config}")

def cmd_train(args):
    """Run incremental training"""
    pipeline = ContinuousLearningPipeline(args.config)
    pipeline.initialize()
    
    results = pipeline.train_incremental(
        new_data_path=args.data,
        strategy=args.strategy,
        old_validation_data_path=args.old_validation_data
    )
    
    print("Training Results:")
    print(f"  Success: {results.get('training_successful', False)}")
    print(f"  Final Accuracy: {results['final_metrics']['accuracy']:.4f}")
    print(f"  Final F1: {results['final_metrics']['f1']:.4f}")
    
    if results.get('forgetting_metrics'):
        print(f"  Old Data Accuracy: {results['forgetting_metrics']['accuracy']:.4f}")
    
    if results.get('accept_update'):
        print(f"  Model Update: Accepted")
        if 'model_version_path' in results:
            print(f"  Model Version: {results['model_version_path']}")
    else:
        print(f"  Model Update: Rejected")
        if 'reasons' in results:
            print(f"  Reasons: {', '.join(results['reasons'])}")

def cmd_predict(args):
    """Make predictions on new data"""
    pipeline = ContinuousLearningPipeline(args.config)
    pipeline.initialize()
    
    if args.text:
        # Single text prediction
        predictions = pipeline.predict([args.text])
        result = predictions[0]
        
        print("Prediction Result:")
        print(f"  Text: {result['text']}")
        print(f"  Predicted Category: {result['predicted_label']}")
        print(f"  Confidence: {result['confidence']:.4f}")
        print("  Top Predictions:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"    {i}. {pred['label']} ({pred['confidence']:.4f})")
    
    elif args.file:
        # Batch prediction from file
        import pandas as pd
        
        if args.file.endswith('.csv'):
            data = pd.read_csv(args.file)
        elif args.file.endswith('.parquet'):
            data = pd.read_parquet(args.file)
        else:
            print("Error: Unsupported file format. Use CSV or Parquet.")
            return
        
        # Find text column
        text_column = None
        for col in ['product_name', 'input_text', 'text', 'description']:
            if col in data.columns:
                text_column = col
                break
        
        if text_column is None:
            print("Error: No recognized text column found.")
            return
        
        texts = data[text_column].astype(str).tolist()
        predictions = pipeline.predict(texts, batch_size=args.batch_size)
        
        # Add predictions to dataframe
        data['predicted_category'] = [p['predicted_label'] for p in predictions]
        data['confidence'] = [p['confidence'] for p in predictions]
        
        # Save results
        output_file = args.output or args.file.replace('.csv', '_predictions.csv').replace('.parquet', '_predictions.csv')
        data.to_csv(output_file, index=False)
        print(f"Predictions saved to: {output_file}")

def cmd_info(args):
    """Show model information"""
    pipeline = ContinuousLearningPipeline(args.config)
    pipeline.initialize()
    
    info = pipeline.get_model_info()
    
    print("Model Information:")
    print(f"  Number of Categories: {info['num_labels']}")
    print(f"  Total Parameters: {info['model_parameters']:,}")
    print(f"  Trainable Parameters: {info['trainable_parameters']:,}")
    print(f"  Replay Buffer Size: {info['replay_buffer_size']}")
    
    print("\n  Categories:")
    for i, category in enumerate(info['categories'], 1):
        print(f"    {i}. {category}")
    
    print("\n  Replay Buffer Distribution:")
    for category, count in info['replay_buffer_categories'].items():
        print(f"    {category}: {count}")

def cmd_export(args):
    """Export model for deployment"""
    pipeline = ContinuousLearningPipeline(args.config)
    pipeline.initialize()
    
    pipeline.export_model_for_deployment(args.output)
    print(f"Model exported to: {args.output}")

def cmd_watch(args):
    """Watch directory for new data and auto-train"""
    pipeline = ContinuousLearningPipeline(args.config)
    pipeline.initialize()
    
    print(f"Watching directory: {args.directory}")
    print(f"Check interval: {args.interval} seconds")
    print("Press Ctrl+C to stop...")
    
    pipeline.watch_for_new_data(args.directory, args.interval)

def cmd_cleanup(args):
    """Clean up old checkpoints and files"""
    pipeline = ContinuousLearningPipeline(args.config)
    pipeline.initialize()
    
    pipeline.cleanup_old_checkpoints(args.keep)
    print(f"Cleanup completed. Kept last {args.keep} checkpoints.")

def cmd_api(args):
    """Start the API server"""
    import uvicorn
    from .api import app
    
    print(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Continuous Learning Pipeline for BERT Product Classification"
    )
    
    parser.add_argument(
        '--config', 
        default='config.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Init command
    parser_init = subparsers.add_parser('init', help='Initialize continuous learning configuration')
    parser_init.set_defaults(func=cmd_init)
    
    # Train command
    parser_train = subparsers.add_parser('train', help='Run incremental training')
    parser_train.add_argument('data', help='Path to new training data')
    parser_train.add_argument(
        '--strategy', 
        choices=['replay', 'knowledge_distillation', 'ewc', 'hybrid'],
        default='knowledge_distillation',
        help='Training strategy'
    )
    parser_train.add_argument(
        '--old-validation-data',
        help='Path to old validation data for forgetting check'
    )
    parser_train.set_defaults(func=cmd_train)
    
    # Predict command
    parser_predict = subparsers.add_parser('predict', help='Make predictions')
    predict_group = parser_predict.add_mutually_exclusive_group(required=True)
    predict_group.add_argument('--text', help='Single text to predict')
    predict_group.add_argument('--file', help='File with texts to predict')
    parser_predict.add_argument('--output', help='Output file for batch predictions')
    parser_predict.add_argument('--batch-size', type=int, default=32, help='Batch size for processing')
    parser_predict.set_defaults(func=cmd_predict)
    
    # Info command
    parser_info = subparsers.add_parser('info', help='Show model information')
    parser_info.set_defaults(func=cmd_info)
    
    # Export command
    parser_export = subparsers.add_parser('export', help='Export model for deployment')
    parser_export.add_argument('output', help='Output directory for exported model')
    parser_export.set_defaults(func=cmd_export)
    
    # Watch command
    parser_watch = subparsers.add_parser('watch', help='Watch directory for new data')
    parser_watch.add_argument('directory', help='Directory to watch')
    parser_watch.add_argument('--interval', type=int, default=300, help='Check interval in seconds')
    parser_watch.set_defaults(func=cmd_watch)
    
    # Cleanup command
    parser_cleanup = subparsers.add_parser('cleanup', help='Clean up old files')
    parser_cleanup.add_argument('--keep', type=int, default=5, help='Number of checkpoints to keep')
    parser_cleanup.set_defaults(func=cmd_cleanup)
    
    # API command
    parser_api = subparsers.add_parser('api', help='Start API server')
    parser_api.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser_api.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser_api.add_argument('--reload', action='store_true', help='Enable auto-reload')
    parser_api.set_defaults(func=cmd_api)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    setup_logging(args.verbose)
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
