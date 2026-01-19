
import argparse
import sys
import os

# Add local directory to path to ensure src can be imported
sys.path.append(os.getcwd())

try:
    from src.classifier.training import basic
    from src.classifier.training import fallback
    from src.classifier.utils.system import setup_logging
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Blacksmith Classifier Training CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--mode", 
        choices=["standard", "fallback"], 
        default="standard", 
        help="Training mode (default: standard)"
    )

    args = parser.parse_args()
    
    setup_logging()

    if args.command == "train":
        if args.mode == "standard":
            print("Starting Standard Training...")
            basic.train()
        elif args.mode == "fallback":
            print("Starting Fallback Training...")
            fallback.train()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
