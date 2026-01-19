
import logging
import sys
import torch
import os

def setup_logging(level=logging.INFO, log_file=None):
    """
    Sets up logging configuration.
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
        
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )

def get_device():
    """
    Returns the appropriate torch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def check_gpu():
    if torch.cuda.is_available():
        logging.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        logging.warning("CUDA not available, using CPU")
        return False
