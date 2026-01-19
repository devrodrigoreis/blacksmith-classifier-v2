# Blacksmith Classifier

This project implements a BERT-based product classifier organized into a modular Python package. It supports standard training using Hugging Face Transformers and a custom fallback training loop.

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.8+
- CUDA-capable GPU (Recommended for training)
- 16GB+ RAM (32GB+ recommended for large datasets)

### 2. Environment Setup
Clone the repository and install dependencies:

```bash
# Create a virtual environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (Linux/Mac)
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

> **Note**: If `requirements.txt` is missing, ensure you have: `torch`, `transformers`, `scikit-learn`, `imbalanced-learn`, `pandas`, `numpy`, `pyyaml`.

## ⚙️ Configuration

The project is configured via `config.yaml`. 

**Critical Hardware Settings**:
The default configuration is optimized for an **NVIDIA RTX 4060 Ti (16GB)**. If running on different hardware, you **MUST** adjust `config.yaml`:

- **VRAM < 12GB**: Reduce `per_device_train_batch_size` to `8` or `16`. Reduce `max_length` to `128` or `64`.
- **Non-NVIDIA 40-series**: Set `bf16: False` and `fp16: True` (standard mixed precision).
- **CPU Only**: Training will be extremely slow. Set `use_cpu: True` (if supported) or ensure CUDA logic handles it implies CPU.

```yaml
training:
  per_device_train_batch_size: 24  # Adjust based on VRAM
  bf16: True                       # Set False for older GPUs (GTX 10xx, RTX 20xx/30xx)
  fp16: False                      # Set True if disabling bf16
```

## 🏃 Usage

The project uses a unified CLI `cli.py` for all operations.

### Standard Training (Recommended)
Uses the Hugging Face Trainer with extended context and optimizations.

```bash
python cli.py train --mode standard
```

### Fallback Training
Uses a custom PyTorch training loop (legacy/debugging).

```bash
python cli.py train --mode fallback
```

## 📁 Project Structure

```
.
├── config.yaml             # Main configuration file
├── cli.py                  # CLI Entry point
├── requirements.txt        # Python dependencies
├── src/
│   └── classifier/
│       ├── config.py       # Config loader
│       ├── data/           # Dataset & Preprocessing logic
│       ├── models/         # Model definitions
│       ├── training/       # Training loops (standard & fallback)
│       └── utils/          # Logging, memory, text utils
├── data/                   # Input CSVs (Required: products.csv or split files)
└── logs/                   # Training logs
```

## ❓ Troubleshooting

- **OOM (Out of Memory)**: Reduce `per_device_train_batch_size` in `config.yaml`.
- **ImportError: No module named 'src'**: Run the script from the root directory: `python cli.py ...`
- **CUDA errors**: Ensure PyTorch matches your CUDA version (check `torch.cuda.is_available()`).
