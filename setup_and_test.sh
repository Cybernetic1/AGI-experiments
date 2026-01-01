#!/bin/bash
#
# GPU Server Quick Setup & Test Script
# Run this after logging into the GPU server
#

set -e

echo "=========================================="
echo "GPU Server Setup - AGI Experiments"
echo "=========================================="

# Check location
echo ""
echo "Current directory:"
pwd
echo ""
echo "Contents:"
ls -la

# Check if repo exists
if [ -d "AGI-experiments" ]; then
    echo ""
    echo "✓ AGI-experiments directory found"
    cd AGI-experiments
else
    echo ""
    echo "✗ AGI-experiments not found. Please run:"
    echo "  git clone <repository-url> AGI-experiments"
    exit 1
fi

# Check Python
echo ""
echo "Python version:"
python3 --version || python --version

# Check GPU
echo ""
echo "Checking GPU..."
nvidia-smi || echo "⚠ nvidia-smi not found (GPU might not be available)"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv || python -m venv venv

# Activate venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA
echo ""
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo ""
echo "Installing dependencies..."
pip install spacy datasets tqdm numpy

# Download spaCy model
echo ""
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Verify GPU in PyTorch
echo ""
echo "Verifying PyTorch GPU..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Test core modules
echo ""
echo "Testing core modules..."
python implicit_graph_wm.py
echo "✓ implicit_graph_wm.py works"

python reversible_logic_rules.py
echo "✓ reversible_logic_rules.py works"

python symmetric_logic_network.py
echo "✓ symmetric_logic_network.py works"

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Ready to train! Run:"
echo ""
echo "  # Quick test (5-10 minutes):"
echo "  python train_symmetric.py --num_stories 1000 --epochs 10 --batch_size 32"
echo ""
echo "  # Medium run (1-2 hours):"
echo "  python train_symmetric.py --num_stories 10000 --epochs 50 --batch_size 64"
echo ""
echo "  # Full run (several hours):"
echo "  python train_symmetric.py --num_stories 50000 --epochs 100 --batch_size 128"
echo ""
