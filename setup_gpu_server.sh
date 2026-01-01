#!/bin/bash
#
# Setup script for GPU server
# Run this after transferring files to GPU server
#

set -e

echo "=========================================="
echo "Setting up Symmetric Logic Network on GPU"
echo "=========================================="

# Check Python version
echo ""
echo "Checking Python version..."
python3 --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo ""
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo ""
echo "Installing dependencies..."
pip install \
    spacy \
    datasets \
    tqdm \
    numpy \
    matplotlib \
    tensorboard

# Download spaCy model
echo ""
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Verify GPU availability
echo ""
echo "Verifying GPU..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Test core modules
echo ""
echo "Testing core modules..."
python implicit_graph_wm.py
python reversible_logic_rules.py
python symmetric_logic_network.py

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To start training:"
echo "  source venv/bin/activate"
echo "  python train_symmetric.py --num_stories 10000 --epochs 50 --batch_size 64"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir runs/"
echo ""
