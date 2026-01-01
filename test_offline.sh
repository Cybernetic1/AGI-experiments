#!/bin/bash
#
# Offline test - works without internet
# Uses synthetic data to verify training pipeline
#

echo "=========================================="
echo "Offline Training Test (No Internet Needed)"
echo "=========================================="
echo ""
echo "This test:"
echo "  - Uses synthetic data (no downloads)"
echo "  - Verifies GPU works"
echo "  - Tests full training pipeline"
echo "  - Takes ~2-3 minutes"
echo ""

# Make sure venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run quick test with synthetic data
echo "Starting training with synthetic data..."
echo ""

python train_symmetric.py \
    --num_stories 100 \
    --epochs 5 \
    --batch_size 8 \
    --hidden_dim 32 \
    --num_rules 4 \
    --output_dir checkpoints/offline_test

echo ""
echo "=========================================="
echo "Offline Test Complete!"
echo "=========================================="
echo ""
echo "If this worked, your setup is correct!"
echo ""
echo "Next steps:"
echo "  1. Fix network issues to download real data"
echo "  2. Or use pre-downloaded dataset"
echo "  3. Then run full training"
echo ""
