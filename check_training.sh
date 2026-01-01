#!/bin/bash
#
# Quick script to check training status and evaluate models
#

echo "=========================================="
echo "Training Status Check"
echo "=========================================="
echo ""

# Check for checkpoint files
echo "Checkpoint files:"
find checkpoints -name "*.pt" -o -name "*.json" 2>/dev/null | sort

if [ ! -f "checkpoints/best_model.pt" ]; then
    echo ""
    echo "⚠️  No best_model.pt found!"
    echo ""
    echo "Possible reasons:"
    echo "1. Training is still running"
    echo "2. Training crashed before saving"
    echo "3. Model never improved (accuracy stayed at 0)"
    echo ""
    echo "To fix:"
    echo "  git pull  # Get latest code that saves based on loss"
    echo "  # Then restart training"
fi

echo ""
echo "Training logs:"
ls -lh checkpoints/*/training_results.json 2>/dev/null || echo "No training logs found"

echo ""
echo "To evaluate (once model exists):"
echo "  python evaluate_detailed.py --checkpoint checkpoints/best_model.pt"
echo ""
