#!/bin/bash
#
# All-in-one training starter
# Run this after setup completes
#

echo "=========================================="
echo "Training Options"
echo "=========================================="
echo ""
echo "Select training configuration:"
echo ""
echo "1. Quick test (1K stories, 10 epochs, ~5-10 minutes)"
echo "2. Medium run (10K stories, 50 epochs, ~1-2 hours)"
echo "3. Large run (50K stories, 100 epochs, ~4-6 hours)"
echo "4. Custom configuration"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "Starting quick test..."
        python train_symmetric.py \
            --num_stories 1000 \
            --epochs 10 \
            --batch_size 32 \
            --hidden_dim 64 \
            --num_rules 8 \
            --output_dir checkpoints/quick_test
        ;;
    2)
        echo ""
        echo "Starting medium run..."
        python train_symmetric.py \
            --num_stories 10000 \
            --epochs 50 \
            --batch_size 64 \
            --hidden_dim 128 \
            --num_rules 16 \
            --output_dir checkpoints/medium_run
        ;;
    3)
        echo ""
        echo "Starting large run..."
        python train_symmetric.py \
            --num_stories 50000 \
            --epochs 100 \
            --batch_size 128 \
            --hidden_dim 256 \
            --num_rules 32 \
            --output_dir checkpoints/large_run
        ;;
    4)
        echo ""
        read -p "Number of stories [default: 10000]: " stories
        stories=${stories:-10000}
        
        read -p "Number of epochs [default: 50]: " epochs
        epochs=${epochs:-50}
        
        read -p "Batch size [default: 64]: " batch_size
        batch_size=${batch_size:-64}
        
        read -p "Hidden dim [default: 128]: " hidden_dim
        hidden_dim=${hidden_dim:-128}
        
        read -p "Number of rules [default: 16]: " num_rules
        num_rules=${num_rules:-16}
        
        echo ""
        echo "Starting custom training..."
        python train_symmetric.py \
            --num_stories $stories \
            --epochs $epochs \
            --batch_size $batch_size \
            --hidden_dim $hidden_dim \
            --num_rules $num_rules \
            --output_dir checkpoints/custom_run
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Results saved to checkpoints/"
echo ""
echo "To evaluate:"
echo "  python evaluate_symmetric.py --checkpoint checkpoints/*/best_model.pt"
echo ""
