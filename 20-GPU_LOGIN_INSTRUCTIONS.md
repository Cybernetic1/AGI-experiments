# GPU Server Login & Setup Instructions

## Step 1: Log into GPU Server

```bash
ssh -p 23163 root@connect.cqa1.seetacloud.com
# Password: XXXXXXXXX
```

## Step 2: Navigate to Repository

```bash
cd AGI-experiments/
ls -la  # Verify files are there
```

## Step 3: Run Setup Script

```bash
# Make sure script is executable
chmod +x setup_and_test.sh

# Run setup
./setup_and_test.sh
```

This will:
- ✓ Check Python version
- ✓ Check GPU availability
- ✓ Create virtual environment
- ✓ Install PyTorch with CUDA
- ✓ Install all dependencies
- ✓ Download spaCy model
- ✓ Test all core modules
- ✓ Verify GPU is working

**Expected time:** 5-10 minutes

## Step 4: Start Training

After setup completes, activate the environment:

```bash
source venv/bin/activate
```

### Quick Test (5-10 minutes):
```bash
python train_symmetric.py \
    --num_stories 1000 \
    --epochs 10 \
    --batch_size 32 \
    --hidden_dim 64 \
    --num_rules 8
```

### Medium Run (1-2 hours):
```bash
python train_symmetric.py \
    --num_stories 10000 \
    --epochs 50 \
    --batch_size 64 \
    --hidden_dim 128 \
    --num_rules 16
```

### Full Run (4-6 hours):
```bash
python train_symmetric.py \
    --num_stories 50000 \
    --epochs 100 \
    --batch_size 128 \
    --hidden_dim 256 \
    --num_rules 32
```

## Step 5: Monitor Training

Training will show progress bars and metrics in real-time:

```
Epoch 1/50
----------------------------------------------------------------------
Training: 100%|████████| 157/157 [00:15<00:00, 10.2it/s, loss=280K, parse=185K, gen=5.1]
Train - Loss: 280648.15, Parse: 185730.10, Gen: 5.11, Cycle: 189825.89
Val - Parse Accuracy: 0.0123
✓ Saved best model (acc: 0.0123)

Epoch 2/50
...
```

You can also save output to log file:

```bash
python train_symmetric.py --num_stories 10000 --epochs 50 2>&1 | tee training.log
```

## Step 6: Check Results

After training completes:

```bash
# View training metrics
cat checkpoints/training_results.json | python -m json.tool

# List saved checkpoints
ls -lh checkpoints/

# Run evaluation
python evaluate_symmetric.py \
    --checkpoint checkpoints/best_model.pt \
    --num_stories 100
```

## Troubleshooting

### If setup fails:

**Check Python:**
```bash
python3 --version  # Should be 3.8+
```

**Check GPU:**
```bash
nvidia-smi  # Should show GPU info
```

**Check CUDA:**
```bash
nvcc --version  # Should show CUDA version
```

### If training is slow:

**Check GPU usage:**
```bash
watch -n 1 nvidia-smi
# Should show ~90-100% GPU utilization
```

**Reduce batch size if OOM:**
```bash
python train_symmetric.py --batch_size 16  # Instead of 64
```

### If imports fail:

```bash
# Make sure venv is activated
source venv/bin/activate

# Reinstall if needed
pip install torch spacy datasets tqdm
```

## Expected GPU Performance

On typical GPU (RTX 3090 / A100):

| Stories | Epochs | Batch Size | Time | GPU Memory |
|---------|--------|------------|------|------------|
| 1K | 10 | 32 | 5-10 min | ~2GB |
| 10K | 50 | 64 | 1-2 hours | ~4GB |
| 50K | 100 | 128 | 4-6 hours | ~8GB |

**Throughput:** ~500-1000 sentences/second (vs. ~50/sec on CPU)

## Success Indicators

Training is working well if:

1. ✓ GPU utilization > 80% (check with `nvidia-smi`)
2. ✓ Losses decreasing each epoch
3. ✓ No crashes or OOM errors
4. ✓ Checkpoints being saved
5. ✓ Training speed ~10-20 batches/second

## Next Steps After Training

1. **Analyze results** - Check training curves, accuracy
2. **Run evaluation** - Test parsing and generation quality
3. **Scale up** - Try larger datasets (50K → 100K → 1M)
4. **Integrate LTM** - Add entity registry for knowledge persistence
5. **Try other datasets** - Wikipedia, BookCorpus, etc.

---

**Ready to go! 🚀**

Just log in, run the setup script, and start training!
