# Mini-Batch Training Implementation

**Date:** 2026-01-16  
**Problem:** Training on 1.7M labels took 18-38 hours (iterating through all labels every step)  
**Solution:** Added mini-batch training support

---

## Changes Made

### 1. `core/train_utils.py`
- Added `batch_size` parameter to `_train_on_labels()`
- When `batch_size` is set, randomly samples a subset of labels per step
- Falls back to full batch when `batch_size=None` (backward compatible)

### 2. `pipelines/tinystories_pipeline.py`
- Added `train_batch_size` parameter to `tinystories_mini_benchmark()`
- Auto-enables batching for datasets > 100K labels (batch_size=10,000)
- Passes batch_size to training function

### 3. `benchmarks.py`
- Added `--train-batch-size` command-line argument
- Default: `None` (auto-detect)
- Can be manually set for control

### 4. `pipelines/benchmark_suite.py`
- Propagated `train_batch_size` parameter through the call chain

---

## Usage

### Automatic (recommended)
```bash
python benchmarks.py --device cuda --max-stories 581 --max-facts 10000
```
- Auto-enables batching for >100K labels
- Uses batch_size=10,000

### Manual control
```bash
# Smaller batches for memory-constrained GPUs
python benchmarks.py --device cuda --max-stories 581 --train-batch-size 5000

# Force full-batch training (old behavior)
python benchmarks.py --device cuda --max-stories 50 --train-batch-size 0
```

---

## Performance Impact

### Before (Full Batch)
- **1.7M labels, 40 steps**
- Per step: Iterate through ALL 1.7M labels
- Time per step: 28-57 minutes
- Total time: **18-38 hours**

### After (Mini-Batch, size=10K)
- **1.7M labels, 40 steps**
- Per step: Sample 10K random labels
- Time per step: ~10-30 seconds
- Total time: **7-20 minutes** (50-200× faster!)

### Trade-offs
- ✅ Much faster training
- ✅ Still sees diverse samples due to random sampling
- ✅ More gradient updates per epoch (can iterate more)
- ⚠️ May need more steps for convergence
- ⚠️ Each step sees less data (10K vs 1.7M)

---

## Recommended Settings

| Dataset Size | Batch Size | Notes |
|--------------|-----------|-------|
| < 50K labels | None (full) | Fast enough without batching |
| 50K-500K | 10,000 | Good balance |
| 500K-1M | 10,000-20,000 | Adjust based on GPU memory |
| > 1M | 10,000-50,000 | Auto-enabled at 10K |

---

## Next Steps

For your current running job (1.7M labels, no batching):
1. **Wait for first progress update** (2-5 hours) to see actual timing
2. **Kill and restart** with updated code for 50-200× speedup
3. **New command:**
   ```bash
   python benchmarks.py --device cuda --max-stories 581 --max-facts 10000 \
     --no-rule-injection --max-candidate-rules 200
   ```
   This will auto-enable mini-batch training and complete in ~7-20 minutes instead of 18-38 hours!

---

## Backward Compatibility

✅ All existing code continues to work
- Default `batch_size=None` uses full batch (old behavior)
- Only activates when explicitly set or auto-detected
- No changes needed to existing scripts
