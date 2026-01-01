# URGENT: Loss Exploding - Stop Training

## Problem

Loss increasing from 7.4M → 11.1M between epochs 2-11.
This means **gradient explosion** - training is unstable.

## Immediate Action

**STOP the current training run** (Ctrl+C)

The model is diverging and won't recover. We need to fix the training parameters.

## Root Causes

1. **Learning rate too high** (0.001 is too aggressive for this model)
2. **No gradient clipping** (gradients can explode)
3. **Batch size might be too large** (less stable updates)

## Quick Fix

Run with these safer parameters:

```bash
python train_symmetric.py \
    --num_stories 1000 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0001 \
    --hidden_dim 128 \
    --num_rules 16
```

**Key changes:**
- `--lr 0.0001` instead of 0.001 (10x smaller)
- `--batch_size 32` instead of 128 (more stable)

## Better Fix (Add Gradient Clipping)

I need to modify `train_symmetric.py` to add:

```python
# After loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

This prevents gradients from exploding.

## What to Watch

**Healthy training:**
- Loss decreases monotonically (or mostly)
- Gradual improvement
- No sudden jumps

**Unhealthy (like now):**
- Loss increases
- Sudden jumps (7M → 11M)
- Will eventually hit NaN

## Expected Behavior

With fixed parameters:
- Epoch 1: Loss ~450K
- Epoch 10: Loss ~350K
- Epoch 50: Loss ~250K

All should be **decreasing**.
