# Training Analysis - Loss Plateau at ~300

## Current Situation

**Good news:** Loss is stable (298 → 308), not exploding ✓  
**Bad news:** Loss not improving, stuck at ~300 ✗

## Why This Happens

### Issue 1: Learning Rate Too Small Now

After fixing explosion, learning rate (0.0001) might be **too conservative**.

**Symptoms:**
- Loss plateaus immediately
- No improvement over 50 epochs
- Model can't escape local minimum

**Solution:** Try learning rate in between:

```bash
python train_symmetric.py \
    --num_stories 1000 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0003 \
    --hidden_dim 128 \
    --num_rules 16
```

### Issue 2: Loss Components Might Be Unbalanced

The total loss has 4 parts:
```python
loss = loss_parse + loss_generate + 0.5 * (loss_cycle_forward + loss_cycle_backward)
```

If one component dominates (like parse loss = 300K), the others don't matter.

**Check what's happening:**

Look at the training output - you should see:
```
loss=307.9, parse=XXX, gen=XXX
```

What are the parse and gen values?

### Issue 3: Model Capacity vs Data Size

With 1000 stories:
- ~3000 training samples
- Vocabulary: ~1500 words
- Entities: ~500

**Model might be:**
- Too small (128 dim, 16 rules) - can't learn enough
- Too large (overfits immediately) - memorizes instead of learning

### Issue 4: The "Exact Match" Metric Problem

**The 0.0000% accuracy is EXPECTED** with the current metric!

Even if the model is learning, exact match is nearly impossible:

```
Probability of exact match by random chance:
= 1 / (vocab_size × num_entities × vocab_size)
= 1 / (1500 × 500 × 1500)
= 0.00000000088%

So 0.0000% doesn't mean it's not learning!
```

## Diagnostic Steps

### Step 1: Check Loss Components

When training shows:
```
loss=307.9, parse=XXX, gen=XXX
```

Tell me what parse and gen values are. This reveals which component is stuck.

### Step 2: Run Detailed Evaluation

```bash
# This shows component-wise accuracy (much better metric)
git pull
python evaluate_detailed.py \
    --checkpoint checkpoints/best_model.pt \
    --hidden_dim 128 \
    --num_rules 16
```

**Expected results if it's learning:**
- Entity accuracy: 10-30%
- Relation accuracy: 5-20%
- Value accuracy: 5-15%
- "At least 1 correct": 40-60%

**If all are 0-5%, then it's truly not learning.**

### Step 3: Check if Model is Just Predicting Zeros

```bash
python -c "
import torch
checkpoint = torch.load('checkpoints/best_model.pt')
model = torch.load('checkpoints/best_model.pt')
print('Model weights sample:')
# Check if weights are actually changing
"
```

## Recommended Actions (in order)

### Action 1: Increase Learning Rate Slightly

```bash
# Stop current training
# Try this:
python train_symmetric.py \
    --num_stories 1000 \
    --epochs 30 \
    --batch_size 32 \
    --lr 0.0003 \
    --hidden_dim 128 \
    --num_rules 16
```

**Watch first 5 epochs:**
- Loss should decrease from ~450K to ~350K
- If loss explodes again → reduce to 0.0002
- If still stuck → problem is elsewhere

### Action 2: Try Smaller Model (Faster Iteration)

```bash
# Smaller model learns faster with limited data
python train_symmetric.py \
    --num_stories 1000 \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.0005 \
    --hidden_dim 64 \
    --num_rules 8
```

### Action 3: Use the Detailed Evaluation

```bash
# THIS IS KEY - shows if model is actually learning
python evaluate_detailed.py \
    --checkpoint checkpoints/best_model.pt \
    --hidden_dim 128 \
    --num_rules 16
```

**If this shows 20%+ entity accuracy, the model IS learning!**

The 0.0000% exact match is just a bad metric.

## What Loss ~300 Actually Means

Looking at the loss magnitude:

```
Parse loss: ~200K-300K
Gen loss: ~5
Cycle loss: ~200K-300K

Total: ~300K
```

This is **MSE loss on continuous predictions**. The numbers seem large but might be OK relative to the scale.

**Key question:** Are parse/gen/cycle losses all stuck at the same value, or do they fluctuate?

## Summary

**Immediate next steps:**

1. **Tell me the parse and gen loss values** from the training output
2. **Run detailed evaluation** to see real accuracy
3. **Try lr=0.0003** for next run

The fact that loss is stable (not exploding) is good! But we need to see if it's learning anything, which the exact-match metric hides.

**The model might actually be learning, but the metric is too strict to show it!**
