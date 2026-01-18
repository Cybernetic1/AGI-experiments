# Fast ILP Comparison - Skip Training Mode

## The Problem

Label generation becomes **exponentially slow** with large datasets:
- 1K facts → ~10 seconds
- 10K facts → ~minutes  
- 50K facts → **hours** (symbolic inference is O(n²) or worse)

For **comparing ILP algorithms**, we don't actually need to train the DLN!

## The Solution

### Option 1: Skip Training Entirely (Fastest)
```bash
# Just compare which rules are discovered (no DLN training)
python test_ilp_comparison.py --max-stories 500 --max-facts 100000 --skip-training

# Output:
# Algorithm       Rules Mined    
# -----------------------------
# frequency       25             
# foil            18             
# confidence      12             
```

**Use case:** Quick comparison of rule discovery algorithms  
**Speed:** ~10 seconds for 500 stories

### Option 2: Sample Facts for Labels (Fast)
```bash
# Mine rules from ALL facts, but only generate labels from first 5K
python test_ilp_comparison.py --max-stories 500 --max-facts 100000 --sample-facts 5000

# Output includes train/eval MSE but much faster
```

**Use case:** Want DLN performance metrics but can't wait hours  
**Speed:** ~1-2 minutes for 500 stories

### Option 3: Full Training (Slow but Complete)
```bash
# Use all facts for everything (SLOW!)
python test_ilp_comparison.py --max-stories 500 --max-facts 100000 --sample-facts 100000
```

**Use case:** Final validation before production  
**Speed:** Could take hours for 50K+ facts

## Recommended Workflow

### Phase 1: Quick Rule Discovery (--skip-training)
```bash
# Test at different scales to see how many rules are discovered
python test_ilp_comparison.py --max-stories 10 --max-facts 10000 --skip-training
python test_ilp_comparison.py --max-stories 50 --max-facts 50000 --skip-training  
python test_ilp_comparison.py --max-stories 200 --max-facts 200000 --skip-training
python test_ilp_comparison.py --max-stories 500 --max-facts 500000 --skip-training
```

**Goal:** See which algorithm finds more rules at different scales  
**Time:** ~10 seconds per test

### Phase 2: Sampled Training (--sample-facts 5000)
```bash
# Once you know which algorithms find good rules, test their quality
python test_ilp_comparison.py --max-stories 200 --max-facts 200000 --sample-facts 5000 --num-runs 3
```

**Goal:** Estimate DLN performance without full label generation  
**Time:** ~5 minutes

### Phase 3: Full Validation (no sampling, small scale)
```bash
# Final test on manageable scale
python test_ilp_comparison.py --max-stories 50 --max-facts 50000 --sample-facts 50000 --num-runs 5
```

**Goal:** Accurate performance metrics for chosen algorithm  
**Time:** ~15-30 minutes

## Why This Works

### For Comparing ILP Algorithms:
We care about:
1. ✅ **Which rules are discovered** (--skip-training shows this)
2. ✅ **How many rules** (--skip-training shows this)
3. ⚠️ **Rule quality** (can estimate with --sample-facts)
4. ❌ **Exact DLN MSE** (need full training, but not critical for comparison)

### Label Generation is O(facts × rules × bindings):
- **1K facts, 10 rules:** ~10K operations
- **10K facts, 20 rules:** ~2M operations  
- **50K facts, 50 rules:** ~125M operations 😱

Sampling facts = linear speedup!

## Command Reference

```bash
# Fastest: Just show rules
python test_ilp_comparison.py --skip-training

# Fast: Sample labels
python test_ilp_comparison.py --sample-facts 5000

# Custom sample size
python test_ilp_comparison.py --sample-facts 10000

# Multiple runs for statistics
python test_ilp_comparison.py --skip-training --num-runs 5

# Large scale, rules only
python test_ilp_comparison.py --max-stories 500 --max-facts 500000 --skip-training

# Medium scale with training
python test_ilp_comparison.py --max-stories 100 --max-facts 100000 --sample-facts 5000
```

## What You'll Learn

### From --skip-training:
- Which algorithm finds more rules
- How rule count scales with dataset size
- Rule diversity (different patterns discovered)
- **Fast iteration** on algorithm parameters

### From --sample-facts N:
- Approximate DLN performance
- Rule quality estimation
- Variance across runs
- **Reasonable speed** for development

### From full training:
- Exact performance metrics
- Production-ready validation
- **Slow but accurate**

## Example Session

```bash
# 1. Quick check: Which algorithms scale?
$ python test_ilp_comparison.py --max-stories 500 --skip-training
# frequency: 45 rules
# foil: 28 rules
# confidence: 15 rules
# → Frequency discovers most rules at scale ✅

# 2. Sample training: Which performs best?
$ python test_ilp_comparison.py --max-stories 500 --sample-facts 5000
# frequency: 0.035 MSE
# foil: 0.052 MSE  
# confidence: 0.089 MSE
# → Frequency also performs best ✅

# 3. Final validation on manageable scale
$ python test_ilp_comparison.py --max-stories 100 --sample-facts 20000 --num-runs 5
# frequency: 0.028 ± 0.004 MSE (most stable) ✅
```

## Bottom Line

**Use `--skip-training` for fast exploration, then `--sample-facts` for validation.**

You rarely need full label generation for comparing algorithms!
