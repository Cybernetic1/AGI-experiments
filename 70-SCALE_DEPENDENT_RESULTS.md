# ILP Scale-Dependent Behavior - Critical Finding

## 🚨 MAJOR DISCOVERY: Results Flip at Scale!

### Small Scale (10 stories) vs Large Scale (500 stories)

| Algorithm   | 10 Stories (Eval MSE) | 500 Stories (Eval MSE) | Winner Changes |
|-------------|-----------------------|------------------------|----------------|
| Frequency   | 0.061 - 0.112         | **0.028503** 🏆        | Gets BETTER |
| FOIL        | 0.032 - 0.061         | 0.043106               | Gets worse |
| Confidence  | 0.037 - 0.043         | 0.111529 ❌            | **Collapses!** |

## Detailed Comparison

### 10 Stories (Small Dataset):
```
Algorithm       Rules    Labels     Eval MSE    
------------------------------------------------
frequency       5        16,194     0.061-0.112  (worst/middle)
foil            5        25,466     0.032-0.061  (best/middle)
confidence      5        16,630     0.037-0.043  (best/middle) ✅ Most stable
```

### 500 Stories (Large Dataset):
```
Algorithm       Rules    Labels     Eval MSE    
------------------------------------------------
frequency       10       41,267     0.028503     🏆 WINNER
foil            5        25,466     0.043106     (middle)
confidence      6        16,746     0.111529     ❌ COLLAPSED
```

## Key Observations

### 1. **Rule Count Mystery** 🤔
- **Expected:** Up to 50 rules (--max-rules 50)
- **Actual:** 10, 5, 6 rules respectively
- **Why?** `min_support=2` is filtering out most candidates
- With 500 stories, patterns need to appear in only 2 facts to qualify
- Either:
  - Few patterns meet this threshold (data too diverse)
  - Bug in rule counting/filtering

### 2. **Frequency Scales Well** ✅
- 10 stories: 5 rules, 16K labels, 0.061-0.112 MSE (bad)
- 500 stories: 10 rules, 41K labels, 0.028503 MSE (excellent!)
- **2× more rules, 2.5× more labels, 4× better MSE**
- **Insight:** Common patterns become more reliable with more data

### 3. **Confidence Collapses at Scale** ❌
- 10 stories: 5 rules, 16K labels, 0.037-0.043 MSE (great)
- 500 stories: 6 rules, 17K labels, 0.111529 MSE (terrible!)
- **Same rule count, same labels, but 3× worse MSE!**
- **Hypothesis:** Confidence threshold (0.3) filters *too aggressively* with diverse data

### 4. **FOIL Stays Consistent** 📊
- 10 stories: 5 rules, 25K labels, 0.032-0.061 MSE
- 500 stories: 5 rules, 25K labels, 0.043106 MSE
- **Exactly same rules and labels!** Same patterns discovered.
- **Consistent but middle-of-the-road performance**

## Why This Happens

### Frequency Algorithm (Wins at Scale):
```python
# Counts co-occurrences
# With more data → more co-occurrences → stronger signal
# Common patterns become statistically significant
```
✅ **Strength:** Leverages statistical power of large datasets  
❌ **Weakness:** Noisy on small datasets (can't distinguish signal from noise)

### Confidence Algorithm (Fails at Scale):
```python
# Filters by P(conclusion | premises) > 0.3
# With diverse data → fewer patterns have >30% confidence
# Threshold may be too restrictive
```
✅ **Strength:** Works well on homogeneous small datasets  
❌ **Weakness:** Too restrictive for diverse large datasets

### FOIL Algorithm (Stable but Limited):
```python
# Information gain = support × -log2(confidence)
# Finds same patterns regardless of scale
# Not discovering new patterns with more data
```
✅ **Strength:** Consistent across scales  
❌ **Weakness:** Not leveraging additional data (same 5 rules!)

## Critical Questions

### 1. Why So Few Rules?
- With 500 stories and max_rules=50, why only 5-10 rules?
- Check `min_support=2` - is this too restrictive? Too lenient?
- Check rule mining logic - are we filtering correctly?

### 2. Why Does Confidence Fail?
- Same 6 rules, similar labels (16K→17K), but MSE explodes (0.04→0.11)
- Is `min_confidence=0.3` wrong for large/diverse datasets?
- Are the 6 rules *different* rules than the 5 from small dataset?
- Is train/eval split different (overfitting)?

### 3. Why Doesn't FOIL Scale?
- Exact same 5 rules and 25K labels on both 10 and 500 stories
- Is FOIL not mining from all 500 stories?
- Is information gain calculation saturating early?
- Are we limiting to first 5000 facts? (Check code!)

## Recommended Investigations

### 1. **Check Facts Loading**
```python
# In test_single_algorithm():
facts = load_tinystories_facts(max_stories=args.max_stories)
print(f"Total facts loaded: {len(facts)}")

# Then use facts[:5000] - WAIT! Is this the bug?
labels_dict = _collect_labels(facts[:5000], rules, log_progress=False)
```
**🚨 POTENTIAL BUG:** We might be limiting to first 5000 facts even with 500 stories!

### 2. **Try Lower min_confidence**
```bash
# Test confidence with lower threshold
python test_ilp_comparison.py --max-stories 500 --max-rules 50 --min-confidence 0.1
```

### 3. **Try Different min_support**
```bash
# Require patterns in at least 10 facts (not just 2)
python test_ilp_comparison.py --max-stories 500 --max-rules 50 --min-support 10
```

### 4. **Check Rule Details**
Add debug output to see which rules are discovered:
```python
print(f"\nRules discovered by {algorithm_name}:")
for i, (rule, support, conf) in enumerate(candidates):
    print(f"  {i+1}. {rule} (support={support}, conf={conf})")
```

## Revised Recommendations

### Based on Scale:

**Small Datasets (<50 stories):**
- Use **Confidence-based** (0.037-0.043 MSE, low variance)
- Or **FOIL** (0.032-0.061 MSE, higher potential)

**Large Datasets (500+ stories):**
- Use **Frequency-based** (0.028503 MSE) ✅
- **NOT Confidence** (collapses to 0.111529)
- FOIL is okay but doesn't leverage scale

**General Production:**
- **Test at target scale!** Don't trust small-scale results
- Consider **adaptive algorithm selection** based on dataset size
- Or **ensemble approach** (combine all three)

## Action Items

1. 🐛 **Fix potential bug:** Check if we're limiting to 5000 facts
2. 🔬 **Investigate Confidence collapse:** Why does it fail at scale?
3. 📊 **Investigate FOIL plateau:** Why same rules for 10 vs 500 stories?
4. ⚙️ **Add hyperparameter options:** --min-confidence flag
5. 📈 **Test intermediate scales:** 50, 100, 200 stories to find transition point

## Meta-Insight

**Scale changes everything!**
- Small-scale experiments can be misleading
- Algorithms have different scaling behaviors
- "Best on small dataset" ≠ "Best on large dataset"
- Always test at production scale before choosing algorithm

This is why real ML research requires large-scale validation! 🎯

## Files to Check

```bash
# Check if we're limiting facts:
grep -n "facts\[:5000\]" test_ilp_comparison.py

# Check min_confidence hardcoding:
grep -n "min_confidence" core/ilp_algorithms.py
```
