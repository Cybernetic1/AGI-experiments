# ILP Algorithm Comparison Framework

## Overview

This framework allows testing and comparing different ILP (Inductive Logic Programming) algorithms for rule discovery in our DLN system. The goal is to determine which algorithms produce the most useful rules and whether adding more rules actually improves performance.

## Problem Statement

**Key Question:** Does adding more rules help DLN performance, or do we hit combinatorial explosion?

Traditional ILP faces combinatorial explosion when generating too many candidate rules. We need to:
1. Test different rule discovery algorithms
2. Track which rules actually contribute to predictions
3. Identify "dead" rules that never fire (potential noise)
4. Compare quality vs. quantity trade-offs

## Components

### 1. Rule Tracker (`core/rule_tracker.py`)

Monitors individual rule performance:

```python
from core.rule_tracker import RuleTracker

tracker = RuleTracker()
tracker.register_rules(rules, rule_type="mined")
tracker.record_firing(rule_id, is_train=True)  # During label generation
tracker.record_prediction(rule_id, pred, target)  # During evaluation
tracker.print_report()  # Shows which rules are useful
```

**Metrics tracked:**
- **Firing frequency:** How often each rule fires during label generation (train/eval)
- **Prediction accuracy:** % of predictions within threshold (0.1)
- **MSE/MAE:** Per-rule prediction errors
- **Dead rules:** Rules that never fire (potential noise)

### 2. ILP Algorithms (`core/ilp_algorithms.py`)

Three algorithms implemented:

#### A. Frequency-Based (Current)
```python
mine_frequency_based(facts, max_rules=10, min_support=2)
```
- Counts co-occurrence frequency of predicate pairs
- Generates: `P1(?x,?y) ∧ P2(?y,?z) → P1_P2_mined(?x,?z)`
- **Pros:** Simple, fast, captures common patterns
- **Cons:** Ignores confidence, may favor overly-common patterns

#### B. FOIL-Style (Information Gain)
```python
mine_foil_style(facts, max_rules=10, min_support=2, min_info_gain=0.01)
```
- Uses information gain: `support × -log2(confidence)`
- Favors rules that are both **frequent** and **informative** (not too obvious)
- **Pros:** Balances frequency with informativeness
- **Cons:** More complex, may miss very common patterns

#### C. Confidence-Based
```python
mine_confidence_based(facts, max_rules=10, min_support=2, min_confidence=0.5)
```
- Confidence = P(P2 follows P1) = count(P1 ∧ P2) / count(P1)
- Filters by minimum confidence threshold
- **Pros:** High predictive power (if P1, then P2 is likely)
- **Cons:** May favor rare but unreliable patterns

### 3. Comparison Test (`test_ilp_comparison.py`)

Runs all algorithms and compares results:

```bash
# Compare all algorithms on 50 stories
python test_ilp_comparison.py --max-stories 50 --max-rules 20

# Test single algorithm
python test_ilp_comparison.py --algorithm foil --max-rules 30

# Larger test
python test_ilp_comparison.py --max-stories 200 --max-rules 50
```

**Output includes:**
1. Rule overlap analysis (which rules appear in multiple algorithms)
2. Training/eval MSE for each algorithm
3. Number of labels generated (measure of rule productivity)
4. Performance summary and winner determination

## Expected Results

### Hypothesis 1: Quality > Quantity
- **Prediction:** FOIL-style will outperform frequency-based
- **Reason:** Information gain filters obvious patterns, keeps informative ones
- **Test:** Compare eval MSE across algorithms

### Hypothesis 2: Diminishing Returns
- **Prediction:** Performance plateaus after N rules (N ≈ 20-30?)
- **Reason:** Core patterns captured early, additional rules add noise
- **Test:** Run with `--max-rules 10,20,50,100` and plot MSE vs. rules

### Hypothesis 3: Dead Rules Exist
- **Prediction:** 20-40% of rules never fire
- **Reason:** Rules mined from subset may not generalize
- **Test:** Use RuleTracker to count zero-firing rules

### Hypothesis 4: Rule Overlap Indicates Quality
- **Prediction:** Rules appearing in all 3 algorithms are most robust
- **Reason:** Agreement across methods suggests true patterns
- **Test:** Check overlap analysis in comparison output

## Usage Example

```bash
# Quick test (50 stories, 20 rules per algorithm)
python test_ilp_comparison.py

# Large-scale comparison (500 stories, 50 rules)
python test_ilp_comparison.py --max-stories 500 --max-rules 50
```

**Expected output:**
```
==================================================================
ILP ALGORITHM COMPARISON TEST
==================================================================
Loading 50 stories...
Loaded 10234 facts from stories

==================================================================
PHASE 1: MINING COMPARISON
==================================================================

[1] Frequency-based mining...
  Generated 20 rules

[2] FOIL-style (information gain)...
  Generated 20 rules

[3] Confidence-based mining...
  Generated 20 rules

Rule overlap:
  Frequency ∩ FOIL: 15 rules
  Frequency ∩ Confidence: 12 rules
  FOIL ∩ Confidence: 10 rules
  All three: 8 rules

==================================================================
PHASE 2: TRAINING COMPARISON
==================================================================

==================================================================
TESTING: FREQUENCY
==================================================================
...
Train MSE: 0.000123, Eval MSE: 0.000456

==================================================================
SUMMARY COMPARISON
==================================================================
Algorithm       Rules    Labels     Train MSE    Eval MSE     Eval MAE    
----------------------------------------------------------------------
frequency       20       45231      0.000123     0.000456     0.002134
foil            20       42890      0.000098     0.000321     0.001876
confidence      20       38765      0.000234     0.000543     0.002345

🏆 Best performing algorithm: FOIL
   Eval MSE: 0.000321, MAE: 0.001876
```

## Next Steps

1. **Run initial comparison:**
   ```bash
   python test_ilp_comparison.py --max-stories 50
   ```

2. **Analyze results:**
   - Which algorithm wins?
   - How many rules overlap?
   - Are there dead rules?

3. **Test scaling:**
   ```bash
   # Does performance improve with more rules?
   python test_ilp_comparison.py --max-rules 10
   python test_ilp_comparison.py --max-rules 20
   python test_ilp_comparison.py --max-rules 50
   ```

4. **Integrate winner:**
   - Replace current mining in `pipelines/tinystories_pipeline.py`
   - Use best-performing algorithm by default
   - Keep others available for comparison

5. **Add rule pruning:**
   - Use RuleTracker to identify dead rules
   - Remove rules with <N firings after training
   - Re-train with pruned rule set

## Integration with Benchmarks

To add rule tracking to existing benchmarks, modify `pipelines/tinystories_pipeline.py`:

```python
from core.rule_tracker import RuleTracker

def run_tinystories_benchmark(..., track_rules=False):
    # ... setup ...
    
    if track_rules:
        tracker = RuleTracker()
        tracker.register_rules(base_rules, "base")
        tracker.register_rules(mined_rules, "mined")
        # ... (record firings during label generation) ...
        tracker.print_report()
```

## References

- **FOIL:** Quinlan & Cameron-Jones (1993) - "FOIL: A midterm report"
- **RIPPER:** Cohen (1995) - "Fast Effective Rule Induction"
- **ClausIE:** Del Corro & Gemulla (2013) - "ClausIE: clause-based open information extraction"
- **Aleph:** Srinivasan (2001) - "The Aleph Manual"

## Key Insight

**The question isn't "can we mine more rules?"** (we can always mine thousands)

**The question is "which rules actually help?"** (quality > quantity)

This framework answers that question empirically.
