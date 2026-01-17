# ILP Rule Tracking - Quick Start Guide

## ✅ IT WORKS! Test Results Available

The test script is now fully functional. See **`68-ILP_TEST_RESULTS.md`** for detailed results.

**TL;DR:** Confidence-based algorithm wins with 45-61% better eval MSE!

## Quick Test

```bash
# Small test (10 stories, 5 rules each)
source venv/bin/activate
python test_ilp_comparison.py --max-stories 10 --max-rules 5

# Medium test (50 stories, 20 rules each)
python test_ilp_comparison.py --max-stories 50 --max-rules 20

# Large test (200 stories, 50 rules each)
python test_ilp_comparison.py --max-stories 200 --max-rules 50
```

## What Was Created

Three new components to track and compare ILP algorithms:

### 1. **Rule Tracker** (`core/rule_tracker.py`)
Monitors individual rule performance:
- Tracks firing frequency (how often each rule is used)
- Measures prediction accuracy per rule
- Identifies "dead" rules that never fire
- Generates detailed performance reports

### 2. **ILP Algorithms** (`core/ilp_algorithms.py`)
Three rule discovery algorithms:

**A. Frequency-Based** (current approach)
- Counts pattern co-occurrences
- Simple and fast
- May favor overly-common patterns

**B. FOIL-Style** (information gain)  
- Uses information theory: `support × -log2(confidence)`
- Balances frequency with informativeness
- Filters obvious patterns

**C. Confidence-Based** (predictive power)
- Confidence = P(conclusion | premises)
- Favors reliable patterns
- Filters by minimum confidence threshold

### 3. **Comparison Test** (`test_ilp_comparison.py`)
Automated benchmarking script to compare algorithms.

## Quick Test (Manual)

Since the test script has some integration issues with the existing codebase, here's a simpler manual test:

```python
# In Python console or notebook
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))

from pipelines.tinystories_pipeline import load_tinystories_facts
from core.ilp_algorithms import compare_algorithms

# Load facts
facts = load_tinystories_facts(max_stories=50)
print(f"Loaded {len(facts)} facts")

# Compare algorithms
results = compare_algorithms(facts, max_rules=20, min_support=2)

# You'll see output like:
# [1] Frequency-based mining... Generated 20 rules
# [2] FOIL-style (information gain)... Generated 20 rules  
# [3] Confidence-based mining... Generated 20 rules
# Rule overlap: Frequency ∩ FOIL: X rules ...
```

## Key Findings From Initial Test

From the small test run (10 stories, 5 rules):

1. **Rule Overlap is Low:**
   - Frequency ∩ FOIL: 1/5 rules
   - Frequency ∩ Confidence: 4/5 rules
   - FOIL ∩ Confidence: 0/5 rules
   - All three: 0/5 rules

2. **Different Algorithms Find Different Patterns:**
   - This is GOOD - means we can combine them
   - FOIL focuses on informative patterns
   - Confidence focuses on predictive patterns
   - Frequency captures common patterns

3. **Label Generation Works:**
   - 5 rules → 16,194 labels (from 1,000 facts)
   - 16× expansion shows rules are productive

## Integration with Existing Benchmarks

To add rule tracking to `pipelines/tinystories_pipeline.py`:

```python
from core.rule_tracker import RuleTracker
from core.ilp_algorithms import mine_foil_style  # or other algorithm

def tinystories_mini_benchmark(..., rule_algorithm='frequency'):
    # ... existing setup ...
    
    # Replace mine_chain_rules with configurable algorithm
    if rule_algorithm == 'frequency':
        mined_rules, mined_preds = mine_chain_rules(all_facts_train, max_rules=50)
    elif rule_algorithm == 'foil':
        from core.ilp_algorithms import mine_foil_style
        mined_rules, mined_preds = mine_foil_style(all_facts_train, max_rules=50)
    # ... etc ...
    
    # Add rule tracking
    tracker = RuleTracker()
    tracker.register_rules(base_rules, "base")
    tracker.register_rules(mined_rules, "mined")
    
    # During label generation, record firings
    # (would need to modify label_utils.py to call tracker.record_firing())
    
    # At end
    tracker.print_report(top_n=10)
```

## Next Steps

1. **Fix test_ilp_comparison.py dependencies**
   - The test script needs to be aligned with the exact API of dln.py and train_utils.py
   - Currently has parameter mismatches

2. **Run comparison on real data:**
   ```bash
   # Once fixed:
   python test_ilp_comparison.py --max-stories 50 --max-rules 20
   python test_ilp_comparison.py --max-stories 200 --max-rules 50
   ```

3. **Test scaling:** Does performance improve with more rules?
   ```bash
   python test_ilp_comparison.py --max-rules 10
   python test_ilp_comparison.py --max-rules 20
   python test_ilp_comparison.py --max-rules 50
   ```

4. **Integrate best algorithm** into benchmarks.py

5. **Add rule pruning:** Remove rules that never fire

## Expected Benefits

**Quality Over Quantity:**
- FOIL-style should outperform frequency on eval MSE
- Fewer rules with higher information content
- Combinator explosion avoided

**Dead Rule Detection:**
- Identify 20-40% of rules that never fire
- Prune these before training
- Cleaner model, faster training

**Algorithm Combination:**
- Take top N rules from each algorithm
- Ensemble approach: diverse rule sources
- Better coverage of pattern space

## Documentation Reference

See `66-ILP_COMPARISON_FRAMEWORK.md` for full details on:
- Algorithm theory and trade-offs
- Expected hypotheses and results
- Integration instructions
- ILP algorithm references (FOIL, RIPPER, etc.)
