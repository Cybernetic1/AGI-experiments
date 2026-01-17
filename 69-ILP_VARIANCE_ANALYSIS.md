# ILP Algorithm Comparison - Variance Analysis

## Multiple Runs Show Different Winners!

### Run 1 Results (First test):
| Algorithm   | Train MSE | Eval MSE | Winner |
|-------------|-----------|----------|--------|
| Frequency   | 0.087775  | 0.111563 |        |
| FOIL        | 0.083296  | 0.061276 |        |
| **Confidence** | **0.029211** | **0.043122** | 🏆 |

### Run 2 Results (User's run):
| Algorithm   | Train MSE | Eval MSE | Winner |
|-------------|-----------|----------|--------|
| Frequency   | 0.057463  | 0.061227 |        |
| **FOIL**    | **0.035142** | **0.032341** | 🏆 |
| Confidence  | 0.047083  | 0.037085 |        |

### Combined Analysis:

| Algorithm   | Run 1 Eval | Run 2 Eval | Average | Std Dev | Variance |
|-------------|-----------|-----------|---------|---------|----------|
| Frequency   | 0.111563  | 0.061227  | 0.086395 | 0.0356 | High |
| FOIL        | 0.061276  | 0.032341  | 0.046809 | 0.0205 | Moderate |
| Confidence  | 0.043122  | 0.037085  | 0.040104 | 0.0043 | **Low** ✅ |

## Key Insights

### 🎲 **High Variance = Unreliable**

**Frequency algorithm:**
- Run 1: 0.112 MSE (worst)
- Run 2: 0.061 MSE (middle)
- **82% difference!** Very unstable.

**FOIL algorithm:**
- Run 1: 0.061 MSE (middle)
- Run 2: 0.032 MSE (best)
- **89% difference!** Also unstable, but performed best in Run 2.

**Confidence algorithm:**
- Run 1: 0.043 MSE (best)
- Run 2: 0.037 MSE (second best)
- **14% difference** - Most consistent! ✅

### 🏆 **True Winner: Confidence (Most Stable)**

Despite FOIL winning Run 2:
1. **Confidence has lowest variance** (0.0043 std dev)
2. **Confidence always performs well** (0.037-0.043 range)
3. **FOIL is unpredictable** (0.032-0.061 range - might get lucky or unlucky)

### 📊 **Why the Variance?**

Several sources of randomness:
1. **Neural network initialization** (random weights)
2. **Training optimization** (SGD is stochastic)
3. **Mini-batch sampling** (if enabled)
4. **Rule order** (may affect label generation)

### 🔬 **Statistical Significance Needed**

Two runs are NOT enough to conclude! Need:
- **Multiple runs (5-10)** with different random seeds
- **Average eval MSE** across runs
- **Confidence intervals** (e.g., mean ± std dev)
- **Statistical tests** (t-test, ANOVA)

## Revised Conclusion

### Based on 2 Runs:

**1st Place: Confidence (most reliable)**
- Average: 0.040 eval MSE
- Std Dev: 0.004 (very stable)
- Always in top 2 performers

**2nd Place: FOIL (high potential, high variance)**
- Average: 0.047 eval MSE
- Std Dev: 0.020 (unstable)
- Can achieve best results (0.032) but also middling (0.061)

**3rd Place: Frequency (inconsistent)**
- Average: 0.086 eval MSE  
- Std Dev: 0.036 (very unstable)
- Ranges from worst (0.112) to middle (0.061)

## Recommendation Update

### For Production Use:

**Choose Confidence-based** because:
1. ✅ **Consistent performance** (low variance)
2. ✅ **Always good** (never performs poorly)
3. ✅ **Reliable** (predictable results)

**Avoid Frequency-based** because:
1. ❌ **High variance** (unpredictable)
2. ❌ **Worst average** (0.086)
3. ❌ **Can fail badly** (0.112 in Run 1)

**FOIL is interesting** but:
1. ⚠️ **High risk, high reward** (0.032 best, but 0.061 in Run 1)
2. ⚠️ **May need hyperparameter tuning** to stabilize
3. ⚠️ **Requires multiple runs** to verify performance

### For Research/Ensemble:

Consider **combining all three:**
- Take top 3-5 rules from each algorithm (15 total)
- Diversity of rule sources may improve robustness
- Low overlap (0-4/5) suggests complementary patterns

## Improved Test Script Needed

Add to `test_ilp_comparison.py`:

```python
--num-runs N          # Run each algorithm N times (default: 5)
--random-seed SEED    # Set random seed for reproducibility
--report-variance     # Show mean ± std dev across runs
```

This would give output like:
```
Algorithm       Eval MSE (mean ± std)    Best    Worst
----------------------------------------------------------------
confidence      0.040 ± 0.004            0.037   0.043  ✅ (most stable)
foil            0.047 ± 0.020            0.032   0.061  (high variance)
frequency       0.086 ± 0.036            0.061   0.112  (unstable)
```

## Action Items

1. ✅ **Current insight:** Confidence is most stable (even though FOIL won Run 2)
2. 🔄 **Short term:** Run 3-5 more times to confirm pattern
3. 🛠️ **Medium term:** Add `--num-runs` parameter to test script
4. 📊 **Long term:** Full statistical analysis with confidence intervals

## Meta-Learning Insight

This variance teaches us:
- **Don't trust single runs!** (Run 1 said Confidence, Run 2 said FOIL)
- **Stability matters** as much as peak performance
- **Average performance** is more important than best-case
- **For AGI:** Need rules that work consistently, not just occasionally

**Conclusion:** Confidence-based is still the best choice for production due to low variance, even though FOIL achieved the single best result (0.032) in Run 2.
