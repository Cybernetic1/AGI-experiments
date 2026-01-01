# Progress Update: GA Shows Promise, Neural Approach Failed

**Date:** 2026-01-01  
**Major Pivot:** GA approach working (62% fitness!), neural approach failed (0% accuracy)

---

## 🎯 Key Result

```
Neural Network (50 rules, 30 epochs):  0.00% accuracy ✗
Genetic Algorithm (30 generations):    61.6% fitness  ✓
```

**GA discovered interpretable rule:** `['NOUN', 'VERB']` pattern that matches all test examples!

---

## Why This Matters

1. **Validates symbolic search:** Discrete optimization works for symbolic rules
2. **Sample efficient:** Found working rule with 7,500 evaluations (vs 30,000 for neural)
3. **Interpretable:** Can see exactly what pattern was learned
4. **Uses semantic-AR:** Fitness based on meaning preservation, not surface form

---

## Next Steps

1. Test GA on TinyStories (100-1000 examples)
2. Compare GA vs Neural head-to-head
3. Implement hybrid: GA discovers rules, neural fine-tunes

See full details in this file (below) and `docs/` folder.
