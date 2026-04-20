# Logic Transformer V2: Summary and Design Decision

## Context

We needed to solve the **cross-premise variable binding problem**: how to enforce constraints like `father(X,Y) ∧ father(Y,Z) → grandfather(X,Z)` where variable Y must be bound consistently across both premises.

Traditional symbolic logic requires constructing all W^J combinations of working memory elements (exponential complexity). Our V1 architecture avoided this by having each premise independently select its best match via soft attention (linear complexity), but this meant **no communication between premises** - they couldn't enforce binding constraints.

---

## Two Approaches Explored

### V2 Heavy: Multi-Head Attention
- **Idea:** Use transformer-style attention to let premise 2's query be modulated by premise 1's selection
- **Implementation:** Full attention machinery with Q/K/V projections, 4-head attention, hidden dimension=32
- **Parameters:** ~4,700 per rule (40.6x increase over V1)
- **Result:** **FAILED** - overfits, worse generalization (-14.2% on test set)

### V2 Lightweight: Binding Matrices
- **Idea:** Directly encode position-wise binding constraints in a learnable (L×L) matrix
- **Implementation:** `B[i,k]` = strength of constraint "prev[i] should match curr[k]"
- **Parameters:** ~10 per rule (1.1x increase over V1)
- **Result:** **SUCCESS** - better generalization (+76.6% on test set)

---

## Why V2 Lightweight Wins

### Key Insight: Logical Binding is Sparse

For `father(X,Y) ∧ father(Y,Z)`:
- Only **ONE position pair** needs to match: premise1.arg2 = premise2.arg0
- Multi-head attention uses **~4,000 parameters** to discover this one constraint
- Binding matrix uses **9 parameters** (one per position pair in 3×3 matrix)

This is a textbook case of **"strong inductive bias beats brute force"**

### Advantages of Binding Matrices:

1. **Sparse structure** - most B[i,k] = 0, only specific pairs matter
2. **Interpretable** - can read off which variables are shared
3. **Efficient** - O(L²) not O(hidden² × heads)
4. **Strong prior** - encodes known structure of logical binding
5. **Better generalization** - less parameters to overfit

---

## Technical Details

### How Binding Works:

```python
# Step 1: Base cylindrification matching (V1 behavior)
match_score[w] = Σ_l (1-γ[l]) × (constant[l] - wm[w,l])²

# Step 2: Add binding penalty
for i, k where B[i,k] ≠ 0:
    match_score[w] += gate × |B[i,k]| × (prev_selection[i] - wm[w,k])²

# Step 3: Soft attention
weights = softmax(-match_scores)
```

The binding penalty creates **asymmetry** - propositions matching the binding constraint get lower scores (better), breaking ties that cylindrification alone couldn't resolve.

### Example:

Working Memory:
```
[0] = [john, father, bob]
[1] = [bob, father, alice]
```

Without binding: both match "father" equally → random selection
With binding (B[2,0]=1.0): WM[1] wins because bob=bob → correct transitive chain

---

## Performance Results

**Transitive Reasoning Task** (50 train, 20 test):

|                  | Params  | Test Loss | Generalization |
|------------------|---------|-----------|----------------|
| V1 (baseline)    | 480     | 0.0423    | -              |
| V2 Heavy         | 19,472  | 0.0483    | -14.2% ⚠️      |
| **V2 Lightweight** | **520** | **0.0099** | **+76.6%** ✓ |

V2 Lightweight is:
- **37x fewer parameters** than V2 Heavy
- **77% better** than V1 on generalization
- **Preserves efficiency** advantage over Transformers

---

## Design Decision: V2 Lightweight is the Recommended Architecture

We recommend using **`logic_transformer_v2_lightweight.py`** as the default "Logic Transformer V2" because:

1. **Minimal overhead** - only 8% parameter increase vs V1
2. **Best performance** - significantly better generalization
3. **Interpretability** - binding matrices are human-readable
4. **Efficiency** - maintains advantage over traditional Transformers
5. **Scalability** - O(L²) not O(hidden²), stays compact as we add more rules

V2 Heavy can be archived as a reference implementation showing that "bigger isn't always better" for structured reasoning tasks.

---

## Implementation Files

- **Core V2 Lightweight:** `logic_transformer_v2_lightweight.py`
- **Comparison (V2 Heavy):** `logic_transformer_v2.py` (for reference)
- **Benchmark:** `compare_logic_transformer_versions.py`
- **Diagram:** `logic_transformer_v2_diagram.pdf`

---

## Preserved Structural Priors

V2 Lightweight maintains ALL the key innovations from V1:

- ✓ **Cylindrification (γ)** - constant vs variable distinction
- ✓ **Explicit rule structure** - J premises → I variables → conclusion
- ✓ **Proposition-level granularity** - not token-level like Transformers
- ✓ **Variable slots** - explicit binding sites with semantic identity
- ✓ **Mixture of experts** - M independent rules

And adds:
- ✓ **Cross-premise binding** - via lightweight binding matrices

---

## Lessons Learned

### 1. Inductive Bias > Parameters
Adding 40x parameters via attention hurt performance. Adding 2% parameters with the right structure helped enormously.

### 2. Match Structure to Domain
Logical binding has known sparse structure - encode that directly rather than hoping a neural network discovers it.

### 3. Interpretability Matters
Being able to inspect binding matrices (B[2,0]=1.0 means Y is shared) is valuable for debugging and understanding.

### 4. Efficiency is a Feature
Staying close to V1's parameter count preserves the core advantage over Transformers - compact, efficient, sample-efficient learning.

---

## Future Directions

### Potential Enhancements:
1. **Binding matrix regularization** - encourage sparsity (L1 penalty)
2. **Symbolic initialization** - pre-populate B based on known rules
3. **Multi-hop binding** - extend to J>2 with chained binding matrices
4. **Learned gate values** - per-rule or per-position gating instead of shared

### Open Questions:
1. How do binding matrices scale to very long proposition chains?
2. Can we learn binding patterns from symbolic rules and transfer to neural learning?
3. What's the right balance of binding strength vs cylindrification?

---

## Conclusion

Logic Transformer V2 Lightweight successfully adds cross-premise variable binding while maintaining the efficiency and interpretability advantages of the original architecture. By encoding the sparse structure of logical binding directly (via binding matrices) rather than using general-purpose attention, we achieve:

- Better generalization
- Fewer parameters  
- Greater interpretability
- Preserved efficiency

This demonstrates a core principle: **domain-specific architectural priors often outperform generic neural mechanisms for structured reasoning tasks.**
