# Architecture Comparison: V2 Heavy vs V2 Lightweight

## Overview

This document compares two approaches to implementing cross-premise variable binding in Logic Transformer V2.

---

## V2 Heavy (logic_transformer_v2.py)

Uses **MULTI-HEAD ATTENTION** for cross-premise binding.

### Components per rule (J=2, L=3, hidden_dim=32):

- Query projections: 2 × Linear(L=3 → hidden=32) = **256 params**
- Key projection: Linear(L=3 → hidden=32) = **128 params**  
- Value projection: Linear(L=3 → L=3) = **12 params**
- Binding networks: 1 × Linear(L=3 → hidden=32) = **128 params**
- **MultiheadAttention(hidden=32, heads=4):**
  - in_proj (Q,K,V): 3×32×32 + 3×32 = **3,168 params** ← MAIN CULPRIT!
  - out_proj: 32×32 + 32 = **1,056 params**

**Total attention overhead: ~4,700 params per rule**

### How it works:

1. Base query from constants (via cylindrification γ)
2. Modulate query with Linear(prev_selection) → adjustment
3. Use MultiheadAttention(query, prev_selection, prev_selection) → refinement
4. Combined query attends to working memory (via key/value projections)
5. Cylindrification scores added to attention scores

**Is it multi-head?** YES - uses nn.MultiheadAttention with 4 heads

---

## V2 Lightweight (logic_transformer_v2_lightweight.py)

Uses **BINDING MATRICES** instead of attention.

### Components per rule (J=2, L=3):

- Binding matrix B: (L, L) = (3, 3) = **9 params**
- Binding gate: scalar = **1 param**

**Total binding overhead: 10 params per rule**

### How it works:

1. **Base cylindrification matching** (same as V1):
   ```
   match_score[w] = Σ_l (1-γ[l]) × (constant[l] - wm[w,l])²
   ```

2. **Add binding constraint**:
   ```
   For each position pair (i,k):
     if binding_matrix[i,k] is significant:
       match_score[w] += gate × |B[i,k]| × (prev_selection[i] - wm[w,k])²
   ```

3. **Soft attention** over working memory:
   ```
   attention_weights = softmax(-match_scores)
   ```

4. **Select proposition**: weighted average using attention_weights

**Is it multi-head?** NO - no attention mechanism at all!

---

## Key Differences

| Aspect              | V2 Heavy         | V2 Lightweight     |
|---------------------|------------------|--------------------|
| Multi-head?         | YES (4 heads)    | NO                 |
| Attention?          | YES (full attn)  | NO (direct match)  |
| Params overhead     | ~4,700/rule      | ~10/rule           |
| Total increase      | 40.6x            | 1.1x               |
| Binding mechanism   | Learned via attn | Explicit matrix    |
| Query modulation?   | YES (complex)    | NO (direct score)  |
| Hidden dimension?   | Required (32+)   | Not needed         |

### What V2 Lightweight Eliminates:

- ✗ Query/Key/Value projections to hidden space
- ✗ Multi-head attention mechanism  
- ✗ Attention weights computation in hidden space
- ✗ Learned combination of multiple attention heads
- ✗ All the associated parameters (3K+ per rule)

### What V2 Lightweight Keeps:

- ✓ Cross-premise binding concept
- ✓ Ability to enforce "arg[i] of premise1 = arg[j] of premise2"
- ✓ Differentiability (binding matrix is learnable)
- ✓ Cylindrification (γ) for constant/variable distinction

---

## Binding Matrix Interpretation

For `father(X,Y) ∧ father(Y,Z) → grandfather(X,Z)`:

```
binding_matrix =     p2.subject  p2.relation  p2.object
                  [                                     ]
     p1.subject   [     0           0            0     ]
     p1.relation  [     0           0            0     ]
     p1.object    [    1.0          0            0     ]  ← Key constraint!
```

This means: premise1.object (Y) should match premise2.subject (Y)

**Much more interpretable than 4-head attention weights!**

---

## Analogy

### V2 Heavy (Multi-head Attention):
"Use a sophisticated neural pattern matcher to figure out which parts of the previous selection should constrain the current selection"

Like: Using a deep neural network to learn `f(prev) → query_adjustment`

### V2 Lightweight (Binding Matrix):
"Directly specify which positions should match across premises"

Like: Using a simple lookup table: `B[i,j]` = "how much prev[i] and curr[j] should match"

The lightweight version is essentially a **STRUCTURED SPARSITY** prior:
"Most position pairs don't need to match, but a few specific ones do"

This is perfect for logical variable binding, where you typically have:
- `father(?X, ?Y)`: variables at positions 0 and 2
- `father(?Y, ?Z)`: variables at positions 0 and 2
- Binding constraint: position 2 of premise1 = position 0 of premise2

You don't need 4 attention heads to learn this - just `B[2,0] = high`!

---

## Mathematical Formulation

### V2 Heavy (Attention-based):
```python
prev_embed = f_query(prev_selection)           # L → hidden
query_adjust = MultiheadAttn(                  # hidden → hidden
  query=f_base(constants),
  key=prev_embed,
  value=prev_embed
)
# Then use adjusted query to attend to working memory
```
**Parameters: O(hidden² × num_heads)** ← Quadratic in hidden dimension!

### V2 Lightweight (Matrix-based):
```python
binding_score[w] = Σ_{i,k} B[i,k] × (prev[i] - wm[w,k])²
```
**Parameters: O(L²)** ← Only quadratic in proposition length!

For typical values (L=3, hidden=32):
- V2 Heavy: O(32² × 4) = ~4,000 params
- V2 Light: O(3²) = 9 params
- **Ratio: 444x parameter reduction!**

---

## Performance Evidence

Transitive reasoning task (50 train, 20 test samples):

|                  | Params  | Train Loss | Test Loss | Generalization |
|------------------|---------|------------|-----------|----------------|
| V1 (baseline)    | 480     | 0.0053     | 0.0423    | baseline       |
| V2 Heavy         | 19,472  | 0.0366     | 0.0483    | -14.2% ⚠️      |
| V2 Lightweight   | 520     | 0.0051     | 0.0099    | **+76.6%** ✓   |

The binding matrix approach is:
- **37x fewer parameters** than attention
- **77% better generalization**
- **Preserves the efficiency advantage** over Transformers

---

## Key Insight: Why Binding Matrices Work Better

Logical variable binding is **SPARSE and STRUCTURED**:

Example: `father(X,Y) ∧ father(Y,Z) → grandfather(X,Z)`
- Format: `[arg0, rel, arg2] ∧ [arg0, rel, arg2]`
- Binding constraint: `premise1.arg2 = premise2.arg0` (only ONE pair!)

Multi-head attention with 4 heads learns a 4×(hidden_dim²) parameter function to essentially discover this **ONE sparse constraint**.

It's like using a sledgehammer to crack a nut - massive overkill!

The binding matrix directly encodes the structure:
"Most positions don't interact (B[i,k]=0), but a few specific ones do"

This SPARSE PRIOR matches the actual structure of logical rules, leading to:
- ✓ Faster learning (less parameters to search)
- ✓ Better generalization (stronger inductive bias)
- ✓ Greater interpretability (can read off binding patterns)

---

## Conclusion

V2 Lightweight completely **ELIMINATES multi-head attention**.

Instead, it uses a DIRECT, INTERPRETABLE binding matrix that:
- ✓ Explicitly represents "which positions should match"
- ✓ Has minimal parameter overhead (~2% increase)
- ✓ Generalizes better due to stronger inductive bias
- ✓ Is fully differentiable (matrix entries are learnable)
- ✓ Maintains the core efficiency advantage over Transformers

**Key realization:** Logical variable binding has KNOWN STRUCTURE (sparse position-wise constraints), so we should ENCODE that structure directly rather than making a neural network discover it from scratch.

This is a perfect example of **"inductive bias beats brute force learning"**!
