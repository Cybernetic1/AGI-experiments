# Binding Mechanism Walkthrough

This document provides a detailed step-by-step walkthrough of how the binding matrix works in Logic Transformer V2 Lightweight.

---

## Example Task

Learn transitive reasoning: `father(X,Y) ∧ father(Y,Z) → grandfather(X,Z)`

### Working Memory:
```
[0] = [john,  father, bob]     ← premise1 should match this
[1] = [bob,   father, alice]   ← premise2 should match this  
[2] = [sue,   mother, bob]
[3] = [alice, child,  bob]
...
```

---

## Premise 1 Matching (Cylindrification Only - same as V1)

### Rule Setup:
```
constants[0]: [?, father, ?]  
γ[0] = [0.9, 0.0, 0.9]
         ^    ^     ^
        var  const var
```

### Match Score Computation:

For each WM proposition `w`, compute:
```
score[w] = (1-γ[0])×(const[0]-wm[w,0])² + (1-γ[1])×(father-wm[w,1])² + (1-γ[2])×(const[2]-wm[w,2])²
         = 0.1×(anything)² + 0.9×(father-wm[w,1])² + 0.1×(anything)²
```

Results:
```
score[0] = 0.1×(...) + 0.9×(father-father)² + 0.1×(...) = LOW  ← good match!
score[1] = 0.1×(...) + 0.9×(father-father)² + 0.1×(...) = LOW  ← also matches
score[2] = 0.1×(...) + 0.9×(father-mother)² + 0.1×(...) = HIGH ← bad match
```

### Soft Attention:
```
attention = softmax(-scores) = [0.3, 0.3, 0.1, 0.1, ...]
```

### Selected Proposition:
```
selected = weighted_avg(WM, attention) ≈ [john, father, bob]
```

**Premise 1 Selection: [john, father, bob]**
- Position 0: john
- Position 1: father
- Position 2: bob

---

## Premise 2 Matching (V2 Lightweight - WITH Binding Constraint)

### Rule Setup:
```
constants[1]: [?, father, ?]  
γ[1] = [0.9, 0.0, 0.9]
```

### Step 1: Base Cylindrification Score

Same as premise 1:
```
cyl_score[0] = LOW  (matches father)
cyl_score[1] = LOW  (matches father)
cyl_score[2] = HIGH (doesn't match father)
```

### Step 2: Binding Matrix

```
Binding Matrix B:
                  premise2.position
                    0    1    2
  premise1.pos 0 [  0    0    0  ]
               1 [  0    0    0  ]
               2 [  1.0  0    0  ]  ← B[2,0] = "arg2 of p1 should match arg1 of p2"
```

**Interpretation:** `B[2,0] = 1.0` means:
- Previous selection's position 2 (bob) should match current proposition's position 0

### Step 3: Binding Penalty Computation

Previous selection: `[john, father, bob]`
- Position 0: john
- Position 1: father  
- Position 2: **bob** ← This is what we need to match!

For each WM proposition `w`:
```
For i=2, k=0 (only non-zero entry in B):
  binding_penalty[w] = gate × B[2,0] × (prev_selection[2] - wm[w,0])²
                     = 0.5 × 1.0 × (bob - wm[w,0])²
```

Computing for each proposition:
```
binding_penalty[0] = 0.5 × (bob - john)²  = HIGH  ← violates binding!
binding_penalty[1] = 0.5 × (bob - bob)²   = 0     ← satisfies binding! ✓
binding_penalty[2] = 0.5 × (bob - sue)²   = HIGH
```

### Step 4: Total Score

```
total_score = cylindrification_score + binding_penalty
```

Results:
```
score[0] = LOW  + HIGH = HIGH  (matches father, but wrong subject)
score[1] = LOW  + 0    = LOW   (matches father AND bob matches!) ✓
score[2] = HIGH + HIGH = VERY HIGH
```

### Step 5: Soft Attention

```
attention = softmax(-scores) ≈ [0.1, 0.7, 0.05, ...]
```

### Selected Proposition:
```
selected ≈ [bob, father, alice]
```

**Premise 2 Selection: [bob, father, alice]**

---

## Variable Capture & Conclusion

### Captured Variables:

Via body networks:
```
From premise 1: X=john, Y=bob    (from [john, father, bob])
From premise 2: Y=bob, Z=alice   (from [bob, father, alice])
```

**Notice: Y is consistent!** (bob in both)

### Head Network Output:
```
[john, grandfather, alice]
```

**SUCCESS!** The binding matrix enforced Y consistency without explicit unification.

---

## Comparison: What if B[2,0] = 0? (No Binding - like V1)

Without binding constraint, premise 2 matching:
```
score[0] = LOW  (matches father)
score[1] = LOW  (matches father)
```

Both have equal scores! Attention might pick either one randomly.

Could select:
- `[john, father, bob]` again (wrong!)
- `[sue, mother, ...]` (wrong!)
- Any proposition with "father" in position 1

The binding matrix **B[2,0]=1.0** creates the **ASYMMETRY** that breaks the tie and forces premise 2 to select a proposition whose subject matches premise 1's object.

---

## The Essence of Cross-Premise Variable Binding

The binding matrix allows the network to learn:

**"When matching premise j+1, prefer WM propositions whose position k contains the same value that premise j selected at position i"**

This is implemented as a soft penalty term that:
1. Measures the mismatch: `(prev[i] - current[k])²`
2. Weights it by binding strength: `B[i,k]`
3. Adds it to the match score
4. Lets softmax naturally favor low-penalty matches

All fully differentiable - gradients flow back to `B[i,k]` to learn the correct binding patterns!

---

## Key Advantages

1. **Explicit Structure**: Binding constraints are represented directly in B, not hidden in attention weights
2. **Sparse**: Most entries of B are zero (only specific position pairs need to match)
3. **Interpretable**: Can inspect B to see which variables are shared across premises
4. **Efficient**: Only L² parameters instead of hidden_dim² × num_heads
5. **Strong Prior**: Encodes the structural assumption that logical binding is position-specific and sparse

This is why V2 Lightweight outperforms both V1 (no binding) and V2 Heavy (over-parameterized attention) on transitive reasoning tasks!
