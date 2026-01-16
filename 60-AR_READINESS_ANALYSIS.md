# Autoregressive Extension: Readiness Analysis

**Date:** 2026-01-16  
**Context:** After achieving perfect logical inference (MSE 0.0000) with 32K parameters  
**Question:** Can we extend to AR (autoregressive language modeling)?

---

## Current Status: What We Have ✅

### 1. **Strong Foundation**
- ✅ DLN with 32K parameters achieving perfect inference
- ✅ Davidsonian parsing (text → logic propositions)
- ✅ Symbolic reasoning engine (rule-based inference)
- ✅ Hybrid consolidation (neural learning + symbolic reasoning)
- ✅ 171× label expansion from symbolic rules
- ✅ Mini-batch training infrastructure

### 2. **Existing AR Components**
```python
# In dln.py (line 56)
self.ar_head = nn.Linear(self.prop_dim, len(predicates))
```
- Already has auxiliary AR head for predicate prediction
- Currently used as auxiliary loss (with `--no-ar-aux` to disable)
- Predicts predicate class from premise encoding

### 3. **Existing AR Experiments**
- `train_semantic_ar_v2.py` - Semantic AR with Davidsonian parsing
- `hybrid_ar_model.py` - Hybrid AR architecture
- `experiment_ar_training.py` - AR training experiments

---

## What AR Requires: Gap Analysis

### Core Requirement: Sequence Generation
**Status:** ⚠️ Partially implemented, needs completion

#### What We Have:
```
Input: Premises (logic propositions)
       ↓
   DLN Encoder (encode_premises)
       ↓
   AR Head (predict next predicate)
       ↓
Output: Predicate class distribution
```

#### What We Need for Full AR:
```
Input: Text sequence (tokens)
       ↓
   Token Embeddings
       ↓
   Transformer/RNN (maintain context)
       ↓
   Logic Layer (optional: inject symbolic reasoning)
       ↓
   Output Head (predict next token)
       ↓
Output: Token distribution → Generate text
```

---

## Key Obstacles & Solutions

### Obstacle 1: **Architecture Mismatch**
**Problem:** Current DLN is designed for logic inference, not sequence generation
- Takes propositions → outputs truth values
- No sequential modeling (no attention, no recurrence)
- No token-level generation

**Solutions:**
1. **Hybrid approach (recommended):**
   ```
   Text → Tokenizer → Seq Model → Logic Layer → Output
   ```
   - Small Transformer/LSTM for sequential context
   - DLN for logical reasoning (inject as auxiliary signal)
   - Combine both for next-token prediction

2. **Pure logical AR:**
   ```
   Text → Davidsonian Parse → Propositions → DLN → Next Prop
   ```
   - Stay fully in logical domain
   - Generate logical forms, not tokens
   - Convert back to text as post-processing

3. **Extend DLN for sequences:**
   - Add positional encoding
   - Add attention mechanism
   - Make it work on token sequences directly

### Obstacle 2: **Training Signal**
**Problem:** Current training uses symbolic labels (truth values), not AR loss

**Current training:**
```python
# Supervised: given facts + rules → predict truth values
loss = MSE(predicted_truth, symbolic_truth)
```

**AR training needs:**
```python
# Unsupervised: given tokens[0:t] → predict tokens[t+1]
loss = CrossEntropy(predicted_token, actual_next_token)
```

**Solution:** Dual objective
```python
# Combined loss
ar_loss = CrossEntropy(pred_token, next_token)       # Generation
logic_loss = MSE(pred_truth, symbolic_truth)        # Reasoning
total_loss = ar_loss + λ * logic_loss
```

### Obstacle 3: **Data Format**
**Problem:** Current pipeline: Text → Logic → Labels (no token sequences)

**What we have:**
```
Story: "Lily found a toy."
  ↓ Davidsonian parsing
Facts: [Proposition("finds", ("lily", "toy"), 1.0)]
  ↓ Symbolic inference
Labels: {("finds_inferred", ("lily", "toy")): 1.0, ...}
```

**What AR needs:**
```
Text: "Lily found a toy."
Tokens: [23, 145, 89, 12, 678]
Training: predict token[i+1] from tokens[0:i]
```

**Solution:** Dual processing
```python
# Keep both paths
text → tokens → AR model → next token
text → logic → DLN → reasoning signal (auxiliary)
```

---

## "Tricks" (Heuristics) for AR Convergence

Based on your logical inference success, here are analogous tricks for AR:

### 1. **Rule-Primed Initialization** ✨
**Logical inference analogue:** Rule injection accelerated convergence

**AR version:**
```python
# Pre-train on synthetic rule-based data
rules = ["If X finds Y, then X has Y", ...]
synthetic_stories = generate_from_rules(rules)
pretrain_on(synthetic_stories)  # Quick convergence on structure
finetune_on(real_stories)       # Learn naturalistic patterns
```

**Why it works:** Rules provide inductive bias, reduce search space

### 2. **Symbolic Scaffolding** ✨
**Logical inference analogue:** 171× label expansion from symbolic rules

**AR version:**
```python
# Use logic as auxiliary signal during AR training
def forward(tokens):
    # Standard AR path
    hidden = transformer(tokens)
    token_logits = output_head(hidden)
    
    # Symbolic auxiliary path
    props = parse_to_logic(tokens)
    logic_signal = dln(props)
    
    # Combine: logic guides token generation
    combined = token_logits + α * logic_signal
    return combined
```

**Why it works:** Logic provides high-level structure, prevents nonsense

### 3. **Hierarchical Training** ✨
**Logical inference analogue:** Train on labels at multiple granularities

**AR version:**
```python
# Stage 1: Train on logical forms only
model.learn_logic(propositions)  # Fast, structured

# Stage 2: Add token generation
model.learn_tokens(text, use_logic=True)  # Guided by logic

# Stage 3: End-to-end fine-tuning
model.finetune(text)  # Full flexibility
```

**Why it works:** Curriculum learning, easier → harder

### 4. **Proposition-Level Chunking** ✨
**Logical inference analogue:** Propositions as atomic units

**AR version:**
```python
# Don't generate token-by-token, generate proposition-by-proposition
chunks = [
    "Lily found a toy",      # Prop 1
    "The toy was red",       # Prop 2
    "She was very happy"     # Prop 3
]
# Train: predict next proposition, then expand to tokens
```

**Why it works:** Higher-level units → faster learning, better coherence

### 5. **Symbolic Constraint Regularization** ✨
**Logical inference analogue:** Symbolic engine ensures consistency

**AR version:**
```python
# Add consistency loss during training
def consistency_loss(generated_text):
    props = parse(generated_text)
    violations = check_logic_violations(props)
    return penalty(violations)

total_loss = ar_loss + β * consistency_loss
```

**Why it works:** Prevents logical contradictions, improves quality

---

## Recommended Path Forward

### Phase 1: **Hybrid AR Prototype** (1-2 weeks)
Build minimal viable AR system using existing components:

```python
class HybridARModel(nn.Module):
    def __init__(self):
        # Small Transformer for sequences
        self.transformer = nn.TransformerEncoder(
            d_model=128, nhead=4, num_layers=2
        )
        # Your proven DLN for logic
        self.dln = SimpleDLN(predicates, args, embed_dim=32)
        # Output head
        self.output = nn.Linear(128 + 32, vocab_size)
    
    def forward(self, tokens, logic_props=None):
        # Sequential encoding
        seq_repr = self.transformer(embed(tokens))
        
        # Optional: inject logic signal
        if logic_props:
            logic_signal = self.dln.encode_premises(logic_props)
            combined = torch.cat([seq_repr, logic_signal], -1)
        else:
            combined = seq_repr
        
        return self.output(combined)
```

**Deliverable:** Generate coherent short sentences with logical consistency

### Phase 2: **Rule-Primed Training** (1 week)
Implement symbolic scaffolding:

1. Generate synthetic training data from rules
2. Pre-train on synthetic (fast convergence)
3. Fine-tune on real TinyStories
4. Measure convergence speedup vs vanilla Transformer

**Deliverable:** Demonstrate faster convergence with rule priming

### Phase 3: **Full Hybrid System** (2-3 weeks)
Integrate all components:

1. Dual-objective training (AR + logic)
2. Proposition-level generation
3. Symbolic consistency regularization
4. End-to-end evaluation

**Deliverable:** Full system beating baseline Transformers on TinyStories

---

## Estimated Effort

| Component | Difficulty | Time | Priority |
|-----------|-----------|------|----------|
| Basic hybrid AR architecture | Medium | 3-5 days | HIGH |
| Dual-objective training | Easy | 1-2 days | HIGH |
| Rule-primed initialization | Medium | 2-3 days | HIGH |
| Proposition-level generation | Hard | 5-7 days | MEDIUM |
| Symbolic consistency reg | Medium | 2-3 days | MEDIUM |
| Full integration & eval | Hard | 5-7 days | HIGH |

**Total estimate:** 3-4 weeks for complete system

---

## Open Questions

1. **Architecture:**
   - Should logic be auxiliary signal or primary path?
   - How to balance AR loss vs logic loss (λ)?
   - Token-level or proposition-level generation?

2. **Training:**
   - How much synthetic data for rule priming?
   - Curriculum: logic-first or AR-first?
   - Mini-batch size for sequences?

3. **Evaluation:**
   - How to measure "logical consistency" of generated text?
   - Perplexity vs semantic coherence trade-off?
   - Comparison metric to Transformers?

4. **Scalability:**
   - Will 32K params be enough for AR?
   - Need to scale up Transformer component?
   - Memory requirements for attention?

---

## Conclusion

**Status:** 🟡 **Ready with caveats**

You have:
- ✅ Strong logical reasoning foundation
- ✅ Excellent parameter efficiency
- ✅ Symbolic scaffolding infrastructure
- ⚠️ Need to add sequential modeling
- ⚠️ Need AR training objective

**Recommended:** Start with **Phase 1 (Hybrid AR Prototype)** to validate the approach before committing to full implementation.

The "tricks" that worked for logical inference (rule injection, symbolic expansion, hierarchical training) should transfer well to AR, giving you a unique advantage over pure neural approaches.

**Next concrete step:** Implement the minimal `HybridARModel` and run initial experiments on small TinyStories subset (100-500 stories).
