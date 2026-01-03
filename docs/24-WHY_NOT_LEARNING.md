# Critical Issue: Model Not Learning (2% Accuracy)

## Current Results
- Entity: 2.04% (basically random)
- Relation: 0.00%
- Value: 0.00%
- **Model is NOT learning**

## Root Cause Analysis

### Problem 1: Loss Scale Imbalance

The loss has 4 components with VASTLY different scales:

```python
loss = loss_parse + loss_generate + 0.5 * (loss_cycle_forward + loss_cycle_backward)

Typical values:
- loss_parse:           200,000-300,000  (MSE on propositions)
- loss_generate:        4-6              (cross-entropy on text)
- loss_cycle_forward:   4-6              (cross-entropy on text)
- loss_cycle_backward:  200,000-300,000  (MSE on propositions)

Total: ~400,000

The parse/cycle losses DOMINATE (99.99% of total)
The generate loss is INVISIBLE (0.001% of total)
```

**Result:** Model ignores text generation completely, focuses only on proposition reconstruction (which is too hard).

### Problem 2: MSE Loss on Discrete Outputs

```python
# We're doing:
loss_parse = MSE(predicted_propositions, target_propositions)

# But propositions are DISCRETE IDs:
predicted: [entity=42, relation=156, value=23]  # After argmax
target:    [entity=45, relation=160, value=20]

# MSE = (42-45)² + (156-160)² + (23-20)²
#     = 9 + 16 + 9 = 34

# This is meaningless! MSE doesn't make sense for categorical data.
```

Should use **cross-entropy** instead.

### Problem 3: Data Processing Issues

Looking at the clamping we added:
```python
entities = torch.clamp(entities, 0, self.num_entities - 1)
relations = torch.clamp(relations, 0, self.vocab_size - 1)
values = torch.clamp(values, 0, self.num_entities - 1)
```

**This means wrong IDs are being SILENTLY FIXED during training!**

If data has entity=505 but max is 500, we clamp to 500 (wrong entity).
The model learns on corrupted data.

### Problem 4: The Symmetric Architecture Assumption

The assumption that parsing and generation are truly "symmetric" might be wrong for this task:

- Parsing: Text (1500 dims) → Logic (3 discrete IDs)
- Generation: Logic (3 discrete IDs) → Text (1500 dims)

These are NOT symmetric! One is compression (many→few), other is expansion (few→many).

## Recommended Fixes

### Fix 1: Separate Loss Weights (CRITICAL)

```python
# Current:
loss = loss_parse + loss_generate + 0.5 * (loss_cycle_f + loss_cycle_b)

# Should be:
loss = 0.01 * loss_parse + loss_generate + 0.01 * (loss_cycle_f + loss_cycle_b)
#      ^^^^                               ^^^^
#      Scale down MSE losses to match cross-entropy scale

# Or better: normalize each loss
loss_parse_norm = loss_parse / 100000  # Bring to ~1-5 range
loss_generate_norm = loss_generate      # Already in ~1-5 range
loss = loss_parse_norm + loss_generate_norm + ...
```

### Fix 2: Use Cross-Entropy for Propositions

```python
# Instead of MSE on propositions, treat each component as classification:

# Predict distributions
entity_logits = model.predict_entity(...)      # (batch, num_entities)
relation_logits = model.predict_relation(...)  # (batch, vocab_size)
value_logits = model.predict_value(...)        # (batch, num_entities)

# Cross-entropy loss
loss_entity = F.cross_entropy(entity_logits, target_entities)
loss_relation = F.cross_entropy(relation_logits, target_relations)
loss_value = F.cross_entropy(value_logits, target_values)

loss_parse = loss_entity + loss_relation + loss_value
```

### Fix 3: Fix Data Processing

The dataset creation is generating invalid IDs. Need to:
1. Ensure all IDs are in valid range BEFORE creating dataset
2. Remove clamping (it hides bugs)
3. Add validation to catch out-of-range IDs

### Fix 4: Simplify Architecture (Start Simple)

The symmetric cycle consistency might be too complex to learn with limited data.

**Try supervised learning ONLY first:**
```python
# Just learn: text → logic (no generation, no cycle)
loss = cross_entropy(predicted_logic, target_logic)
```

Once this works (>50% accuracy), THEN add generation and cycle consistency.

## Quick Test: Baseline Model

Let me create a simple baseline that SHOULD learn:

```python
class SimpleParser(nn.Module):
    """Dead simple: just predict entity from text."""
    
    def __init__(self, vocab_size, num_entities):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 128, batch_first=True)
        self.classifier = nn.Linear(128, num_entities)
    
    def forward(self, text_ids):
        emb = self.embedding(text_ids)
        out, _ = self.lstm(emb)
        out = out[:, -1, :]  # Take last token
        logits = self.classifier(out)
        return logits

# Train with simple cross-entropy
logits = model(text_ids)
loss = F.cross_entropy(logits, target_entities)
```

**If this gets >50% accuracy, the data is fine but our architecture is wrong.**
**If this ALSO fails, the data is corrupted.**

## Immediate Action Plan

1. **Test the baseline model** (see if ANYTHING can learn from this data)
2. **If baseline works:** Fix our architecture (loss scaling, cross-entropy)
3. **If baseline fails:** Fix data processing pipeline

I'll create the baseline test script now.
