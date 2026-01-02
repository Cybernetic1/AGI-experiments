# Quick Start Guide

**Goal:** Train the convergence system on your dataset

---

## What You Have (Working!)

1. **Davidsonian Extractor** - Extracts events + thematic roles
2. **Forward Chainer** - Symbolic inference with common sense
3. **Integrated System** - End-to-end pipeline with learnable weights

---

## Quick Test

```bash
cd /home/yky/misc-programs/AGI-experiments
source venv/bin/activate
python convergence_system.py
```

---

## Next Steps

### 1. Test on Your Dataset (1 day)

```python
from convergence_system import ConvergenceSystem

system = ConvergenceSystem()

# Load your data
sentences = [
    "The cat sat on the mat",
    "John loves Mary",
    # ... your dataset
]

for sentence in sentences:
    result = system.forward(sentence)
    print(result['extracted'])  # Davidsonian triples
    print(result['inferred'])   # After inference
```

### 2. Add More Rules (1-2 days)

Edit `simple_forward_chainer.py`, add to `create_common_sense_rules()`:

```python
# Spatial reasoning
def rule_on_location(bindings):
    x = bindings['?x']
    y = bindings['?y']
    return [Fact(x, 'spatial_rel', f'above_{y}')]

rules.append(Rule(
    name="on_means_above",
    patterns=[
        {'entity': '?x', 'relation': 'on', 'value': '?y'}
    ],
    action=rule_on_location
))
```

### 3. Train Weights (2-3 days)

```python
import torch.optim as optim

system = ConvergenceSystem()
optimizer = optim.Adam(system.parameters(), lr=0.001)

for epoch in range(50):
    for sentence, gold_parse in train_data:
        # Forward pass
        result = system.forward(sentence)
        
        # Compute loss (you define this based on gold_parse)
        loss = compute_loss(result['inferred'], gold_parse)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.item()}")
```

---

## Key Files

- `davidsonian_extraction.py` - Extraction meta-rules (edit to add more)
- `simple_forward_chainer.py` - Inference rules (edit to add common sense)
- `convergence_system.py` - Main system (shouldn't need to edit)

---

## Expected Results

- **Week 1:** 60-70% accuracy (just meta-rules, no training)
- **Week 2:** 75-80% accuracy (after weight training)
- **Week 3:** 80-85% accuracy (with more rules added)

---

## Troubleshooting

**Issue:** Extraction missing patterns
- **Fix:** Add more meta-rules to `davidsonian_extraction.py`

**Issue:** Wrong inferences
- **Fix:** Add/fix rules in `simple_forward_chainer.py`

**Issue:** Slow inference
- **Fix:** Add rule indexing (RETE optimization)

---

## Documentation

See `IMPLEMENTATION_SUMMARY.md` for details
See `SESSION_SUMMARY.md` for theory

---

**You're ready to start training!** 🚀
