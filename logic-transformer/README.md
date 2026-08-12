# Logic Transformer (LT)

This directory contains the standalone, bare-bones implementation of the **Logic Transformer (LT)**, also referred to as the Differentiable Logic Network (DLN). 

It is designed to be fully self-contained so it can be easily ported to new repositories.

## Tri-System Neurosymbolic Architecture

This LT module acts as **System 1** in a broader tri-system cognitive architecture:

1. **System 1 (Logic Transformer):** Fast, differentiable, fuzzy logic rule application using gradient-based representation learning. It processes continuous semantic working memory and unifies patterns softly.
2. **System 2 (Rete Engine):** An exact forward-chaining engine designed for deterministic, discrete, and efficient pattern matching over established facts.
3. **System 3 (ProbLog):** A probabilistic backward-chaining inference engine intended for deep, complex reasoning and goal-directed planning.

## Usage

The module is implemented in pure PyTorch using highly optimized, loop-free tensor operations (`torch.einsum`).

```python
import torch
from logic_transformer import LogicTransformer

# Hyperparameters
prop_length = 64   # Dimension of semantic embeddings
max_props = 20     # Max statements in working memory
output_dim = 64    # Dimension of the conclusion

# Initialize the LT (System 1)
lt = LogicTransformer(
    prop_length=prop_length,
    max_props=max_props,
    output_dim=output_dim,
    num_rules=8,       # Number of learned logical rules
    num_premises=2,    # Premises per rule (e.g., IF A and B...)
    var_slots=3        # Variables available for fuzzy unification
)

# Example Working Memory (Batch Size = 4, 15 facts in memory)
working_memory = torch.randn(4, 15, prop_length)

# Forward pass (Fast, differentiable inference)
conclusions = lt(working_memory, temperature=1.0)
print("Inference output shape:", conclusions.shape) # (4, 64)
```

## Differentiable Mechanics

- **Variables vs Constants:** LT utilizes soft slot-selectors to route attention across working memory, dynamically binding inputs to internal variables.
- **Fuzzy Unification:** Rule conditions are evaluated via L2 distance in continuous space, making all operations (matching, substitution, projection) fully differentiable.
- **Gradient Flow:** Because the forward pass consists entirely of softmax, einsum, and element-wise additions, backpropagation correctly updates both the rule structures and the semantic representations.