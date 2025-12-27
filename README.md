# AGI Experiments: Logic Networks for Language Understanding

## Overview

This repository explores scaling logic-based neural networks from toy domains (Tic-Tac-Toe) to language understanding and AGI. The approach combines:

- **Logic networks** with fuzzy unification and cylindrification (variables)
- **Autoregressive (AR) learning** for world model/prediction
- **Reinforcement Learning (RL)** for goal-directed behavior
- **Vector Quantization (VQ)** for tractable propositional generation
- **Entity memory** for tracking individuals across contexts

## Motivation

Traditional LLMs excel at pattern matching but lack explicit reasoning capabilities. This project investigates whether **learned logic rules** can emerge from combining:
1. Predictive learning (autoregression on text)
2. Task-oriented learning (RL on reasoning/QA tasks)

## Documentation

**Core Concepts**:
- [Logic Network](docs/LOGIC_NETWORK.md) - Core reasoning architecture (fuzzy rules, concepts)
- [Memory Architecture](docs/MEMORY_ARCHITECTURE.md) - WM, LTM, and entity registry
- [Variables and Entities](docs/VARIABLES_AND_ENTITIES.md) - Scoping, binding, and name learning
- [Training Strategy](docs/TRAINING_STRATEGY.md) - NLP preprocessing and AR+RL protocol

**Technical Details**:
- [Scaling Challenges](docs/SCALING_CHALLENGES.md) - Rule retrieval, output space, efficiency
- [First Experiment](docs/FIRST_EXPERIMENT.md) - Dataset size and preparation guide

**Legacy Documents** (being phased out):
- [SCALING_TO_AGI.md](SCALING_TO_AGI.md) - Original technical discussions
- [VARIABLES_VS_ENTITIES.md](VARIABLES_VS_ENTITIES.md) - Original scoping clarifications

## Key Innovations from TTT Experiments

### Successful TTT Results
- ✓ Hierarchical logic network (shared concepts + dual heads)
- ✓ 3-phase training protocol (AR → RL frozen → joint)
- ✓ 100% win rate vs optimal opponent
- ✓ Interpretable learned rules showing position/player awareness

See `../logic-autoencoder-ttt/HIERARCHICAL_RESULTS.md` for full details.

### Scaling Challenges & Solutions

See [SCALING_TO_AGI.md](SCALING_TO_AGI.md) for comprehensive discussion of:

1. **Rule-to-state matching**: Use learnable embedding ρ + approximate nearest neighbors (treat as recommendation problem)
2. **Output space explosion**: Use VQ codebook (8K proposition patterns) + optional autoregressive refinement
3. **Entity tracking**: Integer IDs + property storage (not unique embeddings)
4. **Variable scoping**: γ parameters for constant/variable distinction

## Architecture Components

### Core Logic Network (`neural_logic_core.py`)

Implements fuzzy logic rules with:
- **Fuzzy unification**: Soft attention-based premise matching
- **Cylindrification (γ parameters)**: γ≈0 acts as variable, γ≈1 acts as constant
- **Concept extraction**: Learned rules extract high-level features

```python
rule = LogicRule(
    num_premises=3,
    premise_arity=3,
    symbol_dim=768  # For language embeddings
)
concepts = rule(propositions)  # Extract patterns from propositions
```

### Hierarchical Network (`hierarchical_logic_network.py`)

Combines AR and RL objectives:
- **Shared logic rules**: Extract concepts from working memory
- **AR head**: Predict next state/proposition (world modeling)
- **RL head**: Q-values for action selection (goal achievement)

```python
network = HierarchicalLogicNetwork(
    num_rules=8,
    concept_dim=64,
    ar_output_dim=8192,  # VQ codebook size
    rl_output_dim=action_space
)
```

## Roadmap

### Phase 1: Simple Text Domain (Next)
- [ ] Dataset: Children's books / simple narratives
- [ ] Task: Next sentence prediction (AR) + question answering (RL)
- [ ] Architecture: VQ proposition encoder + logic network
- [ ] Baseline: Compare to GPT-2 scale models

### Phase 2: Entity Tracking & Co-reference
- [ ] Implement entity memory (integer IDs + properties)
- [ ] Variable binding across propositions
- [ ] Test on multi-entity scenarios (e.g., "two cats" example)
- [ ] Evaluate co-reference resolution accuracy

### Phase 3: Rule Retrieval
- [ ] Implement learnable ρ mapping (rules → state space)
- [ ] Train with contrastive learning
- [ ] Scale to 1000+ rules
- [ ] Measure retrieval precision/recall

### Phase 4: Complex Reasoning
- [ ] Multi-hop reasoning tasks (e.g., bAbI tasks)
- [ ] Hierarchical propositions (propositions about propositions)
- [ ] Meta-rules (rules that modify rules)
- [ ] Compare to transformer baselines

### Phase 5: AGI-Scale Experiments
- [ ] Train on web-scale text corpora
- [ ] Evaluate on reasoning benchmarks (ARC, MMLU, etc.)
- [ ] Interpretability analysis (which rules activate when)
- [ ] Transfer learning across domains

## Key Open Questions

### 1. Variables vs Entities (Critical!)

**The Scoping Problem**:
- **Variables (?X)**: Scope limited to single rule, unified during matching
- **Entities (cat_1)**: Persistent across scenario, tracked in entity memory

**Current confusion**: How do these interact? Options:
- A) Variables bind to entity IDs within rule execution
- B) Entity creation is a separate mechanism from variable binding
- C) Both are aspects of the same unification mechanism

**Need to clarify**: 
- When is a new entity created vs. referencing existing?
- How do rules "know" to bind ?X to entity 0 vs. creating entity 3?
- What is the training signal for entity creation decisions?

### 2. Proposition Granularity

How many propositions needed for complex situations?
- Simple: 1-3 propositions
- Medium: 5-10 propositions  
- Complex: 20+ propositions

Trade-off: More propositions = more expressiveness but harder to learn

### 3. VQ Codebook Size

- 512 codes: Sufficient for simple patterns?
- 8K codes: Sweet spot (DALL-E, VQ-VAE-2)?
- 64K codes: Necessary for language diversity?

Needs empirical validation on text data.

### 4. Training Data Requirements

How much data needed for rules to emerge?
- TTT: ~10K games sufficient
- Simple text: 100K sentences?
- Complex reasoning: 1M+ examples?

## Implementation Notes

### Environment Setup

```bash
cd /home/yky/misc-programs/AGI-experiments
python -m venv venv
source venv/bin/activate
pip install torch numpy matplotlib
```

### Starting Point

The core logic network is domain-agnostic and ready for text:
```python
from neural_logic_core import LogicNetwork

# Initialize for language
logic_net = LogicNetwork(
    num_rules=16,
    num_premises_per_rule=4,
    premise_arity=3,
    symbol_dim=768,  # Match text embeddings (e.g., BERT)
    concept_dim=128
)

# Process text propositions
propositions = extract_propositions(text)  # To implement
concepts = logic_net(propositions)
```

## References

### Related Work
- **REALM** (Google): Retrieval-augmented language models
- **VQ-VAE-2**: Vector quantization for image generation
- **Neural Theorem Provers**: End-to-end differentiable logic
- **Neuro-symbolic AI**: Combining neural nets with symbolic reasoning

### Key Papers
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- "Generating Diverse High-Fidelity Images with VQ-VAE-2"
- "Neural Module Networks"
- "Learning to Reason: End-to-End Module Networks for Visual Question Answering"

## Contact & Collaboration

This is an experimental research project. Ideas, suggestions, and collaborations welcome!

## License

(To be determined)
