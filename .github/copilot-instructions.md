# AGI Experiments - Copilot Instructions

## Project Overview

This is a research project building towards AGI by combining differentiable neural networks with symbolic logic reasoning. The core innovation is the **Differentiable Logic Network (DLN)** - making classical logic-based AI systems differentiable and trainable via gradient descent.

**Key architectural components:**
- **DLN**: Fuzzy logic with cylindrification parameters (γ) that bridge variables and constants
- **Autoregressive (AR)** learning for world modeling (like LLMs)
- **Reinforcement Learning (RL)** for goal-directed behavior
- **Hybrid symbolic-neural** architecture for knowledge injection and long-term memory
- **Neo-Davidsonian semantics** for natural language parsing into logical forms
- **Genetic Algorithm (GA) + ILP** for populating initial rule bases

## Build and Test Commands

### Environment Setup
```bash
# Activate virtual environment (already exists)
source venv/bin/activate

# Install dependencies (if needed)
pip install -r requirements.txt
```

### Running Tests
Tests are standalone Python scripts (not pytest). Run directly with the venv activated:

```bash
# Run a specific test
python test_dln_babi.py
python test_ga_semantic_ar.py
python test_ilp_comparison.py

# Run comparison benchmarks
python compare_ilp_rules.py
python compare_babi_fair.py
python compare_ga_baseline.py

# Quick smoke test
python convergence_system.py
```

### Shell Scripts
```bash
# Check training status
./check_training.sh

# Run experiments (various configurations exist)
./run_experiment.sh
./start_training.sh
```

### Training Commands
```bash
# Semantic AR training
python train_semantic_ar.py

# Hybrid GA+ILP+DLN training (full pipeline)
python train_hybrid_semantic_ar.py \
  --discovery-stories 100 \
  --training-stories 500 \
  --ga-generations 10

# DLN training on specific tasks
python train_dln_semantic_ar.py
python train_dln_unification_task.py
```

## Architecture Conventions

### File Organization

**Core logic modules:**
- `logic_core.py` - Base symbolic logic (Proposition, Rule, SymbolicEngine)
- `neural_logic_core.py` - DLN with fuzzy unification and cylindrification
- `dln.py` - SimpleDLN lightweight stub for quick experiments
- `symmetric_logic_network.py` - Symmetric DLN architecture
- `hierarchical_logic_network.py` - Combines AR and RL objectives

**Natural language processing:**
- `davidsonian_extraction.py` - NL → logical form (event-based)
- `entity_registry.py` - Entity tracking across discourse

**Rule discovery:**
- `core/ilp_algorithms.py` - Inductive Logic Programming (frequency, FOIL, confidence)
- `genetic_logic_rules.py` - Genetic algorithm for rule evolution
- `hybrid_ga_ilp_dln.py` - Hybrid ILP+GA+DLN integration

**Pipelines:**
- `pipelines/tinystories_pipeline.py` - TinyStories dataset loading and processing
- `pipelines/benchmark_suite.py` - Benchmark comparisons

**Infrastructure:**
- `rule_store.py` - Rule persistence and management
- `rule_injection.py` - Direct knowledge injection into models
- `label_utils.py` - Label generation from symbolic rules
- `core/train_utils.py` - Training/evaluation utilities

### Key Abstractions

#### Proposition
```python
# From logic_core.py
Proposition(predicate: str, args: Tuple[str, ...], truth: float)
# Example: Proposition("on", ("cat", "mat"), 1.0)
```

#### Rule
```python
# Symbolic rule structure
Rule(
    premises: List[Proposition],  # patterns with ?variables
    conclusion: Proposition,
    weight: float  # confidence/strength
)
```

#### Cylindrification (γ parameters)
The DLN's key innovation - γ ∈ [0,1] controls constant vs variable behavior:
- **γ ≈ 0**: acts as constant (match specific value)
- **γ ≈ 1**: acts as variable (capture any value)

These are learned parameters in `neural_logic_core.py` that enable differentiable unification.

### Dataset Conventions

**TinyStories format** (in `data/processed/tinystories_train.json`):
```json
{
  "text": "The cat sat on the mat.",
  "facts": [
    {"subject": "cat", "relation": "on", "object": "mat"}
  ]
}
```

**bAbI tasks**: Used for reasoning benchmarks (see `test_dln_babi.py`)

**Fact limits**: Many scripts use `max_facts` parameters (e.g., 10000, 50000) to control memory usage

### Naming Patterns

- `test_*.py` - Standalone test/experiment scripts
- `compare_*.py` - Benchmark comparison scripts
- `train_*.py` - Training scripts for specific architectures
- `*_pipeline.py` - Data processing pipelines
- Functions starting with `_` are internal utilities (e.g., `_collect_labels`, `_train_on_labels`)

### Training Conventions

**Semantic AR objective**: Train on logical similarity, not token matching
- Loss computed via graph edit distance between logical propositions
- Core innovation distinguishing this from standard language models

**3-phase training protocol** (from TicTacToe experiments):
1. AR pretraining (world model)
2. RL with frozen AR (goal achievement)
3. Joint fine-tuning

**Hybrid approach**: Symbolic rules fire alongside differentiable rules
- Symbolic: deterministic, gradient-transparent
- Neural: soft weights, fully differentiable

### Common Patterns

**Entity canonicalization**:
```python
def _canon(name: str) -> str:
    """Convert entity names to canonical form via registry"""
    if reg is None:
        return name
    eid = reg.get_or_create_entity(name)
    ent = reg.get_entity(eid)
    return ent.name.lower() if ent else name
```

**Label collection**: Rules infer new facts from existing facts
```python
from label_utils import _collect_labels
labels = _collect_labels(symbolic_engine, facts, rules, max_iters=1)
```

**Training loop structure**:
```python
from core.train_utils import _train_on_labels, _eval_on_labels
dln = SimpleDLN(predicates, args, embed_dim=64)
train_loss = _train_on_labels(dln, train_labels, epochs=100, lr=0.001)
eval_loss = _eval_on_labels(dln, eval_labels)
```

## Documentation Structure

**Quick references:**
- `README.md` - Main overview and roadmap
- `48-QUICK_START.md` - Getting started guide
- `76-HYBRID_ARCHITECTURE_GUIDE.md` - Complete integration guide
- `19-QUICK_REFERENCE.md` - Architecture quick reference

**Core concepts** (in `docs/`):
- `08-LOGIC_NETWORK.md` - DLN architecture
- `03-MEMORY_ARCHITECTURE.md` - Working memory, long-term memory
- `10-VARIABLES_AND_ENTITIES.md` - Variable scoping and entity tracking
- `04-TRAINING_STRATEGY.md` - AR+RL training protocol

**Technical deep-dives** (numbered `.md` files):
- `42-ARCHITECTURE_SYNTHESIS.md` - Architecture overview
- `43-GRADIENT_FLOW_THROUGH_SYMBOLIC.md` - How gradients flow through symbolic rules
- `53-AGI_ARCHITECTURE_MODULES.md` - Module reference
- `65-ILP_MODULE_DESIGN.md` - ILP implementation details
- `75-GA_ILP_DLN_INTEGRATION.md` - Hybrid discovery architecture

**Session summaries**: Files like `47-SESSION_SUMMARY.md`, `51-NL_PARSING_SESSION_SUMMARY.md` document research progress

## Development Notes

### Python Environment
- Uses `venv` (virtual environment in `venv/`)
- No conda/poetry - just pip + requirements.txt
- Python 3.8+ required (some dependencies like Experta need 3.8)

### GPU Support
- Code supports both CPU and GPU (uses `torch.device`)
- GPU setup docs: `14-GPU_SETUP_README.md`, `15-GPU_DEPLOYMENT_CHECKLIST.md`
- Training logs go to `checkpoints/`

### Data Loading
- TinyStories dataset expected at `data/processed/tinystories_train.json`
- Use `download_tinystories.py` and `preprocess_tinystories.py` to prepare data
- Other preprocessors: `preprocess_babi.py`, `preprocess_task*.py`

### Common Gotchas

**Variables in rules**: Use `?x`, `?y` prefix convention for logic variables
```python
# Correct
Proposition("agent", ("?e", "?x"), 1.0)  # ?e and ?x are variables

# Wrong - without ? these are constants
Proposition("agent", ("e", "x"), 1.0)
```

**Entity tracking**: Use `PersistentEntityRegistry` for cross-sentence coreference
```python
from entity_registry import PersistentEntityRegistry
registry = PersistentEntityRegistry()
entity_id = registry.get_or_create_entity("the cat")
```

**Rule application**: Symbolic rules need SymbolicEngine
```python
from logic_core import SymbolicEngine
engine = SymbolicEngine()
engine.add_fact(fact)
engine.add_rule(rule)
new_facts = engine.apply_rules()  # Forward chaining
```

**Memory management**: Use fact limits to prevent explosion
```python
# Always specify max_facts when loading data
facts = load_tinystories_facts(max_stories=100, max_facts=10000)
```

### Output Locations
- Training checkpoints: `checkpoints/`
- Experiment outputs: `outputs/` (various subdirectories)
- Training logs: `training.log`
- Test registries: `test_registry.json`, `test_registry_embeddings.pt`

## Key Research Questions

From README.md and architectural docs:

1. **Variables vs Entities**: How do logic variables (?X) interact with persistent entities (cat_1)?
2. **Proposition granularity**: How many propositions needed for complex reasoning?
3. **VQ codebook size**: 512 vs 8K vs 64K codes for proposition patterns
4. **Rule retrieval at scale**: Using learnable embeddings ρ + approximate nearest neighbors
5. **Reflection**: Can the system extract its own rules from experience?

## Experimental Results to Reference

- **TicTacToe (TTT)**: Solved in ~1 minute using dihedral symmetry (2024)
- **TTT with DLN**: First proof-of-concept of differentiable logic (Xmas 2025)
- **ILP baseline**: Frequency-based ILP achieves 0.031 eval MSE
- **Hybrid GA+ILP**: Expected 30-50% improvement over pure ILP

See numbered `.md` files for detailed results (e.g., `68-ILP_TEST_RESULTS.md`, `70-SCALE_DEPENDENT_RESULTS.md`)
