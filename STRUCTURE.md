# Project Structure

```
AGI-experiments/
├── README.md                      # Main project overview
├── requirements.txt               # Python dependencies
├── venv/                          # Virtual environment
│
├── docs/                          # Documentation (NEW!)
│   ├── MEMORY_ARCHITECTURE.md    # WM, LTM, entity registry
│   ├── VARIABLES_AND_ENTITIES.md # Scoping, binding, names
│   ├── TRAINING_STRATEGY.md      # NLP preprocessing, AR+RL
│   ├── SCALING_CHALLENGES.md     # Rule retrieval, VQ codebook
│   └── FIRST_EXPERIMENT.md       # Dataset size, bAbI setup
│
├── SCALING_TO_AGI.md             # Legacy (kept for reference)
├── VARIABLES_VS_ENTITIES.md      # Legacy (kept for reference)
│
├── neural_logic_core.py           # Core logic network (from TTT)
├── hierarchical_logic_network.py  # AR+RL architecture (from TTT)
│
├── data/                          # Datasets (to be added)
│   ├── babi/                     # bAbI tasks
│   ├── simple_text/              # Children's books
│   └── annotated/                # Entity-annotated corpora
│
├── models/                        # Model implementations (to be added)
│   ├── entity_registry.py        # Entity ID/name management
│   ├── memory.py                 # WM and LTM classes
│   ├── vq_propositions.py        # Vector quantization codebook
│   ├── text_logic_network.py    # Logic network for text
│   └── rule_retrieval.py         # Efficient rule matching (ρ)
│
├── preprocessing/                 # NLP tools (to be added)
│   ├── entity_extraction.py     # Extract entities from text
│   ├── relation_extraction.py   # Extract relations
│   └── to_logical_format.py     # Convert to propositions
│
├── training/                      # Training scripts (to be added)
│   ├── train_ar.py               # Autoregressive training
│   ├── train_rl.py               # RL for QA tasks
│   └── train_hierarchical.py     # Combined AR+RL
│
├── experiments/                   # Results (to be added)
│   ├── phase1_babi/
│   ├── phase2_scaling/
│   └── phase3_complex/
│
└── notebooks/                     # Exploration
    ├── data_exploration.ipynb
    └── rule_visualization.ipynb
```

## Current Status

**Completed**:
- ✓ Core logic network (from TTT)
- ✓ Hierarchical AR+RL architecture
- ✓ Comprehensive documentation in `docs/`
- ✓ Technical roadmap and first experiment plan

**Next Steps** (from [FIRST_EXPERIMENT.md](docs/FIRST_EXPERIMENT.md)):
1. Download bAbI dataset
2. Implement entity registry
3. Implement WM/LTM classes
4. Write preprocessing scripts
5. Run Phase 1 training
2. Implement VQ proposition encoder
3. Acquire simple text dataset
4. Create text proposition extraction pipeline
5. Begin Phase 1 experiments
