# Project Structure

```
AGI-experiments/
├── README.md                      # Main project documentation
├── SCALING_TO_AGI.md              # Technical challenges and solutions
├── VARIABLES_VS_ENTITIES.md       # Clarification on scoping (NEW!)
├── requirements.txt               # Python dependencies
├── venv/                          # Virtual environment
│
├── neural_logic_core.py           # Core logic network (from TTT)
├── hierarchical_logic_network.py  # AR+RL architecture (from TTT)
│
├── data/                          # Datasets (to be added)
│   ├── simple_text/              # Children's books, simple narratives
│   ├── qa_pairs/                 # Question-answering datasets
│   └── annotated/                # Entity-annotated corpora
│
├── models/                        # Model implementations (to be added)
│   ├── entity_memory.py          # Entity tracking system
│   ├── vq_propositions.py        # Vector quantization for propositions
│   ├── text_logic_network.py    # Logic network adapted for text
│   └── rule_retrieval.py         # Efficient rule matching
│
├── training/                      # Training scripts (to be added)
│   ├── train_ar_text.py          # Autoregressive on text
│   ├── train_rl_qa.py            # RL for question answering
│   └── train_hierarchical.py     # Combined AR+RL training
│
├── experiments/                   # Experimental results (to be added)
│   ├── phase1_simple_text/
│   ├── phase2_entity_tracking/
│   └── phase3_rule_retrieval/
│
└── notebooks/                     # Jupyter notebooks for exploration
    ├── explore_data.ipynb
    ├── visualize_rules.ipynb
    └── analyze_concepts.ipynb
```

## Current Status

**Completed**:
- ✓ Core logic network implementation (from TTT experiments)
- ✓ Hierarchical AR+RL architecture
- ✓ Technical documentation (scaling challenges)
- ✓ Clarification document on variables vs entities

**Next Steps**:
1. Implement entity memory module
2. Implement VQ proposition encoder
3. Acquire simple text dataset
4. Create text proposition extraction pipeline
5. Begin Phase 1 experiments
