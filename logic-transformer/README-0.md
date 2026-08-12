# Tri-System Neurosymbolic Architecture

This document describes the broader cognitive architecture into which modules like the Logic Transformer (LT) can be integrated.

The architecture aims to combine neural learning with symbolic reasoning by separating cognition into three distinct systems:

1. **System 1 (Logic Transformer / Neural Logic):**
   - **Role:** Fast, differentiable, fuzzy logic rule application.
   - **Mechanism:** Uses gradient-based representation learning to process continuous semantic working memory and unify patterns softly. Acts as the intuitive, pattern-matching layer that maps natural language or raw perception into latent logical forms.

2. **System 2 (Rete Engine / Forward Chaining):**
   - **Role:** Deterministic, discrete, and efficient pattern matching over established facts.
   - **Mechanism:** Once facts are crystallized or extracted with high confidence from System 1, they are fed into a Rete network. This engine rapidly deduces immediate logical consequences (forward chaining) without the overhead of probabilistic search.

3. **System 3 (ProbLog / Backward Chaining):**
   - **Role:** Deep, complex reasoning and goal-directed planning.
   - **Mechanism:** A probabilistic backward-chaining inference engine intended for answering complex queries, handling uncertainty, and executing multi-step planning tasks that require searching through a large hypothesis space.

By separating these concerns, the architecture leverages the strengths of differentiable neural networks for noisy input handling (System 1) while retaining the correctness, interpretability, and powerful search capabilities of classical symbolic engines (Systems 2 & 3).