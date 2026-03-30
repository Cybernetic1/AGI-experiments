Bottlenecks of AGI world-model learning
=======================================

   a) Data efficiency paradox

     - Humans learn world models from ~few thousand experiences
     - Current ML needs millions of examples
     - Problem: Can't experience everything (long-tail events)

   b) Credit assignment across time

     - Action at t=0 → consequence at t=100
     - Which rule/belief caused success/failure?
     - Problem: Exponential search space for causal chains

   c) Abstraction hierarchy

     - Need rules at multiple levels: physical laws, social norms, individual quirks
     - Must know WHEN to apply which level
     - Problem: Compositional explosion (how rules combine)

   d) Belief revision complexity

     - New evidence contradicts old beliefs
     - Which beliefs to retract? Ripple effects?
     - Problem: Non-monotonic reasoning is undecidable in general

   e) Grounding problem

     - Symbols (words) must map to real-world referents
     - Context-dependent, ambiguous
     - Problem: Your "581 stories" issue—sparse coverage of concept space

Most effective heuristics
=========================

   Top 5 (in my assessment):

   a) Hierarchical abstraction (clustering)

     688 verbs → 20 categories → 5 super-categories
     Learn transitions at abstract level, specialize downward

     - Addresses: Data efficiency, generalization
     - Your insight: Semantic embeddings partially do this

   b) Active learning / curiosity

     RL reward = prediction_error + novelty
     Seek experiences that violate current model

     - Addresses: Credit assignment, data efficiency
     - Your insight: Dissonance-driven learning!

   c) Meta-learning (learning to learn)

     Learn: "How to revise beliefs" not just "what beliefs"
     Second-order rules about rule acquisition

     - Addresses: Belief revision, transfer learning
     - Example: "If emotion conflict, newer observation wins"

   d) Modular world models

     Separate modules: physics, psychology, social, language
     Each module has own rules, limited interfaces

     - Addresses: Compositional explosion, scalability
     - Example: Davidsonian events = modular (agent, patient, manner separate)

   e) Probabilistic + symbolic hybrid

     Symbolic: structure, compositionality
     Probabilistic: uncertainty, soft constraints

     - Addresses: Noise, ambiguity, incomplete info
     - Your current approach: DLN (neural) + rules (symbolic)

Is autoregression the only framework?
=====================================

   No! Alternatives:

   a) World model + planning (Model-Based RL)

     Learn: P(s_t+1 | s_t, a_t) [transition model]
     Use: Plan multi-step ahead, simulate futures

     - Advantage: Explicit causality, counterfactuals
     - Disadvantage: Harder to train, error accumulation

   b) Energy-based models

     Learn: E(state) [low energy = plausible]
     Use: Score coherence, don't generate sequentially

     - Advantage: Bidirectional consistency
     - Disadvantage: Sampling/optimization hard

   c) Declarative memory + retrieval

     Store: Episodes as-is
     Use: Retrieve similar past, adapt to present

     - Advantage: No lossy compression, case-based reasoning
     - Disadvantage: Scales poorly, needs similarity metric

   d) Constraint satisfaction

     Learn: Constraints (rules as hard/soft requirements)
     Use: Find state satisfying all constraints

     - Advantage: Naturally handles contradictions
     - Disadvantage: NP-hard in general

   Your dissonance-based approach is actually closest to (d)!
