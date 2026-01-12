# Meta-Reasoning & Rule Injection (Gist)

## Goal
Provide a lightweight meta-learning loop that lets the symbolic engine invent, score, and selectively transfer rules into the DLN (or keep them symbolic) to improve task success with minimal supervision.

## Components
- **Rule proposer**: ILP-style templates, specialization/generalization, analogy, and paraphrase-driven predicate alignment.
- **Rule scorer (bandit/Bayesian)**: per-rule weight/posterior updated from downstream signals; cheap conjugate updates or UCB/Thompson are preferred online.
- **DLN transfer hook**: explicit action that commits a logical rule to DLN (create/reuse canonical predicate, bind args, set confidence), distinct from normal inference.
- **Lexicon/ontology priors**: type signatures, synonym clusters, compositional templates (quantifiers, modifiers, prepositions, relative clauses) to bias reuse over proliferation.
- **Replay + diversity**: keep high-utility rules, decay or prune low performers; penalize near-duplicates to avoid fragmentation.

## Learning loop (online)
1) Propose candidate rule(s).
2) Score via downstream rollout: validity + task reward + sparsity/length penalty + reuse bonus (reuse existing predicates > spawning new ones).
3) Update rule weight/posterior; keep a small top-K cache.
4) If score surpasses transfer threshold, invoke DLN transfer hook; otherwise keep symbolic-side or discard.
5) Periodically distill: batch re-estimate weights on replay buffer to reduce drift.

## Reward/Signal shaping
- **Validity** (logical consistency, type checks), **task success**, **coverage**, **parsimony** (short, general), **stability** (rule performance variance), **reuse** (canonical predicate alignment).
- Low-confidence mappings route through symbolic fallback or human-in-the-loop; do not silently auto-transfer.

## Seeding to accelerate
- Core predicate clusters (eat/consume/ingest → ingest(agent, patient)).
- Modifier handling (color(x, red); speed(run, fast)).
- Quantifier/scope templates (∃/∀, if/then, ¬).
- Preposition/rel-clause templates (loc_in/has/with/of).
- Canonicalization preference: reuse if similarity > τ; otherwise spawn with low prior.

## Bayesian vs bandit
- Online **bandit/UCB/Thompson** for speed; conjugate Beta/Dirichlet updates are cheap and usually sufficient.
- Reserve heavier Bayesian/hierarchical fits for occasional offline consolidation on the top-K cache.

## Extensions (beyond rule injection)
- Modal/fixpoint/meta-operators for reflection without DLN transfer.
- Cross-module proposals scored by other systems (planning, perception) via the same bandit/Bayesian loop.
- Safety filters: type/scope guards, entropy thresholds before transfer, rollback hooks for bad transfers.

## Minimal ops checklist
- Expose a callable "transfer_rule_to_DLN(rule, confidence)" action.
- Implement per-rule weight + UCB/Thompson update from rollout scores.
- Add reuse-biased reward terms and length penalty.
- Maintain replay + pruning; schedule periodic consolidation.
