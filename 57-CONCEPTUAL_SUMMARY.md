# Project Direction and Acceleration Plan

## Current Position
- Hybrid stack is validated on TinyStories mini benchmark: symbolic forward chaining, DLN, mined rules, RuleStore persistence, optional entity canonicalization, paraconsistency.
- Davidsonian parser exists but not yet the default for TinyStories ingestion; current ingestion uses lightweight SVO extractor.
- GA warm-start available but used only as a seed; main training uses rule-induced labels (predicate-level MSE/MAE), not semantic-AR.
- Backward chaining helper added; not yet wired into training/eval.

## Near-Term Objectives
1) **Parser unification**: Make the Davidsonian extractor the primary ingestion path; keep SVO as a fallback. Normalize predicates (verb lemma + role), add coref/pronoun heuristics, quantifier tags, and type-aware role filtering.
2) **Rule/label efficiency**: Cache symbolic closures; prioritize high-support rules; sample labels per step; use RuleStore nearest-neighbor lookup to avoid full rule sweeps.
3) **Training accelerators**: GA warm-start for rule seeds; symbolic pseudo-labels for dense supervision; curriculum from high-confidence to mined rules; goal-directed negatives from failed backward proofs; optional semantic-AR loss alongside rule MSE.
4) **Consolidation (“grokking”) loop**: Periodically re-run mining, support/lift scoring, contradiction checks, and snapshot a stable rule subset; fine-tune DLN on refreshed labels; optionally distill DLN proposals back into symbolic candidates.
5) **Scale-out targets**: Move beyond TinyStories to ROCStories, CommonGen, WikiHow, MultiWOZ, HotpotQA/StrategyQA contexts, and Wikipedia/Book subsets parsed via Davidsonian; incorporate ATOMIC/ConceptNet/VerbNet as priors for predicate and role normalization.

## Division of Labor (Side-by-Side Engines)
- **Symbolic engine**: High-confidence constraints and pseudo-label generator; paraconsistent, can inject hand-written or mined rules (including natural-language “rule form” sentences).
- **DLN**: Learns soft scores, fills gaps, and proposes candidates; trained on symbolic labels plus optional AR/contrastive/link-prediction objectives.
- Interaction guardrails: keep provenance on labels; resolve conflicts via rule confidence and type checks; use backward chaining for contrastive negatives.

## Practical Next Steps
- Replace TinyStories ingestion with Davidsonian-first path and re-run the benchmark with caching and label sampling.
- Add a coverage report (template hits, propositions/sentence, label density) to guide parser improvements.
- Introduce semantic-AR as an auxiliary loss in the mini benchmark to stabilize early training.
- Add a lightweight rule-sentence detector to auto-ingest explicit NL rules into RuleStore.

## Expected Gains vs LLM-Only Training
- Faster convergence via dense symbolic pseudo-labels, not token-level AR alone.
- Smaller search space thanks to rule priors (GA seeds, mined chains, NL rule ingestion).
- Better interpretability and editability through explicit RuleStore snapshots and consolidation cycles.

## End Goal
A scalable, hybrid pipeline that parses large real-life corpora into normalized predicates, mines and injects rules, trains DLN with symbolic + AR objectives, and periodically consolidates knowledge to maintain logical coherence while expanding coverage.
