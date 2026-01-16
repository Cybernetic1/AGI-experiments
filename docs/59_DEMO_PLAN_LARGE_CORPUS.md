# Large-Corpus Demo Plan (Rule Injection + DLN)

## Objective (1 week)
Show DLN + symbolic rule injection achieving competitive accuracy with fewer parameters than a small Transformer on a mid-size text slice (e.g., 1–5M sentences), with faster label generation via per-rule injection.

## Scope
- Corpus: TinyStories or C4 slice (fixed size, e.g., 1–5M sentences)
- Hardware: single GPU (specify, e.g., A100/L40/3090)
- Baseline: small Transformer (parameter-matched report)
- Metrics: DLN params vs baseline params, train/eval MAE (or task metric), labels/sec, rule acceptance count, collection time, wall-clock

## Workplan (7 days)
1) **Plumbing (Day 1-2)**
   - Add batched/streamed label collection with predicate filter per rule; disk cache on by default.
   - Thread device through RuleInjector; ensure fine-tunes run on GPU.
   - Loader for corpus slice (TinyStories/C4) to produce facts at scale.
2) **Metrics & baselines (Day 3-4)**
   - Instrument metrics: train/eval loss/MAE, label counts, rules accepted, collection time, GPU util snapshot.
   - Add parameter-count reporting for DLN; run a small Transformer baseline on same slice/budget.
3) **Stability & vocab (Day 5)**
   - Robust vocab growth (predicate/arg expansion), skip empty batches, cache reuse, failure guards.
4) **Demo run (Day 6)**
   - Execute full pipeline on chosen slice; capture metrics and logs.
5) **Packaging (Day 7)**
   - Produce notebook/script to rerun; prepare slides summarizing architecture, rule injection flow, metrics, and compression comparison.

## Key levers for compression demo
- Rule injection reduces label sweep (per-rule predicate filter) and reuses canonical predicates, lowering data/compute.
- DLN param count is small; compare against baseline Transformer params and wall-clock.

## Commands (examples)
- Rule-injection TinyStories slice:
  ```bash
  source venv/bin/activate
  python benchmarks.py \
    --device cuda \
    --max-stories 200 \
    --max-facts 5000 \
    --max-candidate-rules 400 \
    --ri-steps 4 --ri-threshold 0.5 --ri-lr 1e-2
  ```
- Injection smoke test:
  ```bash
  source venv/bin/activate
  python - <<'PY'
  from benchmarks import rule_injection_smoke_test
  print(rule_injection_smoke_test())
  PY
  ```

## Dependencies
- PyTorch installed in venv
- GPU drivers/CUDA for cuda device runs
- Corpus files (TinyStories/C4 slice) present under data/processed/

## Risks
- Data load bottlenecks; mitigate with chunked loading and caching.
- Vocab explosion; mitigate with predicate/arg filtering.
- Baseline tuning time; keep baseline small and budgeted.
