import time
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from logic_core import Proposition, Rule, SymbolicEngine

LABEL_CACHE: Dict[Tuple, Dict[Tuple[str, Tuple[str, ...]], float]] = {}
LABEL_CACHE_DIR = Path("data/cache_labels")


def _make_label_cache_key(
    facts: List[Proposition],
    rules: List[Rule],
    predicate_filter: Optional[str],
    rule_batch_size: Optional[int],
    allowed_predicates: Optional[Set[str]],
):
    facts_key = tuple((f.predicate, f.args, f.truth) for f in facts)
    rules_key = tuple(
        (
            tuple((prem.predicate, prem.args, prem.truth) for prem in r.premises),
            (r.conclusion.predicate, r.conclusion.args, r.conclusion.truth),
            r.weight,
        )
        for r in rules
    )
    allowed_key = tuple(sorted(allowed_predicates)) if allowed_predicates else None
    return (facts_key, rules_key, predicate_filter, rule_batch_size, allowed_key)


def _collect_labels(
    facts: List[Proposition],
    rules: List[Rule],
    predicate_filter: Optional[str] = None,
    use_disk_cache: bool = True,
    rule_batch_size: Optional[int] = None,
    allowed_predicates: Optional[Set[str]] = None,
    log_progress: bool = True,
) -> Dict[Tuple[str, Tuple[str, ...]], float]:
    key = _make_label_cache_key(facts, rules, predicate_filter, rule_batch_size, allowed_predicates)
    if key in LABEL_CACHE:
        return LABEL_CACHE[key]

    disk_hit = False
    labels: Dict[Tuple[str, Tuple[str, ...]], float] = {}
    cache_file = None
    if use_disk_cache:
        LABEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key_hash = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()
        cache_file = LABEL_CACHE_DIR / f"labels_{key_hash}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    labels = pickle.load(f)
                    disk_hit = True
                if log_progress:
                    print(f"[labels] loaded {len(labels)} labels from disk cache", flush=True)
            except Exception:
                labels = {}

    if not labels:
        t0 = time.perf_counter()
        if log_progress:
            print(f"[labels] collecting on {len(facts)} facts with {len(rules)} rules (batch={rule_batch_size})...", flush=True)
        # Use max_iters=1 to prevent combinatorial explosion
        eng = SymbolicEngine(max_iters=1)
        batches = [rules] if not rule_batch_size else [rules[i : i + rule_batch_size] for i in range(0, len(rules), rule_batch_size)]
        total_batches = len(batches)
        for idx, batch in enumerate(batches, start=1):
            if log_progress:
                print(f"[labels] === Batch {idx}/{total_batches} === (max_iters=1 to prevent explosion)", flush=True)
            targets = eng.infer(facts, batch, show_progress=log_progress)
            for p in targets:
                if predicate_filter and p.predicate != predicate_filter:
                    continue
                if allowed_predicates and p.predicate not in allowed_predicates:
                    continue
                labels[(p.predicate, p.args)] = p.truth
            if log_progress and total_batches > 1 and (idx % max(1, total_batches // 10) == 0 or idx == total_batches):
                pct = 100.0 * idx / total_batches
                print(f"[labels] Batch {idx}/{total_batches} ({pct:.1f}%) → labels so far: {len(labels)}", flush=True)
        dt = time.perf_counter() - t0
        if log_progress:
            print(f"[labels] generated {len(labels)} labels in {dt:.2f}s", flush=True)
        if use_disk_cache and not disk_hit and cache_file is not None:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(labels, f)
            except Exception:
                pass

    LABEL_CACHE[key] = labels
    return labels


def _collect_labels_filtered(
    facts: List[Proposition],
    rules: List[Rule],
    use_disk_cache: bool = True,
    rule_batch_size: Optional[int] = None,
    allowed_predicates: Optional[Set[str]] = None,
    log_progress: bool = True,
) -> Dict[Tuple[str, Tuple[str, ...]], float]:
    merged: Dict[Tuple[str, Tuple[str, ...]], float] = {}
    total = len(rules)
    for idx, r in enumerate(rules, start=1):
        lbls = _collect_labels(
            facts,
            [r],
            predicate_filter=r.conclusion.predicate,
            use_disk_cache=use_disk_cache,
            rule_batch_size=rule_batch_size,
            allowed_predicates=allowed_predicates,
            log_progress=log_progress,
        )
        merged.update(lbls)
        if log_progress and total > 1 and (idx % max(1, total // 10) == 0 or idx == total):
            pct = 100.0 * idx / total
            print(f"[labels] per-rule progress {idx}/{total} ({pct:.1f}%)", flush=True)
    return merged


__all__ = [
    "_make_label_cache_key",
    "_collect_labels",
    "_collect_labels_filtered",
    "LABEL_CACHE",
    "LABEL_CACHE_DIR",
]
