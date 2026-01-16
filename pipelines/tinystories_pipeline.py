import json
import time
import random
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import torch

from logic_core import Proposition, Rule, SymbolicEngine
from dln import SimpleDLN, _require_torch
from rule_store import RuleStore
from rule_injection import RuleInjector
from entity_registry import PersistentEntityRegistry
from davidsonian_extraction import DavidsonianExtractor
from label_utils import _collect_labels, _collect_labels_filtered
from core.train_utils import _train_on_labels, _eval_on_labels, _mae_on_labels


def load_tinystories_facts(
    max_stories: int = 50,
    max_facts: int = 1000,
    path: str = "data/processed/tinystories_train.json",
    use_entity_registry: bool = False,
    registry: Optional[PersistentEntityRegistry] = None,
    prefer_davidsonian: bool = True,
) -> List[Proposition]:
    fpath = Path(path)
    if not fpath.exists():
        print(f"TinyStories file not found at {path}; skipping TinyStories mini benchmark.")
        return []
    with open(fpath, "r") as f:
        data = json.load(f)

    reg = registry if use_entity_registry else None

    def _canon(name: str) -> str:
        if not name:
            return name
        if reg is None:
            return name
        eid = reg.get_or_create_entity(name)
        ent = reg.get_entity(eid)
        return ent.name.lower() if ent else name

    extractor = DavidsonianExtractor() if prefer_davidsonian else None
    facts: List[Proposition] = []
    stories_seen = 0
    stories_with_dav = 0
    for story in data[:max_stories]:
        stories_seen += 1
        used_dav = False
        if prefer_davidsonian and extractor and story.get("text"):
            props = extractor.extract(story["text"])
            for (ent, rel, val) in props:
                subj = _canon(str(ent))
                obj = _canon(str(val))
                if not rel or not subj or not obj:
                    continue
                facts.append(Proposition(rel, (subj, obj), 1.0))
                used_dav = True
                if len(facts) >= max_facts:
                    break
        if not used_dav:
            for fact in story.get("facts", []):
                subj = _canon(str(fact.get("subject", "")))
                obj = _canon(str(fact.get("object", "")))
                rel = str(fact.get("relation", ""))
                if not subj or not obj or not rel:
                    continue
                facts.append(Proposition(rel, (subj, obj), 1.0))
                if len(facts) >= max_facts:
                    break
        if used_dav:
            stories_with_dav += 1
        if len(facts) >= max_facts:
            break

    print(
        f"TinyStories load: {stories_seen} stories, {stories_with_dav} via Davidsonian, total facts={len(facts)}"
    )
    return facts


def inject_contradiction(facts: List[Proposition], pred: str = "interacts_with", strength: float = 0.8) -> List[Proposition]:
    if not facts:
        return facts
    f0 = facts[0]
    contra = Proposition(f"not_{pred}", f0.args, strength)
    return facts + [contra]


def mine_chain_rules(facts: List[Proposition], max_rules: int = 10, min_support: int = 2) -> Tuple[List[Rule], List[str]]:
    counts: Dict[Tuple[str, str], int] = {}
    facts_2 = [f for f in facts if len(f.args) >= 2]
    by_first: Dict[Tuple[str, str], List[Proposition]] = {}
    for f in facts_2:
        by_first.setdefault((f.predicate, f.args[0]), []).append(f)

    for f1 in facts_2:
        mid = f1.args[1]
        for (pred, a0), lst in by_first.items():
            if a0 != mid:
                continue
            for f2 in lst:
                key = (f1.predicate, f2.predicate)
                counts[key] = counts.get(key, 0) + 1

    sorted_pairs = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    rules: List[Rule] = []
    pred_names: List[str] = []
    for (p1, p2), c in sorted_pairs:
        if c < min_support or len(rules) >= max_rules:
            continue
        concl = f"{p1}_{p2}_mined"
        pred_names.append(concl)
        rules.append(
            Rule(
                [Proposition(p1, ("?x", "?y")), Proposition(p2, ("?y", "?z"))],
                Proposition(concl, ("?x", "?z")),
                1.0,
            )
        )
    return rules, pred_names


def tinystories_mini_benchmark(
    steps: int = 40,
    save_store: bool = False,
    store_path: str = "data/processed/rule_store_tiny.json",
    eval_split: float = 0.2,
    inject_contra: bool = True,
    contra_strength: float = 0.8,
    load_store: bool = True,
    use_mined: bool = True,
    save_mined: bool = False,
    use_entity_registry: bool = True,
    prefer_davidsonian: bool = True,
    device: str = "cpu",
    disk_label_cache: bool = True,
    max_stories: int = 50,
    max_facts: int = 1000,
    use_ar_aux: bool = True,
    ar_weight: float = 0.1,
    max_candidate_rules: int = 200,
    use_rule_injection: bool = True,
    ri_train_steps: int = 4,
    ri_threshold: float = 0.5,
    ri_lr: float = 1e-2,
    label_batch_size: int = 128,
    allowed_predicates: Optional[Set[str]] = None,
):
    _require_torch()
    print(f"[setup] Initializing entity registry (enabled={use_entity_registry})...", flush=True)
    reg = PersistentEntityRegistry(embedding_dim=16) if use_entity_registry else None
    print(f"[setup] Loading TinyStories facts (max_stories={max_stories}, max_facts={max_facts})...", flush=True)
    facts = load_tinystories_facts(
        max_stories=max_stories,
        max_facts=max_facts,
        use_entity_registry=use_entity_registry,
        registry=reg,
        prefer_davidsonian=prefer_davidsonian,
    )
    if not facts:
        return None

    if inject_contra:
        print(f"[setup] Injecting contradiction (strength={contra_strength})...", flush=True)
        facts = inject_contradiction(facts, strength=contra_strength)

    split_idx = max(1, int(len(facts) * (1 - eval_split)))
    train_facts = facts[:split_idx]
    eval_facts = facts[split_idx:] if split_idx < len(facts) else facts[-1:]
    print(f"[setup] Split facts: train={len(train_facts)}, eval={len(eval_facts)}", flush=True)

    print(f"[setup] Building vocabulary...", flush=True)
    relations = sorted({p.predicate for p in facts})
    args_vocab = sorted({a for p in facts for a in p.args})
    predicates = relations + [f"{r}_inferred" for r in relations]
    args = ["<pad>"] + args_vocab
    print(f"[setup] Vocabulary built: {len(relations)} relations, {len(args)} args", flush=True)

    print(f"[setup] Generating base, combo, and negative rules...", flush=True)
    base_rules = [Rule([Proposition(rel, ("?x", "?y"))], Proposition(f"{rel}_inferred", ("?x", "?y")), 1.0) for rel in relations]

    combo_rules: List[Rule] = []
    if len(relations) >= 2:
        for i, r1 in enumerate(relations):
            for r2 in relations[i + 1 :]:
                concl_name = f"{r1}_{r2}_combo"
                predicates.append(concl_name)
                combo_rules.append(
                    Rule(
                        [Proposition(r1, ("?x", "?y")), Proposition(r2, ("?y", "?z"))],
                        Proposition(concl_name, ("?x", "?z")),
                        1.0,
                    )
                )

    neg_rules = [Rule([Proposition(rel, ("?x", "?y"))], Proposition(f"not_{rel}", ("?x", "?y")), 1.0) for rel in relations]
    predicates += [f"not_{rel}" for rel in relations]
    print(f"[setup] Generated {len(base_rules)} base + {len(combo_rules)} combo + {len(neg_rules)} negative rules", flush=True)

    print(f"[setup] Generating narrative rules...", flush=True)
    narrative_rules: List[Rule] = []
    if "gives" in relations and "has" in relations:
        predicates.append("transfer_receives")
        narrative_rules.append(
            Rule(
                [
                    Proposition("gives", ("?giver", "?item")),
                    Proposition("has", ("?giver", "?item")),
                ],
                Proposition("transfer_receives", ("?item", "?giver")),
                1.0,
            )
        )
    if "goes_to" in relations and "has" in relations:
        predicates.append("arrival_possession")
        narrative_rules.append(
            Rule(
                [
                    Proposition("goes_to", ("?who", "?place")),
                    Proposition("has", ("?place", "?thing")),
                ],
                Proposition("arrival_possession", ("?who", "?thing")),
                1.0,
            )
        )
    print(f"[setup] Generated {len(narrative_rules)} narrative rules", flush=True)

    print(f"[setup] Mining chain rules (enabled={use_mined})...", flush=True)
    mined_rules: List[Rule] = []
    mined_pred_names: List[str] = []
    if use_mined:
        mined_rules, mined_pred_names = mine_chain_rules(train_facts, max_rules=10, min_support=2)
        predicates.extend(mined_pred_names)
        print(f"[setup] Mined {len(mined_rules)} chain rules", flush=True)

    all_rules = base_rules + combo_rules + neg_rules + narrative_rules + mined_rules
    print(f"[setup] Total rules: {len(all_rules)}", flush=True)

    print(f"[setup] Initializing DLN model on {device}...", flush=True)
    model = SimpleDLN(predicates, args).to(device)
    print(f"[params] DLN trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"[setup] Initializing RuleStore (load_store={load_store})...", flush=True)
    store = RuleStore(model)
    store_path = Path(store_path)
    if load_store and store_path.exists():
        print(f"[setup] Loading RuleStore from {store_path}...", flush=True)
        store = RuleStore.load(model, store_path, sim_threshold=0.98)
        print(f"[setup] Loaded {len(store.rules)} rules from store", flush=True)
    print(f"[setup] Deduplicating {len(all_rules)} rules via RuleStore...", flush=True)
    deduped_rules: List[Rule] = []
    for r in all_rules:
        added, _ = store.add(r)
        if added:
            deduped_rules.append(r)
    print(f"[setup] Deduplicated to {len(deduped_rules)} unique rules", flush=True)

    print(f"[setup] Building candidate rules list...", flush=True)
    target_preds = set()
    for r in deduped_rules:
        if r.conclusion.predicate.endswith("_inferred") or r.conclusion.predicate.endswith("_combo") or r.conclusion.predicate.startswith("not_"):
            target_preds.add(r.conclusion.predicate)

    candidate_rules: List[Rule] = []
    seen: set = set()
    for pred in target_preds:
        for cr in store.candidates_for_conclusion(pred):
            key = (
                cr.conclusion.predicate,
                tuple(cr.conclusion.args),
                tuple((p.predicate, p.args) for p in cr.premises),
            )
            if key in seen:
                continue
            seen.add(key)
            candidate_rules.append(cr)

    if not candidate_rules:
        candidate_rules = deduped_rules

    if max_candidate_rules and len(candidate_rules) > max_candidate_rules:
        print(f"[setup] Limiting candidates from {len(candidate_rules)} to {max_candidate_rules}", flush=True)
        candidate_rules = candidate_rules[:max_candidate_rules]
    else:
        print(f"[setup] Using {len(candidate_rules)} candidate rules", flush=True)

    accepted = 0
    if use_rule_injection:
        print(f"[rule injection] starting rule injection for {len(candidate_rules)} candidate rules...", flush=True)
        t_ri_start = time.perf_counter()
        injector = RuleInjector(model, store, transfer_threshold=ri_threshold, lr=ri_lr)
        for idx, r in enumerate(candidate_rules, start=1):
            if injector.transfer_rule_to_dln(r, train_facts, confidence=1.0, train_steps=ri_train_steps):
                accepted += 1
            if idx % max(1, len(candidate_rules) // 10) == 0 or idx == len(candidate_rules):
                pct = 100.0 * idx / len(candidate_rules)
                elapsed = time.perf_counter() - t_ri_start
                print(f"[rule injection] progress {idx}/{len(candidate_rules)} ({pct:.1f}%), accepted={accepted}, elapsed={elapsed:.1f}s", flush=True)
        t_ri_total = time.perf_counter() - t_ri_start
        print(f"[rule injection] completed in {t_ri_total:.2f}s, accepted {accepted}/{len(candidate_rules)} rules", flush=True)

    print(f"[label generation] starting label collection for {len(candidate_rules)} candidate rules...", flush=True)
    t0 = time.perf_counter()
    labels = _collect_labels_filtered(
        train_facts,
        candidate_rules,
        use_disk_cache=disk_label_cache,
        rule_batch_size=label_batch_size,
        allowed_predicates=allowed_predicates,
    )
    labels_full = labels
    t_collect = time.perf_counter() - t0
    t_collect_full = t_collect
    print(f"[label generation] completed in {t_collect:.2f}s, generated {len(labels)} labels", flush=True)

    labels = {k: v for k, v in labels.items() if k[0].endswith("_inferred") or k[0].endswith("_combo") or k[0].startswith("not_") or k[0].endswith("_mined")}
    labels_full = {k: v for k, v in labels_full.items() if k[0].endswith("_inferred") or k[0].endswith("_combo") or k[0].startswith("not_") or k[0].endswith("_mined")}
    pruned_only = set(labels.keys()) - set(labels_full.keys())
    full_only = set(labels_full.keys()) - set(labels.keys())
    if not labels:
        print("No labels produced for TinyStories mini benchmark; skipping.")
        return None

    print(f"[training] starting DLN training for {steps} steps on {len(labels)} labels...", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    t1 = time.perf_counter()
    final_mse = _train_on_labels(model, opt, train_facts, labels, steps, device=device, use_ar_aux=use_ar_aux, ar_weight=ar_weight)
    t_train = time.perf_counter() - t1
    print(f"[training] completed in {t_train:.2f}s, final MSE={final_mse:.4f}", flush=True)

    print(f"[evaluation] starting evaluation on {len(eval_facts)} eval facts...", flush=True)
    t_eval_start = time.perf_counter()
    eval_labels = _collect_labels_filtered(
        eval_facts,
        candidate_rules,
        use_disk_cache=disk_label_cache,
        rule_batch_size=label_batch_size,
        allowed_predicates=allowed_predicates,
    )
    eval_labels = {k: v for k, v in eval_labels.items() if k[0].endswith("_inferred") or k[0].endswith("_combo") or k[0].startswith("not_") or k[0].endswith("_mined")}
    t_eval_label_collect = time.perf_counter() - t_eval_start
    print(f"[evaluation] label collection completed in {t_eval_label_collect:.2f}s, {len(eval_labels)} labels", flush=True)
    eval_mse = _eval_on_labels(model, eval_facts, eval_labels, device=device) if eval_labels else float("nan")
    eval_mae = _mae_on_labels(model, eval_facts, eval_labels, device=device) if eval_labels else float("nan")
    t_eval_total = time.perf_counter() - t_eval_start
    print(f"[evaluation] completed in {t_eval_total:.2f}s, eval MSE={eval_mse:.4f}, MAE={eval_mae:.4f} on {len(eval_labels)} labels", flush=True)

    if save_store or save_mined:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store.save(store_path)

    ri_time_msg = ""
    if use_rule_injection and 't_ri_total' in locals():
        ri_time_msg = f", rule_injection={t_ri_total:.1f}s"
    
    print(
        f"TinyStories mini benchmark (inferred + combo + neg): train MSE={final_mse:.4f} on {len(labels)} labels, "
        f"eval MSE={eval_mse:.4f} MAE={eval_mae:.4f} on {len(eval_labels)} labels "
        f"({len(candidate_rules)} rules, injected={accepted if use_rule_injection else 0}, "
        f"timing: label_collect={t_collect:.1f}s, train={t_train:.1f}s, eval={t_eval_total:.1f}s{ri_time_msg})"
    )
    return final_mse, len(labels)


__all__ = [
    "tinystories_mini_benchmark",
    "load_tinystories_facts",
    "inject_contradiction",
    "mine_chain_rules",
]
