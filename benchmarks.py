import json
import time
import random
import math
import argparse
import pickle
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from logic_core import Proposition, Rule, SymbolicEngine
from dln import SimpleDLN, _require_torch, F
from rule_store import RuleStore
from entity_registry import PersistentEntityRegistry
from davidsonian_extraction import DavidsonianExtractor


# Simple in-memory cache for label collection to avoid repeated symbolic inference
LABEL_CACHE: Dict[Tuple, Dict[Tuple[str, Tuple[str, ...]], float]] = {}
LABEL_CACHE_DIR = Path("data/cache_labels")


def _toy_data() -> Tuple[List[Proposition], List[Rule]]:
    facts = [Proposition("A", ("alice",), 1.0), Proposition("A", ("bob",), 0.7)]
    rule = Rule([Proposition("A", ("?x",))], Proposition("B", ("?x",)), 1.0)
    return facts, [rule]


def ga_seed_rules(predicates: List[str], pop_size: int = 6) -> List[Rule]:
    rng = random.Random(0)
    rules: List[Rule] = []
    var = "?x"
    for _ in range(pop_size):
        prem_pred = rng.choice(predicates)
        concl_pred = rng.choice(predicates)
        rules.append(Rule([Proposition(prem_pred, (var,))], Proposition(concl_pred, (var,)), 1.0))
    return rules


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
    """Mining-style discovery: if rel1(x,y) and rel2(y,z) co-occur, propose rel1_rel2_mined(x,z)."""
    counts: Dict[Tuple[str, str], int] = {}
    # index facts by (pred, arg0, arg1) assuming arity>=2
    facts_2 = [f for f in facts if len(f.args) >= 2]
    by_first: Dict[Tuple[str, str], List[Proposition]] = {}
    by_second: Dict[Tuple[str, str], List[Proposition]] = {}
    for f in facts_2:
        by_first.setdefault((f.predicate, f.args[0]), []).append(f)
        by_second.setdefault((f.predicate, f.args[1]), []).append(f)

    # find chains where arg1 of f1 equals arg0 of f2
    for f1 in facts_2:
        mid = f1.args[1]
        for f2 in by_first.get((None, mid), []):
            pass  # unused branch; keep structure simple
        for f2 in by_first.get((f2_key := None, mid), []):
            pass  # placeholder
        # simpler: scan all f2 with arg0==mid
        for (pred, a0), lst in by_first.items():
            if a0 != mid:
                continue
            for f2 in lst:
                key = (f1.predicate, f2.predicate)
                counts[key] = counts.get(key, 0) + 1

    # build rules sorted by support
    sorted_pairs = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    rules: List[Rule] = []
    pred_names: List[str] = []
    for (p1, p2), c in sorted_pairs:
        if c < min_support:
            continue
        if len(rules) >= max_rules:
            break
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
) -> Optional[Tuple[float, int]]:
    _require_torch()
    reg = PersistentEntityRegistry(embedding_dim=16) if use_entity_registry else None
    facts = load_tinystories_facts(max_stories=max_stories, max_facts=max_facts, use_entity_registry=use_entity_registry, registry=reg, prefer_davidsonian=prefer_davidsonian)
    if not facts:
        return None

    if inject_contra:
        facts = inject_contradiction(facts, strength=contra_strength)

    split_idx = max(1, int(len(facts) * (1 - eval_split)))
    train_facts = facts[:split_idx]
    eval_facts = facts[split_idx:] if split_idx < len(facts) else facts[-1:]

    relations = sorted({p.predicate for p in facts})
    args_vocab = sorted({a for p in facts for a in p.args})
    predicates = relations + [f"{r}_inferred" for r in relations]
    args = ["<pad>"] + args_vocab

    base_rules = [Rule([Proposition(rel, ("?x", "?y"))], Proposition(f"{rel}_inferred", ("?x", "?y")), 1.0) for rel in relations]

    combo_rules: List[Rule] = []
    if len(relations) >= 2:
        for i, r1 in enumerate(relations):
            for r2 in relations[i + 1:]:
                concl_name = f"{r1}_{r2}_combo"
                predicates.append(concl_name)
                combo_rules.append(
                    Rule(
                        [Proposition(r1, ("?x", "?y")), Proposition(r2, ("?y", "?z"))],
                        Proposition(concl_name, ("?x", "?z")),
                        1.0,
                    )
                )

    # Negation template to exercise paraconsistency on real data
    neg_rules = [Rule([Proposition(rel, ("?x", "?y"))], Proposition(f"not_{rel}", ("?x", "?y")), 1.0) for rel in relations]
    predicates += [f"not_{rel}" for rel in relations]

    # Additional two-premise templates for narrative patterns
    narrative_rules: List[Rule] = []
    if "gives" in relations and "has" in relations:
        predicates.append("transfer_receives")
        narrative_rules.append(
            Rule([
                Proposition("gives", ("?giver", "?item")),
                Proposition("has", ("?giver", "?item")),
            ], Proposition("transfer_receives", ("?item", "?giver")), 1.0)
        )
    if "goes_to" in relations and "has" in relations:
        predicates.append("arrival_possession")
        narrative_rules.append(
            Rule([
                Proposition("goes_to", ("?who", "?place")),
                Proposition("has", ("?place", "?thing")),
            ], Proposition("arrival_possession", ("?who", "?thing")), 1.0)
        )

    mined_rules: List[Rule] = []
    mined_pred_names: List[str] = []
    if use_mined:
        mined_rules, mined_pred_names = mine_chain_rules(train_facts, max_rules=10, min_support=2)
        predicates.extend(mined_pred_names)

    all_rules = base_rules + combo_rules + neg_rules + narrative_rules + mined_rules

    model = SimpleDLN(predicates, args).to(device)
    store = RuleStore(model)
    store_path = Path(store_path)
    # Try to warm-start rule store if available
    if load_store and store_path.exists():
        store = RuleStore.load(model, store_path, sim_threshold=0.98)
    deduped_rules: List[Rule] = []
    for r in all_rules:
        added, _ = store.add(r)
        if added:
            deduped_rules.append(r)

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
        candidate_rules = candidate_rules[:max_candidate_rules]

    t0 = time.perf_counter()
    labels = _collect_labels(train_facts, candidate_rules, predicate_filter=None, use_disk_cache=disk_label_cache)
    t_collect = time.perf_counter() - t0
    # For comparison: full rule set label collection
    t0_full = time.perf_counter()
    labels_full = _collect_labels(train_facts, deduped_rules, predicate_filter=None, use_disk_cache=disk_label_cache)
    t_collect_full = time.perf_counter() - t0_full

    labels = {k: v for k, v in labels.items() if k[0].endswith("_inferred") or k[0].endswith("_combo") or k[0].startswith("not_") or k[0].endswith("_mined")}
    labels_full = {k: v for k, v in labels_full.items() if k[0].endswith("_inferred") or k[0].endswith("_combo") or k[0].startswith("not_") or k[0].endswith("_mined")}
    # Simple agreement check: compare label sets
    pruned_only = set(labels.keys()) - set(labels_full.keys())
    full_only = set(labels_full.keys()) - set(labels.keys())
    if not labels:
        print("No labels produced for TinyStories mini benchmark; skipping.")
        return None

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    t1 = time.perf_counter()
    final_mse = _train_on_labels(model, opt, train_facts, labels, steps, device=device, use_ar_aux=use_ar_aux, ar_weight=ar_weight)
    t_train = time.perf_counter() - t1

    # Evaluate on held-out split using the same labels generated from rules over eval facts
    eval_labels = _collect_labels(eval_facts, candidate_rules, predicate_filter=None, use_disk_cache=disk_label_cache)
    eval_labels = {k: v for k, v in eval_labels.items() if k[0].endswith("_inferred") or k[0].endswith("_combo") or k[0].startswith("not_") or k[0].endswith("_mined")}
    eval_mse = _eval_on_labels(model, eval_facts, eval_labels, device=device) if eval_labels else float("nan")
    eval_mae = _mae_on_labels(model, eval_facts, eval_labels, device=device) if eval_labels else float("nan")

    if save_store:
        store_path = Path(store_path)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store.save(store_path)
    elif save_mined:
        # save mined rules even if save_store not requested
        store_path = Path(store_path)
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store.save(store_path)

    print(
        f"TinyStories mini benchmark (inferred + combo + neg): train MSE={final_mse:.4f} on {len(labels)} labels, "
        f"eval MSE={eval_mse:.4f} MAE={eval_mae:.4f} on {len(eval_labels)} labels "
        f"({len(candidate_rules)} rules, collect {t_collect:.3f}s vs full {t_collect_full:.3f}s, "
        f"pruned_only={len(pruned_only)}, full_only={len(full_only)})"
    )
    return final_mse, len(labels)


def _make_label_cache_key(facts: List[Proposition], rules: List[Rule], predicate_filter: Optional[str]):
    facts_key = tuple((f.predicate, f.args, f.truth) for f in facts)
    rules_key = tuple(
        (
            tuple((prem.predicate, prem.args, prem.truth) for prem in r.premises),
            (r.conclusion.predicate, r.conclusion.args, r.conclusion.truth),
            r.weight,
        )
        for r in rules
    )
    return (facts_key, rules_key, predicate_filter)


def _collect_labels(
    facts: List[Proposition],
    rules: List[Rule],
    predicate_filter: Optional[str] = None,
    use_disk_cache: bool = True,
) -> Dict[Tuple[str, Tuple[str, ...]], float]:
    key = _make_label_cache_key(facts, rules, predicate_filter)
    if key in LABEL_CACHE:
        return LABEL_CACHE[key]

    disk_hit = False
    labels: Dict[Tuple[str, Tuple[str, ...]], float] = {}
    if use_disk_cache:
        LABEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        key_hash = hashlib.sha1(repr(key).encode("utf-8")).hexdigest()
        cache_file = LABEL_CACHE_DIR / f"labels_{key_hash}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    labels = pickle.load(f)
                    disk_hit = True
            except Exception:
                labels = {}

    if not labels:
        eng = SymbolicEngine()
        targets = eng.infer(facts, rules)
        for p in targets:
            if predicate_filter and p.predicate != predicate_filter:
                continue
            labels[(p.predicate, p.args)] = p.truth
        if use_disk_cache and not disk_hit:
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(labels, f)
            except Exception:
                pass

    LABEL_CACHE[key] = labels
    return labels


def _train_on_labels(model: SimpleDLN, opt: "torch.optim.Optimizer", facts: List[Proposition], labels: Dict[Tuple[str, Tuple[str, ...]], float], steps: int, device: str = "cpu", use_ar_aux: bool = False, ar_weight: float = 0.1) -> float:
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.0
        for (pred, args_tuple), truth in labels.items():
            premises = [p for p in facts if p.args == args_tuple]
            if not premises:
                continue
            out = model(premises, Proposition(pred, args_tuple, truth))
            target = torch.tensor([[truth]], dtype=torch.float32, device=device)
            loss = loss + F.mse_loss(out, target)
            if use_ar_aux and pred in model.pred_vocab:
                prem_repr = model.encode_premises(premises)
                logits = model.ar_head(prem_repr)
                target_idx = torch.tensor([model.pred_vocab[pred]], device=device)
                loss = loss + ar_weight * F.cross_entropy(logits, target_idx)
        loss.backward()
        opt.step()
    with torch.no_grad():
        total = 0.0
        count = 0
        for (pred, args_tuple), truth in labels.items():
            premises = [p for p in facts if p.args == args_tuple]
            if not premises:
                continue
            out = model(premises, Proposition(pred, args_tuple, truth))
            total += F.mse_loss(out, torch.tensor([[truth]], dtype=torch.float32, device=device)).item()
            count += 1
    return total / max(count, 1)


def _eval_on_labels(model: SimpleDLN, facts: List[Proposition], labels: Dict[Tuple[str, Tuple[str, ...]], float], device: str = "cpu") -> float:
    with torch.no_grad():
        total = 0.0
        count = 0
        for (pred, args_tuple), truth in labels.items():
            premises = [p for p in facts if p.args == args_tuple]
            if not premises:
                continue
            out = model(premises, Proposition(pred, args_tuple, truth))
            total += F.mse_loss(out, torch.tensor([[truth]], dtype=torch.float32, device=device)).item()
            count += 1
    return total / max(count, 1)


def _mae_on_labels(model: SimpleDLN, facts: List[Proposition], labels: Dict[Tuple[str, Tuple[str, ...]], float], device: str = "cpu") -> float:
    with torch.no_grad():
        total = 0.0
        count = 0
        for (pred, args_tuple), truth in labels.items():
            premises = [p for p in facts if p.args == args_tuple]
            if not premises:
                continue
            out = model(premises, Proposition(pred, args_tuple, truth)).item()
            total += abs(out - truth)
            count += 1
    return total / max(count, 1)


def symbolic_smoke_test():
    facts, rules = _toy_data()
    eng = SymbolicEngine()
    inferred = eng.infer(facts, rules)
    table = {(p.predicate, p.args): p.truth for p in inferred}
    assert math.isclose(table.get(("B", ("alice",)), 0.0), 1.0, rel_tol=1e-3)
    assert math.isclose(table.get(("B", ("bob",)), 0.0), 0.7, rel_tol=1e-3)
    return table


def rule_store_smoke_test():
    _require_torch()
    facts, rules = _toy_data()
    predicates = ["A", "B", "C"]
    args = ["<pad>", "alice", "bob", "?x"]
    model = SimpleDLN(predicates, args)
    store = RuleStore(model, sim_threshold=0.98)

    base_rule = rules[0]
    added, _ = store.add(base_rule)
    assert added, "Base rule should be accepted"

    dup_added, sim = store.add(base_rule)
    assert dup_added is False and sim >= 0.98

    other_rule = Rule([Proposition("B", ("?x",))], Proposition("A", ("?x",)), 1.0)
    added_other, _ = store.add(other_rule)
    assert added_other, "Different rule should be accepted"

    neigh = store.nearest(other_rule, topk=2)
    assert neigh and neigh[0][0] == other_rule

    tmp_path = Path("/tmp/rule_store_test.json")
    store.save(tmp_path)
    loaded = RuleStore.load(model, tmp_path, sim_threshold=0.98)
    assert len(loaded.rules) == len(store.rules)

    return store


def dln_smoke_test(steps: int = 60):
    _require_torch()
    facts, rules = _toy_data()
    predicates = ["A", "B"]
    args = ["<pad>", "alice", "bob", "?x"]
    model = SimpleDLN(predicates, args)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    labels = _collect_labels(facts, rules, predicate_filter="B")
    _train_on_labels(model, opt, facts, labels, steps)
    return model, labels


def benchmark_ga_vs_random(steps_main: int = 40, steps_ga: int = 20):
    _require_torch()
    torch.manual_seed(0)
    facts, rules = _toy_data()
    predicates = ["A", "B"]
    args = ["<pad>", "alice", "bob", "?x"]

    true_labels = _collect_labels(facts, rules, predicate_filter="B")

    def make_model():
        model = SimpleDLN(predicates, args)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        return model, opt

    base_model, base_opt = make_model()
    base_final = _train_on_labels(base_model, base_opt, facts, true_labels, steps_main)

    ga_model, ga_opt = make_model()
    ga_rules = ga_seed_rules(predicates, pop_size=6)
    ga_labels = _collect_labels(facts, ga_rules)
    if ga_labels:
        _train_on_labels(ga_model, ga_opt, facts, ga_labels, steps_ga)
    ga_final = _train_on_labels(ga_model, ga_opt, facts, true_labels, steps_main)

    print(f"Benchmark MSE → baseline: {base_final:.4f}, GA-seeded: {ga_final:.4f}")
    return base_final, ga_final


def paraconsistency_smoke_test():
    facts = [
        Proposition("P", ("x",), 1.0),
        Proposition("not_P", ("x",), 0.9),
        Proposition("U", ("u",), 1.0),
    ]
    rules = [
        Rule([Proposition("P", ("?x",))], Proposition("Q", ("?x",)), 1.0),
        Rule([Proposition("not_P", ("?x",))], Proposition("R", ("?x",)), 1.0),
        Rule([Proposition("U", ("?u",))], Proposition("V", ("?u",)), 1.0),
    ]
    eng = SymbolicEngine()
    inferred = eng.infer(facts, rules)
    table = {(p.predicate, p.args): p.truth for p in inferred}
    assert math.isclose(table.get(("Q", ("x",)), 0.0), 1.0, rel_tol=1e-3)
    assert math.isclose(table.get(("R", ("x",)), 0.0), 0.9, rel_tol=1e-3)
    assert math.isclose(table.get(("V", ("u",)), 0.0), 1.0, rel_tol=1e-3)
    assert ("Z", ("x",)) not in table
    return table


def paraconsistency_chain_test():
    facts = [
        Proposition("P", ("x",), 1.0),
        Proposition("not_P", ("x",), 0.8),
        Proposition("A", ("a",), 1.0),
    ]
    rules = [
        Rule([Proposition("P", ("?x",))], Proposition("Q", ("?x",)), 1.0),
        Rule([Proposition("not_P", ("?x",))], Proposition("R", ("?x",)), 1.0),
        Rule([Proposition("Q", ("?x",))], Proposition("S", ("?x",)), 1.0),
        Rule([Proposition("R", ("?x",))], Proposition("T", ("?x",)), 1.0),
        Rule([Proposition("A", ("?a",))], Proposition("B", ("?a",)), 1.0),
    ]
    eng = SymbolicEngine()
    inferred = eng.infer(facts, rules)
    table = {(p.predicate, p.args): p.truth for p in inferred}
    assert math.isclose(table.get(("Q", ("x",)), 0.0), 1.0, rel_tol=1e-3)
    assert math.isclose(table.get(("R", ("x",)), 0.0), 0.8, rel_tol=1e-3)
    assert math.isclose(table.get(("S", ("x",)), 0.0), 1.0, rel_tol=1e-3)
    assert math.isclose(table.get(("T", ("x",)), 0.0), 0.8, rel_tol=1e-3)
    assert math.isclose(table.get(("B", ("a",)), 0.0), 1.0, rel_tol=1e-3)
    assert ("Z", ("x",)) not in table
    return table


def run_all_smoke_tests(run_tiny: bool = True, run_ga: bool = True, run_para: bool = True, save_store: bool = False, load_store: bool = True, store_path: str = "data/processed/rule_store_tiny.json", use_mined: bool = True, contra_strength: float = 0.8, save_mined: bool = False, use_entity_registry: bool = True, prefer_davidsonian: bool = True, device: str = "cpu", disk_label_cache: bool = True, max_stories: int = 50, max_facts: int = 1000, use_ar_aux: bool = True, ar_weight: float = 0.1, max_candidate_rules: int = 200):
    sym_table = symbolic_smoke_test()
    if SimpleDLN is None:
        print("PyTorch not installed; skipping DLN smoke test. Symbolic test passed.")
        return
    model, labels = dln_smoke_test()
    _ = rule_store_smoke_test()
    if run_para:
        paraconsistency_smoke_test()
        paraconsistency_chain_test()
    if run_ga:
        benchmark_ga_vs_random()
    if run_tiny:
            tinystories_mini_benchmark(save_store=save_store, load_store=load_store, store_path=store_path, use_mined=use_mined, contra_strength=contra_strength, save_mined=save_mined, use_entity_registry=use_entity_registry, prefer_davidsonian=prefer_davidsonian, device=device, disk_label_cache=disk_label_cache, max_stories=max_stories, max_facts=max_facts, use_ar_aux=use_ar_aux, ar_weight=ar_weight, max_candidate_rules=max_candidate_rules)
    with torch.no_grad():
        for (pred, args_tuple), truth in labels.items():
            premises = [Proposition("A", args_tuple, sym_table[("A", args_tuple)])]
            pred_truth = model(premises, Proposition(pred, args_tuple)).item()
            print(f"Predict {pred}{args_tuple}: {pred_truth:.3f} (target {truth:.3f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid prototype benchmarks")
    parser.add_argument("--no-tiny", action="store_true", help="Skip TinyStories benchmark")
    parser.add_argument("--no-ga", action="store_true", help="Skip GA benchmark")
    parser.add_argument("--no-para", action="store_true", help="Skip paraconsistency tests")
    parser.add_argument("--save-store", action="store_true", help="Save RuleStore after TinyStories run")
    parser.add_argument("--load-store", action="store_true", help="Load RuleStore before TinyStories run")
    parser.add_argument("--store-path", type=str, default="data/processed/rule_store_tiny.json", help="Path for RuleStore load/save")
    parser.add_argument("--no-mined", action="store_true", help="Disable mined rules in TinyStories benchmark")
    parser.add_argument("--contra-strength", type=float, default=0.8, help="Strength of injected contradiction")
    parser.add_argument("--save-mined", action="store_true", help="Save RuleStore when mined rules are generated")
    parser.add_argument("--no-entity-registry", action="store_true", help="Disable entity registry canonicalization during TinyStories load")
    parser.add_argument("--svo-fallback", action="store_true", help="Use SVO extractor only (disable Davidsonian parser) for TinyStories load")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device for DLN (cpu or cuda)")
    parser.add_argument("--no-label-cache", action="store_true", help="Disable disk label cache for symbolic inference")
    parser.add_argument("--max-stories", type=int, default=50, help="Number of TinyStories to load")
    parser.add_argument("--max-facts", type=int, default=1000, help="Cap on facts loaded from TinyStories")
    parser.add_argument("--no-ar-aux", action="store_true", help="Disable auxiliary predicate prediction loss")
    parser.add_argument("--ar-weight", type=float, default=0.1, help="Weight for auxiliary predicate prediction loss")
    parser.add_argument("--max-candidate-rules", type=int, default=200, help="Cap on candidate rules used for label collection")
    args = parser.parse_args()

    run_all_smoke_tests(
        run_tiny=not args.no_tiny,
        run_ga=not args.no_ga,
        run_para=not args.no_para,
        save_store=args.save_store,
        load_store=args.load_store,
        store_path=args.store_path,
        use_mined=not args.no_mined,
        contra_strength=args.contra_strength,
        save_mined=args.save_mined,
        use_entity_registry=not args.no_entity_registry,
        prefer_davidsonian=not args.svo_fallback,
        device=args.device,
        disk_label_cache=not args.no_label_cache,
        max_stories=args.max_stories,
        max_facts=args.max_facts,
        use_ar_aux=not args.no_ar_aux,
        ar_weight=args.ar_weight,
        max_candidate_rules=args.max_candidate_rules,
    )
