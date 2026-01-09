import json
import time
import random
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from logic_core import Proposition, Rule, SymbolicEngine
from dln import SimpleDLN, _require_torch, F
from rule_store import RuleStore


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


def load_tinystories_facts(max_stories: int = 20, max_facts: int = 400, path: str = "data/processed/tinystories_train.json") -> List[Proposition]:
    fpath = Path(path)
    if not fpath.exists():
        print(f"TinyStories file not found at {path}; skipping TinyStories mini benchmark.")
        return []
    with open(fpath, "r") as f:
        data = json.load(f)
    facts: List[Proposition] = []
    for story in data[:max_stories]:
        for fact in story.get("facts", []):
            subj = str(fact.get("subject", ""))
            obj = str(fact.get("object", ""))
            rel = str(fact.get("relation", ""))
            if not subj or not obj or not rel:
                continue
            facts.append(Proposition(rel, (subj, obj), 1.0))
            if len(facts) >= max_facts:
                return facts
    return facts


def tinystories_mini_benchmark(steps: int = 40) -> Optional[Tuple[float, int]]:
    _require_torch()
    facts = load_tinystories_facts()
    if not facts:
        return None

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

    all_rules = base_rules + combo_rules

    model = SimpleDLN(predicates, args)
    store = RuleStore(model)
    deduped_rules: List[Rule] = []
    for r in all_rules:
        added, _ = store.add(r)
        if added:
            deduped_rules.append(r)

    target_preds = set()
    for r in deduped_rules:
        if r.conclusion.predicate.endswith("_inferred") or r.conclusion.predicate.endswith("_combo"):
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

    t0 = time.perf_counter()
    labels = _collect_labels(facts, candidate_rules, predicate_filter=None)
    t_collect = time.perf_counter() - t0

    labels = {k: v for k, v in labels.items() if k[0].endswith("_inferred") or k[0].endswith("_combo")}
    if not labels:
        print("No labels produced for TinyStories mini benchmark; skipping.")
        return None

    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    t1 = time.perf_counter()
    final_mse = _train_on_labels(model, opt, facts, labels, steps)
    t_train = time.perf_counter() - t1

    print(
        f"TinyStories mini benchmark (inferred + combo): MSE={final_mse:.4f} on {len(labels)} labels "
        f"({len(candidate_rules)} rules, collect {t_collect:.3f}s, train {t_train:.3f}s)"
    )
    return final_mse, len(labels)


def _collect_labels(facts: List[Proposition], rules: List[Rule], predicate_filter: Optional[str] = None) -> Dict[Tuple[str, Tuple[str, ...]], float]:
    eng = SymbolicEngine()
    targets = eng.infer(facts, rules)
    labels = {}
    for p in targets:
        if predicate_filter and p.predicate != predicate_filter:
            continue
        labels[(p.predicate, p.args)] = p.truth
    return labels


def _train_on_labels(model: SimpleDLN, opt: "torch.optim.Optimizer", facts: List[Proposition], labels: Dict[Tuple[str, Tuple[str, ...]], float], steps: int) -> float:
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.0
        for (pred, args_tuple), truth in labels.items():
            premises = [p for p in facts if p.args == args_tuple]
            if not premises:
                continue
            out = model(premises, Proposition(pred, args_tuple, truth))
            target = torch.tensor([[truth]], dtype=torch.float32)
            loss = loss + F.mse_loss(out, target)
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
            total += F.mse_loss(out, torch.tensor([[truth]], dtype=torch.float32)).item()
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


def run_all_smoke_tests():
    sym_table = symbolic_smoke_test()
    if SimpleDLN is None:
        print("PyTorch not installed; skipping DLN smoke test. Symbolic test passed.")
        return
    model, labels = dln_smoke_test()
    _ = rule_store_smoke_test()
    paraconsistency_smoke_test()
    paraconsistency_chain_test()
    benchmark_ga_vs_random()
    tinystories_mini_benchmark()
    with torch.no_grad():
        for (pred, args_tuple), truth in labels.items():
            premises = [Proposition("A", args_tuple, sym_table[("A", args_tuple)])]
            pred_truth = model(premises, Proposition(pred, args_tuple)).item()
            print(f"Predict {pred}{args_tuple}: {pred_truth:.3f} (target {truth:.3f})")


if __name__ == "__main__":
    run_all_smoke_tests()
