import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch

from logic_core import Proposition, Rule, SymbolicEngine
from dln import SimpleDLN, _require_torch, F
from rule_store import RuleStore
from rule_injection import RuleInjector
from label_utils import _collect_labels, _collect_labels_filtered
from core.train_utils import _train_on_labels, _eval_on_labels, _mae_on_labels
from pipelines.tinystories_pipeline import (
    tinystories_mini_benchmark,
    load_tinystories_facts,
    inject_contradiction,
    mine_chain_rules,
)


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


def rule_injection_smoke_test():
    _require_torch()
    facts, _ = _toy_data()
    predicates = ["A", "B", "C"]
    args = ["<pad>", "alice", "bob", "?x"]
    model = SimpleDLN(predicates, args)
    store = RuleStore(model, sim_threshold=0.95)
    injector = RuleInjector(model, store, transfer_threshold=0.4, lr=5e-3)
    new_rule = Rule([Proposition("A", ("?x",))], Proposition("C", ("?x",)), 1.0)
    accepted = injector.transfer_rule_to_dln(new_rule, facts, confidence=0.9, train_steps=6)
    assert accepted
    with torch.no_grad():
        alice = Proposition("A", ("alice",), 1.0)
        out = model([alice], Proposition("C", ("alice",), 1.0)).item()
    return out


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


def run_all_smoke_tests(run_tiny: bool = True, run_ga: bool = True, run_para: bool = True, save_store: bool = False, load_store: bool = True, store_path: str = "data/processed/rule_store_tiny.json", use_mined: bool = True, contra_strength: float = 0.8, save_mined: bool = False, use_entity_registry: bool = True, prefer_davidsonian: bool = True, device: str = "cpu", disk_label_cache: bool = True, max_stories: int = 50, max_facts: int = 1000, use_ar_aux: bool = True, ar_weight: float = 0.1, max_candidate_rules: int = 200, use_rule_injection: bool = True, ri_train_steps: int = 4, ri_threshold: float = 0.5, ri_lr: float = 1e-2, label_batch_size: int = 128, allowed_predicates: Optional[Set[str]] = None, train_batch_size: int = None):
    print("=" * 60, flush=True)
    print("STARTING BENCHMARK SUITE", flush=True)
    print("=" * 60, flush=True)
    print(f"[config] Symbolic smoke test", flush=True)
    sym_table = symbolic_smoke_test()
    print(f"[config] Symbolic smoke test PASSED", flush=True)
    if SimpleDLN is None:
        print("PyTorch not installed; skipping DLN smoke test. Symbolic test passed.")
        return
    print(f"[config] DLN smoke test", flush=True)
    model, labels = dln_smoke_test()
    print(f"[config] DLN smoke test PASSED", flush=True)
    print(f"[config] RuleStore smoke test", flush=True)
    _ = rule_store_smoke_test()
    print(f"[config] RuleStore smoke test PASSED", flush=True)
    print(f"[config] RuleInjection smoke test", flush=True)
    _ = rule_injection_smoke_test()
    print(f"[config] RuleInjection smoke test PASSED", flush=True)
    if run_para:
        print(f"[config] Paraconsistency tests", flush=True)
        paraconsistency_smoke_test()
        paraconsistency_chain_test()
        print(f"[config] Paraconsistency tests PASSED", flush=True)
    if run_ga:
        print(f"[config] GA vs Random benchmark", flush=True)
        benchmark_ga_vs_random()
        print(f"[config] GA vs Random benchmark PASSED", flush=True)
    if run_tiny:
        print("=" * 60, flush=True)
        print("STARTING TINYSTORIES BENCHMARK", flush=True)
        print("=" * 60, flush=True)
        tinystories_mini_benchmark(save_store=save_store, load_store=load_store, store_path=store_path, use_mined=use_mined, contra_strength=contra_strength, save_mined=save_mined, use_entity_registry=use_entity_registry, prefer_davidsonian=prefer_davidsonian, device=device, disk_label_cache=disk_label_cache, max_stories=max_stories, max_facts=max_facts, use_ar_aux=use_ar_aux, ar_weight=ar_weight, max_candidate_rules=max_candidate_rules, use_rule_injection=use_rule_injection, ri_train_steps=ri_train_steps, ri_threshold=ri_threshold, ri_lr=ri_lr, label_batch_size=label_batch_size, allowed_predicates=allowed_predicates, train_batch_size=train_batch_size)
        print("=" * 60, flush=True)
        print("TINYSTORIES BENCHMARK COMPLETE", flush=True)
        print("=" * 60, flush=True)
    with torch.no_grad():
        for (pred, args_tuple), truth in labels.items():
            premises = [Proposition("A", args_tuple, sym_table[("A", args_tuple)])]
            pred_truth = model(premises, Proposition(pred, args_tuple)).item()
            print(f"Predict {pred}{args_tuple}: {pred_truth:.3f} (target {truth:.3f})")
    print("=" * 60, flush=True)
    print("ALL BENCHMARKS COMPLETE", flush=True)
    print("=" * 60, flush=True)
