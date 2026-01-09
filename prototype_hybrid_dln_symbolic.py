"""
Prototype hybrid: symbolic fuzzy inference + simple DLN stub.
- Shared logical schema (Proposition, Rule) with fuzzy truth in [0,1]
- Symbolic forward chaining with non-explosive handling of contradictions
- Simple DLN that learns to mirror symbolic outputs
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - runtime guard
    torch = None
    nn = None
    F = None


def _require_torch():
    if torch is None:
        raise ImportError(
            "PyTorch is required for DLN components. Install with `pip install torch`."
        )


@dataclass(frozen=True)
class Proposition:
    predicate: str
    args: Tuple[str, ...]
    truth: float = 1.0


@dataclass(frozen=True)
class Rule:
    premises: List[Proposition]
    conclusion: Proposition
    weight: float = 1.0


def _is_var(token: str) -> bool:
    return token.startswith("?")


def _unify(pattern: Proposition, fact: Proposition) -> Optional[Dict[str, str]]:
    """Unify a pattern with possible variables against a fact."""
    if len(pattern.args) != len(fact.args):
        return None
    subst: Dict[str, str] = {}
    # predicate can also be a variable (second-order-ish pattern)
    if not _is_var(pattern.predicate) and pattern.predicate != fact.predicate:
        return None
    if _is_var(pattern.predicate):
        subst[pattern.predicate] = fact.predicate
    for p_arg, f_arg in zip(pattern.args, fact.args):
        if _is_var(p_arg):
            subst[p_arg] = f_arg
        elif p_arg != f_arg:
            return None
    return subst


def _apply_subst(prop: Proposition, subst: Dict[str, str], truth: float) -> Proposition:
    pred = subst.get(prop.predicate, prop.predicate)
    args = tuple(subst.get(a, a) for a in prop.args)
    return Proposition(pred, args, truth)


class SymbolicEngine:
    """Fuzzy, paraconsistent forward chaining."""

    def __init__(self, max_iters: int = 4):
        self.max_iters = max_iters

    @staticmethod
    def _combine_truth(truths: List[float], rule_weight: float) -> float:
        return max(0.0, min(1.0, min(truths) * rule_weight))

    def infer(self, facts: List[Proposition], rules: List[Rule]) -> List[Proposition]:
        kb: Dict[Tuple[str, Tuple[str, ...]], float] = {}
        for f in facts:
            key = (f.predicate, f.args)
            kb[key] = max(kb.get(key, 0.0), f.truth)

        for _ in range(self.max_iters):
            added = False
            for rule in rules:
                matches: List[Tuple[Dict[str, str], List[float]]] = []
                # enumerate all premise matches
                def backtrack(idx: int, subst: Dict[str, str], tvals: List[float]):
                    if idx == len(rule.premises):
                        matches.append((subst, tvals))
                        return
                    prem = rule.premises[idx]
                    for (pred, args), truth in list(kb.items()):
                        candidate = Proposition(pred, args, truth)
                        s = _unify(prem, candidate)
                        if s is not None:
                            merged = {**subst, **s}
                            backtrack(idx + 1, merged, tvals + [truth])
                backtrack(0, {}, [])

                for subst, tvals in matches:
                    concl = _apply_subst(rule.conclusion, subst, 1.0)
                    new_truth = self._combine_truth(tvals, rule.weight)
                    key = (concl.predicate, concl.args)
                    prev = kb.get(key, 0.0)
                    if new_truth > prev + 1e-6:
                        kb[key] = new_truth
                        added = True
            if not added:
                break

        return [Proposition(pred, args, truth) for (pred, args), truth in kb.items()]


if torch is None:
    class SimpleDLN:
        def __init__(self, *_, **__):
            _require_torch()
        def forward(self, *_, **__):
            _require_torch()
else:
    class SimpleDLN(nn.Module):
        """Lightweight DLN stub that predicts fuzzy truth for conclusions."""

        def __init__(self, predicates: List[str], args: List[str], embed_dim: int = 32):
            _require_torch()
            super().__init__()
            self.pred_vocab = {p: i for i, p in enumerate(predicates)}
            self.arg_vocab = {a: i for i, a in enumerate(args)}
            self.pred_embed = nn.Embedding(len(predicates), embed_dim)
            self.arg_embed = nn.Embedding(len(args), embed_dim)
            self.prop_dim = embed_dim * 3  # predicate + two args
            self.mlp = nn.Sequential(
                nn.Linear(self.prop_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 1),
                nn.Sigmoid(),
            )

        def encode_prop(self, prop: Proposition) -> torch.Tensor:
            pred = torch.tensor([self.pred_vocab[prop.predicate]])
            emb = [self.pred_embed(pred)]
            for i in range(2):
                arg = prop.args[i] if i < len(prop.args) else "<pad>"
                arg_idx = self.arg_vocab.get(arg, 0)
                emb.append(self.arg_embed(torch.tensor([arg_idx])))
            if len(emb) == 2:
                emb.append(self.arg_embed(torch.tensor([0])))
            return torch.cat(emb, dim=-1)

        def forward(self, premises: List[Proposition], conclusion: Proposition) -> torch.Tensor:
            if not premises:
                raise ValueError("Need at least one premise")
            prem_vecs = torch.cat([self.encode_prop(p) for p in premises], dim=0)
            prem_repr = prem_vecs.mean(dim=0, keepdim=True)
            concl_repr = self.encode_prop(conclusion)
            features = torch.cat([prem_repr, concl_repr], dim=-1)
            return self.mlp(features)


class RuleStore:
    """Index of rules with similarity-based dedup and retrieval."""

    def __init__(self, model: "SimpleDLN", sim_threshold: float = 0.98):
        _require_torch()
        self.model = model
        self.sim_threshold = sim_threshold
        self.rules: List[Rule] = []
        self.embs: List[torch.Tensor] = []
        self.by_concl: Dict[str, List[int]] = {}

    def _embed_rule(self, rule: Rule) -> torch.Tensor:
        with torch.no_grad():
            prem_vecs = torch.cat([self.model.encode_prop(p) for p in rule.premises], dim=0)
            prem_repr = prem_vecs.mean(dim=0, keepdim=True)
            concl_repr = self.model.encode_prop(rule.conclusion)
            return torch.cat([prem_repr, concl_repr], dim=-1)

    def add(self, rule: Rule) -> Tuple[bool, float]:
        emb = self._embed_rule(rule)
        max_sim = 0.0
        if self.embs:
            stacked = torch.cat(self.embs, dim=0)
            sims = F.cosine_similarity(emb, stacked, dim=-1)
            max_sim = sims.max().item()
            if max_sim >= self.sim_threshold:
                return False, max_sim
        idx = len(self.rules)
        self.rules.append(rule)
        self.embs.append(emb)
        self.by_concl.setdefault(rule.conclusion.predicate, []).append(idx)
        return True, max_sim

    def nearest(self, rule: Rule, topk: int = 3) -> List[Tuple[Rule, float]]:
        if not self.embs:
            return []
        emb = self._embed_rule(rule)
        stacked = torch.cat(self.embs, dim=0)
        sims = F.cosine_similarity(emb, stacked, dim=-1)
        k = min(topk, sims.numel())
        vals, idxs = torch.topk(sims, k)
        return [(self.rules[i], vals[j].item()) for j, i in enumerate(idxs)]

    def candidates_for_conclusion(self, predicate: str) -> List[Rule]:
        idxs = self.by_concl.get(predicate, [])
        return [self.rules[i] for i in idxs]


def _toy_data() -> Tuple[List[Proposition], List[Rule]]:
    # Simple rule: A(x) -> B(x)
    facts = [Proposition("A", ("alice",), 1.0), Proposition("A", ("bob",), 0.7)]
    rule = Rule([Proposition("A", ("?x",))], Proposition("B", ("?x",)), 1.0)
    return facts, [rule]


def ga_seed_rules(predicates: List[str], pop_size: int = 6) -> List[Rule]:
    """Very small GA-style seeding: sample predicate pairs as candidate rules."""
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
    table = { (p.predicate, p.args): p.truth for p in inferred }
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

    return store


def dln_smoke_test(steps: int = 60):
    _require_torch()
    facts, rules = _toy_data()
    predicates = ["A", "B"]
    args = ["<pad>", "alice", "bob", "?x"]
    model = SimpleDLN(predicates, args)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    eng = SymbolicEngine()

    # create training pairs from symbolic inference labels
    targets = eng.infer(facts, rules)
    labels = { (p.predicate, p.args): p.truth for p in targets if p.predicate == "B" }

    for _ in range(steps):
        opt.zero_grad()
        loss = 0.0
        for (pred, args_tuple), truth in labels.items():
            premises = [p for p in facts if p.predicate == "A" and p.args == args_tuple]
            if not premises:
                continue
            out = model(premises, Proposition(pred, args_tuple, truth))
            target = torch.tensor([[truth]], dtype=torch.float32)
            loss = loss + F.mse_loss(out, target)
        loss.backward()
        opt.step()
    return model, labels


def run_all_smoke_tests():
    sym_table = symbolic_smoke_test()
    if torch is None:
        print("PyTorch not installed; skipping DLN smoke test. Symbolic test passed.")
        return
    model, labels = dln_smoke_test()
    _ = rule_store_smoke_test()
    with torch.no_grad():
        for (pred, args_tuple), truth in labels.items():
            premises = [Proposition("A", args_tuple, sym_table[("A", args_tuple)])]
            pred_truth = model(premises, Proposition(pred, args_tuple)).item()
            print(f"Predict {pred}{args_tuple}: {pred_truth:.3f} (target {truth:.3f})")


if __name__ == "__main__":
    run_all_smoke_tests()
