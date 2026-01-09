from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


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

    def backward_chain(
        self,
        goal: Proposition,
        facts: List[Proposition],
        rules: List[Rule],
        max_depth: int = 3,
    ) -> List[List[Proposition]]:
        """Goal-directed backward chaining for proof sketches.

        Returns a list of proof paths (each a list of propositions). Stops at
        max_depth to avoid exponential blowup. Uses unification on rule
        conclusions to generate subgoals.
        """

        kb: Dict[Tuple[str, Tuple[str, ...]], float] = {
            (f.predicate, f.args): f.truth for f in facts
        }
        proofs: List[List[Proposition]] = []

        def _prove_all(subgoals: List[Proposition], depth: int, trail: List[Proposition]):
            if not subgoals:
                proofs.append(trail)
                return
            if depth < 0:
                return
            first, rest = subgoals[0], subgoals[1:]
            _prove_goal(first, rest, depth, trail)

        def _prove_goal(target: Proposition, remaining: List[Proposition], depth: int, trail: List[Proposition]):
            key = (target.predicate, target.args)
            if key in kb:
                found = Proposition(target.predicate, target.args, kb[key])
                _prove_all(remaining, depth, trail + [found])
            if depth == 0:
                return
            for rule in rules:
                subst = _unify(rule.conclusion, target)
                if subst is None:
                    continue
                new_goals = [_apply_subst(p, subst, 1.0) for p in rule.premises]
                _prove_all(new_goals + remaining, depth - 1, trail + [target])

        _prove_all([goal], max_depth, [])
        return proofs


__all__ = [
    "Proposition",
    "Rule",
    "SymbolicEngine",
    "_apply_subst",
    "_unify",
    "_is_var",
]
