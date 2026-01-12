from typing import Dict, List, Tuple, Optional

import torch
from torch.nn import functional as F

from logic_core import Proposition, Rule, SymbolicEngine
from dln import SimpleDLN, _require_torch
from rule_store import RuleStore


class RuleInjector:
    """Lightweight rule injection: dedup, gate by confidence, fine-tune DLN on inferred labels."""

    def __init__(
        self,
        model: SimpleDLN,
        store: RuleStore,
        engine: Optional[SymbolicEngine] = None,
        transfer_threshold: float = 0.5,
        lr: float = 1e-2,
    ):
        _require_torch()
        self.model = model
        self.store = store
        self.engine = engine or SymbolicEngine()
        self.transfer_threshold = transfer_threshold
        self.lr = lr

    def transfer_rule_to_dln(
        self,
        rule: Rule,
        facts: List[Proposition],
        confidence: float = 1.0,
        train_steps: int = 4,
    ) -> bool:
        """
        Inject a symbolic rule into the DLN.
        - Gate by confidence.
        - Deduplicate via RuleStore.
        - Fine-tune DLN on labels inferred with the new rule (conclusion predicate only).
        Returns True if the rule was accepted and used for training.
        """
        if confidence < self.transfer_threshold:
            return False

        added, _ = self.store.add(rule)
        if not added:
            return False

        labels = self._collect_labels(facts, [rule], predicate_filter=rule.conclusion.predicate)
        if not labels:
            return True

        device = next(self.model.parameters()).device
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self._train_on_labels(opt, facts, labels, train_steps, device)
        return True

    def _collect_labels(
        self,
        facts: List[Proposition],
        rules: List[Rule],
        predicate_filter: Optional[str] = None,
    ) -> Dict[Tuple[str, Tuple[str, ...]], float]:
        labels: Dict[Tuple[str, Tuple[str, ...]], float] = {}
        targets = self.engine.infer(facts, rules)
        for p in targets:
            if predicate_filter and p.predicate != predicate_filter:
                continue
            labels[(p.predicate, p.args)] = p.truth
        return labels

    def _train_on_labels(
        self,
        opt: "torch.optim.Optimizer",
        facts: List[Proposition],
        labels: Dict[Tuple[str, Tuple[str, ...]], float],
        steps: int,
        device: torch.device,
    ) -> None:
        for _ in range(steps):
            opt.zero_grad()
            loss = torch.zeros([], device=device)
            for (pred, args_tuple), truth in labels.items():
                premises = [p for p in facts if p.args == args_tuple]
                if not premises:
                    continue
                out = self.model(premises, Proposition(pred, args_tuple, truth))
                target = torch.tensor([[truth]], dtype=torch.float32, device=device)
                loss = loss + F.mse_loss(out, target)
            if loss.item() == 0.0:
                opt.zero_grad()
                continue
            loss.backward()
            opt.step()


__all__ = ["RuleInjector"]
