import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
from torch.nn import functional as F
from logic_core import Proposition, Rule
from dln import SimpleDLN, _require_torch


class RuleStore:
    """Index of rules with similarity-based dedup, retrieval, and persistence."""

    def __init__(self, model: SimpleDLN, sim_threshold: float = 0.98):
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

    def save(self, path: str) -> None:
        data = {
            "rules": [
                {
                    "premises": [p.__dict__ for p in r.premises],
                    "conclusion": r.conclusion.__dict__,
                    "weight": r.weight,
                }
                for r in self.rules
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, model: SimpleDLN, path: str, sim_threshold: float = 0.98) -> "RuleStore":
        rs = cls(model, sim_threshold=sim_threshold)
        fpath = Path(path)
        if not fpath.exists():
            return rs
        with open(fpath, "r") as f:
            data = json.load(f)
        for entry in data.get("rules", []):
            prem = [Proposition(**p) for p in entry.get("premises", [])]
            concl = Proposition(**entry.get("conclusion", {}))
            rule = Rule(prem, concl, entry.get("weight", 1.0))
            rs.add(rule)
        return rs


__all__ = ["RuleStore"]
