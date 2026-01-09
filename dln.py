from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from logic_core import Proposition


def _require_torch():
    if torch is None:
        raise ImportError(
            "PyTorch is required for DLN components. Install with `pip install torch`."
        )


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
        device = self.pred_embed.weight.device
        pred = torch.tensor([self.pred_vocab[prop.predicate]], device=device)
        emb = [self.pred_embed(pred)]
        for i in range(2):
            arg = prop.args[i] if i < len(prop.args) else "<pad>"
            arg_idx = self.arg_vocab.get(arg, 0)
            emb.append(self.arg_embed(torch.tensor([arg_idx], device=device)))
        if len(emb) == 2:
            emb.append(self.arg_embed(torch.tensor([0], device=device)))
        return torch.cat(emb, dim=-1)

    def forward(self, premises: List[Proposition], conclusion: Proposition) -> torch.Tensor:
        if not premises:
            raise ValueError("Need at least one premise")
        prem_vecs = torch.cat([self.encode_prop(p) for p in premises], dim=0)
        prem_repr = prem_vecs.mean(dim=0, keepdim=True)
        concl_repr = self.encode_prop(conclusion)
        features = torch.cat([prem_repr, concl_repr], dim=-1)
        return self.mlp(features)


__all__ = ["SimpleDLN", "_require_torch", "F"]
