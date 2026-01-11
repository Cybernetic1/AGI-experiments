from typing import Dict, List, Optional, Union
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

    def __init__(
        self,
        predicates: List[str],
        args: List[str],
        embed_dim: int = 32,
        preset_pred_embeddings: Optional[Dict[str, Union[torch.Tensor, List[float]]]] = None,
        preset_arg_embeddings: Optional[Dict[str, Union[torch.Tensor, List[float]]]] = None,
        freeze_presets: bool = False,
    ):
        _require_torch()
        super().__init__()
        self.pred_vocab = {p: i for i, p in enumerate(predicates)}
        self.arg_vocab = {a: i for i, a in enumerate(args)}
        self.pred_names = predicates
        self.arg_names = args
        self.embed_dim = embed_dim
        self.pred_embed = nn.Embedding(len(predicates), embed_dim)
        self.arg_embed = nn.Embedding(len(args), embed_dim)
        self.register_buffer(
            "frozen_pred_mask",
            torch.zeros(len(predicates), dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "frozen_arg_mask",
            torch.zeros(len(args), dtype=torch.bool),
            persistent=False,
        )
        self._apply_presets(preset_pred_embeddings, preset_arg_embeddings, freeze_presets)
        self._install_freeze_hooks()
        self.prop_dim = embed_dim * 3  # predicate + two args
        self.mlp = nn.Sequential(
            nn.Linear(self.prop_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
            nn.Sigmoid(),
        )
        # Auxiliary AR-style head: predict predicate class from premises encoding
        self.ar_head = nn.Linear(self.prop_dim, len(predicates))

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

    def encode_premises(self, premises: List[Proposition]) -> torch.Tensor:
        if not premises:
            raise ValueError("Need at least one premise")
        prem_vecs = torch.cat([self.encode_prop(p) for p in premises], dim=0)
        return prem_vecs.mean(dim=0, keepdim=True)

    def forward(self, premises: List[Proposition], conclusion: Proposition) -> torch.Tensor:
        prem_repr = self.encode_premises(premises)
        concl_repr = self.encode_prop(conclusion)
        features = torch.cat([prem_repr, concl_repr], dim=-1)
        return self.mlp(features)

    def _set_embedding_row(
        self,
        table: nn.Embedding,
        name_to_idx: Dict[str, int],
        frozen_mask: torch.Tensor,
        name: str,
        vec: Union[torch.Tensor, List[float]],
        freeze: bool,
    ) -> None:
        if name not in name_to_idx:
            return
        idx = name_to_idx[name]
        with torch.no_grad():
            v = vec
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=table.weight.dtype, device=table.weight.device)
            if v.numel() != table.embedding_dim:
                raise ValueError(f"Preset embedding for {name} has dim {v.numel()}, expected {table.embedding_dim}")
            table.weight[idx].copy_(v.to(table.weight.device))
        if freeze:
            frozen_mask[idx] = True

    def _apply_presets(
        self,
        pred_embs: Optional[Dict[str, Union[torch.Tensor, List[float]]]],
        arg_embs: Optional[Dict[str, Union[torch.Tensor, List[float]]]],
        freeze: bool,
    ) -> None:
        if pred_embs:
            for name, vec in pred_embs.items():
                self._set_embedding_row(self.pred_embed, self.pred_vocab, self.frozen_pred_mask, name, vec, freeze)
        if arg_embs:
            for name, vec in arg_embs.items():
                self._set_embedding_row(self.arg_embed, self.arg_vocab, self.frozen_arg_mask, name, vec, freeze)

    def _install_freeze_hooks(self) -> None:
        def _mask_grad(grad: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            if grad is None or not mask.any():
                return grad
            return grad * (~mask).unsqueeze(1)

        self.pred_embed.weight.register_hook(lambda g: _mask_grad(g, self.frozen_pred_mask))
        self.arg_embed.weight.register_hook(lambda g: _mask_grad(g, self.frozen_arg_mask))

    def export_embeddings(self) -> Dict[str, Dict[str, List[float]]]:
        """Return current embedding tables keyed by predicate/arg names."""
        pred = {name: self.pred_embed.weight[idx].detach().cpu().tolist() for name, idx in self.pred_vocab.items()}
        args = {name: self.arg_embed.weight[idx].detach().cpu().tolist() for name, idx in self.arg_vocab.items()}
        return {"predicates": pred, "args": args}

    def load_embeddings(
        self,
        pred_embs: Optional[Dict[str, Union[torch.Tensor, List[float]]]] = None,
        arg_embs: Optional[Dict[str, Union[torch.Tensor, List[float]]]] = None,
        freeze: bool = False,
    ) -> None:
        """Load embeddings by name; optionally freeze the provided rows."""
        self._apply_presets(pred_embs, arg_embs, freeze)


__all__ = ["SimpleDLN", "_require_torch", "F"]
