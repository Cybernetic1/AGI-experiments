from typing import Dict, List, Tuple
import random

import torch
from torch.nn import functional as F

from logic_core import Proposition
from dln import SimpleDLN


def _train_on_labels(
    model: SimpleDLN,
    opt: "torch.optim.Optimizer",
    facts: List[Proposition],
    labels: Dict[Tuple[str, Tuple[str, ...]], float],
    steps: int,
    device: str = "cpu",
    use_ar_aux: bool = False,
    ar_weight: float = 0.1,
    batch_size: int = None,
) -> float:
    # Enable mini-batch training for large label sets
    use_batching = batch_size is not None and len(labels) > batch_size
    if use_batching:
        label_list = list(labels.items())
        print(f"[train] Using mini-batch training: {len(labels)} labels, batch_size={batch_size}", flush=True)
    
    for i in range(steps):
        opt.zero_grad()
        loss = 0.0
        
        # Select batch of labels
        if use_batching:
            batch_items = random.sample(label_list, min(batch_size, len(label_list)))
        else:
            batch_items = labels.items()
        
        for (pred, args_tuple), truth in batch_items:
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
        if (i + 1) % max(1, steps // 5) == 0:
            print(f"[train] step {i+1}/{steps}, loss={loss.item():.4f}", flush=True)
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


def _eval_on_labels(
    model: SimpleDLN,
    facts: List[Proposition],
    labels: Dict[Tuple[str, Tuple[str, ...]], float],
    device: str = "cpu",
) -> float:
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


def _mae_on_labels(
    model: SimpleDLN,
    facts: List[Proposition],
    labels: Dict[Tuple[str, Tuple[str, ...]], float],
    device: str = "cpu",
) -> float:
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


__all__ = ["_train_on_labels", "_eval_on_labels", "_mae_on_labels"]
