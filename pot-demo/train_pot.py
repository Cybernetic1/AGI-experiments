"""
Train a minimal multi-label baseline on exported PoT logical-form pairs.

This is a lightweight end-to-end check that the spaCy front-end and the
logical-form supervision can be consumed by a learner.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import random
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def prop_token(prop: Dict[str, Sequence[str]]) -> str:
    pred = str(prop.get("pred", "")).strip()
    args = [str(a).strip() for a in prop.get("args", [])]
    return "|".join([pred, *args])


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_vocab(rows):
    counter = Counter()
    for row in rows:
        counter.update(prop_token(p) for p in row["input_props"])
        counter.update(prop_token(p) for p in row["target_props"])
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_props(props, vocab):
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    for prop in props:
        idx = vocab.get(prop_token(prop), vocab["<unk>"])
        vec[idx] += 1.0
    return vec


def split_rows(rows, holdout=0.2, seed=0):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    cut = max(1, int(len(rows) * (1.0 - holdout))) if len(rows) > 1 else len(rows)
    return rows[:cut], rows[cut:]


class PoTMultiLabel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def multi_hot(indices, size):
    y = torch.zeros(size, dtype=torch.float32)
    for idx in indices:
        y[idx] = 1.0
    return y


def evaluate(model, rows, vocab, threshold=0.5):
    model.eval()
    exact = 0
    total = 0
    f1_sum = 0.0
    by_family = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for row in rows:
            x = encode_props(row["input_props"], vocab).unsqueeze(0)
            logits = model(x)[0]
            probs = torch.sigmoid(logits)
            gold = sorted({prop_token(p) for p in row["target_props"]})
            gold_ids = {vocab.get(tok, 1) for tok in gold if tok in vocab}
            k = max(1, len(gold_ids))
            pred_ids = set(torch.topk(probs, k=min(k, probs.numel())).indices.tolist())
            pred_ids = {idx for idx in pred_ids if idx not in {0, 1}}
            if pred_ids == gold_ids:
                exact += 1
            tp = len(pred_ids & gold_ids)
            prec = tp / max(1, len(pred_ids))
            rec = tp / max(1, len(gold_ids))
            f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
            f1_sum += f1
            total += 1
            fam = row.get("family", "unknown")
            by_family[fam][0] += int(pred_ids == gold_ids)
            by_family[fam][1] += 1
    return {
        "exact": exact / max(1, total),
        "f1": f1_sum / max(1, total),
        "by_family": {k: v[0] / max(1, v[1]) for k, v in by_family.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Train a minimal PoT baseline")
    parser.add_argument("--data", default="pot-demo/data/lt_pairs.jsonl")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    rows = load_rows(data_path)
    if not rows:
        raise ValueError(f"No examples found in {data_path}")

    train_rows, eval_rows = split_rows(rows, args.holdout, args.seed)
    vocab = build_vocab(rows)

    model = PoTMultiLabel(len(vocab), args.hidden, len(vocab))
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_features = [encode_props(row["input_props"], vocab) for row in train_rows]
    train_targets = []
    for row in train_rows:
        target_ids = [vocab.get(prop_token(p), 1) for p in row["target_props"] if prop_token(p) in vocab]
        train_targets.append(multi_hot(target_ids, len(vocab)))

    print(f"Rows: train={len(train_rows)} eval={len(eval_rows)} vocab={len(vocab)}")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        order = list(range(len(train_rows)))
        random.shuffle(order)
        for idx in order:
            x = train_features[idx].unsqueeze(0)
            y = train_targets[idx].unsqueeze(0)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            metrics = evaluate(model, eval_rows, vocab)
            print(
                f"Epoch {epoch + 1:02d} | loss={total_loss / max(1, len(train_rows)):.4f} "
                f"| exact={metrics['exact']:.3f} | f1={metrics['f1']:.3f}"
            )

    metrics = evaluate(model, eval_rows, vocab)
    print(f"Final exact: {metrics['exact']:.3f}")
    print(f"Final F1: {metrics['f1']:.3f}")
    for family, acc in sorted(metrics["by_family"].items()):
        print(f"  {family}: {acc:.3f}")


if __name__ == "__main__":
    main()
