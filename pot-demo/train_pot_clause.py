"""
Train a clause-aware PoT decoder on exported logical-form pairs.

Each logical form is predicted as a sequence of whole clauses rather than
individual tokens. This keeps the output space much smaller and better aligned
with the structure LT should ultimately learn.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import random
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def clause_text(prop: Dict[str, Sequence[str]]) -> str:
    pred = str(prop.get("pred", "")).strip()
    args = [str(a).strip() for a in prop.get("args", [])]
    return f"{pred}({', '.join(args)})."


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def split_rows(rows, holdout=0.2, seed=0):
    rows = list(rows)
    rng = random.Random(seed)
    rng.shuffle(rows)
    cut = max(1, int(len(rows) * (1.0 - holdout))) if len(rows) > 1 else len(rows)
    return rows[:cut], rows[cut:]


def build_input_vocab(rows):
    counter = Counter()
    for row in rows:
        for prop in row["input_props"]:
            counter.update([clause_text(prop)])
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def build_clause_vocab(rows):
    counter = Counter()
    for row in rows:
        counter.update(clause_text(prop) for prop in row["target_props"])
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_input(row, vocab):
    tokens = ["<bos>"] + [clause_text(prop) for prop in row["input_props"]] + ["<eos>"]
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokens]


def encode_target(row, vocab):
    tokens = ["<bos>"] + [clause_text(prop) for prop in row["target_props"]] + ["<eos>"]
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokens]


class PoTClauseDecoder(nn.Module):
    def __init__(self, input_vocab: int, clause_vocab: int, hidden_dim: int = 128):
        super().__init__()
        self.input_embed = nn.Embedding(input_vocab, hidden_dim)
        self.output_embed = nn.Embedding(clause_vocab, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, clause_vocab)

    def forward(self, input_ids, clause_ids):
        enc_emb = self.input_embed(input_ids)
        _, h = self.encoder(enc_emb)
        h0 = torch.tanh(self.context_proj(h[-1])).unsqueeze(0)
        dec_emb = self.output_embed(clause_ids[:, :-1])
        out, _ = self.decoder(dec_emb, h0)
        return self.output(out)

    def beam_decode(self, input_ids, bos_id: int, eos_id: int, beam_width: int = 5, max_len: int = 32):
        enc_emb = self.input_embed(input_ids)
        _, h = self.encoder(enc_emb)
        start_h = torch.tanh(self.context_proj(h[-1])).unsqueeze(0)
        beams = [([bos_id], 0.0, start_h)]
        finished = []
        for _ in range(max_len):
            new_beams = []
            for seq, score, h_state in beams:
                if seq[-1] == eos_id:
                    finished.append((seq, score))
                    continue
                token = torch.tensor([[seq[-1]]], dtype=torch.long, device=input_ids.device)
                emb = self.output_embed(token)
                out, h_next = self.decoder(emb, h_state)
                logits = self.output(out[:, -1])
                log_probs = F.log_softmax(logits, dim=-1)[0]
                top_scores, top_ids = torch.topk(log_probs, k=min(beam_width, log_probs.numel()))
                for tok_score, tok_id in zip(top_scores.tolist(), top_ids.tolist()):
                    new_beams.append((seq + [tok_id], score + tok_score, h_next))
            if not new_beams:
                break
            new_beams.sort(key=lambda item: item[1], reverse=True)
            beams = new_beams[:beam_width]
        candidates = finished + [(seq, score) for seq, score, _ in beams]
        best_seq = max(candidates, key=lambda item: item[1])[0] if candidates else [bos_id, eos_id]
        return [tok for tok in best_seq[1:] if tok != eos_id]


def decode_sequence(ids, inv_vocab):
    return [inv_vocab[i] for i in ids if i in inv_vocab]


def evaluate(model, rows, input_vocab, clause_vocab):
    model.eval()
    exact = 0
    clause_acc_sum = 0.0
    total = 0
    by_family = defaultdict(lambda: [0, 0])
    inv_clause_vocab = {i: t for t, i in clause_vocab.items()}
    with torch.no_grad():
        for row in rows:
            input_ids = torch.tensor([encode_input(row, input_vocab)], dtype=torch.long)
            gold_ids = encode_target(row, clause_vocab)
            pred_ids = model.beam_decode(input_ids, clause_vocab["<bos>"], clause_vocab["<eos>"], beam_width=5)
            pred_tokens = decode_sequence(pred_ids, inv_clause_vocab)
            gold_tokens = decode_sequence(gold_ids[1:-1], inv_clause_vocab)
            if pred_tokens == gold_tokens:
                exact += 1
            common = sum(1 for a, b in zip(pred_tokens, gold_tokens) if a == b)
            denom = max(len(pred_tokens), len(gold_tokens), 1)
            clause_acc_sum += common / denom
            total += 1
            fam = row.get("family", "unknown")
            by_family[fam][0] += int(pred_tokens == gold_tokens)
            by_family[fam][1] += 1
    return {
        "exact": exact / max(1, total),
        "clause_acc": clause_acc_sum / max(1, total),
        "by_family": {k: v[0] / max(1, v[1]) for k, v in by_family.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Train a clause-aware PoT decoder")
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
    input_vocab = build_input_vocab(rows)
    clause_vocab = build_clause_vocab(rows)
    model = PoTClauseDecoder(len(input_vocab), len(clause_vocab), args.hidden)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_inputs = [encode_input(row, input_vocab) for row in train_rows]
    train_targets = [encode_target(row, clause_vocab) for row in train_rows]
    max_len = max(len(t) for t in train_targets)

    print(f"Rows: train={len(train_rows)} eval={len(eval_rows)} in_vocab={len(input_vocab)} clause_vocab={len(clause_vocab)} max_len={max_len}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        order = list(range(len(train_rows)))
        random.shuffle(order)
        for idx in order:
            input_ids = torch.tensor([train_inputs[idx]], dtype=torch.long)
            target = train_targets[idx]
            padded = target + [clause_vocab["<pad>"]] * (max_len - len(target))
            clause_ids = torch.tensor([padded], dtype=torch.long)
            logits = model(input_ids, clause_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                clause_ids[:, 1:].reshape(-1),
                ignore_index=clause_vocab["<pad>"],
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            metrics = evaluate(model, eval_rows, input_vocab, clause_vocab)
            print(
                f"Epoch {epoch + 1:02d} | loss={total_loss / max(1, len(train_rows)):.4f} "
                f"| exact={metrics['exact']:.3f} | clause_acc={metrics['clause_acc']:.3f}",
                flush=True,
            )

    metrics = evaluate(model, eval_rows, input_vocab, clause_vocab)
    print(f"Final exact: {metrics['exact']:.3f}", flush=True)
    print(f"Final clause_acc: {metrics['clause_acc']:.3f}", flush=True)
    for family, acc in sorted(metrics["by_family"].items()):
        print(f"  {family}: {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
