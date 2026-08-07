"""
Train a small autoregressive PoT decoder on exported logical-form pairs.

This is the next step after the bag-of-clauses baseline: predict the ordered
logical-form token sequence from spaCy-derived input propositions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import random
import re
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


TOKEN_RE = re.compile(r"\?[a-z]+\d+|[A-Za-z_]+|\d+|[(),.]")


def clause_text(prop: Dict[str, Sequence[str]]) -> str:
    pred = str(prop.get("pred", "")).strip()
    args = [str(a).strip() for a in prop.get("args", [])]
    return f"{pred}({', '.join(args)})."


def tokenize_form(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text))


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


def build_vocab(rows):
    counter = Counter()
    for row in rows:
        for prop in row["input_props"]:
            counter.update(tokenize_form(clause_text(prop)))
        for prop in row["target_props"]:
            counter.update(tokenize_form(clause_text(prop)))
    vocab = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
    for token, _ in counter.most_common():
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_input(props, vocab):
    tokens = ["<bos>"]
    for prop in props:
        tokens.extend(tokenize_form(clause_text(prop)))
    tokens.append("<eos>")
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokens], tokens


def encode_target(props, vocab):
    tokens = ["<bos>"]
    for prop in props:
        tokens.extend(tokenize_form(clause_text(prop)))
    tokens.append("<eos>")
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokens], tokens


class PoTSeqDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, token_ids):
        enc_emb = self.token_embed(input_ids)
        _, h = self.encoder(enc_emb)
        h0 = torch.tanh(self.context_proj(h[-1])).unsqueeze(0)
        emb = self.token_embed(token_ids[:, :-1])
        out, _ = self.gru(emb, h0)
        return self.output(out)

    def greedy_decode(self, input_ids, bos_id: int, eos_id: int, max_len: int = 256):
        enc_emb = self.token_embed(input_ids)
        _, h = self.encoder(enc_emb)
        h = torch.tanh(self.context_proj(h[-1])).unsqueeze(0)
        token = torch.tensor([[bos_id]], dtype=torch.long, device=input_ids.device)
        out_tokens = []
        for _ in range(max_len):
            emb = self.token_embed(token[:, -1:])
            out, h = self.gru(emb, h)
            logits = self.output(out[:, -1])
            next_id = int(logits.argmax(dim=-1).item())
            if next_id == eos_id:
                break
            out_tokens.append(next_id)
            token = torch.cat([token, torch.tensor([[next_id]], device=input_ids.device)], dim=1)
        return out_tokens

    def beam_decode(self, input_ids, bos_id: int, eos_id: int, beam_width: int = 5, max_len: int = 256):
        enc_emb = self.token_embed(input_ids)
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
                emb = self.token_embed(token)
                out, h_next = self.gru(emb, h_state)
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


def decode_tokens(ids, inv_vocab):
    return [inv_vocab[i] for i in ids if i in inv_vocab]


def evaluate(model, rows, vocab):
    model.eval()
    exact = 0
    token_acc_sum = 0.0
    total = 0
    by_family = defaultdict(lambda: [0, 0])
    inv_vocab = {i: t for t, i in vocab.items()}
    with torch.no_grad():
        for row in rows:
            input_ids, _ = encode_input(row["input_props"], vocab)
            input_tensor = torch.tensor([input_ids], dtype=torch.long)
            gold_ids, gold_tokens = encode_target(row["target_props"], vocab)
            pred_ids = model.beam_decode(input_tensor, vocab["<bos>"], vocab["<eos>"], beam_width=5)
            pred_tokens = decode_tokens(pred_ids, inv_vocab)
            gold_seq = gold_tokens[1:-1]
            if pred_tokens == gold_seq:
                exact += 1
            common = sum(1 for a, b in zip(pred_tokens, gold_seq) if a == b)
            denom = max(len(gold_seq), len(pred_tokens), 1)
            token_acc_sum += common / denom
            total += 1
            fam = row.get("family", "unknown")
            by_family[fam][0] += int(pred_tokens == gold_seq)
            by_family[fam][1] += 1
    return {
        "exact": exact / max(1, total),
        "token_acc": token_acc_sum / max(1, total),
        "by_family": {k: v[0] / max(1, v[1]) for k, v in by_family.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Train a sequence PoT decoder")
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
    model = PoTSeqDecoder(len(vocab), args.hidden)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_inputs = [encode_input(row["input_props"], vocab)[0] for row in train_rows]
    train_targets = [encode_target(row["target_props"], vocab)[0] for row in train_rows]
    max_len = max(len(t) for t in train_targets)

    print(f"Rows: train={len(train_rows)} eval={len(eval_rows)} vocab={len(vocab)} max_len={max_len}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        order = list(range(len(train_rows)))
        random.shuffle(order)
        for idx in order:
            input_ids = torch.tensor([train_inputs[idx]], dtype=torch.long)
            target = train_targets[idx]
            padded = target + [vocab["<pad>"]] * (max_len - len(target))
            token_ids = torch.tensor([padded], dtype=torch.long)
            logits = model(input_ids, token_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                token_ids[:, 1:].reshape(-1),
                ignore_index=vocab["<pad>"],
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            metrics = evaluate(model, eval_rows, vocab)
            print(
                f"Epoch {epoch + 1:02d} | loss={total_loss / max(1, len(train_rows)):.4f} "
                f"| exact={metrics['exact']:.3f} | token_acc={metrics['token_acc']:.3f}",
                flush=True,
            )

    metrics = evaluate(model, eval_rows, vocab)
    print(f"Final exact: {metrics['exact']:.3f}", flush=True)
    print(f"Final token_acc: {metrics['token_acc']:.3f}", flush=True)
    for family, acc in sorted(metrics["by_family"].items()):
        print(f"  {family}: {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
