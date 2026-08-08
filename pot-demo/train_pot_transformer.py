"""
Train a Transformer encoder-decoder baseline over clause-level tokens.
Each input clause is a single token; target is a sequence of clause tokens.
This is a minimal, self-contained baseline for comparison with the pointer model.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import json
import random
import numpy as np
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def clause_text(prop: Dict[str, List[str]]) -> str:
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
    tokens = [clause_text(prop) for prop in row["input_props"]]
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokens]


def encode_target(row, vocab):
    tokens = [clause_text(prop) for prop in row["target_props"]]
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokens]


class ClauseTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512, dropout=0.1, max_len=64):
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=None):
        # src: (S, N), tgt: (T, N)
        device = src.device
        S, N = src.shape
        T, _ = tgt.shape
        src_pos = torch.arange(S, device=device).unsqueeze(1)
        tgt_pos = torch.arange(T, device=device).unsqueeze(1)
        src_emb = self.src_tok_emb(src) * (self.d_model ** 0.5) + self.pos_emb(src_pos)
        tgt_emb = self.tgt_tok_emb(tgt) * (self.d_model ** 0.5) + self.pos_emb(tgt_pos)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=src_key_padding_mask)
        logits = self.out(out)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src, src_key_padding_mask, bos_id, eos_id, max_len=32, device=None):
        device = device or src.device
        S = src.size(0)
        N = src.size(1)
        # encode
        src_pos = torch.arange(S, device=device).unsqueeze(1)
        src_emb = self.src_tok_emb(src) * (self.d_model ** 0.5) + self.pos_emb(src_pos)
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
        ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        for i in range(max_len):
            T = ys.size(0)
            tgt_pos = torch.arange(T, device=device).unsqueeze(1)
            tgt_emb = self.tgt_tok_emb(ys) * (self.d_model ** 0.5) + self.pos_emb(tgt_pos)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
            out = self.transformer.decoder(tgt_emb, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask)
            logits = self.out(out)  # (T, N, V)
            next_tok = logits[-1].argmax(dim=-1).unsqueeze(0)  # (1, N)
            ys = torch.cat([ys, next_tok], dim=0)
            if (next_tok == eos_id).all():
                break
        return ys[1:].squeeze(1).tolist()


def evaluate(model, rows, src_vocab, tgt_vocab, device):
    model.eval()
    exact = 0
    clause_acc_sum = 0.0
    total = 0
    by_family = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for row in rows:
            input_ids = torch.tensor(encode_input(row, src_vocab), dtype=torch.long, device=device).unsqueeze(1)  # (S,1)
            tgt_tokens = [clause_text(p) for p in row["target_props"]]
            pred_ids = model.greedy_decode(input_ids, None, bos_id=1, eos_id=2, max_len= max(1, len(tgt_tokens) + 4), device=device)
            # map ids to tokens
            inv = {v:k for k,v in tgt_vocab.items()}
            pred_tokens = [inv.get(i, "<unk>") for i in pred_ids if i != 0 and i != 1]
            # strip eos if present
            if pred_tokens and pred_tokens[-1] == inv.get(2):
                pred_tokens = pred_tokens[:-1]
            if pred_tokens == tgt_tokens:
                exact += 1
            common = sum(1 for a,b in zip(pred_tokens, tgt_tokens) if a == b)
            denom = max(len(pred_tokens), len(tgt_tokens), 1)
            clause_acc_sum += common / denom
            total += 1
            fam = row.get("family", "unknown")
            by_family[fam][0] += int(pred_tokens == tgt_tokens)
            by_family[fam][1] += 1
    return {"exact": exact / max(1, total), "clause_acc": clause_acc_sum / max(1, total), "by_family": {k: v[0] / max(1, v[1]) for k, v in by_family.items()}}


def main():
    parser = argparse.ArgumentParser(description="Train Transformer clause baseline")
    parser.add_argument("--data", default="pot-demo/data/lt_pairs.jsonl")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--agreement-only", action="store_true")
    args = parser.parse_args()

    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        except Exception:
            pass
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                print("Warning: could not enable deterministic cuDNN settings", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")
    rows = load_rows(data_path)
    train_rows, eval_rows = split_rows(rows, args.holdout, args.seed)
    if args.agreement_only:
        train_rows = [r for r in train_rows if r.get("agreement")]
        eval_rows = [r for r in eval_rows if r.get("agreement")]
    if not train_rows or not eval_rows:
        raise ValueError("No usable rows after filtering")

    src_vocab = build_input_vocab(rows)
    tgt_vocab = build_clause_vocab(rows)
    inv_tgt = {v:k for k,v in tgt_vocab.items()}
    model = ClauseTransformer(len(src_vocab), len(tgt_vocab), d_model=args.d_model, nhead=args.nhead, num_encoder_layers=args.layers, num_decoder_layers=args.layers).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=5e-4)

    print(f"Rows: train={len(train_rows)} eval={len(eval_rows)} src_vocab={len(src_vocab)} tgt_vocab={len(tgt_vocab)}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        order = list(range(len(train_rows)))
        random.shuffle(order)
        for idx in order:
            row = train_rows[idx]
            src_ids = torch.tensor(encode_input(row, src_vocab), dtype=torch.long, device=device).unsqueeze(1)  # (S,1)
            tgt_ids = encode_target(row, tgt_vocab)
            tgt_in = [1] + tgt_ids  # bos + target
            tgt_out = tgt_ids + [2]  # target + eos
            tgt_in_t = torch.tensor(tgt_in, dtype=torch.long, device=device).unsqueeze(1)  # (T,1)
            tgt_out_t = torch.tensor(tgt_out, dtype=torch.long, device=device)
            T = tgt_in_t.size(0)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
            logits = model(src_ids, tgt_in_t, src_key_padding_mask=None, tgt_key_padding_mask=None, tgt_mask=tgt_mask)  # (T, N, V)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out_t.reshape(-1), ignore_index=0)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            metrics = evaluate(model, eval_rows, src_vocab, tgt_vocab, device)
            print(f"Epoch {epoch+1:02d} | loss={total_loss/len(train_rows):.4f} | exact={metrics['exact']:.3f} | clause_acc={metrics['clause_acc']:.3f}", flush=True)

    metrics = evaluate(model, eval_rows, src_vocab, tgt_vocab, device)
    print(f"Final exact: {metrics['exact']:.3f}", flush=True)
    print(f"Final clause_acc: {metrics['clause_acc']:.3f}", flush=True)
    for family, acc in sorted(metrics["by_family"].items()):
        print(f"  {family}: {acc:.3f}", flush=True)


if __name__ == '__main__':
    main()
