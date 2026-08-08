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
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple

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
    tokens = [clause_text(prop) for prop in row["input_props"]]
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokens]


def encode_decoder_inputs(target_positions, bos_index: int):
    return [bos_index] + target_positions


def build_target_positions(row) -> Optional[List[int]]:
    input_tokens = [clause_text(prop) for prop in row["input_props"]]
    used = [False] * len(input_tokens)
    positions: List[int] = []
    for prop in row["target_props"]:
        tok = clause_text(prop)
        pos = next((i for i, in_tok in enumerate(input_tokens) if not used[i] and in_tok == tok), None)
        if pos is None:
            return None
        used[pos] = True
        positions.append(pos)
    return positions


class PoTPointerDecoder(nn.Module):
    def __init__(self, input_vocab: int, max_positions: int, hidden_dim: int = 128):
        super().__init__()
        self.input_embed = nn.Embedding(input_vocab, hidden_dim)
        self.position_embed = nn.Embedding(max_positions + 2, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.eos_proj = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, decoder_input_ids):
        enc_emb = self.input_embed(input_ids)
        enc_out, h = self.encoder(enc_emb)
        h0 = torch.tanh(self.context_proj(h[-1])).unsqueeze(0)
        dec_emb = self.position_embed(decoder_input_ids)
        out, _ = self.decoder(dec_emb, h0)
        query = self.query_proj(out)
        copy_logits = torch.bmm(query, enc_out.transpose(1, 2))
        eos_logits = self.eos_proj(out)
        return torch.cat([copy_logits, eos_logits], dim=-1)

    def beam_decode(
        self,
        input_ids,
        bos_id: int,
        eos_id: int,
        beam_width: int = 5,
        max_steps: int = 32,
        fixed_length: Optional[int] = None,
    ):
        enc_emb = self.input_embed(input_ids)
        enc_out, h = self.encoder(enc_emb)
        start_h = torch.tanh(self.context_proj(h[-1])).unsqueeze(0)
        bos = self.position_embed(torch.tensor([[bos_id]], dtype=torch.long, device=input_ids.device))
        beams = [([], 0.0, start_h, bos, torch.zeros(enc_out.size(1), dtype=torch.bool, device=input_ids.device), -1)]
        finished = []
        eos_index = enc_out.size(1)
        steps = fixed_length if fixed_length is not None else max_steps
        for step in range(steps):
            new_beams = []
            for seq, score, h_state, last_embed, used_mask, last_pos in beams:
                if len(seq) > 0 and seq[-1] == eos_index:
                    finished.append((seq, score))
                    continue
                out, h_next = self.decoder(last_embed, h_state)
                copy_logits = torch.bmm(self.query_proj(out[:, -1:]), enc_out.transpose(1, 2)).squeeze(1)
                if last_pos >= 0:
                    copy_logits[:, : last_pos + 1] = float("-inf")
                copy_logits[:, : used_mask.numel()] = copy_logits[:, : used_mask.numel()].masked_fill(used_mask.unsqueeze(0), float("-inf"))
                eos_logits = self.eos_proj(out[:, -1])
                logits = torch.cat([copy_logits, eos_logits], dim=-1)
                if fixed_length is not None and step + 1 < fixed_length:
                    logits[:, eos_index] = float("-inf")
                if fixed_length is not None and step + 1 == fixed_length:
                    logits[:, eos_index] = float("-inf")
                log_probs = F.log_softmax(logits, dim=-1)[0]
                top_scores, top_ids = torch.topk(log_probs, k=min(beam_width, log_probs.numel()))
                for tok_score, tok_id in zip(top_scores.tolist(), top_ids.tolist()):
                    if tok_id == eos_index:
                        new_beams.append((seq + [tok_id], score + tok_score, h_next, last_embed, used_mask, last_pos))
                    else:
                        next_embed = self.position_embed(torch.tensor([[tok_id]], dtype=torch.long, device=input_ids.device))
                        next_mask = used_mask.clone()
                        next_mask[tok_id] = True
                        new_beams.append((seq + [tok_id], score + tok_score, h_next, next_embed, next_mask, tok_id))
            if not new_beams:
                break
            new_beams.sort(key=lambda item: item[1], reverse=True)
            beams = new_beams[:beam_width]
        candidates = finished + [(seq, score) for seq, score, _, _, _, _ in beams]
        best_seq = max(candidates, key=lambda item: item[1])[0] if candidates else [eos_id]
        return [tok for tok in best_seq if tok != eos_id]


def decode_positions(row, positions):
    input_tokens = [clause_text(prop) for prop in row["input_props"]]
    return [input_tokens[i] for i in positions if 0 <= i < len(input_tokens)]


def evaluate(model, rows, input_vocab):
    model.eval()
    exact = 0
    clause_acc_sum = 0.0
    total = 0
    by_family = defaultdict(lambda: [0, 0])
    with torch.no_grad():
        for row in rows:
            target_positions = build_target_positions(row)
            if target_positions is None:
                continue
            input_ids = torch.tensor([encode_input(row, input_vocab)], dtype=torch.long)
            gold_tokens = [clause_text(prop) for prop in row["target_props"]]
            pred_ids = model.beam_decode(
                input_ids,
                max(len(row["input_props"]) for row in rows) + 1,
                len(row["input_props"]),
                beam_width=5,
                max_steps=len(row["input_props"]),
                fixed_length=len(row["input_props"]),
            )
            pred_tokens = decode_positions(row, pred_ids)
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
    parser.add_argument("--agreement-only", action="store_true", help="Train only on rows where parser and gold agree")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic PyTorch/CUDA behavior and seed RNGs")
    args = parser.parse_args()

    # Seed RNGs for reproducibility when deterministic requested
    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        try:
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.seed)
        except Exception:
            pass
        # Try to enable fully deterministic algorithms where available
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                print("Warning: could not enable deterministic cuDNN settings", flush=True)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    rows = load_rows(data_path)
    if not rows:
        raise ValueError(f"No examples found in {data_path}")

    train_rows, eval_rows = split_rows(rows, args.holdout, args.seed)
    if args.agreement_only:
        train_rows = [row for row in train_rows if row.get("agreement")]
        eval_rows = [row for row in eval_rows if row.get("agreement")]
    train_rows = [row for row in train_rows if build_target_positions(row) is not None]
    eval_rows = [row for row in eval_rows if build_target_positions(row) is not None]
    if not train_rows or not eval_rows:
        raise ValueError("No usable rows after filtering; generate cleaner data or drop --agreement-only")

    input_vocab = build_input_vocab(rows)
    max_positions = max(len(row["input_props"]) for row in rows)
    model = PoTPointerDecoder(len(input_vocab), max_positions, args.hidden)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)

    train_inputs = [encode_input(row, input_vocab) for row in train_rows]
    train_targets = [build_target_positions(row) for row in train_rows]
    max_len = max(len(t) for t in train_targets if t is not None)

    best_exact = -1.0
    best_epoch = -1

    print(f"Rows: train={len(train_rows)} eval={len(eval_rows)} in_vocab={len(input_vocab)} max_len={max_len}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        order = list(range(len(train_rows)))
        random.shuffle(order)
        for idx in order:
            row = train_rows[idx]
            input_ids = torch.tensor([train_inputs[idx]], dtype=torch.long)
            target_positions = train_targets[idx]
            assert target_positions is not None
            decoder_input_ids = torch.tensor([encode_decoder_inputs(target_positions, max_positions + 1)], dtype=torch.long)
            logits = model(input_ids, decoder_input_ids)
            targets = torch.tensor([target_positions + [input_ids.size(1)]], dtype=torch.long)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            optim.zero_grad()
            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            total_loss += loss.item()

        # scheduler step
        try:
            scheduler.step()
        except Exception:
            pass

        metrics = evaluate(model, eval_rows, input_vocab)
        print(
            f"Epoch {epoch + 1:02d} | loss={total_loss / max(1, len(train_rows)):.4f} "
            f"| exact={metrics['exact']:.3f} | clause_acc={metrics['clause_acc']:.3f}",
            flush=True,
        )

        # save best checkpoint by exact-match
        if metrics["exact"] > best_exact:
            best_exact = metrics["exact"]
            best_epoch = epoch
            ck = {
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "epoch": epoch,
                "exact": best_exact,
            }
            torch.save(ck, f"/tmp/pot_clause_seed{args.seed}_best.pt")

    metrics = evaluate(model, eval_rows, input_vocab)
    print(f"Final exact: {metrics['exact']:.3f}", flush=True)
    print(f"Final clause_acc: {metrics['clause_acc']:.3f}", flush=True)
    for family, acc in sorted(metrics["by_family"].items()):
        print(f"  {family}: {acc:.3f}", flush=True)


if __name__ == "__main__":
    main()
