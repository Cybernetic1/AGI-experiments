"""
Train a DLN-powered PoT decoder on exported logical-form pairs.
Replaces the GRU encoder with a LogicNetwork to process the input working memory.
"""

import sys
from pathlib import Path
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent dir to path to import neural_logic_core
sys.path.append(str(Path(__file__).resolve().parent.parent))
from neural_logic_core_vectorized import VectorizedLogicNetwork as LogicNetwork

# Import training utilities from baseline
import importlib.util
_spec_tpc = importlib.util.spec_from_file_location("tpc", str(Path(__file__).resolve().parent / "train_pot_clause.py"))
tpc = importlib.util.module_from_spec(_spec_tpc)
_spec_tpc.loader.exec_module(tpc)


class DLNPointerDecoder(nn.Module):
    def __init__(self, input_vocab: int, max_positions: int, hidden_dim: int = 128, num_rules: int = 8):
        super().__init__()
        self.input_embed = nn.Embedding(input_vocab, hidden_dim)
        self.position_embed = nn.Embedding(max_positions + 2, hidden_dim)
        
        # DLN acts as the encoder, reading the set of embedded clauses
        self.dln = LogicNetwork(
            prop_length=hidden_dim,
            num_props=max_positions,
            output_dim=hidden_dim,
            num_rules=num_rules,
            num_premises=2,
            var_slots=3
        )
        
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.eos_proj = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, decoder_input_ids):
        enc_emb = self.input_embed(input_ids)
        seq_len = enc_emb.size(1)
        if seq_len < self.dln.W:
            padding = torch.zeros(enc_emb.size(0), self.dln.W - seq_len, enc_emb.size(2), device=enc_emb.device)
            dln_input = torch.cat([enc_emb, padding], dim=1)
        else:
            dln_input = enc_emb[:, :self.dln.W, :]
        dln_out = self.dln(dln_input)
        h0 = torch.tanh(self.context_proj(dln_out)).unsqueeze(0)
        dec_emb = self.position_embed(decoder_input_ids)
        out, _ = self.decoder(dec_emb, h0)
        query = self.query_proj(out)
        copy_logits = torch.bmm(query, enc_emb.transpose(1, 2))
        eos_logits = self.eos_proj(out)
        return torch.cat([copy_logits, eos_logits], dim=-1)

    def beam_decode(self, input_ids, bos_id: int, eos_id: int, beam_width: int = 5, max_steps: int = 32, fixed_length=None):
        enc_emb = self.input_embed(input_ids)
        seq_len = enc_emb.size(1)
        if seq_len < self.dln.W:
            padding = torch.zeros(enc_emb.size(0), self.dln.W - seq_len, enc_emb.size(2), device=enc_emb.device)
            dln_input = torch.cat([enc_emb, padding], dim=1)
        else:
            dln_input = enc_emb[:, :self.dln.W, :]
        dln_out = self.dln(dln_input)
        start_h = torch.tanh(self.context_proj(dln_out)).unsqueeze(0)
        bos = self.position_embed(torch.tensor([[bos_id]], dtype=torch.long, device=input_ids.device))
        beams = [([], 0.0, start_h, bos, torch.zeros(enc_emb.size(1), dtype=torch.bool, device=input_ids.device), -1)]
        finished = []
        eos_index = enc_emb.size(1)
        steps = fixed_length if fixed_length is not None else max_steps
        for step in range(steps):
            new_beams = []
            for seq, score, h_state, last_embed, used_mask, last_pos in beams:
                if len(seq) > 0 and seq[-1] == eos_index:
                    finished.append((seq, score))
                    continue
                out, h_next = self.decoder(last_embed, h_state)
                copy_logits = torch.bmm(self.query_proj(out[:, -1:]), enc_emb.transpose(1, 2)).squeeze(1)
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

def main():
    parser = argparse.ArgumentParser(description="Train a DLN-powered PoT pointer decoder")
    parser.add_argument("--data", default="/tmp/pot_lt_pairs_clean_balanced2.jsonl")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num-rules", type=int, default=8)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agreement-only", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    if args.deterministic:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    rows = tpc.load_rows(Path(args.data))
    train_rows, eval_rows = tpc.split_rows(rows, args.holdout, args.seed)
    
    if args.agreement_only:
        train_rows = [row for row in train_rows if row.get("agreement")]
        eval_rows = [row for row in eval_rows if row.get("agreement")]
        
    train_rows = [row for row in train_rows if tpc.build_target_positions(row) is not None]
    eval_rows = [row for row in eval_rows if tpc.build_target_positions(row) is not None]

    input_vocab = tpc.build_input_vocab(rows)
    max_positions = max(len(row["input_props"]) for row in rows)
    
    model = DLNPointerDecoder(len(input_vocab), max_positions, args.hidden, args.num_rules)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)

    train_inputs = [tpc.encode_input(row, input_vocab) for row in train_rows]
    train_targets = [tpc.build_target_positions(row) for row in train_rows]
    
    best_exact = -1.0

    print(f"DLN Pointer Rows: train={len(train_rows)} eval={len(eval_rows)} in_vocab={len(input_vocab)}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        order = list(range(len(train_rows)))
        random.shuffle(order)
        for idx in order:
            row = train_rows[idx]
            input_ids = torch.tensor([train_inputs[idx]], dtype=torch.long)
            target_positions = train_targets[idx]
            decoder_input_ids = torch.tensor([tpc.encode_decoder_inputs(target_positions, max_positions + 1)], dtype=torch.long)
            logits = model(input_ids, decoder_input_ids)
            targets = torch.tensor([target_positions + [input_ids.size(1)]], dtype=torch.long)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            total_loss += loss.item()

        scheduler.step()

        metrics = tpc.evaluate(model, eval_rows, input_vocab)
        if metrics["exact"] > best_exact:
            best_exact = metrics["exact"]
            
        print(f"Epoch {epoch + 1:02d} | loss={total_loss / max(1, len(train_rows)):.4f} | exact={metrics['exact']:.3f} | clause_acc={metrics['clause_acc']:.3f}", flush=True)

    print(f"Final best exact: {best_exact:.3f}", flush=True)

if __name__ == "__main__":
    main()
