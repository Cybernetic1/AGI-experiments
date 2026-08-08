"""
Compare Parameter Efficiency: Transformer vs LT (DLN Seq) on PoT dataset.
We dynamically scale down the hidden dimensions and layers/rules to observe
how accuracy degrades as parameter capacity hits 'starvation' levels.
"""

import sys
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import DLN Vectorized
sys.path.append(str(Path(__file__).resolve().parent.parent))
from neural_logic_core_vectorized import VectorizedLogicNetwork

# Import data loaders from baseline
import importlib.util
_spec_tpt = importlib.util.spec_from_file_location("tpt", str(Path(__file__).resolve().parent / "train_pot_transformer.py"))
tpt = importlib.util.module_from_spec(_spec_tpt)
_spec_tpt.loader.exec_module(tpt)

# ---------------------------------------------------------------------------
# 1. DLN Autoregressive Sequence Decoder
# ---------------------------------------------------------------------------
class DLNSeqDecoder(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, hidden_dim, num_rules, max_len=20):
        super().__init__()
        self.max_len = max_len
        self.src_emb = nn.Embedding(src_vocab, hidden_dim)
        self.tgt_emb = nn.Embedding(tgt_vocab, hidden_dim)
        
        self.dln = VectorizedLogicNetwork(
            prop_length=hidden_dim,
            num_props=max_len,
            output_dim=hidden_dim,
            num_rules=num_rules,
            num_premises=2,
            var_slots=3
        )
        
        self.decoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, tgt_vocab)

    def forward(self, src_ids, tgt_ids):
        # enc
        enc_emb = self.src_emb(src_ids)
        seq_len = enc_emb.size(1)
        if seq_len < self.max_len:
            padding = torch.zeros(enc_emb.size(0), self.max_len - seq_len, enc_emb.size(2), device=enc_emb.device)
            dln_input = torch.cat([enc_emb, padding], dim=1)
        else:
            dln_input = enc_emb[:, :self.max_len, :]
            
        dln_out = self.dln(dln_input) # (B, hidden_dim)
        h0 = dln_out.unsqueeze(0)     # (1, B, hidden_dim)
        
        # dec
        dec_emb = self.tgt_emb(tgt_ids)
        out, _ = self.decoder(dec_emb, h0)
        return self.out_proj(out)

    @torch.no_grad()
    def greedy_decode(self, src_ids, bos_id, eos_id, max_len=32):
        device = src_ids.device
        enc_emb = self.src_emb(src_ids)
        seq_len = enc_emb.size(1)
        if seq_len < self.max_len:
            padding = torch.zeros(enc_emb.size(0), self.max_len - seq_len, enc_emb.size(2), device=device)
            dln_input = torch.cat([enc_emb, padding], dim=1)
        else:
            dln_input = enc_emb[:, :self.max_len, :]
            
        dln_out = self.dln(dln_input)
        h = dln_out.unsqueeze(0)
        
        ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        for _ in range(max_len):
            dec_emb = self.tgt_emb(ys[:, -1:])
            out, h = self.decoder(dec_emb, h)
            logits = self.out_proj(out[:, -1])
            next_tok = logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == eos_id:
                break
        return ys[0, 1:].tolist() # omit bos

# ---------------------------------------------------------------------------
# 2. Traditional Transformer Sequence Decoder
# ---------------------------------------------------------------------------
class TinyTransformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, hidden_dim, num_layers, nhead, max_len=32):
        super().__init__()
        self.d_model = hidden_dim
        self.src_emb = nn.Embedding(src_vocab, hidden_dim)
        self.tgt_emb = nn.Embedding(tgt_vocab, hidden_dim)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim * 4, batch_first=True, dropout=0.1
        )
        self.out_proj = nn.Linear(hidden_dim, tgt_vocab)

    def forward(self, src_ids, tgt_ids):
        device = src_ids.device
        S, T = src_ids.size(1), tgt_ids.size(1)
        src_pos = torch.arange(S, device=device).unsqueeze(0)
        tgt_pos = torch.arange(T, device=device).unsqueeze(0)
        
        src_seq = self.src_emb(src_ids) * (self.d_model ** 0.5) + self.pos_emb(src_pos)
        tgt_seq = self.tgt_emb(tgt_ids) * (self.d_model ** 0.5) + self.pos_emb(tgt_pos)
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
        out = self.transformer(src_seq, tgt_seq, tgt_mask=tgt_mask)
        return self.out_proj(out)

    @torch.no_grad()
    def greedy_decode(self, src_ids, bos_id, eos_id, max_len=32):
        device = src_ids.device
        S = src_ids.size(1)
        src_pos = torch.arange(S, device=device).unsqueeze(0)
        src_seq = self.src_emb(src_ids) * (self.d_model ** 0.5) + self.pos_emb(src_pos)
        memory = self.transformer.encoder(src_seq)
        
        ys = torch.tensor([[bos_id]], dtype=torch.long, device=device)
        for _ in range(max_len):
            T = ys.size(1)
            tgt_pos = torch.arange(T, device=device).unsqueeze(0)
            tgt_seq = self.tgt_emb(ys) * (self.d_model ** 0.5) + self.pos_emb(tgt_pos)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)
            out = self.transformer.decoder(tgt_seq, memory, tgt_mask=tgt_mask)
            logits = self.out_proj(out[:, -1])
            next_tok = logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == eos_id:
                break
        return ys[0, 1:].tolist()

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_eval(model, train_rows, eval_rows, src_vocab, tgt_vocab, epochs=25, lr=1e-3):
    device = torch.device('cpu')
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_rows)))
        random.shuffle(order)
        for idx in order:
            row = train_rows[idx]
            src_ids = torch.tensor([tpt.encode_input(row, src_vocab)], dtype=torch.long, device=device)
            tgt_target = tpt.encode_target(row, tgt_vocab)
            tgt_in = torch.tensor([[1] + tgt_target], dtype=torch.long, device=device)  # 1=bos
            tgt_out = torch.tensor([tgt_target + [2]], dtype=torch.long, device=device) # 2=eos
            
            logits = model(src_ids, tgt_in)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()

    # Eval
    model.eval()
    exact = 0
    with torch.no_grad():
        for row in eval_rows:
            src_ids = torch.tensor([tpt.encode_input(row, src_vocab)], dtype=torch.long, device=device)
            gold = tpt.encode_target(row, tgt_vocab)
            pred = model.greedy_decode(src_ids, bos_id=1, eos_id=2, max_len=len(gold)+4)
            # Remove EOS if present
            if pred and pred[-1] == 2:
                pred = pred[:-1]
            if pred == gold:
                exact += 1
    return exact / max(1, len(eval_rows))

def main():
    print("="*60)
    print("PARAMETER EFFICIENCY SWEEP: TRANSFORMER vs LT (DLN)")
    print("="*60)

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    rows = tpt.load_rows(Path('/tmp/pot_diverse_lt_pairs.jsonl'))
    train_rows, eval_rows = tpt.split_rows(rows, holdout=0.2, seed=0)
    
    # We DO NOT filter by agreement anymore, because the diverse dataset expects disagreement
    # train_rows = [r for r in train_rows if r.get('agreement')]
    # eval_rows = [r for r in eval_rows if r.get('agreement')]

    src_vocab = tpt.build_input_vocab(rows)
    tgt_vocab = tpt.build_clause_vocab(rows)
    
    # Configurations: Target distinct parameter budgets
    configs = [
        {"name": "Large (~600k-800k)", "hidden": 128, "tf_layers": 2, "tf_nhead": 4, "dln_rules": 8},
        {"name": "Medium (~150k)",    "hidden": 64,  "tf_layers": 2, "tf_nhead": 2, "dln_rules": 4},
        {"name": "Small (~40k)",      "hidden": 32,  "tf_layers": 1, "tf_nhead": 2, "dln_rules": 2},
        {"name": "Tiny (~10k)",       "hidden": 16,  "tf_layers": 1, "tf_nhead": 1, "dln_rules": 1},
    ]

    for conf in configs:
        print(f"\n--- {conf['name']} ---")
        
        # Build Transformer
        tf_model = TinyTransformer(len(src_vocab), len(tgt_vocab), conf['hidden'], conf['tf_layers'], conf['tf_nhead'])
        tf_params = count_params(tf_model)
        
        # Build LT (DLN)
        lt_model = DLNSeqDecoder(len(src_vocab), len(tgt_vocab), conf['hidden'], conf['dln_rules'])
        lt_params = count_params(lt_model)
        
        print(f"Transformer Params: {tf_params:,}")
        print(f"LT (DLN) Params:    {lt_params:,}")
        
        print("Training Transformer...")
        tf_acc = train_and_eval(tf_model, train_rows, eval_rows, src_vocab, tgt_vocab, epochs=25)
        
        print("Training LT (DLN)...")
        lt_acc = train_and_eval(lt_model, train_rows, eval_rows, src_vocab, tgt_vocab, epochs=25)
        
        print(f"Result -> Transformer: {tf_acc*100:.1f}% | LT (DLN): {lt_acc*100:.1f}%")

if __name__ == "__main__":
    main()
