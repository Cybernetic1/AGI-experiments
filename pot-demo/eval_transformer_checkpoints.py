"""
Load best checkpoints saved at /tmp/transformer_seed{seed}_best.pt and evaluate on holdout.
If a checkpoint for a seed is missing, optionally train with debug trainer to produce it.
"""
from __future__ import annotations
from pathlib import Path
import json, random, argparse
import numpy as np
import torch
import torch.nn as nn
from collections import Counter, defaultdict

# Reuse ClauseTransformer from debug file to ensure compatibility
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
            logits = self.out(out)
            next_tok = logits[-1].argmax(dim=-1).unsqueeze(0)
            ys = torch.cat([ys, next_tok], dim=0)
            if (next_tok == eos_id).all():
                break
        return ys[1:].squeeze(1).tolist()


def clause_text(prop):
    pred = str(prop.get('pred','')).strip()
    args = [str(a).strip() for a in prop.get('args',[])]
    return f"{pred}({', '.join(args)})."


def load_rows(path):
    rows=[]
    with open(path,'r',encoding='utf8') as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_input_vocab(rows):
    counter=Counter()
    for row in rows:
        for p in row['input_props']:
            counter.update([clause_text(p)])
    vocab={'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3}
    for t,_ in counter.most_common():
        if t not in vocab:
            vocab[t]=len(vocab)
    return vocab


def build_clause_vocab(rows):
    counter=Counter()
    for row in rows:
        for p in row['target_props']:
            counter.update([clause_text(p)])
    vocab={'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3}
    for t,_ in counter.most_common():
        if t not in vocab:
            vocab[t]=len(vocab)
    return vocab


def evaluate_checkpoint(checkpoint_path, data_path):
    rows = load_rows(data_path)
    train = rows
    src_vocab = build_input_vocab(rows)
    tgt_vocab = build_clause_vocab(rows)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClauseTransformer(len(src_vocab), len(tgt_vocab)).to(device)
    ck = torch.load(checkpoint_path,map_location=device)
    model.load_state_dict(ck['model_state'])
    # evaluate on holdout split: we'll reproduce split with seed used in filename
    # extract seed
    name = Path(checkpoint_path).name
    seed=None
    import re
    m=re.search(r'seed(\d+)', name)
    if m:
        seed=int(m.group(1))
    else:
        seed=0
    # split
    train_rows, eval_rows = [], []
    rows_all = rows.copy()
    rng = random.Random(seed)
    rng.shuffle(rows_all)
    cut = max(1, int(len(rows_all)*(1.0-0.2)))
    train_rows = rows_all[:cut]
    eval_rows = rows_all[cut:]
    # evaluate
    def inv_vocab(v):
        return {v:k for k,v in v.items()}
    inv_tgt = inv_vocab(tgt_vocab)
    model.eval()
    exact=0
    clause_acc_sum=0.0
    total=0
    with torch.no_grad():
        for row in eval_rows:
            src_ids = torch.tensor([src_vocab.get(clause_text(p), src_vocab['<unk>']) for p in row['input_props']],dtype=torch.long).unsqueeze(1).to(device)
            tgt_tokens = [clause_text(p) for p in row['target_props']]
            pred_ids = model.greedy_decode(src_ids, None, bos_id=1, eos_id=2, max_len=max(1,len(tgt_tokens)+4), device=device)
            pred_tokens=[inv_tgt.get(i,'<unk>') for i in pred_ids if i not in (0,1)]
            if pred_tokens and pred_tokens[-1]==inv_tgt.get(2):
                pred_tokens=pred_tokens[:-1]
            if pred_tokens==tgt_tokens:
                exact+=1
            common=sum(1 for a,b in zip(pred_tokens,tgt_tokens) if a==b)
            denom=max(len(pred_tokens),len(tgt_tokens),1)
            clause_acc_sum+=common/denom
            total+=1
    return {'path':str(checkpoint_path),'exact':exact/total,'clause_acc':clause_acc_sum/total,'seed':seed}

if __name__=='__main__':
    import glob
    data='/tmp/pot_lt_pairs_clean_balanced2.jsonl'
    results=[]
    for s in range(10):
        ck=Path(f'/tmp/transformer_seed{s}_best.pt')
        if not ck.exists():
            # produce by running debug trainer
            print(f'Checkpoint missing for seed {s}, training debug to produce it...')
            import subprocess
            subprocess.check_call(['python','pot-demo/train_pot_transformer_debug.py','--data',data,'--epochs','20','--seed',str(s),'--agreement-only','--deterministic'])
        print(f'Evaluating seed {s}')
        res=evaluate_checkpoint(f'/tmp/transformer_seed{s}_best.pt',data)
        print(res)
        results.append(res)
    print('\nSUMMARY')
    exs=[r['exact'] for r in results]
    import statistics
    print('mean_exact',statistics.mean(exs),'std',statistics.pstdev(exs))
    # save
    Path('/tmp/transformer_ckpt_eval.json').write_text(json.dumps(results,indent=2))
    print('Saved /tmp/transformer_ckpt_eval.json')
