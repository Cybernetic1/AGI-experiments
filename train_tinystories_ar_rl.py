#!/usr/bin/env python3
"""
Train Real DLN on TinyStories using TTT's AR+RL methodology
============================================================

Adapts the 3-phase training that worked for TTT:
  Phase 1: AR - Learn to predict next fact from previous facts
  Phase 2: RL - Learn which facts lead to coherent stories (reward)
  Phase 3: Joint - Refine both objectives together

This is what actually worked on TTT, not supervised classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json
import random
from typing import List, Dict, Tuple
import argparse

from neural_logic_core import LogicNetwork
from logic_core import Proposition


def load_story_sequences(max_stories=50):
    """Load stories as sequences of facts (like TTT game sequences)."""
    data_path = Path("data/processed/tinystories_train.json")
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found")
    
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    sequences = []
    for story in data[:max_stories]:
        facts = []
        for fact in story.get('facts', []):
            rel = fact.get('relation', '')
            subj = fact.get('subject', '')
            obj = fact.get('object', '')
            if rel and subj and obj:
                facts.append(Proposition(rel, [subj, obj]))
        
        if len(facts) >= 3:  # Need at least 3 facts for sequence
            sequences.append(facts)
    
    return sequences


def build_vocabularies(sequences):
    """Build vocabularies from sequences."""
    predicates = set()
    args = set()
    
    for seq in sequences:
        for prop in seq:
            predicates.add(prop.predicate)
            for arg in prop.args:
                args.add(arg)
    
    return sorted(predicates), sorted(args)


class DLNWithHeads(nn.Module):
    """
    Real DLN with dual heads (like TTT hierarchical model):
    - AR head: Predicts next fact
    - RL head: Evaluates sequence coherence (Q-value)
    """
    
    def __init__(self, pred_vocab, arg_vocab, num_rules=6, embed_dim=8):
        super().__init__()
        
        self.pred_vocab = pred_vocab
        self.arg_vocab = arg_vocab
        self.embed_dim = embed_dim
        
        # Embeddings
        self.pred_embed = nn.Embedding(len(pred_vocab), embed_dim)
        self.arg_embed = nn.Embedding(len(arg_vocab), embed_dim)
        
        self.prop_length = embed_dim * 3  # [pred, arg1, arg2]
        
        # Core DLN - learns concepts from sequences
        self.dln = LogicNetwork(
            prop_length=self.prop_length,
            num_props=5,  # Context window: last 5 facts
            output_dim=embed_dim * 4,  # Shared representation
            num_rules=num_rules,
            num_premises=3,
            var_slots=2
        )
        
        # AR head - predicts next fact
        self.ar_pred_head = nn.Linear(embed_dim * 4, len(pred_vocab))
        self.ar_arg1_head = nn.Linear(embed_dim * 4, len(arg_vocab))
        self.ar_arg2_head = nn.Linear(embed_dim * 4, len(arg_vocab))
        
        # RL head - evaluates sequence quality
        self.rl_head = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 1)  # Q-value
        )
    
    def encode_prop(self, prop, device):
        """Encode proposition to vector."""
        pred_idx = self.pred_vocab.get(prop.predicate, 0)
        arg1_idx = self.arg_vocab.get(prop.args[0] if len(prop.args) > 0 else "", 0)
        arg2_idx = self.arg_vocab.get(prop.args[1] if len(prop.args) > 1 else "", 0)
        
        pred_emb = self.pred_embed(torch.tensor([pred_idx], device=device))
        arg1_emb = self.arg_embed(torch.tensor([arg1_idx], device=device))
        arg2_emb = self.arg_embed(torch.tensor([arg2_idx], device=device))
        
        return torch.cat([pred_emb, arg1_emb, arg2_emb], dim=-1).squeeze(0)
    
    def forward_ar(self, context_facts):
        """
        AR forward: predict next fact given context.
        
        Args:
            context_facts: List of Propositions (context)
        Returns:
            pred_logits, arg1_logits, arg2_logits
        """
        device = next(self.parameters()).device
        
        # Encode context (pad if needed)
        context_vecs = []
        for fact in context_facts[-5:]:  # Last 5 facts
            context_vecs.append(self.encode_prop(fact, device))
        
        # Pad if less than 5
        while len(context_vecs) < 5:
            context_vecs.insert(0, torch.zeros(self.prop_length, device=device))
        
        working_memory = torch.stack(context_vecs).unsqueeze(0)  # (1, 5, prop_length)
        
        # Get shared representation
        shared_repr = self.dln(working_memory).squeeze(0)  # (embed_dim * 4)
        
        # Predict next fact
        pred_logits = self.ar_pred_head(shared_repr)
        arg1_logits = self.ar_arg1_head(shared_repr)
        arg2_logits = self.ar_arg2_head(shared_repr)
        
        return pred_logits, arg1_logits, arg2_logits
    
    def forward_rl(self, context_facts):
        """
        RL forward: evaluate sequence quality.
        
        Args:
            context_facts: List of Propositions
        Returns:
            q_value: Quality score
        """
        device = next(self.parameters()).device
        
        # Encode context
        context_vecs = []
        for fact in context_facts[-5:]:
            context_vecs.append(self.encode_prop(fact, device))
        
        while len(context_vecs) < 5:
            context_vecs.insert(0, torch.zeros(self.prop_length, device=device))
        
        working_memory = torch.stack(context_vecs).unsqueeze(0)
        
        # Get shared representation
        shared_repr = self.dln(working_memory).squeeze(0)
        
        # Evaluate quality
        q_value = self.rl_head(shared_repr)
        
        return q_value


def phase1_ar_training(model, sequences, pred_vocab, arg_vocab, epochs=20, lr=0.001, device='cpu'):
    """
    Phase 1: Train AR head to predict next fact.
    Freezes RL head, trains DLN + AR head.
    """
    print("\n" + "="*70)
    print("PHASE 1: Concept Formation (AR Training)")
    print("="*70)
    
    model = model.to(device)
    
    # Freeze RL head
    for param in model.rl_head.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        random.shuffle(sequences)
        
        for seq in sequences:
            if len(seq) < 2:
                continue
            
            # Random split point
            split = random.randint(1, len(seq) - 1)
            context = seq[:split]
            target = seq[split]
            
            # Get target indices
            target_pred = pred_vocab.get(target.predicate, 0)
            target_arg1 = arg_vocab.get(target.args[0] if len(target.args) > 0 else "", 0)
            target_arg2 = arg_vocab.get(target.args[1] if len(target.args) > 1 else "", 0)
            
            # Forward
            optimizer.zero_grad()
            pred_logits, arg1_logits, arg2_logits = model.forward_ar(context)
            
            # Loss
            loss = (criterion(pred_logits.unsqueeze(0), torch.tensor([target_pred], device=device)) +
                   criterion(arg1_logits.unsqueeze(0), torch.tensor([target_arg1], device=device)) +
                   criterion(arg2_logits.unsqueeze(0), torch.tensor([target_arg2], device=device)))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Check if predicted correctly
            pred_correct = (pred_logits.argmax() == target_pred)
            arg1_correct = (arg1_logits.argmax() == target_arg1)
            arg2_correct = (arg2_logits.argmax() == target_arg2)
            
            if pred_correct and arg1_correct and arg2_correct:
                correct += 1
            total += 1
        
        avg_loss = total_loss / len(sequences)
        accuracy = correct / total * 100 if total > 0 else 0
        
        print(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.1f}%")
    
    # Unfreeze RL head
    for param in model.rl_head.parameters():
        param.requires_grad = True
    
    return model


def phase2_rl_training(model, sequences, episodes=100, lr=0.001, device='cpu'):
    """
    Phase 2: Train RL head to evaluate sequence quality.
    Freezes DLN concepts, trains RL head only.
    """
    print("\n" + "="*70)
    print("PHASE 2: Concept Valuation (RL Training)")
    print("="*70)
    
    model = model.to(device)
    
    # Freeze DLN and AR head
    for param in model.dln.parameters():
        param.requires_grad = False
    for param in model.ar_pred_head.parameters():
        param.requires_grad = False
    for param in model.ar_arg1_head.parameters():
        param.requires_grad = False
    for param in model.ar_arg2_head.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.rl_head.parameters(), lr=lr)
    
    rewards = []
    
    for episode in range(episodes):
        model.train()
        
        # Sample a sequence
        seq = random.choice(sequences)
        
        # Reward: longer coherent sequences are better
        # Simple heuristic: reward = length (can be improved)
        reward = len(seq) / 10.0  # Normalize
        
        # Q-learning: predict value of sequence
        q_value = model.forward_rl(seq)
        
        # Target: actual reward
        target = torch.tensor([reward], dtype=torch.float32, device=device)
        
        # TD loss
        loss = F.mse_loss(q_value, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        rewards.append(reward)
        
        if (episode + 1) % 20 == 0:
            avg_reward = sum(rewards[-20:]) / 20
            print(f"Episode {episode+1:3d}: Avg Reward = {avg_reward:.3f}, Loss = {loss.item():.4f}")
    
    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
    
    return model


def phase3_joint_training(model, sequences, iterations=50, lr=0.0005, device='cpu'):
    """
    Phase 3: Joint refinement of AR and RL.
    Both objectives trained together.
    """
    print("\n" + "="*70)
    print("PHASE 3: Joint Refinement (AR + RL)")
    print("="*70)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    ce_criterion = nn.CrossEntropyLoss()
    
    for iteration in range(iterations):
        model.train()
        total_ar_loss = 0
        total_rl_loss = 0
        
        random.shuffle(sequences)
        
        for seq in sequences[:50]:  # Subsample for speed
            if len(seq) < 2:
                continue
            
            split = random.randint(1, len(seq) - 1)
            context = seq[:split]
            target = seq[split]
            
            # AR loss
            pred_vocab_dict = {p: i for i, p in enumerate(model.pred_vocab)}
            arg_vocab_dict = {a: i for i, a in enumerate(model.arg_vocab)}
            
            target_pred = pred_vocab_dict.get(target.predicate, 0)
            target_arg1 = arg_vocab_dict.get(target.args[0] if len(target.args) > 0 else "", 0)
            target_arg2 = arg_vocab_dict.get(target.args[1] if len(target.args) > 1 else "", 0)
            
            pred_logits, arg1_logits, arg2_logits = model.forward_ar(context)
            
            ar_loss = (ce_criterion(pred_logits.unsqueeze(0), torch.tensor([target_pred], device=device)) +
                      ce_criterion(arg1_logits.unsqueeze(0), torch.tensor([target_arg1], device=device)) +
                      ce_criterion(arg2_logits.unsqueeze(0), torch.tensor([target_arg2], device=device)))
            
            # RL loss
            reward = len(seq) / 10.0
            q_value = model.forward_rl(seq)
            rl_loss = F.mse_loss(q_value, torch.tensor([reward], dtype=torch.float32, device=device))
            
            # Combined loss
            loss = ar_loss + 0.1 * rl_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_ar_loss += ar_loss.item()
            total_rl_loss += rl_loss.item()
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration+1:2d}: AR Loss = {total_ar_loss/50:.4f}, RL Loss = {total_rl_loss/50:.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stories", type=int, default=100)
    parser.add_argument("--num-rules", type=int, default=6)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    print("="*70)
    print("TRAIN REAL DLN ON TINYSTORIES (TTT METHODOLOGY)")
    print("="*70)
    
    device = torch.device(args.device)
    
    # Load data as sequences
    print(f"\nLoading {args.stories} story sequences...")
    sequences = load_story_sequences(max_stories=args.stories)
    print(f"  Loaded {len(sequences)} sequences")
    print(f"  Avg facts per sequence: {sum(len(s) for s in sequences) / len(sequences):.1f}")
    
    # Build vocabularies
    predicates, args_list = build_vocabularies(sequences)
    pred_vocab = {p: i for i, p in enumerate(predicates)}
    arg_vocab = {a: i for i, a in enumerate(args_list)}
    
    print(f"  Predicates: {len(predicates)}")
    print(f"  Arguments: {len(args_list)}")
    
    # Create model
    print(f"\nCreating DLN with {args.num_rules} rules...")
    model = DLNWithHeads(pred_vocab, arg_vocab, num_rules=args.num_rules, embed_dim=8)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # 3-phase training (like TTT)
    model = phase1_ar_training(model, sequences, pred_vocab, arg_vocab, 
                               epochs=20, lr=0.001, device=device)
    
    model = phase2_rl_training(model, sequences, episodes=100, lr=0.001, device=device)
    
    model = phase3_joint_training(model, sequences, iterations=50, lr=0.0005, device=device)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\nNext step: Evaluate model and compare with Transformer baseline")


if __name__ == "__main__":
    main()
