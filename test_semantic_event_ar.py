"""
Semantic Event-Level AR with Embedding Similarity
==================================================

Key changes from exact-match version:
1. Uses pre-trained word embeddings for verbs (GloVe or fastText)
2. Cosine similarity loss instead of cross-entropy
3. Evaluation based on semantic closeness, not exact match

This allows model to learn "next action is emotion-related" rather than
memorizing exact verb sequences.

Usage:
    python test_semantic_event_ar.py --scale full --device cuda --epochs 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import time
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Set
import argparse
from collections import defaultdict
import urllib.request
import zipfile

from pipelines.tinystories_pipeline import load_tinystories_facts
from logic_core import Proposition
from dln import SimpleDLN
from test_event_level_ar import (
    group_facts_into_events, 
    create_event_sequences,
    create_event_ar_training_data,
    SCALE_CONFIGS
)


def download_glove_embeddings(embed_dim: int = 50, cache_dir: str = "data/embeddings") -> Dict[str, np.ndarray]:
    """
    Download and load GloVe embeddings.
    
    Available dimensions: 50, 100, 200, 300
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    glove_file = cache_path / f"glove.6B.{embed_dim}d.txt"
    
    if not glove_file.exists():
        print(f"  Downloading GloVe {embed_dim}d embeddings...")
        # Use smaller glove.6B for simplicity (400K vocab, trained on Wikipedia)
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        zip_path = cache_path / "glove.6B.zip"
        
        if not zip_path.exists():
            print(f"    Fetching from {url}...")
            urllib.request.urlretrieve(url, zip_path)
            print(f"    ✅ Downloaded to {zip_path}")
        
        print(f"    Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extract(f"glove.6B.{embed_dim}d.txt", cache_path)
        print(f"    ✅ Extracted")
    
    # Load embeddings
    print(f"  Loading GloVe embeddings from {glove_file}...")
    embeddings = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            embeddings[word] = vector
    
    print(f"  ✅ Loaded {len(embeddings):,} word embeddings ({embed_dim}d)")
    return embeddings


def create_verb_embedding_matrix(
    verbs: List[str],
    glove_embeddings: Dict[str, np.ndarray],
    embed_dim: int
) -> Tuple[torch.Tensor, int]:
    """
    Create embedding matrix for verbs, using GloVe when available.
    
    Returns:
        (embedding_matrix, num_oov)
    """
    vocab_size = len(verbs)
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    
    num_oov = 0  # Out of vocabulary
    
    for i, verb in enumerate(verbs):
        if verb in glove_embeddings:
            embedding_matrix[i] = glove_embeddings[verb]
        else:
            # Random initialization for OOV words
            embedding_matrix[i] = np.random.normal(0, 0.1, embed_dim).astype(np.float32)
            num_oov += 1
    
    return torch.from_numpy(embedding_matrix), num_oov


class SemanticEventAR(nn.Module):
    """
    Event-level AR model with semantic similarity objective.
    """
    
    def __init__(
        self,
        predicates: List[str],
        args: List[str],
        verbs: List[str],
        embed_dim: int = 64,
        verb_embeddings: torch.Tensor = None
    ):
        super().__init__()
        
        self.pred_vocab = {p: i for i, p in enumerate(predicates)}
        self.arg_vocab = {a: i for i, a in enumerate(args)}
        self.verb_vocab = {v: i for i, v in enumerate(verbs)}
        self.verb_names = verbs
        
        self.embed_dim = embed_dim
        
        # Predicate and argument embeddings (learned)
        self.pred_embed = nn.Embedding(len(predicates), embed_dim)
        self.arg_embed = nn.Embedding(len(args), embed_dim)
        
        # Verb embeddings (pre-trained, fine-tunable)
        if verb_embeddings is not None:
            self.verb_embed = nn.Embedding.from_pretrained(
                verb_embeddings,
                freeze=False  # Allow fine-tuning
            )
        else:
            self.verb_embed = nn.Embedding(len(verbs), embed_dim)
        
        # Context encoder
        prop_dim = embed_dim * 3  # pred + 2 args
        self.context_encoder = nn.Sequential(
            nn.Linear(prop_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU()
        )
        
        # Prediction head: context → verb embedding space
        self.pred_head = nn.Linear(embed_dim, embed_dim)
    
    def encode_prop(self, prop: Proposition, device: str = "cpu") -> torch.Tensor:
        """Encode a proposition into embedding space."""
        pred_idx = self.pred_vocab.get(prop.predicate, 0)
        pred_emb = self.pred_embed(torch.tensor([pred_idx], device=device))
        
        arg_embs = []
        for i in range(2):
            arg = prop.args[i] if i < len(prop.args) else "<pad>"
            arg_idx = self.arg_vocab.get(arg, 0)
            arg_embs.append(self.arg_embed(torch.tensor([arg_idx], device=device)))
        
        return torch.cat([pred_emb, arg_embs[0], arg_embs[1]], dim=-1)
    
    def encode_context(self, props: List[Proposition], device: str = "cpu") -> torch.Tensor:
        """Encode context propositions."""
        if not props:
            return torch.zeros(1, self.embed_dim, device=device)
        
        prop_embs = torch.cat([self.encode_prop(p, device) for p in props], dim=0)
        # Mean pooling over propositions
        context_repr = prop_embs.mean(dim=0, keepdim=True)
        return self.context_encoder(context_repr)
    
    def forward(self, context_props: List[Proposition], target_verb: str, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning predicted and target verb embeddings.
        
        Returns:
            (pred_embedding, target_embedding) for cosine loss
        """
        # Encode context
        context_emb = self.encode_context(context_props, device)
        
        # Predict verb embedding
        pred_verb_emb = self.pred_head(context_emb)
        
        # Get target verb embedding
        if target_verb not in self.verb_vocab:
            # OOV target - use zero embedding
            target_verb_emb = torch.zeros_like(pred_verb_emb)
        else:
            target_idx = self.verb_vocab[target_verb]
            target_verb_emb = self.verb_embed(torch.tensor([target_idx], device=device))
        
        return pred_verb_emb, target_verb_emb
    
    def predict_verb(self, context_props: List[Proposition], device: str = "cpu", top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict most likely verbs based on cosine similarity.
        
        Returns:
            List of (verb, similarity_score) tuples
        """
        context_emb = self.encode_context(context_props, device)
        pred_verb_emb = self.pred_head(context_emb)
        
        # Compute cosine similarity with all verb embeddings
        all_verb_embs = self.verb_embed.weight  # (num_verbs, embed_dim)
        
        # Normalize for cosine similarity
        pred_norm = F.normalize(pred_verb_emb, dim=-1)
        verb_norm = F.normalize(all_verb_embs, dim=-1)
        
        similarities = torch.matmul(pred_norm, verb_norm.T).squeeze(0)  # (num_verbs,)
        
        # Top-k
        top_similarities, top_indices = torch.topk(similarities, min(top_k, len(self.verb_names)))
        
        results = []
        for idx, sim in zip(top_indices, top_similarities):
            results.append((self.verb_names[idx.item()], sim.item()))
        
        return results


def train_semantic_event_ar(
    model: SemanticEventAR,
    training_pairs: List[Tuple[List[List[Proposition]], List[Proposition]]],
    device: str = "cpu",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    verbose: bool = True
) -> Dict:
    """
    Train with cosine similarity loss.
    """
    model.train()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Split train/eval
    split_idx = int(0.8 * len(training_pairs))
    train_pairs = training_pairs[:split_idx]
    eval_pairs = training_pairs[split_idx:]
    
    if verbose:
        print(f"  Training on {len(train_pairs)} pairs, evaluating on {len(eval_pairs)} pairs")
    
    history = []
    start_time = time.time()
    
    for epoch in range(epochs):
        random.shuffle(train_pairs)
        
        # Training
        total_loss = 0.0
        total_similarity = 0.0
        num_batches = 0
        
        for i in range(0, len(train_pairs), batch_size):
            batch = train_pairs[i:i+batch_size]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_sim = 0.0
            batch_count = 0
            
            for context_events, target_event in batch:
                if not context_events or not target_event:
                    continue
                
                try:
                    # Flatten context
                    context_props = []
                    for event in context_events:
                        context_props.extend(event)
                    
                    if not context_props:
                        continue
                    
                    # Get target verb
                    target_verb = None
                    for prop in target_event:
                        if prop.predicate == 'type' and len(prop.args) >= 2:
                            target_verb = prop.args[1]
                            break
                    
                    if not target_verb:
                        continue
                    
                    # Forward pass
                    pred_emb, target_emb = model(context_props, target_verb, device)
                    
                    # Cosine similarity loss (maximize similarity = minimize negative similarity)
                    pred_norm = F.normalize(pred_emb, dim=-1)
                    target_norm = F.normalize(target_emb, dim=-1)
                    similarity = F.cosine_similarity(pred_norm, target_norm, dim=-1)
                    
                    loss = 1.0 - similarity  # Range [0, 2], minimize to maximize similarity
                    
                    batch_loss = batch_loss + loss
                    batch_sim = batch_sim + similarity.item()
                    batch_count += 1
                    
                except Exception as e:
                    continue
            
            if batch_count > 0:
                batch_loss = batch_loss / batch_count
                batch_loss.backward()
                optimizer.step()
                
                total_loss += batch_loss.item()
                total_similarity += batch_sim / batch_count
                num_batches += 1
        
        train_loss = total_loss / max(num_batches, 1)
        train_sim = total_similarity / max(num_batches, 1)
        
        # Evaluation
        model.eval()
        eval_loss = 0.0
        eval_sim = 0.0
        eval_top1 = 0  # Exact match in top-1
        eval_top5 = 0  # Target in top-5
        eval_count = 0
        
        with torch.no_grad():
            for context_events, target_event in eval_pairs:
                if not context_events or not target_event:
                    continue
                
                try:
                    context_props = []
                    for event in context_events:
                        context_props.extend(event)
                    
                    if not context_props:
                        continue
                    
                    target_verb = None
                    for prop in target_event:
                        if prop.predicate == 'type' and len(prop.args) >= 2:
                            target_verb = prop.args[1]
                            break
                    
                    if not target_verb:
                        continue
                    
                    # Forward
                    pred_emb, target_emb = model(context_props, target_verb, device)
                    
                    # Similarity
                    pred_norm = F.normalize(pred_emb, dim=-1)
                    target_norm = F.normalize(target_emb, dim=-1)
                    similarity = F.cosine_similarity(pred_norm, target_norm, dim=-1)
                    
                    eval_sim += similarity.item()
                    eval_loss += (1.0 - similarity).item()
                    
                    # Top-k accuracy
                    predictions = model.predict_verb(context_props, device, top_k=5)
                    pred_verbs = [v for v, _ in predictions]
                    
                    if pred_verbs and pred_verbs[0] == target_verb:
                        eval_top1 += 1
                    if target_verb in pred_verbs:
                        eval_top5 += 1
                    
                    eval_count += 1
                    
                except Exception:
                    continue
        
        eval_loss = eval_loss / max(eval_count, 1)
        eval_sim = eval_sim / max(eval_count, 1)
        eval_top1_acc = eval_top1 / max(eval_count, 1)
        eval_top5_acc = eval_top5 / max(eval_count, 1)
        
        model.train()
        
        elapsed = time.time() - start_time
        
        if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
            print(f"    Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Train Sim={train_sim:.4f}, "
                  f"Eval Sim={eval_sim:.4f}, Top-1={eval_top1_acc:.4f}, Top-5={eval_top5_acc:.4f}, Time={elapsed:.1f}s")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_similarity': train_sim,
            'eval_loss': eval_loss,
            'eval_similarity': eval_sim,
            'eval_top1_acc': eval_top1_acc,
            'eval_top5_acc': eval_top5_acc,
            'time': elapsed
        })
    
    training_time = time.time() - start_time
    
    final_metrics = history[-1] if history else {}
    
    return {
        'training_time': training_time,
        'history': history,
        **final_metrics
    }


def main():
    parser = argparse.ArgumentParser(description='Semantic Event-Level AR with Embeddings')
    parser.add_argument('--scale', default='full', choices=list(SCALE_CONFIGS.keys()),
                       help='Dataset scale')
    parser.add_argument('--corpus', default='data/processed/tinystories_train.json',
                       help='Corpus path')
    parser.add_argument('--embed-dim', type=int, default=50,
                       choices=[50, 100, 200, 300],
                       help='GloVe embedding dimension')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--context-size', type=int, default=3, help='Context events')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'], help='Device')
    parser.add_argument('--output-dir', default='outputs/semantic_event_ar',
                       help='Output directory')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    verbose = not args.quiet
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if verbose:
        print("\n" + "="*70)
        print("SEMANTIC EVENT-LEVEL AR (with Pre-trained Embeddings)")
        print("="*70)
    
    # Load GloVe embeddings
    if verbose:
        print("\n[1] Loading GloVe embeddings...")
    
    glove_embeddings = download_glove_embeddings(embed_dim=args.embed_dim)
    
    # Load corpus
    if verbose:
        print(f"\n[2] Loading corpus ({args.scale})...")
    
    config = SCALE_CONFIGS[args.scale]
    facts = load_tinystories_facts(
        max_stories=config['stories'],
        max_facts=config['facts'],
        path=args.corpus
    )
    
    if verbose:
        print(f"  ✅ Loaded {len(facts):,} facts")
    
    # Create training data
    if verbose:
        print(f"\n[3] Creating event-level training pairs...")
    
    training_pairs = create_event_ar_training_data(
        facts,
        context_size=args.context_size,
        max_samples=999999
    )
    
    if verbose:
        print(f"  ✅ Created {len(training_pairs):,} training pairs")
    
    # Extract vocabularies
    all_predicates = set(f.predicate for f in facts)
    all_args = set()
    all_verbs = set()
    
    for f in facts:
        all_args.update(f.args)
        if f.predicate == 'type' and len(f.args) >= 2:
            all_verbs.add(f.args[1])
    
    verb_list = sorted(list(all_verbs))
    
    if verbose:
        print(f"  Vocabulary: {len(all_predicates)} predicates, {len(all_args)} entities, {len(verb_list)} verbs")
    
    # Create verb embedding matrix
    if verbose:
        print(f"\n[4] Creating verb embedding matrix...")
    
    verb_embedding_matrix, num_oov = create_verb_embedding_matrix(
        verb_list,
        glove_embeddings,
        args.embed_dim
    )
    
    if verbose:
        print(f"  ✅ Verb embeddings: {len(verb_list)} verbs, {num_oov} OOV ({100*num_oov/len(verb_list):.1f}%)")
    
    # Create model
    if verbose:
        print(f"\n[5] Creating semantic AR model...")
    
    model = SemanticEventAR(
        predicates=list(all_predicates),
        args=list(all_args),
        verbs=verb_list,
        embed_dim=args.embed_dim,
        verb_embeddings=verb_embedding_matrix
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Model parameters: {num_params:,}")
    
    # Train
    if verbose:
        print(f"\n[6] Training semantic event-level AR...")
    
    results = train_semantic_event_ar(
        model,
        training_pairs,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=verbose
    )
    
    # Save
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'verb_vocab': model.verb_vocab,
        'pred_vocab': model.pred_vocab,
        'arg_vocab': model.arg_vocab,
    }, output_path / 'semantic_model.pt')
    
    # Save results
    with open(output_path / 'results.json', 'w') as f:
        json.dump({
            'config': vars(args),
            'facts': len(facts),
            'training_pairs': len(training_pairs),
            'verbs': len(verb_list),
            'oov_verbs': num_oov,
            **results
        }, f, indent=2)
    
    if verbose:
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"  Final eval similarity: {results.get('eval_similarity', 0):.4f}")
        print(f"  Final Top-1 accuracy: {results.get('eval_top1_acc', 0):.4f}")
        print(f"  Final Top-5 accuracy: {results.get('eval_top5_acc', 0):.4f}")
        print(f"  Training time: {results['training_time']:.1f}s")
        print(f"\n✅ Results saved to {output_path}")


if __name__ == '__main__':
    main()
