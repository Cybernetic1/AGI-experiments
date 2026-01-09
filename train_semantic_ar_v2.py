"""
Semantic Autoregressive Training v2
====================================

Integrates Davidsonian parsing with Differentiable Logic Network (DLN):
1. Davidsonian NL->Logic extraction (symbolic meta-rules)
2. SymmetricLogicNetwork (differentiable bidirectional logic)
3. Semantic AR objective (predict next sentence's logic form)

This version uses the full DLN architecture instead of simple MLP.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from symmetric_logic_network import SymmetricLogicNetwork
from davidsonian_extraction import DavidsonianExtractor
import json
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm


class TinyStoriesDataset(Dataset):
    """Load preprocessed TinyStories data."""
    def __init__(self, data_path='data/processed/tinystories_train.json', max_samples=1000):
        with open(data_path) as f:
            stories = json.load(f)
        
        # Each story has: text, entities, facts
        self.stories = stories[:max_samples]
        
        # Build vocabulary from all texts
        self.word_to_idx = {'<pad>': 0, '<unk>': 1}
        for story in self.stories:
            words = story['text'].lower().split()
            for word in words:
                if word not in self.word_to_idx:
                    self.word_to_idx[word] = len(self.word_to_idx)
        
        self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
        
        # Tokenize all texts
        self.samples = []
        for i in range(len(self.stories) - 1):  # -1 because we need next story
            input_text = self.stories[i]['text']
            target_text = self.stories[i + 1]['text']
            
            input_ids = [self.word_to_idx.get(w.lower(), 1) for w in input_text.split()]
            target_ids = [self.word_to_idx.get(w.lower(), 1) for w in target_text.split()]
            
            # Pad/truncate to fixed length
            max_len = 50
            input_ids = (input_ids + [0] * max_len)[:max_len]
            target_ids = (target_ids + [0] * max_len)[:max_len]
            
            self.samples.append((input_ids, target_ids))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_ids, target_ids = self.samples[idx]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)


class SemanticARModel(nn.Module):
    """
    Semantic AR model using Differentiable Logic Network.
    
    Flow:
    1. Input text -> Davidsonian parser -> Logic propositions
    2. Logic propositions -> DLN encoder -> Latent representation
    3. Latent -> DLN decoder -> Predicted next logic propositions
    4. Loss: Similarity between predicted and actual next logic
    """
    def __init__(self, vocab_size, hidden_dim=64, num_rules=8, prop_length=5):
        super().__init__()
        
        # Davidsonian parser (symbolic meta-rules)
        self.parser = DavidsonianExtractor()
        
        # Differentiable Logic Network (symmetric bidirectional)
        self.dln = SymmetricLogicNetwork(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_rules=num_rules,
            prop_length=prop_length,
            num_entities=1000
        )
        
        # Predictor: predicts next logic state from current logic state
        self.logic_predictor = nn.Sequential(
            nn.Linear(hidden_dim * prop_length, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * prop_length)
        )
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.prop_length = prop_length
        
        # Build reverse vocab for decoding
        self.idx_to_word = {}
        
    def set_vocab(self, word_to_idx):
        """Set vocabulary mappings."""
        self.word_to_idx = word_to_idx
        self.idx_to_word = {v: k for k, v in word_to_idx.items()}
        
    def text_to_logic(self, text: str) -> List[Tuple[str, str, str]]:
        """Convert text to logic propositions using Davidsonian parser."""
        return self.parser.extract(text)
    
    def logic_to_tensor(self, logic_props: List[Tuple[str, str, str]]) -> torch.Tensor:
        """
        Convert logic propositions to tensor format.
        Each proposition is [entity_id, relation_id, value_id].
        Pad to prop_length.
        """
        tensor = torch.zeros(self.prop_length, 3, dtype=torch.long)
        
        for i, (entity, relation, value) in enumerate(logic_props[:self.prop_length]):
            # Map to vocabulary indices (use hash for entities/values)
            entity_id = abs(hash(entity)) % 1000  # Entity space
            relation_id = abs(hash(relation)) % self.vocab_size  # Relation from vocab
            value_id = abs(hash(value)) % 1000  # Value space
            
            tensor[i] = torch.tensor([entity_id, relation_id, value_id])
        
        return tensor
    
    def forward(self, input_ids: torch.Tensor, target_ids: torch.Tensor = None):
        """
        Forward pass for semantic AR.
        
        Args:
            input_ids: (batch, seq_len) - Current sentence tokens
            target_ids: (batch, seq_len) - Next sentence tokens (for training)
        
        Returns:
            predicted_logic: (batch, prop_length, hidden_dim) - Predicted next logic encoding
            target_logic: (batch, prop_length, hidden_dim) - Actual next logic encoding (if target_ids provided)
        """
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Convert input text to logic using DLN's parse function
        # Note: DLN expects text_ids, outputs propositions
        input_logic = self.dln.parse(input_ids)  # (batch, prop_length, 3)
        
        # Encode logic propositions into latent space
        input_logic_encoded = self.dln.encode_logic(input_logic)  # (batch, prop_length, hidden_dim)
        
        # Flatten for prediction
        input_flat = input_logic_encoded.reshape(batch_size, -1)  # (batch, prop_length * hidden_dim)
        
        # Predict next logic state
        predicted_flat = self.logic_predictor(input_flat)  # (batch, prop_length * hidden_dim)
        predicted_logic = predicted_flat.reshape(batch_size, self.prop_length, self.hidden_dim)
        
        if target_ids is not None:
            # Parse target using DLN
            target_logic = self.dln.parse(target_ids)  # (batch, prop_length, 3)
            target_logic_encoded = self.dln.encode_logic(target_logic)  # (batch, prop_length, hidden_dim)
            return predicted_logic, target_logic_encoded
        else:
            return predicted_logic, None
    
    def compute_loss(self, predicted_logic, target_logic):
        """
        Compute semantic similarity loss between predicted and target logic.
        Uses cosine similarity averaged over propositions.
        """
        # Normalize
        pred_norm = F.normalize(predicted_logic, p=2, dim=-1)  # (batch, prop_length, hidden_dim)
        target_norm = F.normalize(target_logic, p=2, dim=-1)
        
        # Cosine similarity per proposition
        similarity = (pred_norm * target_norm).sum(dim=-1)  # (batch, prop_length)
        
        # Average over propositions and batch
        avg_similarity = similarity.mean()
        
        # Convert to loss (1 - similarity for minimization)
        loss = 1.0 - avg_similarity
        
        return loss, avg_similarity


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_sim = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for input_ids, target_ids in pbar:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        predicted_logic, target_logic = model(input_ids, target_ids)
        
        # Compute loss
        loss, similarity = model.compute_loss(predicted_logic, target_logic)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_sim += similarity.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': loss.item(),
            'sim': f"{similarity.item():.3f}"
        })
    
    avg_loss = total_loss / num_batches
    avg_sim = total_sim / num_batches
    
    return avg_loss, avg_sim


def show_example(model, dataset, device):
    """Show an example prediction."""
    model.eval()
    
    # Get a random sample
    import random
    idx = random.randint(0, len(dataset) - 1)
    input_ids, target_ids = dataset[idx]
    
    # Decode to text
    input_text = ' '.join([dataset.idx_to_word.get(int(i), '<unk>') for i in input_ids if i != 0])
    target_text = ' '.join([dataset.idx_to_word.get(int(i), '<unk>') for i in target_ids if i != 0])
    
    print(f"\n{'='*60}")
    print(f"Input:  {input_text}")
    print(f"Target: {target_text}")
    
    # Parse using Davidsonian extractor
    input_logic = model.text_to_logic(input_text)
    target_logic = model.text_to_logic(target_text)
    
    print(f"\nInput Logic (Davidsonian):")
    for prop in input_logic:
        print(f"  {prop}")
    
    print(f"\nTarget Logic (Davidsonian):")
    for prop in target_logic:
        print(f"  {prop}")
    
    # Get model prediction
    with torch.no_grad():
        input_batch = input_ids.unsqueeze(0).to(device)
        target_batch = target_ids.unsqueeze(0).to(device)
        
        predicted_logic, target_logic_enc = model(input_batch, target_batch)
        loss, similarity = model.compute_loss(predicted_logic, target_logic_enc)
    
    print(f"\nSemantic Similarity: {similarity.item():.3f}")
    print(f"Loss: {loss.item():.3f}")
    print(f"{'='*60}\n")


def main():
    print("="*60)
    print("Semantic AR Training v2 - with Differentiable Logic Network")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading TinyStories dataset...")
    dataset = TinyStoriesDataset(max_samples=1000)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    vocab_size = len(dataset.word_to_idx)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training samples: {len(dataset)}")
    
    # Create model
    print("\nInitializing Semantic AR Model with DLN...")
    model = SemanticARModel(
        vocab_size=vocab_size,
        hidden_dim=64,
        num_rules=8,
        prop_length=5
    ).to(device)
    
    model.set_vocab(dataset.word_to_idx)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        avg_loss, avg_sim = train_epoch(model, dataloader, optimizer, device, epoch)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Avg Loss: {avg_loss:.4f}")
        print(f"  Avg Similarity: {avg_sim:.3f}")
        
        # Show example every few epochs
        if epoch % 2 == 0:
            show_example(model, dataset, device)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    
    # Final examples
    print("\nFinal Examples:")
    for _ in range(3):
        show_example(model, dataset, device)
    
    # Save model
    torch.save(model.state_dict(), 'checkpoints/semantic_ar_v2.pt')
    print("\nModel saved to checkpoints/semantic_ar_v2.pt")


if __name__ == '__main__':
    main()
