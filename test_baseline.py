"""
Baseline test: Super simple model to verify data is learnable.

If this simple model can't learn, the data is corrupted.
If this works but our complex model doesn't, our architecture is wrong.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_symmetric import TinyStoriesLogicDataset
from tqdm import tqdm


class SimpleEntityClassifier(nn.Module):
    """Simplest possible model: Text → Entity (first component only)."""
    
    def __init__(self, vocab_size, num_entities, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_entities)
    
    def forward(self, text_ids):
        emb = self.embedding(text_ids)  # (batch, seq_len, hidden)
        out, _ = self.lstm(emb)         # (batch, seq_len, hidden)
        out = out[:, -1, :]             # Take last hidden state
        logits = self.classifier(out)   # (batch, num_entities)
        return logits


def train_baseline():
    print("=" * 70)
    print("Baseline Test: Simple Entity Classifier")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = TinyStoriesLogicDataset(num_stories=1000, max_seq_len=20)
    
    print(f"Vocabulary: {len(dataset.vocab)}")
    print(f"Entities: {dataset.next_entity_id}")
    print(f"Samples: {len(dataset)}")
    
    # Split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create simple model
    model = SimpleEntityClassifier(
        vocab_size=len(dataset.vocab),
        num_entities=dataset.next_entity_id,
        hidden_dim=128
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    print("\nTraining baseline model...")
    best_acc = 0
    
    for epoch in range(20):
        # Train
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            text_ids = batch['text_ids'].to(device)
            propositions = batch['propositions'].to(device)
            
            # Target: first entity only
            target_entities = propositions[:, 0, 0].long()
            
            # Skip padding
            mask = target_entities != 0
            if mask.sum() == 0:
                continue
            
            text_ids = text_ids[mask]
            target_entities = target_entities[mask]
            
            # Forward
            optimizer.zero_grad()
            logits = model(text_ids)
            loss = F.cross_entropy(logits, target_entities)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                text_ids = batch['text_ids'].to(device)
                propositions = batch['propositions'].to(device)
                
                target_entities = propositions[:, 0, 0].long()
                
                # Skip padding
                mask = target_entities != 0
                if mask.sum() == 0:
                    continue
                
                text_ids = text_ids[mask]
                target_entities = target_entities[mask]
                
                logits = model(text_ids)
                preds = torch.argmax(logits, dim=-1)
                
                correct += (preds == target_entities).sum().item()
                total += len(target_entities)
        
        acc = correct / total if total > 0 else 0
        best_acc = max(best_acc, acc)
        
        print(f"Epoch {epoch+1:2d} - Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Acc: {acc*100:.2f}%")
        
        if acc > 0.5:
            print(f"✓ Baseline working! Achieved {acc*100:.1f}% accuracy.")
            break
    
    print("\n" + "=" * 70)
    print(f"Best Accuracy: {best_acc*100:.2f}%")
    print("=" * 70)
    
    if best_acc > 0.5:
        print("\n✓ GOOD NEWS: Data is learnable!")
        print("  Problem is in our complex architecture.")
        print("  Fix: Simplify architecture or fix loss scaling")
    elif best_acc > 0.1:
        print("\n⚠ MEDIUM: Data is somewhat learnable")
        print("  Model learns but not well.")
        print("  Fix: More data or better preprocessing")
    else:
        print("\n✗ BAD NEWS: Even simple model can't learn")
        print("  Data is corrupted or too noisy.")
        print("  Fix: Debug data processing pipeline")
        print("\nDEBUG INFO:")
        print(f"  Vocabulary size: {len(dataset.vocab)}")
        print(f"  Entity count: {dataset.next_entity_id}")
        print(f"  Random baseline: {100/dataset.next_entity_id:.2f}%")


if __name__ == "__main__":
    train_baseline()
