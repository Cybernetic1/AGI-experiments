"""
Transformer baseline for next-word prediction on TinyStories.
Test various model sizes to establish performance vs parameter count.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import argparse
from collections import Counter
import math

class TinyStoriesDataset(Dataset):
    def __init__(self, stories, vocab, seq_length=32):
        self.vocab = vocab
        self.seq_length = seq_length
        self.examples = []
        
        # Tokenize and create training examples
        for story in stories:
            tokens = story.lower().split()
            # Convert to indices
            indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
            
            # Create overlapping sequences
            for i in range(len(indices) - seq_length):
                self.examples.append({
                    'input': indices[i:i+seq_length],
                    'target': indices[i+1:i+seq_length+1]
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return (
            torch.tensor(ex['input'], dtype=torch.long),
            torch.tensor(ex['target'], dtype=torch.long)
        )

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 5000, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x: (batch, seq_len)
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoder[:, :seq_len, :]
        
        # Create causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.output(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)  # (batch, seq_len, vocab_size)
        
        # Reshape for loss calculation
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        predictions = outputs.argmax(dim=-1)
        total_correct += (predictions == targets).sum().item()
        total_tokens += targets.numel()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * total_correct / total_tokens
    perplexity = math.exp(avg_loss)
    
    return avg_loss, accuracy, perplexity

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
            
            predictions = outputs.argmax(dim=-1)
            total_correct += (predictions == targets).sum().item()
            total_tokens += targets.numel()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * total_correct / total_tokens
    perplexity = math.exp(avg_loss)
    
    return avg_loss, accuracy, perplexity

def build_vocab(stories, max_vocab=5000):
    """Build vocabulary from stories."""
    word_counts = Counter()
    for story in stories:
        words = story.lower().split()
        word_counts.update(words)
    
    # Keep most common words
    most_common = word_counts.most_common(max_vocab - 2)  # Reserve space for special tokens
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in most_common:
        vocab[word] = len(vocab)
    
    return vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--num_stories', type=int, default=200)
    parser.add_argument('--seq_length', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=5000)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load TinyStories
    print(f"\nLoading {args.num_stories} stories from TinyStories...")
    all_stories = []
    with open('data/tinystories/stories_10000.txt', 'r') as f:
        story = []
        for line in f:
            line = line.strip()
            if line.startswith('<|endoftext|>'):
                if story:
                    all_stories.append(' '.join(story))
                    story = []
                    if len(all_stories) >= args.num_stories:
                        break
            elif line:
                story.append(line)
    
    # Split train/test
    split = int(0.8 * len(all_stories))
    train_stories = all_stories[:split]
    test_stories = all_stories[split:]
    
    print(f"Train stories: {len(train_stories)}")
    print(f"Test stories: {len(test_stories)}")
    
    # Build vocabulary
    print(f"\nBuilding vocabulary (max {args.vocab_size} words)...")
    vocab = build_vocab(all_stories, max_vocab=args.vocab_size)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create datasets
    train_dataset = TinyStoriesDataset(train_stories, vocab, seq_length=args.seq_length)
    test_dataset = TinyStoriesDataset(test_stories, vocab, seq_length=args.seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    
    # Create model
    print(f"\nCreating Transformer model...")
    print(f"  d_model: {args.d_model}")
    print(f"  num_layers: {args.num_layers}")
    print(f"  nhead: {args.nhead}")
    print(f"  dim_feedforward: {args.dim_feedforward}")
    
    model = TransformerLM(
        vocab_size=len(vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"  Total parameters: {num_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    best_test_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_ppl = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_ppl = evaluate(model, test_loader, criterion, device)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.1f}%, Train PPL={train_ppl:.2f} | "
                  f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.1f}%, Test PPL={test_ppl:.2f}")
    
    # Final results
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"Model: Transformer")
    print(f"  Layers: {args.num_layers}, d_model: {args.d_model}")
    print(f"  Parameters: {num_params:,}")
    print(f"  Best test accuracy: {best_test_acc:.1f}%")
    print(f"  Final test accuracy: {test_acc:.1f}%")
    print(f"  Final test perplexity: {test_ppl:.2f}")
    print("="*80)

if __name__ == '__main__':
    main()
