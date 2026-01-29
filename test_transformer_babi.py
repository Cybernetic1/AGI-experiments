"""
Transformer baseline for bAbI Task 1 (Question Answering).
Compare different model sizes: parameters vs accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import argparse
from collections import Counter
import time

class BaBIDataset(Dataset):
    def __init__(self, data_file, max_facts=10):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        self.max_facts = max_facts
        self.vocab = self._build_vocab()
        self.examples = self._prepare_examples()
    
    def _build_vocab(self):
        """Build vocabulary from all text"""
        all_words = []
        for item in self.data:
            # Facts
            for fact in item['facts']:
                all_words.extend(fact['text'].lower().split())
            # Questions and answers
            for q in item['questions']:
                all_words.extend(q['text'].lower().split())
                all_words.append(q['answer'].lower())
        
        word_counts = Counter(all_words)
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, _ in word_counts.most_common():
            if word not in vocab:
                vocab[word] = len(vocab)
        return vocab
    
    def _prepare_examples(self):
        """Convert to input/output examples"""
        examples = []
        for item in self.data:
            # Take only first max_facts
            facts = item['facts'][:self.max_facts]
            
            for q in item['questions']:
                # Concatenate facts + question
                text_parts = [f['text'] for f in facts] + [q['text']]
                text = ' '.join(text_parts).lower()
                
                # Convert to indices
                tokens = text.split()
                input_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in tokens]
                
                # Pad to fixed length
                max_len = 100
                if len(input_ids) < max_len:
                    input_ids += [self.vocab['<PAD>']] * (max_len - len(input_ids))
                else:
                    input_ids = input_ids[:max_len]
                
                # Target is answer word
                answer = q['answer'].lower()
                target_id = self.vocab.get(answer, self.vocab['<UNK>'])
                
                examples.append({
                    'input': input_ids,
                    'target': target_id
                })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return (
            torch.tensor(ex['input'], dtype=torch.long),
            torch.tensor(ex['target'], dtype=torch.long)
        )

class TransformerQA(nn.Module):
    """Transformer for question answering"""
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output classification head
        self.output_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) * (self.d_model ** 0.5)
        x = self.transformer(x)
        
        # Take last position for answer
        x = x[:, -1, :]  # (batch, d_model)
        logits = self.output_head(x)  # (batch, vocab_size)
        return logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        _, predicted = outputs.max(1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)
    
    return total_loss / len(dataloader), 100.0 * correct / total

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    return 100.0 * correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', default='data/processed/task1_train.json')
    parser.add_argument('--test_file', default='data/processed/task1_test.json')
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_dataset = BaBIDataset(args.train_file)
    test_dataset = BaBIDataset(args.test_file)
    
    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    print(f"Vocabulary size: {len(train_dataset.vocab)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Create model
    model = TransformerQA(
        vocab_size=len(train_dataset.vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"\nModel: {args.num_layers} layers, {args.d_model} dim")
    print(f"Parameters: {num_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training
    print("\nTraining...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        if epoch % 5 == 0:
            test_acc = evaluate(model, test_loader, device)
            print(f"Epoch {epoch:2d}: Loss={train_loss:.4f}, Train={train_acc:.1f}%, Test={test_acc:.1f}%")
    
    # Final evaluation
    final_test_acc = evaluate(model, test_loader, device)
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Parameters:   {num_params:,}")
    print(f"Test Accuracy: {final_test_acc:.1f}%")
    print(f"Training time: {elapsed:.1f}s")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
