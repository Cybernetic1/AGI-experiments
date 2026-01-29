"""
Compare original DLN vs vectorized DLN on bAbI Task 1
Tests both speed and accuracy to verify vectorization preserves performance
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import time
from neural_logic_core import LogicNetwork
from neural_logic_core_vectorized import VectorizedLogicNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class bAbIDataset(Dataset):
    def __init__(self, json_path, vocab=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Build vocabulary
        if vocab is None:
            self.vocab = {'<PAD>': 0, '<UNK>': 1}
            for item in self.data:
                for fact in item['facts']:
                    for word in fact['text'].lower().split():
                        if word not in self.vocab:
                            self.vocab[word] = len(self.vocab)
                for q in item['questions']:
                    for word in q['text'].lower().split():
                        if word not in self.vocab:
                            self.vocab[word] = len(self.vocab)
                    if q['answer'] not in self.vocab:
                        self.vocab[q['answer']] = len(self.vocab)
        else:
            self.vocab = vocab
        
        self.vocab_size = len(self.vocab)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Encode facts (context)
        fact_ids = []
        for fact in item['facts']:
            words = fact['text'].lower().split()
            ids = [self.vocab.get(w, 1) for w in words]
            fact_ids.extend(ids)
        
        # Pad/truncate to fixed length
        max_len = 100
        if len(fact_ids) > max_len:
            fact_ids = fact_ids[:max_len]
        else:
            fact_ids += [0] * (max_len - len(fact_ids))
        
        # Use first question
        q = item['questions'][0]
        answer_id = self.vocab.get(q['answer'], 1)
        
        return torch.tensor(fact_ids), torch.tensor(answer_id)


class DLNWrapper(nn.Module):
    """Wrapper to use DLN for classification"""
    def __init__(self, vocab_size, embed_dim, num_rules, dln_class):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Create DLN
        prop_length = embed_dim
        self.dln = dln_class(
            prop_length=prop_length,
            num_props=10,
            output_dim=embed_dim * 2,
            num_rules=num_rules,
            num_premises=3,
            var_slots=2
        )
        
        # Classification head
        self.classifier = nn.Linear(embed_dim * 2, vocab_size)
    
    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        # Convert to propositions (simple: take chunks)
        batch_size = embedded.size(0)
        prop_length = embedded.size(2)
        
        # Reshape to [batch, num_props, prop_length]
        # Pad or truncate to get exactly num_props
        num_props = 10
        seq_len = embedded.size(1)
        props_per_seq = seq_len // num_props
        
        if props_per_seq > 0:
            # Take evenly spaced propositions
            indices = torch.linspace(0, seq_len-1, num_props, device=embedded.device).long()
            props = embedded[:, indices, :]  # [batch, num_props, prop_length]
        else:
            # Pad if too short
            props = torch.zeros(batch_size, num_props, prop_length, device=embedded.device)
            props[:, :seq_len, :] = embedded
        
        # Run through DLN
        dln_output = self.dln(props)  # [batch, output_dim]
        
        # Classify
        logits = self.classifier(dln_output)  # [batch, vocab_size]
        return logits


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for facts, answers in dataloader:
        facts = facts.to(device)
        answers = answers.to(device)
        
        optimizer.zero_grad()
        outputs = model(facts)
        loss = criterion(outputs, answers)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (predicted == answers).sum().item()
        total += answers.size(0)
    
    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for facts, answers in dataloader:
            facts = facts.to(device)
            answers = answers.to(device)
            
            outputs = model(facts)
            _, predicted = outputs.max(1)
            correct += (predicted == answers).sum().item()
            total += answers.size(0)
    
    return 100. * correct / total


def run_experiment(dln_class, name, num_rules=5):
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    # Load data
    train_dataset = bAbIDataset('data/processed/task1_train.json')
    test_dataset = bAbIDataset('data/processed/task1_test.json', vocab=train_dataset.vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create model
    vocab_size = train_dataset.vocab_size
    embed_dim = 48
    model = DLNWrapper(vocab_size, embed_dim, num_rules, dln_class).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    print("\nTraining...")
    start_time = time.time()
    
    for epoch in range(1, 21):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
        
        if epoch % 5 == 0:
            test_acc = evaluate(model, test_loader)
            print(f"Epoch {epoch:2d}: Loss={train_loss:.4f}, Train={train_acc:.1f}%, Test={test_acc:.1f}%")
    
    train_time = time.time() - start_time
    
    # Final evaluation
    final_test_acc = evaluate(model, test_loader)
    
    print(f"\nFINAL RESULTS:")
    print(f"  Parameters:  {num_params:,}")
    print(f"  Test Acc:    {final_test_acc:.1f}%")
    print(f"  Train time:  {train_time:.1f}s")
    
    return {
        'name': name,
        'params': num_params,
        'accuracy': final_test_acc,
        'time': train_time
    }


if __name__ == '__main__':
    results = []
    
    # Test original DLN
    result_orig = run_experiment(LogicNetwork, "Original DLN", num_rules=5)
    results.append(result_orig)
    
    # Test vectorized DLN
    result_vec = run_experiment(LogicNetworkVectorized, "Vectorized DLN", num_rules=5)
    results.append(result_vec)
    
    # Compare
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Params':>12} {'Accuracy':>10} {'Time':>10}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['name']:<20} {r['params']:>12,} {r['accuracy']:>9.1f}% {r['time']:>9.1f}s")
    
    # Speedup
    if len(results) == 2:
        speedup = results[0]['time'] / results[1]['time']
        acc_diff = results[1]['accuracy'] - results[0]['accuracy']
        print(f"\nSpeedup: {speedup:.1f}×")
        print(f"Accuracy difference: {acc_diff:+.1f}%")
