"""
Vectorized DLN Parameter Sweep on bAbI Task 1
Tests with varying numbers of rules: 3, 5, 7, 10, 15, 20
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import time
from neural_logic_core_vectorized import VectorizedLogicNetwork

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class bAbIDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        all_words = []
        for item in self.data:
            for fact in item['facts']:
                all_words.extend(fact['text'].lower().split())
            for q in item['questions']:
                all_words.extend(q['text'].lower().split())
                all_words.append(q['answer'].lower())
        
        vocab = sorted(set(all_words))
        self.word2idx = {w: i+1 for i, w in enumerate(vocab)}
        self.word2idx['<PAD>'] = 0
        self.vocab_size = len(self.word2idx)
        
        all_answers = [q['answer'].lower() for item in self.data for q in item['questions']]
        self.answer_vocab = sorted(set(all_answers))
        self.answer2idx = {a: i for i, a in enumerate(self.answer_vocab)}
        self.num_answers = len(self.answer_vocab)
        
    def __len__(self):
        return sum(len(item['questions']) for item in self.data)
    
    def __getitem__(self, idx):
        count = 0
        for story in self.data:
            if count + len(story['questions']) > idx:
                q_idx = idx - count
                question = story['questions'][q_idx]
                facts = story['facts']
                break
            count += len(story['questions'])
        
        fact_texts = [f['text'].lower() for f in facts]
        question_text = question['text'].lower()
        all_text = ' '.join(fact_texts + [question_text])
        tokens = all_text.split()
        encoded = [self.word2idx.get(w, 0) for w in tokens[:20]]
        padded = encoded + [0] * (20 - len(encoded))
        
        answer = question['answer'].lower()
        answer_idx = self.answer2idx[answer]
        
        return torch.tensor(padded, dtype=torch.long), torch.tensor(answer_idx, dtype=torch.long)


class DLNQAModel(nn.Module):
    def __init__(self, vocab_size, num_answers, embed_dim=48, num_rules=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dln = VectorizedLogicNetwork(
            prop_length=embed_dim,
            num_props=20,
            output_dim=embed_dim * 2,
            num_rules=num_rules,
            num_premises=3,
            var_slots=2
        )
        self.classifier = nn.Linear(embed_dim * 2, num_answers)
        
    def forward(self, x):
        embedded = self.embedding(x)
        dln_out = self.dln(embedded)
        return self.classifier(dln_out)


def train_and_evaluate(num_rules, epochs=20):
    print(f"\n{'='*60}")
    print(f"DLN with {num_rules} rules")
    print(f"{'='*60}")
    
    train_dataset = bAbIDataset('data/processed/task1_train.json')
    test_dataset = bAbIDataset('data/processed/task1_test.json')
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = DLNQAModel(
        vocab_size=train_dataset.vocab_size,
        num_answers=train_dataset.num_answers,
        num_rules=num_rules
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    dln_params = sum(p.numel() for p in model.dln.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} (DLN: {dln_params:,})")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        
        if epoch % 5 == 0:
            model.eval()
            test_correct = 0
            test_total = 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    test_correct += predicted.eq(targets).sum().item()
                    test_total += targets.size(0)
            
            train_acc = 100.0 * correct / total
            test_acc = 100.0 * test_correct / test_total
            print(f"Epoch {epoch:2d}: Loss={total_loss/len(train_loader):.4f}, "
                  f"Train={train_acc:.1f}%, Test={test_acc:.1f}%")
    
    train_time = time.time() - start_time
    
    model.eval()
    final_correct = 0
    final_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            final_correct += predicted.eq(targets).sum().item()
            final_total += targets.size(0)
    
    final_acc = 100.0 * final_correct / final_total
    
    return {
        'num_rules': num_rules,
        'params': total_params,
        'dln_params': dln_params,
        'accuracy': final_acc,
        'train_time': train_time
    }


if __name__ == '__main__':
    rule_counts = [3, 5, 7, 10, 15, 20]
    results = []
    
    for num_rules in rule_counts:
        result = train_and_evaluate(num_rules, epochs=20)
        results.append(result)
    
    print(f"\n{'='*75}")
    print("DLN PARAMETER SWEEP - FINAL RESULTS")
    print(f"{'='*75}")
    print(f"{'Rules':<8} {'Total Params':<15} {'DLN Params':<15} {'Accuracy':<12} {'Time'}")
    print("-" * 75)
    
    for r in results:
        print(f"{r['num_rules']:<8} {r['params']:<15,} {r['dln_params']:<15,} "
              f"{r['accuracy']:<12.1f}% {r['train_time']:.1f}s")
    
    print(f"{'='*75}")
    
    with open('dln_sweep_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to dln_sweep_results.json")
