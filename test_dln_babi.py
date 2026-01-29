"""
Real DLN test on bAbI Task 1 (Question Answering).
Uses the actual neural_logic_core.py implementation with cylindrification.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import argparse
from collections import Counter
import time
from neural_logic_core_vectorized import VectorizedLogicNetwork

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
        """Convert to logic propositions"""
        examples = []
        for item in self.data:
            # Take only first max_facts
            facts = item['facts'][:self.max_facts]
            
            for q in item['questions']:
                # Convert each fact to a proposition (bag of word embeddings)
                fact_props = []
                for fact in facts:
                    tokens = fact['text'].lower().split()
                    token_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in tokens]
                    fact_props.append(token_ids)
                
                # Question as proposition
                q_tokens = q['text'].lower().split()
                q_ids = [self.vocab.get(w, self.vocab['<UNK>']) for w in q_tokens]
                
                # Pad all propositions to same length
                max_prop_len = 15
                for prop in fact_props:
                    while len(prop) < max_prop_len:
                        prop.append(self.vocab['<PAD>'])
                    if len(prop) > max_prop_len:
                        prop[:] = prop[:max_prop_len]
                
                while len(q_ids) < max_prop_len:
                    q_ids.append(self.vocab['<PAD>'])
                if len(q_ids) > max_prop_len:
                    q_ids = q_ids[:max_prop_len]
                
                # Pad to fixed number of facts
                while len(fact_props) < self.max_facts:
                    fact_props.append([self.vocab['<PAD>']] * max_prop_len)
                
                # Combine facts + question
                all_props = fact_props + [q_ids]
                
                # Target is answer word
                answer = q['answer'].lower()
                target_id = self.vocab.get(answer, self.vocab['<UNK>'])
                
                examples.append({
                    'props': all_props,
                    'target': target_id
                })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        return (
            torch.tensor(ex['props'], dtype=torch.long),  # [num_props, prop_len]
            torch.tensor(ex['target'], dtype=torch.long)
        )

class DLN_QA(nn.Module):
    """Wrapper around LogicNetwork for QA task"""
    def __init__(self, vocab_size, embed_dim=32, num_rules=10, num_props=11):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_props = num_props
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Proposition length = embed_dim (we'll average the word embeddings)
        prop_length = embed_dim
        
        # Logic network (Vectorized!)
        self.dln = VectorizedLogicNetwork(
            prop_length=prop_length,
            num_props=num_props,
            output_dim=embed_dim,
            num_rules=num_rules,
            num_premises=3,
            var_slots=2
        )
        
        # Output head: DLN output -> vocabulary logits
        self.output_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, props):
        """
        props: [batch_size, num_props, prop_len] - token indices
        """
        batch_size = props.shape[0]
        
        # Embed tokens: [batch_size, num_props, prop_len, embed_dim]
        embedded = self.embedding(props)
        
        # Average over token dimension: [batch_size, num_props, embed_dim]
        prop_vectors = embedded.mean(dim=2)
        
        # DLN processes propositions
        dln_output = self.dln(prop_vectors)  # [batch_size, embed_dim]
        
        # Predict answer
        logits = self.output_head(dln_output)  # [batch_size, vocab_size]
        
        return logits

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for props, targets in dataloader:
        props = props.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits = model(props)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Accuracy
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    
    return total_loss / len(dataloader), correct / total * 100

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for props, targets in dataloader:
            props = props.to(device)
            targets = targets.to(device)
            
            logits = model(props)
            pred = logits.argmax(dim=1)
            
            correct += (pred == targets).sum().item()
            total += targets.size(0)
    
    return correct / total * 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_rules', type=int, default=10, help='Number of DLN rules')
    parser.add_argument('--embed_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"DLN bAbI Task 1 - Question Answering")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    train_dataset = BaBIDataset('data/processed/task1_train.json', max_facts=10)
    test_dataset = BaBIDataset('data/processed/task1_test.json', max_facts=10)
    
    print(f"Vocabulary size: {len(train_dataset.vocab)}")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Test examples: {len(test_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = DLN_QA(
        vocab_size=len(train_dataset.vocab),
        embed_dim=args.embed_dim,
        num_rules=args.num_rules,
        num_props=11  # 10 facts + 1 question
    ).to(args.device)
    
    num_params = count_parameters(model)
    print(f"\nDLN Model ({args.num_rules} rules):")
    print(f"  Total parameters: {num_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Train
    print("\nTraining...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, args.device)
        
        if epoch % 5 == 0:
            test_acc = evaluate(model, test_loader, args.device)
            print(f"Epoch {epoch:2d}: Loss={train_loss:.4f}, Train={train_acc:.1f}%, Test={test_acc:.1f}%")
    
    # Final evaluation
    final_test_acc = evaluate(model, test_loader, args.device)
    training_time = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"  Parameters:  {num_params:,}")
    print(f"  Test Acc:    {final_test_acc:.1f}%")
    print(f"  Train time:  {training_time:.1f}s")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
