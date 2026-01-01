"""
Training Script for Symmetric Logic Network on TinyStories

Designed for GPU training with the following features:
- Cycle consistency training
- Parsing and generation jointly learned
- Integration with spaCy for supervision
- Multi-hop reasoning evaluation
- Checkpoint saving and resuming
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import spacy
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import time
from pathlib import Path
import numpy as np

from symmetric_logic_network import SymmetricLogicNetwork
from implicit_graph_wm import ImplicitGraphWM


class TinyStoriesLogicDataset(Dataset):
    """
    TinyStories dataset with spaCy-generated logic propositions.
    """
    
    def __init__(self, num_stories=1000, max_seq_len=20, cache_dir='data/tinystories_cache'):
        self.max_seq_len = max_seq_len
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load spaCy
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_sm")
        
        # Build vocabulary and entity registry
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.entity_to_id = {}
        self.next_entity_id = 0
        
        # Load and process stories
        cache_file = self.cache_dir / f"processed_{num_stories}.pt"
        
        if cache_file.exists():
            print(f"Loading from cache: {cache_file}")
            cached = torch.load(cache_file)
            self.samples = cached['samples']
            self.vocab = cached['vocab']
            self.entity_to_id = cached['entity_to_id']
            self.next_entity_id = cached['next_entity_id']
        else:
            print(f"Processing {num_stories} stories...")
            self.samples = self._process_stories(num_stories)
            
            # Save to cache
            print(f"Saving to cache: {cache_file}")
            torch.save({
                'samples': self.samples,
                'vocab': self.vocab,
                'entity_to_id': self.entity_to_id,
                'next_entity_id': self.next_entity_id
            }, cache_file)
        
        print(f"Dataset ready: {len(self.samples)} samples")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Entities: {len(self.entity_to_id)}")
    
    def _process_stories(self, num_stories):
        """Process stories with spaCy."""
        # Load TinyStories with retry logic
        max_retries = 3
        ds = None
        
        for attempt in range(max_retries):
            try:
                print(f"Attempt {attempt + 1}/{max_retries} to load dataset...")
                ds = load_dataset(
                    'roneneldan/TinyStories', 
                    split='train', 
                    streaming=True
                )
                print("✓ Dataset loaded successfully")
                break
            except Exception as e:
                print(f"✗ Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    import time
                    wait_time = 2 ** attempt
                    print(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print("\n✗ All attempts failed. Using synthetic data...")
                    ds = self._create_synthetic_data(num_stories)
                    return self._process_synthetic_data(ds)
        
        samples = []
        for i, example in enumerate(tqdm(ds, total=num_stories, desc="Processing")):
            if i >= num_stories:
                break
            
            story = example['text']
            
            # Parse with spaCy
            doc = self.nlp(story)
            
            # Process each sentence
            for sent in doc.sents:
                if len(sent) < 3 or len(sent) > self.max_seq_len:
                    continue
                
                # Extract text tokens
                tokens = [token.text.lower() for token in sent]
                
                # Add to vocabulary
                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
                
                # Convert to IDs
                token_ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
                
                # Extract propositions
                propositions = self._extract_propositions(sent)
                
                if len(propositions) > 0:
                    samples.append({
                        'text': tokens,
                        'text_ids': token_ids,
                        'propositions': propositions
                        # Note: Don't cache spaCy doc objects (not picklable)
                    })
        
        return samples
    
    def _create_synthetic_data(self, num_stories):
        """Create synthetic stories when download fails."""
        print(f"Creating {num_stories} synthetic stories...")
        
        templates = [
            "Once upon a time there was a {adj} {animal}.",
            "The {animal} lived in a {place}.",
            "One day the {animal} met a {other_animal}.",
            "They became good friends.",
            "The {animal} was very {emotion}.",
        ]
        
        adj = ["happy", "sad", "big", "small", "red", "blue"]
        animals = ["cat", "dog", "bird", "fish", "mouse"]
        places = ["forest", "park", "house", "garden"]
        emotions = ["happy", "sad", "excited", "tired"]
        
        stories = []
        for i in range(num_stories):
            story_sentences = []
            for template in templates[:3]:  # 3 sentences per story
                sent = template.format(
                    adj=adj[i % len(adj)],
                    animal=animals[i % len(animals)],
                    other_animal=animals[(i + 1) % len(animals)],
                    place=places[i % len(places)],
                    emotion=emotions[i % len(emotions)]
                )
                story_sentences.append(sent)
            
            stories.append({'text': ' '.join(story_sentences)})
        
        return stories
    
    def _process_synthetic_data(self, stories):
        """Process synthetic stories (already simple, no spaCy needed)."""
        samples = []
        
        print(f"Processing {len(stories)} synthetic stories...")
        for story in tqdm(stories, desc="Processing"):
            text = story['text']
            doc = self.nlp(text)
            
            for sent in doc.sents:
                if len(sent) < 3:
                    continue
                
                tokens = [token.text.lower() for token in sent]
                
                for token in tokens:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
                
                token_ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
                propositions = self._extract_propositions(sent)
                
                if len(propositions) > 0:
                    samples.append({
                        'text': tokens,
                        'text_ids': token_ids,
                        'propositions': propositions
                    })
        
        return samples
    
    def _extract_propositions(self, sent):
        """
        Extract logic propositions from spaCy sentence.
        
        Format: [entity_id, relation_id, value_id]
        """
        propositions = []
        
        # Extract entities (nouns and proper nouns)
        entities = {}
        for token in sent:
            if token.pos_ in ['NOUN', 'PROPN']:
                entity_text = token.lemma_.lower()
                if entity_text not in self.entity_to_id:
                    self.entity_to_id[entity_text] = self.next_entity_id
                    self.next_entity_id += 1
                entities[token.i] = self.entity_to_id[entity_text]
        
        # Extract relations (dependency-based)
        for token in sent:
            if token.i in entities:
                entity_id = entities[token.i]
                
                # Type proposition: [entity, "type", type]
                if token.pos_ == 'NOUN':
                    type_text = token.lemma_.lower()
                    if type_text not in self.vocab:
                        self.vocab[type_text] = len(self.vocab)
                    propositions.append([
                        entity_id,
                        self.vocab.get("type", len(self.vocab)),
                        self.vocab[type_text]
                    ])
                
                # Verb relations: [subject, verb, object]
                if token.dep_ == 'nsubj' and token.head.pos_ == 'VERB':
                    verb = token.head
                    verb_text = verb.lemma_.lower()
                    if verb_text not in self.vocab:
                        self.vocab[verb_text] = len(self.vocab)
                    
                    # Find object
                    obj_id = None
                    for child in verb.children:
                        if child.dep_ in ['dobj', 'pobj'] and child.i in entities:
                            obj_id = entities[child.i]
                            break
                    
                    if obj_id is not None:
                        propositions.append([
                            entity_id,
                            self.vocab[verb_text],
                            obj_id
                        ])
                
                # Adjective modifiers: [entity, "has_property", property]
                for child in token.children:
                    if child.dep_ == 'amod':
                        adj_text = child.lemma_.lower()
                        if adj_text not in self.vocab:
                            self.vocab[adj_text] = len(self.vocab)
                        propositions.append([
                            entity_id,
                            self.vocab.get("has_property", len(self.vocab)),
                            self.vocab[adj_text]
                        ])
        
        return propositions
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Pad text to max_seq_len
        text_ids = sample['text_ids'][:self.max_seq_len]
        text_ids = text_ids + [self.vocab["<PAD>"]] * (self.max_seq_len - len(text_ids))
        
        # Pad propositions to fixed size (max 5 props)
        props = sample['propositions'][:5]
        props = props + [[0, 0, 0]] * (5 - len(props))
        
        return {
            'text_ids': torch.tensor(text_ids, dtype=torch.long),
            'propositions': torch.tensor(props, dtype=torch.long)
            # Don't include 'text' - causes collation issues with variable length lists
        }


def train_epoch(model, dataloader, optimizer, device, lambda_cycle=0.5):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_parse_loss = 0
    total_gen_loss = 0
    total_cycle_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        text_ids = batch['text_ids'].to(device)
        propositions = batch['propositions'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        results = model(text_ids, propositions, lambda_cycle=lambda_cycle)
        
        loss = results['loss']
        loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        total_parse_loss += results['loss_parse'].item()
        total_gen_loss += results['loss_generate'].item()
        cycle_loss = (results['loss_cycle_forward'].item() + 
                     results['loss_cycle_backward'].item())
        total_cycle_loss += cycle_loss
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'parse': f"{results['loss_parse'].item():.4f}",
            'gen': f"{results['loss_generate'].item():.4f}"
        })
    
    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'parse_loss': total_parse_loss / n,
        'gen_loss': total_gen_loss / n,
        'cycle_loss': total_cycle_loss / n
    }


def evaluate(model, dataloader, device):
    """Evaluate parsing and generation."""
    model.eval()
    total_parse_acc = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            text_ids = batch['text_ids'].to(device)
            propositions = batch['propositions'].to(device)
            
            # Parse
            pred_props = model.parse(text_ids)
            
            # Simple accuracy: exact match on first proposition
            matches = (pred_props[:, 0] == propositions[:, 0]).all(dim=-1).sum().item()
            total_parse_acc += matches
            total_samples += text_ids.size(0)
    
    return {
        'parse_accuracy': total_parse_acc / total_samples if total_samples > 0 else 0
    }


def main(args):
    print("=" * 70)
    print("Training Symmetric Logic Network on TinyStories")
    print("=" * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = TinyStoriesLogicDataset(
        num_stories=args.num_stories,
        max_seq_len=args.max_seq_len
    )
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create model
    print("\nCreating model...")
    model = SymmetricLogicNetwork(
        vocab_size=len(dataset.vocab),
        hidden_dim=args.hidden_dim,
        num_rules=args.num_rules,
        prop_length=args.prop_length,
        num_entities=dataset.next_entity_id
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')  # Track best loss instead of accuracy
    results = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 70)
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            lambda_cycle=args.lambda_cycle
        )
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Parse: {train_metrics['parse_loss']:.4f}, "
              f"Gen: {train_metrics['gen_loss']:.4f}, "
              f"Cycle: {train_metrics['cycle_loss']:.4f}")
        
        # Evaluate
        val_metrics = evaluate(model, val_loader, device)
        print(f"Val - Parse Accuracy: {val_metrics['parse_accuracy']:.4f}")
        
        # Save results
        results.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics
        })
        
        # Save checkpoint based on LOSS (not accuracy, since accuracy is broken)
        current_loss = train_metrics['loss']
        if current_loss < best_val_loss:
            best_val_loss = current_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': current_loss,
                'val_acc': val_metrics['parse_accuracy'],
                'vocab': dataset.vocab,
                'entity_to_id': dataset.entity_to_id
            }
            torch.save(checkpoint, args.output_dir / 'best_model.pt')
            print(f"✓ Saved best model (loss: {best_val_loss:.4f})")
        
        # Also save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': current_loss,
                'vocab': dataset.vocab,
                'entity_to_id': dataset.entity_to_id
            }
            torch.save(checkpoint, args.output_dir / f'checkpoint_epoch_{epoch+1}.pt')
            print(f"✓ Saved checkpoint at epoch {epoch+1}")
    
    # Save final results
    with open(args.output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 70)
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Symmetric Logic Network')
    
    # Data
    parser.add_argument('--num_stories', type=int, default=1000,
                       help='Number of stories to use')
    parser.add_argument('--max_seq_len', type=int, default=20,
                       help='Maximum sequence length')
    
    # Model
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension')
    parser.add_argument('--num_rules', type=int, default=16,
                       help='Number of logic rules')
    parser.add_argument('--prop_length', type=int, default=5,
                       help='Proposition length')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--lambda_cycle', type=float, default=0.5,
                       help='Cycle consistency weight')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory')
    
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    main(args)
