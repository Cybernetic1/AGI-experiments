"""
Better evaluation metrics - less strict than exact match.
Shows component-wise accuracy and approximate matching.
"""

import torch
from train_symmetric import TinyStoriesLogicDataset, main
from symmetric_logic_network import SymmetricLogicNetwork
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path


def evaluate_detailed(model, dataloader, device):
    """Evaluate with detailed component-wise metrics."""
    model.eval()
    
    metrics = {
        'exact_match': 0,      # All 3 components match
        'entity_correct': 0,   # First component matches
        'relation_correct': 0, # Second component matches
        'value_correct': 0,    # Third component matches
        'two_of_three': 0,     # At least 2 components match
        'one_of_three': 0,     # At least 1 component matches
        'total_samples': 0
    }
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Detailed Evaluation"):
            text_ids = batch['text_ids'].to(device)
            propositions = batch['propositions'].to(device)
            
            # Parse
            pred_props = model.parse(text_ids)
            
            batch_size = text_ids.size(0)
            
            for i in range(batch_size):
                # Get first proposition (non-padding)
                true_prop = propositions[i, 0]
                pred_prop = pred_props[i, 0]
                
                # Skip padding
                if true_prop[0] == 0:
                    continue
                
                metrics['total_samples'] += 1
                
                # Check each component
                entity_match = (pred_prop[0] == true_prop[0]).item()
                relation_match = (pred_prop[1] == true_prop[1]).item()
                value_match = (pred_prop[2] == true_prop[2]).item()
                
                if entity_match:
                    metrics['entity_correct'] += 1
                if relation_match:
                    metrics['relation_correct'] += 1
                if value_match:
                    metrics['value_correct'] += 1
                
                # Count matches
                num_matches = sum([entity_match, relation_match, value_match])
                
                if num_matches == 3:
                    metrics['exact_match'] += 1
                if num_matches >= 2:
                    metrics['two_of_three'] += 1
                if num_matches >= 1:
                    metrics['one_of_three'] += 1
    
    # Convert to percentages
    n = metrics['total_samples']
    if n > 0:
        return {
            'exact_match': metrics['exact_match'] / n * 100,
            'entity_acc': metrics['entity_correct'] / n * 100,
            'relation_acc': metrics['relation_correct'] / n * 100,
            'value_acc': metrics['value_correct'] / n * 100,
            'two_of_three': metrics['two_of_three'] / n * 100,
            'one_of_three': metrics['one_of_three'] / n * 100,
            'total_samples': n
        }
    return {}


def main_eval(args):
    print("=" * 70)
    print("Detailed Evaluation")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    vocab = checkpoint['vocab']
    entity_to_id = checkpoint['entity_to_id']
    
    print(f"Vocabulary: {len(vocab)}")
    print(f"Entities: {len(entity_to_id)}")
    
    # Create small test dataset
    print("\nCreating test dataset...")
    dataset = TinyStoriesLogicDataset(num_stories=100, max_seq_len=20)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = SymmetricLogicNetwork(
        vocab_size=len(vocab),
        hidden_dim=args.hidden_dim,
        num_rules=args.num_rules,
        prop_length=5,
        num_entities=len(entity_to_id)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded")
    
    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate_detailed(model, dataloader, device)
    
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"\nTotal samples: {metrics['total_samples']}")
    print(f"\nComponent-wise Accuracy:")
    print(f"  Entity (1st component):   {metrics['entity_acc']:.2f}%")
    print(f"  Relation (2nd component): {metrics['relation_acc']:.2f}%")
    print(f"  Value (3rd component):    {metrics['value_acc']:.2f}%")
    print(f"\nOverall Accuracy:")
    print(f"  Exact match (all 3):      {metrics['exact_match']:.2f}%")
    print(f"  2 out of 3 correct:       {metrics['two_of_three']:.2f}%")
    print(f"  At least 1 correct:       {metrics['one_of_three']:.2f}%")
    print("=" * 70)
    
    # Interpretation
    print("\nInterpretation:")
    if metrics['exact_match'] > 50:
        print("  ✓ Excellent! Model has learned the task well.")
    elif metrics['two_of_three'] > 50:
        print("  ✓ Good! Model is learning but needs more training.")
    elif metrics['one_of_three'] > 50:
        print("  ⚠ Fair. Model is learning but accuracy is low.")
    else:
        print("  ✗ Poor. Model needs much more training or larger capacity.")
    
    if metrics['entity_acc'] > metrics['relation_acc']:
        print("  → Model finds entities easier than relations (expected)")
    
    if metrics['value_acc'] < 30:
        print("  → Values are hardest to predict (most variation)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_rules', type=int, default=16)
    
    args = parser.parse_args()
    main_eval(args)
