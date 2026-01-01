"""
Evaluation Script for Symmetric Logic Network

Tests:
1. Parsing accuracy (NL → Logic)
2. Generation quality (Logic → NL)
3. Cycle consistency (round-trip)
4. Multi-hop reasoning
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import spacy
from pathlib import Path
import json
from tqdm import tqdm
import argparse

from symmetric_logic_network import SymmetricLogicNetwork
from train_symmetric import TinyStoriesLogicDataset


def evaluate_parsing(model, dataloader, device, vocab_inv):
    """Evaluate parsing accuracy with detailed metrics."""
    model.eval()
    
    metrics = {
        'exact_match': 0,
        'entity_acc': 0,
        'relation_acc': 0,
        'value_acc': 0,
        'total': 0
    }
    
    examples = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating parsing"):
            text_ids = batch['text_ids'].to(device)
            propositions = batch['propositions'].to(device)
            
            # Parse
            pred_props = model.parse(text_ids)
            
            batch_size = text_ids.size(0)
            for i in range(batch_size):
                # Get first valid proposition (non-padding)
                true_prop = propositions[i, 0]
                pred_prop = pred_props[i, 0]
                
                if true_prop[0] == 0:  # Skip padding
                    continue
                
                metrics['total'] += 1
                
                # Exact match
                if (pred_prop == true_prop).all():
                    metrics['exact_match'] += 1
                
                # Component accuracy
                if pred_prop[0] == true_prop[0]:
                    metrics['entity_acc'] += 1
                if pred_prop[1] == true_prop[1]:
                    metrics['relation_acc'] += 1
                if pred_prop[2] == true_prop[2]:
                    metrics['value_acc'] += 1
                
                # Save example
                if len(examples) < 10:
                    text = ' '.join([vocab_inv.get(t.item(), '<UNK>') 
                                   for t in text_ids[i] if t.item() != 0])
                    examples.append({
                        'text': text,
                        'true': true_prop.cpu().tolist(),
                        'pred': pred_prop.cpu().tolist()
                    })
    
    # Compute percentages
    n = metrics['total']
    if n > 0:
        metrics['exact_match'] = metrics['exact_match'] / n
        metrics['entity_acc'] = metrics['entity_acc'] / n
        metrics['relation_acc'] = metrics['relation_acc'] / n
        metrics['value_acc'] = metrics['value_acc'] / n
    
    return metrics, examples


def evaluate_generation(model, dataloader, device, vocab_inv):
    """Evaluate generation quality."""
    model.eval()
    
    metrics = {
        'perplexity': 0,
        'total': 0
    }
    
    examples = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating generation"):
            text_ids = batch['text_ids'].to(device)
            propositions = batch['propositions'].to(device)
            
            # Generate
            text_logits = model.generate(propositions)
            
            # Compute perplexity
            # Match sequence lengths
            min_len = min(text_logits.size(1), text_ids.size(1))
            loss = F.cross_entropy(
                text_logits[:, :min_len].reshape(-1, text_logits.size(-1)),
                text_ids[:, :min_len].reshape(-1),
                ignore_index=0  # Ignore padding
            )
            metrics['perplexity'] += torch.exp(loss).item()
            metrics['total'] += 1
            
            # Save examples
            if len(examples) < 10:
                i = 0
                true_text = ' '.join([vocab_inv.get(t.item(), '<UNK>') 
                                    for t in text_ids[i] if t.item() != 0])
                pred_ids = torch.argmax(text_logits[i], dim=-1)
                pred_text = ' '.join([vocab_inv.get(t.item(), '<UNK>') 
                                    for t in pred_ids if t.item() != 0])
                examples.append({
                    'propositions': propositions[i, 0].cpu().tolist(),
                    'true_text': true_text,
                    'pred_text': pred_text
                })
    
    # Average perplexity
    if metrics['total'] > 0:
        metrics['perplexity'] = metrics['perplexity'] / metrics['total']
    
    return metrics, examples


def evaluate_cycle_consistency(model, dataloader, device, vocab_inv):
    """Evaluate cycle consistency: text → logic → text'."""
    model.eval()
    
    metrics = {
        'text_similarity': 0,
        'logic_similarity': 0,
        'total': 0
    }
    
    examples = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating cycles"):
            text_ids = batch['text_ids'].to(device)
            propositions = batch['propositions'].to(device)
            
            # Forward cycle: text → logic → text'
            pred_logic = model.parse(text_ids)
            recon_text_logits = model.generate(pred_logic)
            recon_text_ids = torch.argmax(recon_text_logits, dim=-1)
            
            # Backward cycle: logic → text → logic'
            pred_text_logits = model.generate(propositions)
            pred_text_ids = torch.argmax(pred_text_logits, dim=-1)
            recon_logic = model.parse(pred_text_ids)
            
            batch_size = text_ids.size(0)
            for i in range(batch_size):
                metrics['total'] += 1
                
                # Text similarity (token overlap)
                min_len = min(text_ids.size(1), recon_text_ids.size(1))
                text_sim = (text_ids[i, :min_len] == recon_text_ids[i, :min_len]).float().mean().item()
                metrics['text_similarity'] += text_sim
                
                # Logic similarity
                logic_sim = (propositions[i] == recon_logic[i]).float().mean().item()
                metrics['logic_similarity'] += logic_sim
                
                # Save example
                if len(examples) < 5:
                    orig_text = ' '.join([vocab_inv.get(t.item(), '<UNK>') 
                                        for t in text_ids[i] if t.item() != 0])
                    recon_text = ' '.join([vocab_inv.get(t.item(), '<UNK>') 
                                         for t in recon_text_ids[i] if t.item() != 0])
                    examples.append({
                        'original_text': orig_text,
                        'reconstructed_text': recon_text,
                        'text_similarity': text_sim
                    })
    
    # Averages
    n = metrics['total']
    if n > 0:
        metrics['text_similarity'] = metrics['text_similarity'] / n
        metrics['logic_similarity'] = metrics['logic_similarity'] / n
    
    return metrics, examples


def evaluate_multi_hop(model, device):
    """Test multi-hop reasoning capability."""
    # Create test cases manually
    # Format: [entity_id, relation_id, value_id]
    
    # Test case: cat on mat, mat on floor → cat above floor
    test_cases = [
        {
            'name': 'Transitive "on" relation',
            'premises': [
                [1, 10, 2],  # entity_1 on entity_2
                [2, 10, 3],  # entity_2 on entity_3
            ],
            'expected': [1, 11, 3],  # entity_1 above entity_3 (11 = "above")
            'description': 'If A on B and B on C, then A above C'
        }
    ]
    
    model.eval()
    results = []
    
    print("\nTesting multi-hop reasoning:")
    print("-" * 60)
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print(f"Description: {test['description']}")
        
        # Create working memory with premises
        # TODO: Implement multi-hop inference in model
        # For now, just check if model can handle premise encoding
        
        premises = torch.tensor([test['premises']], dtype=torch.long).to(device)
        
        with torch.no_grad():
            # Encode premises
            encoding = model.encode_logic(premises)
            print(f"✓ Premises encoded: {encoding.shape}")
        
        results.append({
            'test': test['name'],
            'status': 'encoded',
            'note': 'Multi-hop inference not yet implemented in model'
        })
    
    return results


def main(args):
    print("=" * 70)
    print("Evaluating Symmetric Logic Network")
    print("=" * 70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    vocab = checkpoint['vocab']
    vocab_inv = {v: k for k, v in vocab.items()}
    entity_to_id = checkpoint['entity_to_id']
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Entities: {len(entity_to_id)}")
    
    # Create model
    model = SymmetricLogicNetwork(
        vocab_size=len(vocab),
        hidden_dim=args.hidden_dim,
        num_rules=args.num_rules,
        prop_length=args.prop_length,
        num_entities=len(entity_to_id)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model loaded (epoch {checkpoint['epoch']})")
    
    # Load dataset
    print("\nLoading test data...")
    dataset = TinyStoriesLogicDataset(
        num_stories=args.num_stories,
        max_seq_len=20
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate parsing
    print("\n" + "=" * 70)
    print("1. Parsing Evaluation (NL → Logic)")
    print("=" * 70)
    parse_metrics, parse_examples = evaluate_parsing(model, dataloader, device, vocab_inv)
    
    print(f"\nResults:")
    print(f"  Exact match:     {parse_metrics['exact_match']:.4f}")
    print(f"  Entity accuracy: {parse_metrics['entity_acc']:.4f}")
    print(f"  Relation accuracy: {parse_metrics['relation_acc']:.4f}")
    print(f"  Value accuracy:  {parse_metrics['value_acc']:.4f}")
    
    print(f"\nExamples:")
    for ex in parse_examples[:3]:
        print(f"  Text: {ex['text']}")
        print(f"  True: {ex['true']}")
        print(f"  Pred: {ex['pred']}")
        print()
    
    # Evaluate generation
    print("\n" + "=" * 70)
    print("2. Generation Evaluation (Logic → NL)")
    print("=" * 70)
    gen_metrics, gen_examples = evaluate_generation(model, dataloader, device, vocab_inv)
    
    print(f"\nResults:")
    print(f"  Perplexity: {gen_metrics['perplexity']:.4f}")
    
    print(f"\nExamples:")
    for ex in gen_examples[:3]:
        print(f"  Props: {ex['propositions']}")
        print(f"  True: {ex['true_text']}")
        print(f"  Pred: {ex['pred_text']}")
        print()
    
    # Evaluate cycle consistency
    print("\n" + "=" * 70)
    print("3. Cycle Consistency Evaluation")
    print("=" * 70)
    cycle_metrics, cycle_examples = evaluate_cycle_consistency(model, dataloader, device, vocab_inv)
    
    print(f"\nResults:")
    print(f"  Text similarity (round-trip): {cycle_metrics['text_similarity']:.4f}")
    print(f"  Logic similarity (round-trip): {cycle_metrics['logic_similarity']:.4f}")
    
    print(f"\nExamples:")
    for ex in cycle_examples[:2]:
        print(f"  Original: {ex['original_text']}")
        print(f"  Reconstructed: {ex['reconstructed_text']}")
        print(f"  Similarity: {ex['text_similarity']:.4f}")
        print()
    
    # Multi-hop reasoning
    print("\n" + "=" * 70)
    print("4. Multi-hop Reasoning")
    print("=" * 70)
    multi_hop_results = evaluate_multi_hop(model, device)
    
    # Save results
    results = {
        'parsing': parse_metrics,
        'generation': gen_metrics,
        'cycle_consistency': cycle_metrics,
        'multi_hop': multi_hop_results,
        'examples': {
            'parsing': parse_examples,
            'generation': gen_examples,
            'cycle': cycle_examples
        }
    }
    
    output_file = Path(args.output_dir) / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Parsing accuracy:      {parse_metrics['exact_match']:.4f}")
    print(f"Generation perplexity: {gen_metrics['perplexity']:.4f}")
    print(f"Cycle consistency:     {cycle_metrics['text_similarity']:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Symmetric Logic Network')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--num_stories', type=int, default=100,
                       help='Number of test stories')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden dimension (must match checkpoint)')
    parser.add_argument('--num_rules', type=int, default=16,
                       help='Number of rules (must match checkpoint)')
    parser.add_argument('--prop_length', type=int, default=5,
                       help='Proposition length (must match checkpoint)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')
    
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
