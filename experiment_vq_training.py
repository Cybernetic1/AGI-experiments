"""
Train VQ-VAE on bAbI Task 1 propositions.

This experiment demonstrates:
1. Learning a codebook of common proposition patterns
2. Compressing propositions from continuous → discrete
3. Memory compression for Long-Term Memory storage
4. Integration with entity registry
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

from vector_quantization import PropositionVQVAE, train_vq_vae_step
from entity_registry import PersistentEntityRegistry


def load_data(filepath: str) -> List[Dict]:
    """Load preprocessed JSON data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_propositions_from_data(data: List[Dict], 
                                   registry: PersistentEntityRegistry) -> List[tuple]:
    """
    Extract all propositions from dataset with global entity IDs.
    
    Returns:
        List of (subj_id, rel_id, obj_id) tuples
    """
    propositions = []
    
    # Relation mapping
    relation_map = {
        'located_at': 0,
        'has_object': 1,
        'is_a': 2,
    }
    
    for story in data:
        # Register entities
        local_to_global = {}
        for local_id, name in story["entities"].items():
            entity_type = None
            if name.lower() in ['bathroom', 'hallway', 'garden', 'office', 'bedroom', 'kitchen']:
                entity_type = 'location'
            elif name[0].isupper():
                entity_type = 'person'
            
            global_id = registry.get_or_create_entity(name, entity_type)
            local_to_global[name] = global_id
        
        # Extract propositions
        for fact in story["facts"]:
            subject = fact.get("subject", "")
            obj = fact.get("object", "")
            
            # Normalize by removing articles
            obj_normalized = ' '.join(w for w in obj.split() if w.lower() not in ['the', 'a', 'an'])
            
            if subject in local_to_global and obj_normalized in local_to_global:
                subj_id = local_to_global[subject]
                obj_id = local_to_global[obj_normalized]
                rel_id = relation_map.get(fact.get("relation", "located_at"), 0)
                
                propositions.append((subj_id, rel_id, obj_id))
    
    return propositions


def create_proposition_batches(propositions: List[tuple], 
                               entity_embeddings: nn.Embedding,
                               relation_embeddings: nn.Embedding,
                               batch_size: int = 32) -> List[tuple]:
    """
    Convert propositions to embedding batches.
    
    Returns:
        List of (subj_batch, rel_batch, obj_batch) tensors
    """
    batches = []
    
    for i in range(0, len(propositions), batch_size):
        batch_props = propositions[i:i+batch_size]
        
        # Extract IDs
        subj_ids = torch.tensor([p[0] for p in batch_props])
        rel_ids = torch.tensor([p[1] for p in batch_props])
        obj_ids = torch.tensor([p[2] for p in batch_props])
        
        # Get embeddings
        subj_emb = entity_embeddings(subj_ids)
        rel_emb = relation_embeddings(rel_ids)
        obj_emb = entity_embeddings(obj_ids)
        
        batches.append((subj_emb, rel_emb, obj_emb))
    
    return batches


def train_vq_on_babi(num_epochs: int = 50, batch_size: int = 32,
                     codebook_size: int = 512, learning_rate: float = 1e-3,
                     dataset: str = "babi"):
    """Train VQ-VAE on bAbI Task 1 or TinyStories propositions.
    
    Args:
        dataset: "babi" or "tinystories"
    """
    
    print("=" * 60)
    print(f"Training VQ-VAE on {dataset.upper()}")
    print("=" * 60)
    
    # Load data based on dataset choice
    print("\nLoading data...")
    if dataset == "babi":
        train_data = load_data("data/processed/task1_train.json")
        test_data = load_data("data/processed/task1_test.json")
    else:  # tinystories
        train_data = load_data("data/processed/tinystories_train.json")
        test_data = load_data("data/processed/tinystories_test.json")
    print(f"Train stories: {len(train_data)}")
    print(f"Test stories: {len(test_data)}")
    
    # Create entity registry
    print("\nCreating entity registry...")
    registry = PersistentEntityRegistry(embedding_dim=64)
    
    # Extract propositions
    print("Extracting propositions...")
    train_props = extract_propositions_from_data(train_data, registry)
    test_props = extract_propositions_from_data(test_data, registry)
    print(f"Train propositions: {len(train_props)}")
    print(f"Test propositions: {len(test_props)}")
    print(f"Unique entities: {len(registry.entities)}")
    
    # Analyze proposition patterns
    print("\nProposition patterns:")
    pattern_counts = Counter([(p[1],) for p in train_props])  # Group by relation
    for pattern, count in pattern_counts.most_common():
        print(f"  Relation {pattern[0]}: {count} occurrences")
    
    # Create embeddings (shared with registry)
    entity_embeddings = registry.entity_embeddings
    relation_embeddings = nn.Embedding(10, 64)
    
    # Create VQ-VAE model
    print(f"\nCreating VQ-VAE with codebook size {codebook_size}...")
    vq_vae = PropositionVQVAE(
        embedding_dim=64,
        hidden_dim=128,
        code_dim=64,
        num_codes=codebook_size,
        commitment_cost=0.25
    )
    
    print(f"Model parameters: {sum(p.numel() for p in vq_vae.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(list(vq_vae.parameters()) + 
                          list(relation_embeddings.parameters()), 
                          lr=learning_rate)
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    history = {
        'train_loss': [],
        'train_recon': [],
        'test_loss': [],
        'test_recon': [],
    }
    
    for epoch in range(num_epochs):
        # Create batches
        train_batches = create_proposition_batches(
            train_props, entity_embeddings, relation_embeddings, batch_size
        )
        
        # Train
        vq_vae.train()
        epoch_losses = []
        epoch_recon = []
        
        for subj, rel, obj in train_batches:
            losses = train_vq_vae_step(vq_vae, subj, rel, obj, optimizer)
            epoch_losses.append(losses['total_loss'])
            epoch_recon.append(losses['recon_loss'])
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_recon = sum(epoch_recon) / len(epoch_recon)
        history['train_loss'].append(avg_loss)
        history['train_recon'].append(avg_recon)
        
        # Evaluate on test set
        if (epoch + 1) % 5 == 0:
            vq_vae.eval()
            test_batches = create_proposition_batches(
                test_props, entity_embeddings, relation_embeddings, batch_size
            )
            
            test_losses = []
            test_recon = []
            
            with torch.no_grad():
                for subj, rel, obj in test_batches:
                    (subj_r, rel_r, obj_r), codes, vq_losses = vq_vae(subj, rel, obj)
                    recon_loss = (
                        nn.functional.mse_loss(subj_r, subj) +
                        nn.functional.mse_loss(rel_r, rel) +
                        nn.functional.mse_loss(obj_r, obj)
                    ) / 3.0
                    total_loss = recon_loss + vq_losses['total_vq']
                    test_losses.append(total_loss.item())
                    test_recon.append(recon_loss.item())
            
            avg_test_loss = sum(test_losses) / len(test_losses)
            avg_test_recon = sum(test_recon) / len(test_recon)
            history['test_loss'].append(avg_test_loss)
            history['test_recon'].append(avg_test_recon)
            
            # Codebook usage stats
            usage = vq_vae.get_codebook_usage()
            
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {avg_loss:.4f} | "
                  f"Test Loss: {avg_test_loss:.4f} | "
                  f"Recon: {avg_recon:.4f} | "
                  f"Codebook: {usage['active_codes']}/{codebook_size} "
                  f"({usage['usage_rate']*100:.1f}%)")
        else:
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f}")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    usage = vq_vae.get_codebook_usage()
    print(f"\nCodebook Usage:")
    print(f"  Active codes: {usage['active_codes']} / {codebook_size}")
    print(f"  Usage rate: {usage['usage_rate']*100:.1f}%")
    print(f"  Most used code: {usage['most_used_code']} ({usage['most_used_count']} times)")
    print(f"  Mean usage: {usage['mean_usage']:.1f}")
    
    # Analyze learned codes
    print("\nAnalyzing learned proposition patterns...")
    vq_vae.eval()
    with torch.no_grad():
        # Encode all training propositions
        all_codes = []
        for subj, rel, obj in create_proposition_batches(
            train_props[:1000], entity_embeddings, relation_embeddings, batch_size=100
        ):
            codes = vq_vae.encode(subj, rel, obj)
            all_codes.extend(codes.tolist())
        
        code_counts = Counter(all_codes)
        print(f"\nTop 10 most common codes:")
        for code, count in code_counts.most_common(10):
            print(f"  Code {code:3d}: {count:4d} occurrences ({count/len(all_codes)*100:.1f}%)")
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    torch.save({
        'vq_vae_state_dict': vq_vae.state_dict(),
        'relation_embeddings_state_dict': relation_embeddings.state_dict(),
        'codebook_size': codebook_size,
        'embedding_dim': 64,
        'history': history,
        'usage_stats': usage,
    }, models_dir / "proposition_vq_vae.pt")
    
    print(f"\n✓ Model saved to {models_dir / 'proposition_vq_vae.pt'}")
    
    # Save registry
    registry.save("models/vq_entity_registry.json")
    print(f"✓ Entity registry saved")
    
    # Plot training curves
    print("\nGenerating training plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train')
    if history['test_loss']:
        test_epochs = list(range(4, num_epochs, 5))  # Start from epoch 5 (index 4)
        if len(test_epochs) > len(history['test_loss']):
            test_epochs = test_epochs[:len(history['test_loss'])]
        axes[0].plot(test_epochs, history['test_loss'], label='Test', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('VQ-VAE Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(history['train_recon'], label='Train')
    if history['test_recon']:
        axes[1].plot(test_epochs, history['test_recon'], label='Test', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/vq_training.png', dpi=150, bbox_inches='tight')
    print(f"✓ Training plot saved to models/vq_training.png")
    
    return vq_vae, registry, history


if __name__ == "__main__":
    import sys
    
    # Check command line args
    dataset = sys.argv[1] if len(sys.argv) > 1 else "babi"
    
    # Train VQ-VAE
    vq_vae, registry, history = train_vq_on_babi(
        num_epochs=50,
        batch_size=32,
        codebook_size=512,
        learning_rate=1e-3,
        dataset=dataset
    )
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. ✓ VQ codebook trained on Task 1 propositions")
    print("  2. TODO: Integrate VQ with working memory compression")
    print("  3. TODO: Use VQ codes for Long-Term Memory storage")
    print("  4. TODO: Train AR model to predict next VQ code")
    print("=" * 60)
