"""
Autoregressive (AR) Training for Proposition Generation

This experiment trains an AR model to predict the next proposition's VQ code
given the context of previous propositions in a story.

Architecture:
- Input: Sequence of previous VQ codes + entity embeddings
- Processing: Transformer/LSTM to model temporal dependencies
- Output: Next VQ code (classification over codebook)

This enables the model to learn narrative patterns and generate coherent
sequences of propositions.
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt

from vector_quantization import PropositionVQVAE
from entity_registry import PersistentEntityRegistry


class PropositionalTransformer(nn.Module):
    """
    Transformer-based AR model for proposition generation.
    
    Predicts next VQ code given sequence of previous propositions.
    """
    
    def __init__(self, codebook_size: int = 512, embedding_dim: int = 64,
                 hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 4,
                 max_seq_len: int = 20):
        super().__init__()
        
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # VQ code embeddings (one for each code in codebook)
        self.code_embeddings = nn.Embedding(codebook_size, embedding_dim)
        
        # Positional encoding
        self.pos_embeddings = nn.Embedding(max_seq_len, embedding_dim)
        
        # Project to hidden dim
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output: predict next VQ code
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, codebook_size)
        )
    
    def forward(self, code_sequence: torch.Tensor, 
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Predict next VQ code given sequence.
        
        Args:
            code_sequence: (batch_size, seq_len) - VQ codes
            mask: Optional attention mask
        
        Returns:
            logits: (batch_size, seq_len, codebook_size) - next code predictions
        """
        batch_size, seq_len = code_sequence.shape
        
        # Embed codes
        code_emb = self.code_embeddings(code_sequence)  # (B, L, E)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=code_sequence.device).unsqueeze(0)
        pos_emb = self.pos_embeddings(positions)  # (1, L, E)
        
        x = code_emb + pos_emb  # (B, L, E)
        
        # Project to hidden dim
        x = self.input_projection(x)  # (B, L, H)
        
        # Create causal mask (prevent looking ahead)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        # Transform
        x = self.transformer(x, mask=causal_mask, is_causal=True)  # (B, L, H)
        
        # Predict next codes
        logits = self.output_head(x)  # (B, L, codebook_size)
        
        return logits
    
    def generate(self, initial_codes: torch.Tensor, max_len: int = 10,
                 temperature: float = 1.0) -> torch.Tensor:
        """
        Generate a sequence of VQ codes autoregressively.
        
        Args:
            initial_codes: (batch_size, initial_len) - starting codes
            max_len: Maximum sequence length to generate
            temperature: Sampling temperature
        
        Returns:
            generated_codes: (batch_size, max_len)
        """
        self.eval()
        batch_size = initial_codes.shape[0]
        generated = initial_codes.clone()
        
        with torch.no_grad():
            for _ in range(max_len - initial_codes.shape[1]):
                # Predict next token
                logits = self.forward(generated)  # (B, L, C)
                next_token_logits = logits[:, -1, :] / temperature  # (B, C)
                
                # Sample
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
                
                # Append
                generated = torch.cat([generated, next_token], dim=1)
                
                if generated.shape[1] >= self.max_seq_len:
                    break
        
        return generated


def load_data(filepath: str) -> List[Dict]:
    """Load preprocessed JSON data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_story_sequences(data: List[Dict], vq_vae: PropositionVQVAE,
                           registry: PersistentEntityRegistry,
                           entity_embeddings: nn.Embedding,
                           relation_embeddings: nn.Embedding) -> List[torch.Tensor]:
    """
    Extract sequences of VQ codes from stories.
    
    Each story becomes a sequence of codes.
    Returns:
        List of code sequences (variable length)
    """
    sequences = []
    
    relation_map = {'located_at': 0, 'has_object': 1, 'is_a': 2}
    
    for story in data:
        # Register entities
        local_to_global = {}
        for local_id, name in story["entities"].items():
            entity_type = None
            if name.lower() in ['bathroom', 'hallway', 'garden', 'office', 'bedroom', 'kitchen']:
                entity_type = 'location'
            elif name and name[0].isupper():
                entity_type = 'person'
            
            global_id = registry.get_or_create_entity(name, entity_type)
            local_to_global[name] = global_id
        
        # Extract propositions and encode to VQ codes
        story_codes = []
        for fact in story["facts"]:
            subject = fact.get("subject", "")
            obj = fact.get("object", "")
            obj_normalized = ' '.join(w for w in obj.split() if w.lower() not in ['the', 'a', 'an'])
            
            if subject in local_to_global and obj_normalized in local_to_global:
                subj_id = local_to_global[subject]
                obj_id = local_to_global[obj_normalized]
                rel_id = relation_map.get(fact.get("relation", "located_at"), 0)
                
                # Get embeddings
                subj_emb = entity_embeddings(torch.tensor([subj_id]))
                rel_emb = relation_embeddings(torch.tensor([rel_id]))
                obj_emb = entity_embeddings(torch.tensor([obj_id]))
                
                # Encode to VQ code
                with torch.no_grad():
                    code = vq_vae.encode(subj_emb, rel_emb, obj_emb)
                    story_codes.append(code.item())
        
        if len(story_codes) >= 2:  # Need at least 2 for AR training
            sequences.append(torch.tensor(story_codes))
    
    return sequences


def create_ar_batches(sequences: List[torch.Tensor], max_seq_len: int = 20,
                     batch_size: int = 32) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create batches for AR training.
    
    Returns:
        List of (input_sequences, target_codes) tuples
    """
    batches = []
    
    # Prepare all training pairs
    all_inputs = []
    all_targets = []
    
    for seq in sequences:
        # For each position, predict next token
        for i in range(1, len(seq)):
            # Input: seq[0:i], Target: seq[i]
            input_seq = seq[:i]
            target = seq[i]
            
            # Pad if needed
            if len(input_seq) < max_seq_len:
                padded = torch.zeros(max_seq_len, dtype=torch.long)
                padded[:len(input_seq)] = input_seq
                all_inputs.append(padded)
            else:
                all_inputs.append(input_seq[-max_seq_len:])
            
            all_targets.append(target)
    
    # Shuffle
    indices = torch.randperm(len(all_inputs))
    
    # Create batches
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_inputs = torch.stack([all_inputs[idx] for idx in batch_indices])
        batch_targets = torch.stack([all_targets[idx] for idx in batch_indices])
        batches.append((batch_inputs, batch_targets))
    
    return batches


def train_ar_model(num_epochs: int = 30, batch_size: int = 32,
                  codebook_size: int = 512, learning_rate: float = 1e-3,
                  dataset: str = "babi"):
    """Train autoregressive model on bAbI or TinyStories.
    
    Args:
        dataset: "babi" or "tinystories"
    """
    
    print("=" * 60)
    print(f"Training Autoregressive Model on {dataset.upper()}")
    print("=" * 60)
    
    # Load VQ-VAE
    print("\nLoading trained VQ-VAE...")
    checkpoint = torch.load("models/proposition_vq_vae.pt")
    
    vq_vae = PropositionVQVAE(
        embedding_dim=checkpoint['embedding_dim'],
        hidden_dim=128,
        code_dim=64,
        num_codes=checkpoint['codebook_size'],
        commitment_cost=0.25
    )
    vq_vae.load_state_dict(checkpoint['vq_vae_state_dict'])
    vq_vae.eval()
    
    relation_embeddings = nn.Embedding(10, 64)
    relation_embeddings.load_state_dict(checkpoint['relation_embeddings_state_dict'])
    
    print(f"✓ VQ-VAE loaded (codebook size: {codebook_size})")
    
    # Load entity registry
    print("Loading entity registry...")
    registry = PersistentEntityRegistry(embedding_dim=64)
    registry.load("models/vq_entity_registry.json")
    entity_embeddings = registry.entity_embeddings
    print(f"✓ Registry loaded ({len(registry.entities)} entities)")
    
    # Load data
    print("\nLoading data...")
    if dataset == "babi":
        train_data = load_data("data/processed/task1_train.json")
        test_data = load_data("data/processed/task1_test.json")
    else:  # tinystories
        train_data = load_data("data/processed/tinystories_train.json")
        test_data = load_data("data/processed/tinystories_test.json")
    print(f"Train stories: {len(train_data)}")
    print(f"Test stories: {len(test_data)}")
    
    # Extract sequences
    print("\nExtracting VQ code sequences...")
    train_sequences = extract_story_sequences(
        train_data, vq_vae, registry, entity_embeddings, relation_embeddings
    )
    test_sequences = extract_story_sequences(
        test_data, vq_vae, registry, entity_embeddings, relation_embeddings
    )
    
    seq_lengths = [len(s) for s in train_sequences]
    print(f"Train sequences: {len(train_sequences)}")
    print(f"Test sequences: {len(test_sequences)}")
    print(f"Avg sequence length: {sum(seq_lengths)/len(seq_lengths):.1f}")
    print(f"Max sequence length: {max(seq_lengths)}")
    
    # Create AR model
    print(f"\nCreating AR transformer...")
    ar_model = PropositionalTransformer(
        codebook_size=codebook_size,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        max_seq_len=20
    )
    
    print(f"Model parameters: {sum(p.numel() for p in ar_model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(ar_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }
    
    for epoch in range(num_epochs):
        # Create batches
        train_batches = create_ar_batches(train_sequences, max_seq_len=20, batch_size=batch_size)
        
        # Train
        ar_model.train()
        epoch_losses = []
        epoch_correct = 0
        epoch_total = 0
        
        for inputs, targets in tqdm(train_batches, desc=f"Epoch {epoch+1}", leave=False):
            optimizer.zero_grad()
            
            # Forward pass
            logits = ar_model(inputs)  # (B, L, C)
            
            # Only predict at the last position of each input
            last_positions = (inputs != 0).sum(dim=1) - 1  # Find last non-padding position
            batch_size = inputs.shape[0]
            
            # Gather predictions at last positions
            last_logits = logits[torch.arange(batch_size), last_positions]  # (B, C)
            
            # Loss
            loss = criterion(last_logits, targets)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Accuracy
            predictions = torch.argmax(last_logits, dim=-1)
            epoch_correct += (predictions == targets).sum().item()
            epoch_total += batch_size
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_acc = epoch_correct / epoch_total
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(avg_acc)
        
        # Evaluate
        if (epoch + 1) % 5 == 0:
            ar_model.eval()
            test_batches = create_ar_batches(test_sequences, max_seq_len=20, batch_size=batch_size)
            
            test_losses = []
            test_correct = 0
            test_total = 0
            
            with torch.no_grad():
                for inputs, targets in test_batches:
                    logits = ar_model(inputs)
                    last_positions = (inputs != 0).sum(dim=1) - 1
                    batch_size = inputs.shape[0]
                    last_logits = logits[torch.arange(batch_size), last_positions]
                    
                    loss = criterion(last_logits, targets)
                    test_losses.append(loss.item())
                    
                    predictions = torch.argmax(last_logits, dim=-1)
                    test_correct += (predictions == targets).sum().item()
                    test_total += batch_size
            
            avg_test_loss = sum(test_losses) / len(test_losses)
            avg_test_acc = test_correct / test_total
            history['test_loss'].append(avg_test_loss)
            history['test_acc'].append(avg_test_acc)
            
            print(f"Epoch {epoch+1:3d} | "
                  f"Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc*100:.1f}% | "
                  f"Test Loss: {avg_test_loss:.4f} | Test Acc: {avg_test_acc*100:.1f}%")
        else:
            print(f"Epoch {epoch+1:3d} | Train Loss: {avg_loss:.4f} | Train Acc: {avg_acc*100:.1f}%")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal Train Accuracy: {history['train_acc'][-1]*100:.1f}%")
    print(f"Final Test Accuracy: {history['test_acc'][-1]*100:.1f}%")
    
    # Save model
    models_dir = Path("models")
    torch.save({
        'ar_model_state_dict': ar_model.state_dict(),
        'codebook_size': codebook_size,
        'embedding_dim': 64,
        'hidden_dim': 128,
        'history': history,
    }, models_dir / "proposition_ar_model.pt")
    
    print(f"\n✓ Model saved to {models_dir / 'proposition_ar_model.pt'}")
    
    # Test generation
    print("\n" + "=" * 60)
    print("Testing Generation")
    print("=" * 60)
    
    ar_model.eval()
    with torch.no_grad():
        # Take a test sequence
        test_seq = test_sequences[0]
        print(f"\nOriginal sequence: {test_seq.tolist()}")
        
        # Generate from first code
        initial = test_seq[:1].unsqueeze(0)
        generated = ar_model.generate(initial, max_len=min(len(test_seq), 10), temperature=1.0)
        print(f"Generated sequence: {generated[0].tolist()}")
    
    # Plot training curves
    print("\nGenerating training plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    if history['test_loss']:
        test_epochs = list(range(4, num_epochs, 5))
        if len(test_epochs) > len(history['test_loss']):
            test_epochs = test_epochs[:len(history['test_loss'])]
        axes[0].plot(test_epochs, history['test_loss'], label='Test', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('AR Model Training Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot([a*100 for a in history['train_acc']], label='Train')
    if history['test_acc']:
        axes[1].plot(test_epochs, [a*100 for a in history['test_acc']], label='Test', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Next Code Prediction Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('models/ar_training.png', dpi=150, bbox_inches='tight')
    print(f"✓ Training plot saved to models/ar_training.png")
    
    return ar_model, history


if __name__ == "__main__":
    import sys
    
    dataset = sys.argv[1] if len(sys.argv) > 1 else "babi"
    
    ar_model, history = train_ar_model(
        num_epochs=30,
        batch_size=32,
        codebook_size=512,
        learning_rate=1e-3,
        dataset=dataset
    )
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. ✓ VQ codebook trained")
    print("  2. ✓ AR model trained to predict next VQ code")
    print("  3. TODO: Integrate AR + RL for question answering")
    print("  4. TODO: Combine with Long-Term Memory")
    print("=" * 60)
