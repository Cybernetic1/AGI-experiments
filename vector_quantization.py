"""
Vector Quantization (VQ) for Proposition Compression

This module implements VQ-VAE style quantization to compress propositions
from continuous embeddings to discrete codes. This solves the combinatorial
explosion problem in the output space.

Key Benefits:
- Reduces output space from vocab³ (billions) to codebook_size (8K)
- Enables tractable softmax over propositions
- Learns common proposition patterns
- Compresses Long-Term Memory storage
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VQCodebook(nn.Module):
    """
    Vector Quantization codebook for proposition compression.
    
    Based on VQ-VAE (van den Oord et al., 2017) adapted for logical propositions.
    Each code represents a common proposition pattern/template.
    """
    
    def __init__(self, num_codes: int = 8192, code_dim: int = 64, 
                 commitment_cost: float = 0.25):
        """
        Args:
            num_codes: Size of codebook (number of discrete codes)
            code_dim: Dimensionality of each code vector
            commitment_cost: Weight for commitment loss (β in VQ-VAE paper)
        """
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        
        # Codebook: (num_codes, code_dim)
        self.codebook = nn.Embedding(num_codes, code_dim)
        
        # Initialize codebook with uniform distribution
        self.codebook.weight.data.uniform_(-1/num_codes, 1/num_codes)
        
        # Track codebook usage for analysis
        self.register_buffer('code_usage', torch.zeros(num_codes))
    
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Quantize continuous embeddings to discrete codes.
        
        Args:
            z: Input embeddings (batch_size, code_dim) or (batch_size, seq_len, code_dim)
        
        Returns:
            z_q: Quantized embeddings (same shape as z)
            codes: Discrete code indices (batch_size,) or (batch_size, seq_len)
            losses: Dict with 'commitment' and 'codebook' losses
        """
        # Handle both 2D and 3D inputs
        original_shape = z.shape
        if len(z.shape) == 3:
            batch_size, seq_len, _ = z.shape
            z_flat = z.reshape(-1, self.code_dim)  # (batch_size * seq_len, code_dim)
        else:
            z_flat = z
            batch_size = z.shape[0]
            seq_len = None
        
        # Compute distances to all codebook entries
        # ||z - e||² = ||z||² + ||e||² - 2⟨z, e⟩
        z_squared = torch.sum(z_flat ** 2, dim=1, keepdim=True)  # (N, 1)
        codebook_squared = torch.sum(self.codebook.weight ** 2, dim=1)  # (num_codes,)
        distances = z_squared + codebook_squared - 2 * torch.matmul(
            z_flat, self.codebook.weight.t()
        )  # (N, num_codes)
        
        # Find nearest code for each input
        codes_flat = torch.argmin(distances, dim=1)  # (N,)
        
        # Look up quantized vectors
        z_q_flat = self.codebook(codes_flat)  # (N, code_dim)
        
        # Reshape back to original shape
        if seq_len is not None:
            z_q = z_q_flat.reshape(batch_size, seq_len, self.code_dim)
            codes = codes_flat.reshape(batch_size, seq_len)
        else:
            z_q = z_q_flat
            codes = codes_flat
        
        # Compute VQ-VAE losses
        # 1. Codebook loss: Move codebook entries toward encodings
        codebook_loss = F.mse_loss(z_q.detach(), z)
        
        # 2. Commitment loss: Encourage encoder to commit to codes
        commitment_loss = F.mse_loss(z_q, z.detach()) * self.commitment_cost
        
        # Straight-through estimator: Copy gradients from z_q to z
        z_q = z + (z_q - z).detach()
        
        # Update usage statistics (for monitoring)
        with torch.no_grad():
            self.code_usage += torch.bincount(
                codes_flat, minlength=self.num_codes
            ).float()
        
        losses = {
            'codebook': codebook_loss,
            'commitment': commitment_loss,
            'total_vq': codebook_loss + commitment_loss
        }
        
        return z_q, codes, losses
    
    def lookup(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Look up code vectors from indices.
        
        Args:
            codes: Code indices (batch_size,) or (batch_size, seq_len)
        
        Returns:
            Code vectors (batch_size, code_dim) or (batch_size, seq_len, code_dim)
        """
        return self.codebook(codes)
    
    def get_usage_stats(self) -> dict:
        """Get statistics about codebook usage."""
        usage = self.code_usage.cpu().numpy()
        active_codes = (usage > 0).sum()
        
        return {
            'active_codes': int(active_codes),
            'total_codes': self.num_codes,
            'usage_rate': float(active_codes / self.num_codes),
            'most_used_code': int(usage.argmax()),
            'most_used_count': int(usage.max()),
            'mean_usage': float(usage.mean()),
        }
    
    def reset_usage_stats(self):
        """Reset usage statistics."""
        self.code_usage.zero_()


class PropositionVQEncoder(nn.Module):
    """
    Encoder for propositions → VQ codes.
    
    Takes proposition embeddings (subject, relation, object) and compresses
    them to a single discrete code representing the proposition pattern.
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128, 
                 code_dim: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.code_dim = code_dim
        
        # Encode (subj, rel, obj) → code
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, code_dim)
        )
    
    def forward(self, subj_emb: torch.Tensor, rel_emb: torch.Tensor, 
                obj_emb: torch.Tensor) -> torch.Tensor:
        """
        Encode a proposition to a continuous vector (pre-quantization).
        
        Args:
            subj_emb: Subject embeddings (batch_size, embedding_dim)
            rel_emb: Relation embeddings (batch_size, embedding_dim)
            obj_emb: Object embeddings (batch_size, embedding_dim)
        
        Returns:
            Encoded vector (batch_size, code_dim)
        """
        # Concatenate all components
        prop_vec = torch.cat([subj_emb, rel_emb, obj_emb], dim=-1)
        
        # Encode to code_dim
        return self.encoder(prop_vec)


class PropositionVQDecoder(nn.Module):
    """
    Decoder for VQ codes → propositions.
    
    Takes discrete codes and reconstructs proposition embeddings.
    """
    
    def __init__(self, code_dim: int = 64, hidden_dim: int = 128, 
                 embedding_dim: int = 64):
        super().__init__()
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        
        # Decode code → (subj, rel, obj)
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim * 3)
        )
    
    def forward(self, code_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode a code vector to proposition components.
        
        Args:
            code_vec: Code vectors (batch_size, code_dim)
        
        Returns:
            subj_emb: Subject embeddings (batch_size, embedding_dim)
            rel_emb: Relation embeddings (batch_size, embedding_dim)
            obj_emb: Object embeddings (batch_size, embedding_dim)
        """
        # Decode
        prop_vec = self.decoder(code_vec)  # (batch_size, embedding_dim * 3)
        
        # Split into components
        prop_vec = prop_vec.view(-1, 3, self.embedding_dim)
        subj_emb = prop_vec[:, 0, :]
        rel_emb = prop_vec[:, 1, :]
        obj_emb = prop_vec[:, 2, :]
        
        return subj_emb, rel_emb, obj_emb


class PropositionVQVAE(nn.Module):
    """
    Complete VQ-VAE for proposition compression.
    
    End-to-end model that encodes propositions to discrete codes
    and decodes them back. Can be trained on proposition datasets
    to learn common patterns.
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128,
                 code_dim: int = 64, num_codes: int = 8192,
                 commitment_cost: float = 0.25):
        super().__init__()
        
        self.encoder = PropositionVQEncoder(embedding_dim, hidden_dim, code_dim)
        self.codebook = VQCodebook(num_codes, code_dim, commitment_cost)
        self.decoder = PropositionVQDecoder(code_dim, hidden_dim, embedding_dim)
    
    def forward(self, subj_emb: torch.Tensor, rel_emb: torch.Tensor, 
                obj_emb: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], 
                                                  torch.Tensor, dict]:
        """
        Full forward pass: encode → quantize → decode.
        
        Args:
            subj_emb: Subject embeddings (batch_size, embedding_dim)
            rel_emb: Relation embeddings (batch_size, embedding_dim)
            obj_emb: Object embeddings (batch_size, embedding_dim)
        
        Returns:
            reconstructed: Tuple of (subj_recon, rel_recon, obj_recon)
            codes: Discrete code indices (batch_size,)
            losses: Dictionary of VQ losses
        """
        # Encode
        z = self.encoder(subj_emb, rel_emb, obj_emb)
        
        # Quantize
        z_q, codes, vq_losses = self.codebook(z)
        
        # Decode
        subj_recon, rel_recon, obj_recon = self.decoder(z_q)
        
        return (subj_recon, rel_recon, obj_recon), codes, vq_losses
    
    def encode(self, subj_emb: torch.Tensor, rel_emb: torch.Tensor, 
               obj_emb: torch.Tensor) -> torch.Tensor:
        """Encode propositions to discrete codes."""
        z = self.encoder(subj_emb, rel_emb, obj_emb)
        _, codes, _ = self.codebook(z)
        return codes
    
    def decode_from_codes(self, codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode codes to propositions."""
        z_q = self.codebook.lookup(codes)
        return self.decoder(z_q)
    
    def get_codebook_usage(self) -> dict:
        """Get codebook usage statistics."""
        return self.codebook.get_usage_stats()


def train_vq_vae_step(model: PropositionVQVAE,
                      subj_emb: torch.Tensor,
                      rel_emb: torch.Tensor, 
                      obj_emb: torch.Tensor,
                      optimizer: torch.optim.Optimizer) -> dict:
    """
    Single training step for VQ-VAE.
    
    Returns:
        Dictionary with all losses
    """
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    (subj_recon, rel_recon, obj_recon), codes, vq_losses = model(
        subj_emb, rel_emb, obj_emb
    )
    
    # Reconstruction loss
    recon_loss = (
        F.mse_loss(subj_recon, subj_emb) +
        F.mse_loss(rel_recon, rel_emb) +
        F.mse_loss(obj_recon, obj_emb)
    ) / 3.0
    
    # Total loss
    total_loss = recon_loss + vq_losses['total_vq']
    
    # Backward
    total_loss.backward()
    optimizer.step()
    
    return {
        'total_loss': total_loss.item(),
        'recon_loss': recon_loss.item(),
        'codebook_loss': vq_losses['codebook'].item(),
        'commitment_loss': vq_losses['commitment'].item(),
    }


if __name__ == "__main__":
    print("VQ-VAE for Proposition Compression")
    print("=" * 50)
    
    # Create model
    vq_vae = PropositionVQVAE(
        embedding_dim=64,
        hidden_dim=128,
        code_dim=64,
        num_codes=256,  # Small for testing
        commitment_cost=0.25
    )
    
    print(f"Model parameters: {sum(p.numel() for p in vq_vae.parameters()):,}")
    print(f"Codebook size: {vq_vae.codebook.num_codes}")
    
    # Test forward pass
    batch_size = 4
    subj = torch.randn(batch_size, 64)
    rel = torch.randn(batch_size, 64)
    obj = torch.randn(batch_size, 64)
    
    (subj_r, rel_r, obj_r), codes, losses = vq_vae(subj, rel, obj)
    
    print(f"\nTest batch size: {batch_size}")
    print(f"Codes: {codes}")
    print(f"Reconstruction MSE: {F.mse_loss(subj_r, subj).item():.4f}")
    print(f"VQ losses: {losses}")
    
    # Test encode/decode
    codes_only = vq_vae.encode(subj, rel, obj)
    subj_d, rel_d, obj_d = vq_vae.decode_from_codes(codes_only)
    
    print(f"\nEncode-decode test:")
    print(f"Original codes: {codes}")
    print(f"Encoded codes:  {codes_only}")
    print(f"Match: {torch.equal(codes, codes_only)}")
    
    print("\n✓ VQ-VAE module ready!")
