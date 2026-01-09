"""
Variable-Length Differentiable Logic Network

Handles Davidsonian semantic graphs with variable numbers of propositions.
Uses auto-regressive generation for propositions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional


class VariableLengthDLN(nn.Module):
    """
    DLN that can output variable numbers of propositions for Davidsonian semantics.
    
    Key idea: Generate propositions auto-regressively until END token is produced.
    """
    
    def __init__(self,
                 vocab_size: int = 1000,
                 hidden_dim: int = 128,
                 num_entities: int = 100,
                 max_propositions: int = 20):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_entities = num_entities
        self.max_propositions = max_propositions
        
        # Special tokens
        self.END_PROP_ID = 0  # Special entity ID meaning "no more propositions"
        
        # Text encoder
        self.text_embedder = nn.Embedding(vocab_size, hidden_dim)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=2
        )
        
        # Proposition decoder (auto-regressive)
        self.prop_embedder = nn.Linear(hidden_dim * 3, hidden_dim)  # [e, r, v] embedding
        self.prop_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=2
        )
        
        # Output heads for each component of proposition [entity, relation, value]
        self.entity_output = nn.Linear(hidden_dim, num_entities)
        self.relation_output = nn.Linear(hidden_dim, vocab_size)
        self.value_output = nn.Linear(hidden_dim, num_entities)
        
        # Reverse: Logic → Text decoder
        self.text_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4, batch_first=True),
            num_layers=2
        )
        self.text_output = nn.Linear(hidden_dim, vocab_size)
        
    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode input text.
        
        Args:
            text_ids: (batch, seq_len)
        Returns:
            encoded: (batch, seq_len, hidden_dim)
        """
        text_ids = torch.clamp(text_ids, 0, self.vocab_size - 1)
        embedded = self.text_embedder(text_ids)
        encoded = self.text_encoder(embedded)
        return encoded
    
    def encode_proposition(self, entity_id: int, relation_id: int, value_id: int) -> torch.Tensor:
        """
        Encode a single proposition [entity, relation, value] to embedding.
        
        Returns:
            embedding: (hidden_dim,)
        """
        # Simple one-hot encoding for now (could use learned embeddings)
        e_vec = F.one_hot(torch.tensor(entity_id), self.num_entities).float()
        r_vec = F.one_hot(torch.tensor(relation_id), self.vocab_size).float()
        v_vec = F.one_hot(torch.tensor(value_id), self.num_entities).float()
        
        combined = torch.cat([e_vec, r_vec, v_vec])
        embedding = self.prop_embedder(combined)
        return embedding
    
    def parse_autoregressive(self, text_ids: torch.Tensor) -> List[List[Tuple[int, int, int]]]:
        """
        Parse text to variable-length list of propositions.
        
        Args:
            text_ids: (batch, seq_len)
        
        Returns:
            List of proposition lists, one per batch item.
            Each proposition is (entity_id, relation_id, value_id).
        """
        batch_size = text_ids.size(0)
        
        # Encode text
        text_encoded = self.encode_text(text_ids)  # (batch, seq_len, hidden_dim)
        
        # Auto-regressively generate propositions
        all_propositions = []
        
        for b in range(batch_size):
            propositions = []
            text_context = text_encoded[b:b+1]  # (1, seq_len, hidden_dim)
            
            # Start token (empty proposition)
            prop_history = torch.zeros(1, 1, self.hidden_dim, device=text_ids.device)
            
            for step in range(self.max_propositions):
                # Decode next proposition
                decoded = self.prop_decoder(prop_history, text_context)  # (1, step+1, hidden_dim)
                current = decoded[:, -1, :]  # (1, hidden_dim) - last position
                
                # Predict components
                entity_logits = self.entity_output(current)  # (1, num_entities)
                relation_logits = self.relation_output(current)  # (1, vocab_size)
                value_logits = self.value_output(current)  # (1, num_entities)
                
                entity_id = torch.argmax(entity_logits, dim=-1).item()
                relation_id = torch.argmax(relation_logits, dim=-1).item()
                value_id = torch.argmax(value_logits, dim=-1).item()
                
                # Check for END token
                if entity_id == self.END_PROP_ID:
                    break
                
                propositions.append((entity_id, relation_id, value_id))
                
                # Update history with this proposition
                prop_emb = self.encode_proposition(entity_id, relation_id, value_id)
                prop_emb = prop_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, hidden_dim)
                prop_history = torch.cat([prop_history, prop_emb], dim=1)
            
            all_propositions.append(propositions)
        
        return all_propositions
    
    def generate_text(self, propositions: torch.Tensor, max_length: int = 20) -> torch.Tensor:
        """
        Generate text from propositions (variable length).
        
        Args:
            propositions: (batch, num_props, 3) - may have padding
        
        Returns:
            text_logits: (batch, max_length, vocab_size)
        """
        batch_size = propositions.size(0)
        
        # Encode propositions
        # For simplicity, use simple embedding (could be improved)
        entities = propositions[:, :, 0].long()
        relations = propositions[:, :, 1].long()
        values = propositions[:, :, 2].long()
        
        # Clamp to valid ranges
        entities = torch.clamp(entities, 0, self.num_entities - 1)
        relations = torch.clamp(relations, 0, self.vocab_size - 1)
        values = torch.clamp(values, 0, self.num_entities - 1)
        
        # Simple encoding (sum of components)
        entity_emb = F.one_hot(entities, self.num_entities).float()
        relation_emb = F.one_hot(relations, self.vocab_size).float()
        value_emb = F.one_hot(values, self.num_entities).float()
        
        # Project to hidden_dim
        prop_encoded = self.prop_embedder(
            torch.cat([entity_emb, relation_emb, value_emb], dim=-1)
        )  # (batch, num_props, hidden_dim)
        
        # Auto-regressive text generation
        # Start with BOS token
        text_history = torch.zeros(batch_size, 1, self.hidden_dim, device=propositions.device)
        
        all_logits = []
        for step in range(max_length):
            decoded = self.text_decoder(text_history, prop_encoded)  # (batch, step+1, hidden_dim)
            current = decoded[:, -1:, :]  # (batch, 1, hidden_dim)
            
            logits = self.text_output(current)  # (batch, 1, vocab_size)
            all_logits.append(logits)
            
            # Use argmax token as next input
            next_token = torch.argmax(logits, dim=-1)  # (batch, 1)
            next_emb = self.text_embedder(next_token)  # (batch, 1, hidden_dim)
            text_history = torch.cat([text_history, next_emb], dim=1)
        
        text_logits = torch.cat(all_logits, dim=1)  # (batch, max_length, vocab_size)
        return text_logits
    
    def forward(self, text_ids: torch.Tensor) -> Dict[str, any]:
        """
        Parse text to propositions (variable length).
        
        Returns:
            Dictionary with propositions and metadata.
        """
        propositions = self.parse_autoregressive(text_ids)
        
        return {
            'propositions': propositions,  # List of lists
            'num_propositions': [len(p) for p in propositions]
        }


def compute_graph_similarity(pred_props: List[Tuple], target_props: List[Tuple]) -> float:
    """
    Compute similarity between two proposition graphs.
    
    Uses Jaccard similarity on the set of propositions.
    """
    if len(pred_props) == 0 and len(target_props) == 0:
        return 1.0
    
    pred_set = set(pred_props)
    target_set = set(target_props)
    
    intersection = len(pred_set & target_set)
    union = len(pred_set | target_set)
    
    return intersection / union if union > 0 else 0.0


# Example usage
if __name__ == "__main__":
    model = VariableLengthDLN(vocab_size=100, hidden_dim=64, num_entities=50)
    
    # Example: "The girl ran quickly"
    text_ids = torch.tensor([[10, 20, 30, 40]])  # (1, 4)
    
    result = model(text_ids)
    print(f"Parsed {result['num_propositions'][0]} propositions:")
    for i, prop in enumerate(result['propositions'][0]):
        print(f"  Prop {i+1}: entity={prop[0]}, relation={prop[1]}, value={prop[2]}")
