"""
Data Flow: Tokens → Propositions → Logic Network → Predictions

This demonstrates how raw text tokens and NLP-parsed propositions
are combined in the hybrid architecture.
"""
import torch
import torch.nn as nn
import spacy
from typing import List, Dict


def visualize_data_flow():
    """
    Show step-by-step how data flows through the hybrid system.
    """
    print("=" * 70)
    print("DATA FLOW: Tokens → Propositions → Logic Network")
    print("=" * 70)
    
    # Example story
    story = "Lily found a needle. She gave it to her mom. Mom was happy."
    
    print("\n1. RAW INPUT (what user provides)")
    print("-" * 70)
    print(f"Text: {story}")
    
    # Parse with spaCy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(story)
    
    print("\n2. TOKENIZATION (for AR model)")
    print("-" * 70)
    tokens = [token.text for token in doc]
    token_ids = list(range(len(tokens)))  # Simplified IDs
    print(f"Tokens: {tokens}")
    print(f"IDs:    {token_ids}")
    print(f"Shape:  (batch=1, seq_len={len(tokens)})")
    print(f"→ These go into: Token Transformer (AR prediction)")
    
    print("\n3. PROPOSITION EXTRACTION (for logic network)")
    print("-" * 70)
    propositions = []
    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                subject = None
                obj = None
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subject = child.text
                    if child.dep_ in ["dobj", "pobj"]:
                        obj = child.text
                
                if subject and obj:
                    propositions.append({
                        'subject': subject,
                        'verb': token.text,
                        'object': obj,
                        'subject_token_idx': token.i,  # Link back to tokens!
                        'verb_token_idx': token.i,
                        'object_token_idx': [c for c in token.children if c.dep_ == "dobj"][0].i if obj else -1
                    })
    
    print(f"Extracted {len(propositions)} propositions:")
    for i, prop in enumerate(propositions):
        print(f"  Prop {i}: {prop['subject']} --{prop['verb']}--> {prop['object']}")
        print(f"    Token positions: subject@{prop['subject_token_idx']}, verb@{prop['verb_token_idx']}, object@{prop['object_token_idx']}")
    
    print("\n4. ENCODE PROPOSITIONS AS VQ CODES")
    print("-" * 70)
    print("Each proposition (subj, rel, obj) is encoded to a discrete code:")
    print("  encode_proposition(Lily, found, needle) → VQ code 42")
    print("  encode_proposition(She, gave, it) → VQ code 127")
    vq_codes = [42, 127]  # Simulated VQ codes
    print(f"VQ codes: {vq_codes}")
    print(f"Shape: (batch=1, num_props={len(vq_codes)})")
    print(f"→ These go into: Logic Network (symbolic reasoning)")
    
    print("\n5. PARALLEL PROCESSING")
    print("-" * 70)
    print("Path A: Token-level AR")
    print("  Input: token_ids [0,1,2,3,4,5,6,7,8,9,10,11,12]")
    print("  Process: Transformer(tokens)")
    print("  Output: hidden_states (batch, seq_len, hidden_dim)")
    print("  → Predicts next token autoregressively")
    print()
    print("Path B: Proposition-level Logic")
    print("  Input: vq_codes [42, 127]")
    print("  Process: LogicNetwork(codes)")
    print("  Output: reasoning_states (batch, num_props, hidden_dim)")
    print("  → Applies logical rules and constraints")
    
    print("\n6. ALIGNMENT: Connect Propositions to Tokens")
    print("-" * 70)
    print("Key insight: Each proposition spans multiple tokens!")
    print()
    print("Proposition 0 (VQ code 42): 'Lily found needle'")
    print("  Spans tokens: [0, 1, 3] → 'Lily', 'found', 'needle'")
    print()
    print("Proposition 1 (VQ code 127): 'She gave it'")
    print("  Spans tokens: [5, 6, 7] → 'She', 'gave', 'it'")
    print()
    print("Alignment matrix:")
    print("  Token 0 (Lily)   ← aligned to → Prop 0")
    print("  Token 1 (found)  ← aligned to → Prop 0")
    print("  Token 3 (needle) ← aligned to → Prop 0")
    print("  Token 5 (She)    ← aligned to → Prop 1")
    print("  Token 6 (gave)   ← aligned to → Prop 1")
    print("  Token 7 (it)     ← aligned to → Prop 1")
    
    print("\n7. FUSION: Combine Token & Proposition Info")
    print("-" * 70)
    print("For each token position, attend to relevant propositions:")
    print()
    print("Predicting token 4 ('a'):")
    print("  Token hidden state: [0.2, -0.5, 0.8, ...]  (from Path A)")
    print("  Query: What token comes after 'needle'?")
    print("  Attention to propositions:")
    print("    Prop 0 relevance: 0.9 (high - we're in this proposition)")
    print("    Prop 1 relevance: 0.1 (low - not started yet)")
    print("  Fused representation:")
    print("    = token_state + 0.9*prop_0_reasoning + 0.1*prop_1_reasoning")
    print("  Final prediction: softmax(fused) → 'a' (50%), 'the' (30%), ...")
    
    print("\n8. CONCRETE EXAMPLE WITH TENSORS")
    print("-" * 70)
    
    # Simulate the actual tensor operations
    batch_size = 1
    seq_len = 13  # Number of tokens
    num_props = 2  # Number of propositions
    hidden_dim = 64
    
    # Path A: Token representations
    token_hidden = torch.randn(batch_size, seq_len, hidden_dim)
    print(f"Token hidden states: {token_hidden.shape}")
    
    # Path B: Proposition representations
    prop_codes = torch.tensor([[42, 127]])  # VQ codes
    prop_embeddings = nn.Embedding(512, hidden_dim)  # Codebook
    prop_hidden = prop_embeddings(prop_codes)
    print(f"Proposition hidden: {prop_hidden.shape}")
    
    # Create alignment mask (which tokens belong to which propositions)
    alignment = torch.zeros(batch_size, seq_len, num_props)
    alignment[0, [0, 1, 3], 0] = 1.0  # Tokens 0,1,3 → Prop 0
    alignment[0, [5, 6, 7], 1] = 1.0  # Tokens 5,6,7 → Prop 1
    print(f"Alignment mask: {alignment.shape}")
    
    # Attention: Each token queries relevant propositions
    attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
    attended_props, attention_weights = attention(
        token_hidden,  # queries: what am I predicting?
        prop_hidden,   # keys: what propositions exist?
        prop_hidden    # values: proposition reasoning
    )
    print(f"Attended propositions: {attended_props.shape}")
    print(f"Attention weights: {attention_weights.shape}")
    
    # Combine
    combined = torch.cat([token_hidden, attended_props], dim=-1)
    print(f"Combined representation: {combined.shape}")
    
    # Final prediction
    vocab_size = 1000
    output_layer = nn.Linear(hidden_dim * 2, vocab_size)
    logits = output_layer(combined)
    print(f"Final logits: {logits.shape}")
    print(f"→ For each token position, predict next token from vocab")
    
    print("\n9. TRAINING SIGNAL")
    print("-" * 70)
    print("Loss = Token AR Loss + Proposition Consistency Loss")
    print()
    print("Token AR Loss (standard cross-entropy):")
    print("  Predict token 1 given token 0")
    print("  Predict token 2 given tokens 0-1")
    print("  ... (standard language modeling)")
    print()
    print("Proposition Consistency Loss:")
    print("  If proposition says 'Lily found needle'")
    print("  Then token sequence must be consistent with this")
    print("  Penalize predictions that violate logical constraints")
    
    print("\n10. BENEFITS OF THIS ARCHITECTURE")
    print("-" * 70)
    print("✓ No information loss - trains on all tokens")
    print("✓ Symbolic reasoning - uses propositions when available")
    print("✓ Soft fallback - if NLP fails, pure AR still works")
    print("✓ Interpretable - can inspect which props influence predictions")
    print("✓ Scalable - propositions are optional auxiliary signal")


def show_logic_network_integration():
    """
    Show how traditional logic rules work with propositions.
    """
    print("\n" + "=" * 70)
    print("LOGIC NETWORK INTEGRATION")
    print("=" * 70)
    
    print("\nTraditional Logic Network (from your existing code):")
    print("-" * 70)
    print("""
class LogicNetwork:
    def forward(self, propositions):
        # propositions: list of (subject_id, relation_id, object_id)
        
        # Add to working memory
        for prop in propositions:
            self.working_memory.append(prop)
        
        # Apply logic rules
        # Rule 1: Transitivity
        # If (A, has, B) and (B, in, C) → infer (A, in, C)
        
        # Rule 2: Consistency checking
        # If (A, at, B) and (A, at, C) and B≠C → contradiction!
        
        # Return: Updated memory state
    """)
    
    print("\nHow propositions feed into this:")
    print("-" * 70)
    print("1. Extract from tokens: 'Lily found needle' → (Lily, found, needle)")
    print("2. Encode with VQ: (Lily, found, needle) → code 42")
    print("3. Pass to Logic Network: LogicNetwork([42, ...])")
    print("4. Logic rules operate on codes:")
    print("   - Decode 42 → (entity_5, relation_3, entity_12)")
    print("   - Check working memory for conflicts")
    print("   - Apply inference rules")
    print("   - Update entity states")
    print("5. Return reasoning results → attend back to tokens")
    
    print("\nKey difference from proposition-only approach:")
    print("-" * 70)
    print("OLD: Propositions are PRIMARY data (lose 80% of text)")
    print("NEW: Propositions are AUXILIARY signal (keep 100% of text)")
    print()
    print("OLD: Model only sees: 'girl sees needle' → 'Lily interacts mom'")
    print("NEW: Model sees: All 17 tokens + 2 extracted propositions")


if __name__ == "__main__":
    visualize_data_flow()
    show_logic_network_integration()
    
    print("\n" + "=" * 70)
    print("SUMMARY: The hybrid architecture feeds data as follows:")
    print("  Raw text → Tokens (100% coverage)")
    print("  Tokens → NLP Parser → Propositions (partial extraction)")
    print("  Propositions → VQ Encoder → Codes")
    print("  Codes → Logic Network → Reasoning")
    print("  Token states + Reasoning states → Attention fusion")
    print("  Fused state → Predict next token")
    print("=" * 70)
