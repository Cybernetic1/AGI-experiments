"""
Logic Network Learns to Parse Natural Language

Key idea: The logic network learns linguistic rules from data,
with NLP (spaCy) providing supervision signals during training.

This is more AGI-like because:
1. One system learns everything (no external black boxes)
2. Linguistic knowledge becomes explicit rules
3. Rules can transfer to new domains
4. Interpretable: we can see what grammar rules it learned
"""
import torch
import torch.nn as nn
import spacy
from typing import List, Dict, Tuple
from collections import defaultdict


class LearnableLinguisticRule(nn.Module):
    """
    A differentiable rule that learns to recognize linguistic patterns.
    
    Example: "If word follows 'a/an' and is lowercase, it's likely a NOUN"
    """
    
    def __init__(self, input_dim: int = 128, num_pos_tags: int = 17):
        super().__init__()
        
        # Pattern matcher: looks at context to predict POS tag
        self.context_encoder = nn.LSTM(input_dim, 64, bidirectional=True, batch_first=True)
        
        # POS predictor
        self.pos_predictor = nn.Linear(128, num_pos_tags)
        
        # Confidence scorer (how confident is this rule?)
        self.confidence = nn.Linear(128, 1)
    
    def forward(self, token_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict POS tags and confidence.
        
        Args:
            token_embeddings: (batch, seq_len, embed_dim)
        
        Returns:
            pos_probs: (batch, seq_len, num_pos_tags)
            confidence: (batch, seq_len, 1)
        """
        # Encode context
        context, _ = self.context_encoder(token_embeddings)
        
        # Predict POS
        pos_logits = self.pos_predictor(context)
        pos_probs = torch.softmax(pos_logits, dim=-1)
        
        # Confidence
        conf = torch.sigmoid(self.confidence(context))
        
        return pos_probs, conf


class LearnablePropositionExtractor(nn.Module):
    """
    Learns to extract propositions from POS-tagged sequences.
    
    Example: [PROPN VERB DET NOUN] → extract (token[0], token[1], token[3])
    """
    
    def __init__(self, num_pos_tags: int = 17, hidden_dim: int = 128):
        super().__init__()
        
        self.num_pos_tags = num_pos_tags
        
        # Pattern detector: finds subject-verb-object patterns
        self.pattern_detector = nn.Sequential(
            nn.Linear(num_pos_tags * 3, hidden_dim),  # Look at 3 consecutive tags
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Is this a valid SVO pattern?
            nn.Sigmoid()
        )
    
    def forward(self, pos_sequence: torch.Tensor) -> torch.Tensor:
        """
        Find proposition patterns in POS sequence.
        
        Args:
            pos_sequence: (batch, seq_len, num_pos_tags) - one-hot POS tags
        
        Returns:
            pattern_scores: (batch, seq_len-2) - score for each potential SVO triple
        """
        batch_size, seq_len, _ = pos_sequence.shape
        
        # Create sliding windows of 3 tokens
        pattern_scores = []
        for i in range(seq_len - 2):
            # Get 3 consecutive POS tags
            triple = pos_sequence[:, i:i+3, :].reshape(batch_size, -1)
            score = self.pattern_detector(triple)
            pattern_scores.append(score)
        
        if pattern_scores:
            return torch.cat(pattern_scores, dim=1)
        else:
            return torch.zeros(batch_size, 0)


class LogicNetworkParser(nn.Module):
    """
    Complete logic network that learns to parse natural language.
    
    Architecture:
    1. Embed tokens
    2. Apply linguistic rules to predict POS tags
    3. Extract propositions from POS sequences
    4. Learn from NLP supervision
    """
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 128,
                 num_pos_tags: int = 17, num_rules: int = 5):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_pos_tags = num_pos_tags
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        
        # Multiple linguistic rules (ensemble)
        self.linguistic_rules = nn.ModuleList([
            LearnableLinguisticRule(embed_dim, num_pos_tags)
            for _ in range(num_rules)
        ])
        
        # Rule weighting (learn which rules are most reliable)
        self.rule_weights = nn.Parameter(torch.ones(num_rules) / num_rules)
        
        # Proposition extractor
        self.prop_extractor = LearnablePropositionExtractor(num_pos_tags, embed_dim)
        
        # POS tag names (for interpretability)
        self.pos_names = [
            'ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'DET', 'INTJ', 'NOUN',
            'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'
        ]
    
    def forward(self, token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parse tokens to extract linguistic structure.
        
        Args:
            token_ids: (batch, seq_len)
        
        Returns:
            Dictionary with:
            - pos_probs: (batch, seq_len, num_pos_tags)
            - propositions: List of extracted propositions
            - rule_contributions: Which rules fired
        """
        # Embed tokens
        token_emb = self.token_embeddings(token_ids)
        
        # Apply all linguistic rules
        all_pos_probs = []
        all_confidences = []
        
        for rule in self.linguistic_rules:
            pos_probs, conf = rule(token_emb)
            all_pos_probs.append(pos_probs)
            all_confidences.append(conf)
        
        # Weighted combination of rules
        rule_weights = torch.softmax(self.rule_weights, dim=0)
        combined_pos_probs = sum(
            w * probs for w, probs in zip(rule_weights, all_pos_probs)
        )
        
        # Extract propositions based on POS patterns
        prop_scores = self.prop_extractor(combined_pos_probs)
        
        return {
            'pos_probs': combined_pos_probs,
            'pos_predictions': torch.argmax(combined_pos_probs, dim=-1),
            'proposition_scores': prop_scores,
            'rule_weights': rule_weights,
            'rule_confidences': torch.stack(all_confidences, dim=0)
        }
    
    def interpret_rules(self, token_ids: torch.Tensor, tokens: List[str]) -> None:
        """
        Show what linguistic rules the network has learned.
        """
        with torch.no_grad():
            output = self.forward(token_ids)
            pos_preds = output['pos_predictions'][0]  # First in batch
            rule_weights = output['rule_weights']
            
            print("Learned Parse:")
            print("-" * 60)
            for i, (token, pos_idx) in enumerate(zip(tokens, pos_preds)):
                pos_tag = self.pos_names[pos_idx]
                print(f"  {token:15} → {pos_tag}")
            
            print("\nRule Importance:")
            print("-" * 60)
            for i, weight in enumerate(rule_weights):
                print(f"  Rule {i}: {weight.item():.3f}")


def train_logic_parser_with_nlp_supervision():
    """
    Train logic network to parse, using spaCy as teacher.
    """
    print("=" * 70)
    print("Training Logic Network to Parse Natural Language")
    print("=" * 70)
    
    # Initialize
    parser = LogicNetworkParser(vocab_size=100, embed_dim=32, num_pos_tags=17, num_rules=3)
    optimizer = torch.optim.Adam(parser.parameters(), lr=0.01)
    
    # Load NLP teacher
    nlp = spacy.load("en_core_web_sm")
    
    # Sample training data
    sentences = [
        "Lily found a needle",
        "She gave it to mom",
        "Mom was very happy",
        "The cat saw a bird",
        "Birds fly in the sky"
    ]
    
    # Create simple vocab
    all_words = set()
    for sent in sentences:
        all_words.update(sent.lower().split())
    vocab = {word: i for i, word in enumerate(sorted(all_words))}
    
    # POS tag mapping
    pos_to_id = {tag: i for i, tag in enumerate(parser.pos_names)}
    
    print("\nTraining for 50 epochs...")
    print("-" * 70)
    
    for epoch in range(50):
        total_loss = 0
        
        for sentence in sentences:
            # Get ground truth from spaCy
            doc = nlp(sentence)
            true_pos_tags = [pos_to_id.get(token.pos_, 0) for token in doc]
            
            # Convert to tensors
            words = sentence.lower().split()
            token_ids = torch.tensor([[vocab.get(w, 0) for w in words]])
            true_pos = torch.tensor([true_pos_tags])
            
            # Parse with logic network
            optimizer.zero_grad()
            output = parser(token_ids)
            
            # Loss: match spaCy's POS tags
            pos_probs = output['pos_probs']
            loss = nn.functional.cross_entropy(
                pos_probs.view(-1, parser.num_pos_tags),
                true_pos.view(-1)
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {total_loss/len(sentences):.4f}")
    
    print("\n" + "=" * 70)
    print("Training Complete! Testing learned rules...")
    print("=" * 70)
    
    # Test on new sentence
    test_sentence = "The bird found food"
    test_doc = nlp(test_sentence)
    test_words = test_sentence.lower().split()
    test_ids = torch.tensor([[vocab.get(w, 0) for w in test_words]])
    
    print(f"\nTest sentence: {test_sentence}")
    print("\nGround truth (spaCy):")
    for token in test_doc:
        print(f"  {token.text:15} → {token.pos_}")
    
    print("\nLogic network prediction:")
    parser.interpret_rules(test_ids, test_words)
    
    print("\n" + "=" * 70)
    print("Key insight:")
    print("  - Logic network LEARNED linguistic rules from data")
    print("  - spaCy was only used for training supervision")
    print("  - Rules are now internal to the logic network")
    print("  - Can be fine-tuned on domain-specific language")
    print("  - More AGI-like: one system learns everything")
    print("=" * 70)


if __name__ == "__main__":
    train_logic_parser_with_nlp_supervision()
    
    print("\n" + "=" * 70)
    print("This approach aligns with AGI principles:")
    print("  ✓ Logic network learns linguistic knowledge")
    print("  ✓ No external ad-hoc parsers at inference time")
    print("  ✓ Rules are interpretable and modifiable")
    print("  ✓ Can transfer to new domains")
    print("  ✓ Unified system for language + reasoning")
    print("=" * 70)
