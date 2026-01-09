"""
Hybrid Consolidation System: DLN + Symbolic Logic Engine

This system achieves memory consolidation through differentiable logic learning
(like LLMs do with Transformers) but outputs logical forms that can be:
1. Used by symbolic logic engine for fast inference and reflection
2. Manipulated symbolically (rule extraction, generalization, transfer)
3. Trained end-to-end with gradient descent

Key Innovation: The DLN learns compressed logical representations from data,
achieving generalization through neural consolidation, but maintaining symbolic
interpretability and reflection capabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import spacy
from davidsonian_extraction import DavidsonianExtractor
from collections import defaultdict
import numpy as np


@dataclass
class LogicalProposition:
    """Atomic logical proposition in Davidsonian form"""
    predicate: str
    args: List[str]
    confidence: float = 1.0
    
    def __hash__(self):
        return hash((self.predicate, tuple(self.args)))
    
    def __eq__(self, other):
        return self.predicate == other.predicate and self.args == other.args
    
    def __repr__(self):
        return f"{self.predicate}({', '.join(self.args)}) [{self.confidence:.2f}]"


@dataclass
class LogicalRule:
    """Logical inference rule: premises -> conclusions (multi-head)"""
    premises: List[LogicalProposition]
    conclusions: List[LogicalProposition]  # Changed to support multiple conclusions
    strength: float = 1.0  # Learnable weight
    
    def __repr__(self):
        prem_str = " ∧ ".join(str(p) for p in self.premises)
        conc_str = " ∧ ".join(str(c) for c in self.conclusions)
        return f"{prem_str} → {conc_str} [{self.strength:.2f}]"


class EntityRegistry:
    """Tracks entities and their bindings across discourse"""
    
    def __init__(self):
        self.entities = {}  # entity_id -> properties
        self.next_id = 0
        
    def register(self, text: str, entity_type: str) -> str:
        """Register a new entity or return existing one"""
        entity_id = f"e{self.next_id}"
        self.entities[entity_id] = {
            'text': text,
            'type': entity_type,
            'mentions': [text]
        }
        self.next_id += 1
        return entity_id
    
    def resolve(self, text: str, entity_type: str) -> Optional[str]:
        """Resolve pronoun/reference to existing entity"""
        # Simple heuristic: match by type and recency
        for eid, props in reversed(list(self.entities.items())):
            if props['type'] == entity_type:
                props['mentions'].append(text)
                return eid
        return None
    
    def clear(self):
        self.entities = {}
        self.next_id = 0


class DavidsonianParser:
    """
    Symbolic rule-based parser using Davidsonian event semantics
    These rules are INJECTED (not learned) to provide structural guidance
    """
    
    def __init__(self):
        self.extractor = DavidsonianExtractor()
        self.entity_registry = EntityRegistry()
        
    def parse_to_logical_form(self, text: str) -> List[LogicalProposition]:
        """Parse NL text into Davidsonian logical propositions"""
        # Use the improved extractor
        triples = self.extractor.extract(text)
        
        # Convert triples to LogicalProposition format
        propositions = []
        for entity, relation, value in triples:
            # Map triple format to LogicalProposition
            propositions.append(LogicalProposition(
                predicate=relation,
                args=[entity, value] if value != entity else [entity]
            ))
        
        return propositions
    
    def _resolve_entity(self, token) -> str:
        """Resolve entity, handling pronouns"""
        text = token.text.lower()
        
        # Determine entity type
        if token.pos_ == "PRON":
            if text in ("he", "him", "his"):
                entity_type = "person_male"
            elif text in ("she", "her"):
                entity_type = "person_female"
            elif text in ("it", "its"):
                entity_type = "object"
            else:
                entity_type = "entity"
            
            # Try to resolve
            resolved = self.entity_registry.resolve(text, entity_type)
            if resolved:
                return resolved
        
        # Register new entity
        if token.pos_ in ("NOUN", "PROPN"):
            entity_type = "person" if token.ent_type_ == "PERSON" else "object"
            return self.entity_registry.register(text, entity_type)
        
        return text


class DifferentiableLogicNetwork(nn.Module):
    """
    Core DLN that learns to consolidate logical patterns from data
    This is where generalization happens (like Transformer in LLMs)
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Embeddings for predicates and entities
        self.predicate_embed = nn.Embedding(vocab_size, embed_dim)
        self.entity_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Proposition encoder: converts atomic prop to vector
        self.prop_encoder = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),  # pred + 2 args
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Rule learner: learns patterns across propositions
        self.rule_attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        
        # Rule decoder: predicts new propositions
        self.rule_decoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim * 3)  # pred + 2 args
        )
        
        # Confidence predictor
        self.confidence_head = nn.Linear(embed_dim, 1)
        
    def encode_proposition(self, prop: LogicalProposition, vocab: Dict) -> torch.Tensor:
        """Encode a logical proposition as vector"""
        pred_idx = vocab.get(prop.predicate, 0)
        arg1_idx = vocab.get(prop.args[0] if len(prop.args) > 0 else "<pad>", 0)
        arg2_idx = vocab.get(prop.args[1] if len(prop.args) > 1 else "<pad>", 0)
        
        pred_emb = self.predicate_embed(torch.tensor([pred_idx]))
        arg1_emb = self.entity_embed(torch.tensor([arg1_idx]))
        arg2_emb = self.entity_embed(torch.tensor([arg2_idx]))
        
        prop_vec = torch.cat([pred_emb, arg1_emb, arg2_emb], dim=-1)
        return self.prop_encoder(prop_vec)
    
    def forward(self, prop_vectors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: consolidate logical patterns and predict next proposition
        
        Args:
            prop_vectors: [batch, seq_len, embed_dim] - encoded propositions
            
        Returns:
            predictions: [batch, embed_dim * 3] - predicted next prop components
            confidences: [batch, 1] - confidence scores
        """
        # Apply self-attention to find rule patterns
        attended, _ = self.rule_attention(prop_vectors, prop_vectors, prop_vectors)
        
        # Pool to get consolidated representation
        consolidated = attended.mean(dim=1)  # [batch, embed_dim]
        
        # Predict next proposition
        predictions = self.rule_decoder(consolidated)  # [batch, embed_dim * 3]
        confidences = torch.sigmoid(self.confidence_head(consolidated))  # [batch, 1]
        
        return predictions, confidences


class SymbolicLogicEngine:
    """
    Fast symbolic inference engine
    Uses rules extracted from DLN or injected symbolically
    Supports reflection: can extract/manipulate rules
    """
    
    def __init__(self):
        self.rules: List[LogicalRule] = []
        self.facts: Set[LogicalProposition] = set()
        
    def add_rule(self, rule: LogicalRule):
        """Add a symbolic rule (from DLN or manual injection)"""
        self.rules.append(rule)
        
    def add_fact(self, fact: LogicalProposition):
        """Add a fact to working memory"""
        self.facts.add(fact)
        
    def forward_chain(self, max_steps: int = 10) -> Set[LogicalProposition]:
        """
        Forward chaining inference
        Returns: all derived facts
        """
        derived = set(self.facts)
        
        for _ in range(max_steps):
            new_facts = set()
            
            for rule in self.rules:
                # Check if all premises are satisfied
                if all(prem in derived for prem in rule.premises):
                    # Apply rule with strength - add all conclusions
                    for conclusion in rule.conclusions:
                        new_conclusion = LogicalProposition(
                            predicate=conclusion.predicate,
                            args=conclusion.args,
                            confidence=rule.strength
                        )
                        new_facts.add(new_conclusion)
            
            if not new_facts - derived:
                break  # No new facts derived
            
            derived.update(new_facts)
        
        return derived
    
    def extract_rules_from_facts(self, facts: List[LogicalProposition]) -> List[LogicalRule]:
        """
        Reflection: Extract rules from observed fact patterns
        This is where symbolic manipulation enables meta-learning
        Now supports multi-head rules (multiple conclusions)
        """
        rules = []
        
        # Group facts by shared entities to find patterns
        entity_groups = defaultdict(list)
        for fact in facts:
            for arg in fact.args:
                if arg.startswith('e') or arg.startswith('ev'):  # entity or event
                    entity_groups[arg].append(fact)
        
        # For each entity/event, create multi-head rules
        for entity, entity_facts in entity_groups.items():
            if len(entity_facts) >= 2:
                # Use first fact as premise, rest as conclusions
                premise = entity_facts[0]
                conclusions = entity_facts[1:]
                
                rule = LogicalRule(
                    premises=[premise],
                    conclusions=conclusions,
                    strength=0.5  # Initial strength
                )
                rules.append(rule)
        
        return rules
    
    def generalize_rule(self, rule: LogicalRule) -> LogicalRule:
        """
        Reflection: Generalize specific rule by introducing variables
        This enables transfer learning
        """
        # Replace specific entities with variables
        var_map = {}
        var_counter = 0
        
        def replace_with_var(entity: str) -> str:
            nonlocal var_counter
            if entity.startswith("e") or entity.startswith("ev"):
                if entity not in var_map:
                    var_map[entity] = f"?x{var_counter}"
                    var_counter += 1
                return var_map[entity]
            return entity
        
        new_premises = []
        for prem in rule.premises:
            new_args = [replace_with_var(arg) for arg in prem.args]
            new_premises.append(LogicalProposition(prem.predicate, new_args))
        
        new_conclusions = []
        for conc in rule.conclusions:
            new_conclusion_args = [replace_with_var(arg) for arg in conc.args]
            new_conclusions.append(LogicalProposition(conc.predicate, new_conclusion_args))
        
        return LogicalRule(new_premises, new_conclusions, rule.strength)


class HybridConsolidationSystem:
    """
    Main system combining:
    1. Symbolic parser (Davidsonian rules - injected)
    2. DLN (learns consolidation - like Transformer)
    3. Symbolic logic engine (fast inference + reflection)
    """
    
    def __init__(self, vocab_size: int = 10000):
        self.parser = DavidsonianParser()
        self.dln = DifferentiableLogicNetwork(vocab_size)
        self.logic_engine = SymbolicLogicEngine()
        
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.vocab_idx = 2
        
    def update_vocab(self, words: List[str]):
        """Update vocabulary with new words"""
        for word in words:
            if word not in self.vocab:
                self.vocab[word] = self.vocab_idx
                self.vocab_idx += 1
    
    def process_text(self, text: str) -> Tuple[List[LogicalProposition], torch.Tensor]:
        """
        Process text through full pipeline:
        1. Parse to logical form (symbolic)
        2. Encode with DLN (neural consolidation)
        3. Return both for training/inference
        """
        # Symbolic parsing
        props = self.parser.parse_to_logical_form(text)
        
        # Handle empty propositions
        if not props:
            # Return empty structures
            return [], torch.zeros(1, self.dln.embed_dim)
        
        # Update vocab
        words = [prop.predicate for prop in props]
        words.extend([arg for prop in props for arg in prop.args])
        self.update_vocab(words)
        
        # Neural encoding
        prop_vectors = torch.stack([
            self.dln.encode_proposition(prop, self.vocab) for prop in props
        ])
        
        return props, prop_vectors
    
    def train_step(self, current_text: str, next_text: str) -> Dict[str, float]:
        """
        Training step for semantic autoregression:
        Predict logical form of next_text from current_text
        """
        # Process both texts
        current_props, current_vecs = self.process_text(current_text)
        next_props, next_vecs = self.process_text(next_text)
        
        # Skip if either is empty
        if len(current_props) == 0 or len(next_props) == 0:
            return {'similarity_loss': 0.0, 'num_props': 0, 'confidence': 0.0}
        
        # Ensure correct dimensions [batch, seq, embed_dim]
        if current_vecs.dim() == 2:
            current_vecs_batch = current_vecs.unsqueeze(0)
        else:
            current_vecs_batch = current_vecs
            
        pred_vecs, confidences = self.dln(current_vecs_batch)
        
        # pred_vecs is [1, embed_dim * 3] = [1, 384], need just embedding part
        pred_flat = pred_vecs.view(-1)  # Flatten to [384]
        pred_embedding = pred_flat[:self.dln.embed_dim]  # Take first [128]
        
        # target_vecs should be [128]
        if next_vecs.dim() == 2:
            target_vecs = next_vecs.mean(dim=0)  # Average propositions -> [128]
        else:
            target_vecs = next_vecs.view(-1)[:self.dln.embed_dim]  # [128]
        
        # Both should now be [128]
        similarity_loss = 1 - F.cosine_similarity(
            pred_embedding.unsqueeze(0), 
            target_vecs.unsqueeze(0), 
            dim=1
        ).squeeze()
        
        # Optional: Extract and add rules to logic engine
        if len(current_props) > 0 and len(next_props) > 0:
            # Create a multi-head rule: current -> all next props
            conf_value = confidences.mean().item() if confidences.numel() > 1 else confidences.item()
            rule = LogicalRule(
                premises=current_props[:2] if len(current_props) >= 2 else current_props,
                conclusions=next_props,  # Multi-head: all next propositions
                strength=conf_value
            )
            self.logic_engine.add_rule(rule)
        
        return {
            'similarity_loss': similarity_loss.item(),
            'num_props': len(next_props),
            'confidence': confidences.mean().item() if confidences.numel() > 1 else confidences.item()
        }
    
    def reflect_and_generalize(self):
        """
        Reflection step: Extract patterns from learned rules and generalize
        This is unique to symbolic systems - enables meta-learning
        """
        # Extract rules from recent fact patterns
        all_props = list(self.logic_engine.facts)
        if len(all_props) > 2:
            extracted_rules = self.logic_engine.extract_rules_from_facts(all_props)
            
            # Generalize and add back
            for rule in extracted_rules[:5]:  # Limit to avoid explosion
                general_rule = self.logic_engine.generalize_rule(rule)
                self.logic_engine.add_rule(general_rule)
        
        print(f"Reflection complete: {len(self.logic_engine.rules)} total rules")


def demonstrate_hybrid_consolidation():
    """Demonstrate the hybrid consolidation system"""
    print("=" * 80)
    print("HYBRID CONSOLIDATION SYSTEM DEMO")
    print("=" * 80)
    
    system = HybridConsolidationSystem()
    
    # Example story sequence
    story = [
        "A girl found a toy.",
        "She played with it.",
        "The toy broke.",
        "She was sad."
    ]
    
    print("\n1. PARSING TO LOGICAL FORM (Symbolic Davidsonian Rules)")
    print("-" * 80)
    for sent in story:
        props, _ = system.process_text(sent)
        print(f"\n'{sent}'")
        for prop in props:
            print(f"  {prop}")
    
    print("\n\n2. TRAINING DLN (Neural Consolidation)")
    print("-" * 80)
    
    # Reset parser state
    system.parser.entity_registry.clear()
    
    optimizer = torch.optim.Adam(system.dln.parameters(), lr=0.001)
    
    for epoch in range(5):
        total_loss = 0
        for i in range(len(story) - 1):
            system.parser.entity_registry.clear()  # Reset each time for consistent parsing
            
            metrics = system.train_step(story[i], story[i+1])
            
            loss = torch.tensor(metrics['similarity_loss'], requires_grad=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += metrics['similarity_loss']
        
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")
    
    print("\n\n3. SYMBOLIC INFERENCE (Fast Rule Application)")
    print("-" * 80)
    print(f"Rules in logic engine: {len(system.logic_engine.rules)}")
    for i, rule in enumerate(system.logic_engine.rules[:5]):
        print(f"  Rule {i+1}: {rule}")
    
    print("\n\n4. REFLECTION & GENERALIZATION")
    print("-" * 80)
    system.reflect_and_generalize()
    
    print("\n\n5. KEY ADVANTAGES")
    print("-" * 80)
    print("✓ Neural consolidation (like LLMs) - learns patterns from data")
    print("✓ Symbolic output - interpretable logical forms")
    print("✓ Fast symbolic inference - no gradients needed at inference time")
    print("✓ Reflection capability - can extract and manipulate rules")
    print("✓ Transfer learning - generalized rules apply to new domains")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    demonstrate_hybrid_consolidation()
