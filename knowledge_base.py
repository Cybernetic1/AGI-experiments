"""
Knowledge Base with Common Sense Reasoning
==========================================

Implements symbolic inference using Experta (RETE algorithm).
Supports reification: Parse NL common sense → symbolic rules.
"""

from experta import *
from typing import List, Tuple, Dict, Any


class Event(Fact):
    """Event with thematic roles (Davidsonian)."""
    pass


class Entity(Fact):
    """Entity (person, object, location)."""
    pass


class Inference(Fact):
    """Inferred fact from reasoning."""
    pass


class CommonSenseKB(KnowledgeEngine):
    """
    Knowledge base with common sense reasoning rules.
    Uses RETE algorithm (via Experta) for efficient pattern matching.
    """
    
    # ========== Common Sense Rules (Reified Knowledge) ==========
    
    @Rule(Event(type='guillotine', agent=MATCH.x))
    def rule_guillotine_kills(self, x):
        """If someone is guillotined, they die."""
        self.declare(Inference(type='dead', entity=x))
        print(f"  [INFERENCE] {x} is dead (guillotined)")
    
    @Rule(Event(type='behead', agent=MATCH.x))
    def rule_behead_kills(self, x):
        """If someone is beheaded, they die."""
        self.declare(Inference(type='dead', entity=x))
        print(f"  [INFERENCE] {x} is dead (beheaded)")
    
    @Rule(Inference(type='dead', entity=MATCH.x))
    def rule_dead_cannot_act(self, x):
        """If someone is dead, they cannot perform future actions."""
        self.declare(Inference(type='cannot_act', entity=x))
        print(f"  [INFERENCE] {x} cannot act (dead)")
    
    @Rule(Event(type='execute', patient=MATCH.x))
    def rule_execute_kills(self, x):
        """If someone is executed, they die."""
        self.declare(Inference(type='dead', entity=x))
        print(f"  [INFERENCE] {x} is dead (executed)")
    
    # ========== Davidsonian Inference Rules ==========
    
    @Rule(Event(type=MATCH.action, agent=MATCH.agent),
          NOT(Inference(type='cannot_act', entity=MATCH.agent)))
    def rule_agent_can_act(self, action, agent):
        """If agent performs action and is not dead, they can act."""
        self.declare(Inference(type='can_do', entity=agent, action=action))
    
    @Rule(Event(type=MATCH.action, patient=MATCH.patient))
    def rule_patient_affected(self, action, patient):
        """If patient is affected by action, record it."""
        self.declare(Inference(type='affected_by', entity=patient, action=action))
    
    # ========== Location and Spatial Reasoning ==========
    
    @Rule(Event(type='sit', agent=MATCH.x, location=MATCH.loc))
    def rule_sitting_at_location(self, x, loc):
        """If X sits at location, X is at location."""
        self.declare(Inference(type='at_location', entity=x, location=loc))
        print(f"  [INFERENCE] {x} is at {loc}")
    
    @Rule(Event(type='go', agent=MATCH.x, location=MATCH.loc))
    def rule_going_to_location(self, x, loc):
        """If X goes to location, X will be at location."""
        self.declare(Inference(type='at_location', entity=x, location=loc))
        print(f"  [INFERENCE] {x} is at {loc}")


class ReificationEngine:
    """
    Parse natural language common sense → symbolic rules.
    
    Example:
    NL: "If someone is guillotined, they die"
    → Rule: guillotine(X) → dead(X)
    """
    
    def __init__(self, kb: CommonSenseKB):
        self.kb = kb
        self.custom_rules = []
    
    def add_common_sense_text(self, text: str):
        """
        Parse common sense from natural language.
        This is a simplified version - can be extended with better NL parsing.
        """
        text_lower = text.lower()
        
        # Pattern: "if X then Y"
        if "guillotine" in text_lower and "die" in text_lower:
            print(f"  [REIFIED] guillotine(X) → dead(X)")
            # Already implemented in KB
            
        elif "behead" in text_lower and "die" in text_lower:
            print(f"  [REIFIED] behead(X) → dead(X)")
            # Already implemented in KB
            
        elif "dead" in text_lower and "cannot" in text_lower:
            print(f"  [REIFIED] dead(X) → cannot_act(X)")
            # Already implemented in KB
        
        else:
            print(f"  [REIFIED] Unknown pattern: {text}")


def integrate_davidsonian_with_kb(propositions: List[Tuple[str, str, str]], 
                                   kb: CommonSenseKB):
    """
    Convert Davidsonian propositions to Experta facts and run inference.
    """
    print("\n" + "="*70)
    print("Adding Davidsonian propositions to KB")
    print("="*70)
    
    for entity, relation, value in propositions:
        if relation == "type" and value in ["love", "give", "sit", "guillotine", "behead", "execute", "go"]:
            # This is an event
            # Find all related propositions for this event
            event_props = {r: v for e, r, v in propositions if e == entity}
            print(f"  Event {entity}: {event_props}")
            kb.declare(Event(**event_props))
        
        elif relation == "type" and value == "entity":
            # This is an entity
            entity_props = {r: v for e, r, v in propositions if e == entity and r != "type"}
            if entity_props:
                print(f"  Entity {entity}: {entity_props}")
                kb.declare(Entity(name=entity, **entity_props))
    
    print("\n" + "="*70)
    print("Running Inference (RETE algorithm)")
    print("="*70)
    
    kb.run()
    
    print("\n" + "="*70)
    print("Final Knowledge Base State")
    print("="*70)
    
    print(f"\nTotal facts: {len(kb.facts)}")
    
    print("\nInferred facts:")
    for fact in kb.facts:
        if isinstance(fact, Inference):
            print(f"  {fact}")


def test_knowledge_base():
    """Test the knowledge base with common sense reasoning."""
    from davidsonian_extraction import DavidsonianExtractor
    
    print("="*70)
    print("Knowledge Base Test: Common Sense Reasoning")
    print("="*70)
    
    # Test 1: "Mary was guillotined"
    print("\n" + "="*70)
    print("Test 1: Mary Queen of Scots")
    print("="*70)
    
    extractor = DavidsonianExtractor()
    kb = CommonSenseKB()
    kb.reset()
    
    sentence = "Mary was guillotined yesterday"
    print(f"\nSentence: {sentence}")
    
    propositions = extractor.extract(sentence)
    print(f"\nExtracted {len(propositions)} propositions")
    
    integrate_davidsonian_with_kb(propositions, kb)
    
    # Test 2: "The cat sat on the mat"
    print("\n" + "="*70)
    print("Test 2: Cat on the mat")
    print("="*70)
    
    extractor = DavidsonianExtractor()
    kb = CommonSenseKB()
    kb.reset()
    
    sentence = "The cat sat on the mat"
    print(f"\nSentence: {sentence}")
    
    propositions = extractor.extract(sentence)
    integrate_davidsonian_with_kb(propositions, kb)
    
    # Test 3: Reification
    print("\n" + "="*70)
    print("Test 3: Reification (Adding Common Sense)")
    print("="*70)
    
    kb = CommonSenseKB()
    kb.reset()
    reifier = ReificationEngine(kb)
    
    common_sense = [
        "If someone is guillotined, they die",
        "If someone is beheaded, they die",
        "If someone is dead, they cannot perform actions",
    ]
    
    print("\nAdding common sense rules:")
    for rule_text in common_sense:
        print(f"  {rule_text}")
        reifier.add_common_sense_text(rule_text)
    
    print("\n✓ Rules already implemented in KB")


if __name__ == "__main__":
    test_knowledge_base()
