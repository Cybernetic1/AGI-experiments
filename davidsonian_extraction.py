"""
Davidsonian Event Extraction using Meta-Rules
==============================================

Implements Neo-Davidsonian semantics for NL parsing.
Uses symbolic meta-rules to extract events with thematic roles.
"""

import spacy
from typing import List, Dict, Tuple, Any

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


class DavidsonianExtractor:
    """
    Extract propositions using Davidsonian event semantics.
    
    Core idea: Every verb denotes an event with thematic roles:
    - agent (subject)
    - patient (object)
    - recipient (indirect object)
    - manner (adverbs)
    - location (prepositional phrases with 'in', 'at')
    - instrument (prepositional phrases with 'with')
    - time (temporal expressions)
    """
    
    def __init__(self):
        self.event_counter = 0
        self.entity_counter = 0
        
    def create_event_id(self):
        """Generate unique event ID."""
        self.event_counter += 1
        return f"e{self.event_counter}"
    
    def create_entity_id(self, text):
        """Generate entity ID (or reuse for known entities)."""
        # For now, use text as entity ID (simple)
        # Later: implement entity registry
        return text.lower()
    
    def extract(self, sentence: str) -> List[Tuple[str, str, str]]:
        """
        Extract Davidsonian propositions from sentence.
        
        Returns: List of (entity, relation, value) triples
        """
        doc = nlp(sentence)
        propositions = []
        
        # Meta-Rule 1: For each verb, create an event
        for token in doc:
            if token.pos_ == "VERB":
                event_id = self.create_event_id()
                
                # Event type (the verb itself)
                propositions.append((event_id, "type", token.lemma_))
                
                # Meta-Rule 2: Subject → Agent
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        entity_id = self.create_entity_id(child.text)
                        propositions.append((event_id, "agent", entity_id))
                        propositions.append((entity_id, "type", "entity"))
                    
                    # Meta-Rule 3: Direct object → Patient
                    elif child.dep_ == "dobj":
                        entity_id = self.create_entity_id(child.text)
                        propositions.append((event_id, "patient", entity_id))
                        propositions.append((entity_id, "type", "entity"))
                    
                    # Meta-Rule 4: Indirect object → Recipient
                    elif child.dep_ == "dative":
                        entity_id = self.create_entity_id(child.text)
                        propositions.append((event_id, "recipient", entity_id))
                        propositions.append((entity_id, "type", "entity"))
                    
                    # Meta-Rule 5: Adverb → Manner
                    elif child.dep_ == "advmod":
                        propositions.append((event_id, "manner", child.text.lower()))
                    
                    # Meta-Rule 6: Prepositional phrases
                    elif child.dep_ == "prep":
                        prep_type = child.text.lower()
                        
                        # Find object of preposition
                        for prep_child in child.children:
                            if prep_child.dep_ == "pobj":
                                entity_id = self.create_entity_id(prep_child.text)
                                propositions.append((entity_id, "type", "entity"))
                                
                                # Location markers
                                if prep_type in ["in", "at", "on"]:
                                    propositions.append((event_id, "location", entity_id))
                                
                                # Instrument markers
                                elif prep_type == "with":
                                    propositions.append((event_id, "instrument", entity_id))
                                
                                # Recipient markers (alternative to dative)
                                elif prep_type == "to":
                                    propositions.append((event_id, "recipient", entity_id))
                                
                                # Time markers
                                elif prep_type in ["during", "after", "before"]:
                                    propositions.append((event_id, "time", entity_id))
                
                # Meta-Rule 7: Tense (simple version)
                if token.tag_ in ["VBD", "VBN"]:  # Past tense
                    propositions.append((event_id, "tense", "past"))
                elif token.tag_ in ["VBZ", "VBP"]:  # Present
                    propositions.append((event_id, "tense", "present"))
                elif token.tag_ == "VB":  # Base/future
                    propositions.append((event_id, "tense", "present"))
        
        return propositions
    
    def extract_with_details(self, sentence: str) -> Dict[str, Any]:
        """
        Extract propositions with detailed information for debugging.
        """
        propositions = self.extract(sentence)
        doc = nlp(sentence)
        
        return {
            'sentence': sentence,
            'propositions': propositions,
            'dependencies': [(token.text, token.dep_, token.head.text) 
                           for token in doc],
            'pos_tags': [(token.text, token.pos_) for token in doc],
        }


def test_extractor():
    """Test the Davidsonian extractor."""
    extractor = DavidsonianExtractor()
    
    test_sentences = [
        "John loves Mary",
        "John quickly gave Mary the book",
        "The cat sat on the mat",
        "Mary was guillotined yesterday",
    ]
    
    print("=" * 70)
    print("Davidsonian Extraction Test")
    print("=" * 70)
    
    for sentence in test_sentences:
        print(f"\nSentence: {sentence}")
        print("-" * 70)
        
        result = extractor.extract_with_details(sentence)
        
        print("Propositions:")
        for entity, relation, value in result['propositions']:
            print(f"  [{entity}, {relation}, {value}]")
        
        print("\nDependencies:")
        for token, dep, head in result['dependencies']:
            print(f"  {token} --{dep}--> {head}")


if __name__ == "__main__":
    test_extractor()
