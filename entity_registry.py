"""
Persistent Entity Registry - Core AGI Component

This registry maintains a global database of all entities encountered,
with their properties, relationships, and embeddings. Entities persist
across stories and can accumulate knowledge over time.

Key features:
- Unique entity IDs that persist across contexts
- Entity embeddings for similarity comparison
- Property tracking (type, attributes, etc.)
- Relationship storage
- Entity merging/deduplication
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
import json


class Entity:
    """Represents a single entity with all its properties and relationships."""
    
    def __init__(self, entity_id: int, name: str, entity_type: Optional[str] = None):
        self.id = entity_id
        self.name = name
        self.entity_type = entity_type
        self.properties = {}  # {property_name: value}
        self.relations = defaultdict(set)  # {relation_type: {target_entity_ids}}
        self.mentions = []  # All names/aliases this entity has been referred to
        self.embedding = None  # Learned embedding
        self.created_at = None
        self.last_seen = None
    
    def add_property(self, property_name: str, value):
        """Add or update a property."""
        self.properties[property_name] = value
    
    def add_relation(self, relation_type: str, target_id: int):
        """Add a relation to another entity."""
        self.relations[relation_type].add(target_id)
    
    def add_mention(self, mention: str):
        """Record an alias/mention of this entity."""
        if mention not in self.mentions:
            self.mentions.append(mention)
    
    def to_dict(self) -> dict:
        """Serialize entity to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type,
            "properties": self.properties,
            "relations": {k: list(v) for k, v in self.relations.items()},
            "mentions": self.mentions
        }
    
    @classmethod
    def from_dict(cls, data: dict):
        """Deserialize entity from dictionary."""
        entity = cls(data["id"], data["name"], data.get("type"))
        entity.properties = data.get("properties", {})
        entity.relations = defaultdict(set, {
            k: set(v) for k, v in data.get("relations", {}).items()
        })
        entity.mentions = data.get("mentions", [])
        return entity


class PersistentEntityRegistry:
    """
    Global registry of all entities encountered.
    
    This is the core component that enables persistent knowledge
    across different stories, tasks, and contexts.
    """
    
    def __init__(self, embedding_dim: int = 64, similarity_threshold: float = 0.8):
        self.embedding_dim = embedding_dim
        self.similarity_threshold = similarity_threshold
        
        # Core storage
        self.entities: Dict[int, Entity] = {}  # {entity_id: Entity}
        self.next_id = 0
        
        # Lookup indices
        self.name_to_id: Dict[str, int] = {}  # {name: entity_id}
        self.type_index: Dict[str, Set[int]] = defaultdict(set)  # {type: {entity_ids}}
        
        # Neural components
        self.entity_embeddings = nn.Embedding(10000, embedding_dim)  # Support up to 10k entities
        
        # Statistics
        self.total_lookups = 0
        self.total_creations = 0
        self.total_merges = 0
    
    def get_or_create_entity(self, name: str, entity_type: Optional[str] = None, 
                            context: Optional[Dict] = None) -> int:
        """
        Get existing entity ID or create new one.
        
        Args:
            name: Entity name/mention
            entity_type: Optional type hint (e.g., "person", "location")
            context: Additional context for entity resolution
        
        Returns:
            entity_id: Unique integer ID for this entity
        """
        name = name.strip()
        
        # Exact name match
        if name in self.name_to_id:
            entity_id = self.name_to_id[name]
            self.total_lookups += 1
            return entity_id
        
        # Check for similar entities (case-insensitive, aliases)
        name_lower = name.lower()
        for existing_name, eid in self.name_to_id.items():
            if existing_name.lower() == name_lower:
                # Same entity, different case
                self.entities[eid].add_mention(name)
                self.name_to_id[name] = eid
                self.total_lookups += 1
                return eid
        
        # Check if it's a pronoun reference (requires context)
        if context and name.lower() in ['he', 'she', 'it', 'they', 'him', 'her', 'them']:
            # Would need coreference resolution here
            # For now, create separate entity
            pass
        
        # Create new entity
        entity_id = self._create_new_entity(name, entity_type)
        self.total_creations += 1
        return entity_id
    
    def _create_new_entity(self, name: str, entity_type: Optional[str] = None) -> int:
        """Create a new entity in the registry."""
        entity_id = self.next_id
        self.next_id += 1
        
        entity = Entity(entity_id, name, entity_type)
        entity.add_mention(name)
        
        self.entities[entity_id] = entity
        self.name_to_id[name] = entity_id
        
        if entity_type:
            self.type_index[entity_type].add(entity_id)
        
        # Initialize embedding
        entity.embedding = self.entity_embeddings(torch.tensor(entity_id))
        
        return entity_id
    
    def get_entity(self, entity_id: int) -> Optional[Entity]:
        """Retrieve entity by ID."""
        return self.entities.get(entity_id)
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Retrieve entity by name."""
        entity_id = self.name_to_id.get(name)
        if entity_id is not None:
            return self.entities[entity_id]
        return None
    
    def add_property(self, entity_id: int, property_name: str, value):
        """Add property to an entity."""
        if entity_id in self.entities:
            self.entities[entity_id].add_property(property_name, value)
    
    def add_relation(self, subject_id: int, relation_type: str, object_id: int):
        """Add a relation between two entities."""
        if subject_id in self.entities:
            self.entities[subject_id].add_relation(relation_type, object_id)
    
    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type."""
        entity_ids = self.type_index.get(entity_type, set())
        return [self.entities[eid] for eid in entity_ids]
    
    def get_entity_embedding(self, entity_id: int) -> Optional[torch.Tensor]:
        """Get the embedding for an entity."""
        if entity_id in self.entities:
            return self.entities[entity_id].embedding
        return None
    
    def find_similar_entities(self, entity_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find entities similar to the given entity based on embeddings.
        
        Returns:
            List of (entity_id, similarity_score) tuples
        """
        if entity_id not in self.entities:
            return []
        
        query_embedding = self.entities[entity_id].embedding
        similarities = []
        
        for eid, entity in self.entities.items():
            if eid == entity_id:
                continue
            
            if entity.embedding is not None:
                sim = torch.cosine_similarity(
                    query_embedding.unsqueeze(0),
                    entity.embedding.unsqueeze(0)
                ).item()
                similarities.append((eid, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def merge_entities(self, keep_id: int, merge_id: int):
        """
        Merge two entities (when we discover they're the same).
        
        All properties, relations, and mentions from merge_id
        are transferred to keep_id, and merge_id is removed.
        """
        if keep_id not in self.entities or merge_id not in self.entities:
            return
        
        keep_entity = self.entities[keep_id]
        merge_entity = self.entities[merge_id]
        
        # Merge properties
        keep_entity.properties.update(merge_entity.properties)
        
        # Merge relations
        for rel_type, targets in merge_entity.relations.items():
            keep_entity.relations[rel_type].update(targets)
        
        # Merge mentions
        for mention in merge_entity.mentions:
            keep_entity.add_mention(mention)
            self.name_to_id[mention] = keep_id
        
        # Remove merged entity
        del self.entities[merge_id]
        self.total_merges += 1
    
    def query_relations(self, subject_id: int, relation_type: str) -> Set[int]:
        """Query relations of a specific type for an entity."""
        if subject_id in self.entities:
            return self.entities[subject_id].relations.get(relation_type, set())
        return set()
    
    def query_property(self, entity_id: int, property_name: str):
        """Query a specific property of an entity."""
        if entity_id in self.entities:
            return self.entities[entity_id].properties.get(property_name)
        return None
    
    def save(self, filepath: str):
        """Save registry to disk."""
        data = {
            "next_id": self.next_id,
            "entities": {eid: entity.to_dict() for eid, entity in self.entities.items()},
            "name_to_id": self.name_to_id,
            "type_index": {k: list(v) for k, v in self.type_index.items()},
            "stats": {
                "total_lookups": self.total_lookups,
                "total_creations": self.total_creations,
                "total_merges": self.total_merges
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Save embeddings separately (binary)
        torch.save(self.entity_embeddings.state_dict(), 
                  filepath.replace('.json', '_embeddings.pt'))
    
    def load(self, filepath: str):
        """Load registry from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.next_id = data["next_id"]
        self.name_to_id = data["name_to_id"]
        self.type_index = defaultdict(set, {
            k: set(v) for k, v in data["type_index"].items()
        })
        
        # Restore entities
        self.entities = {}
        for eid_str, entity_data in data["entities"].items():
            eid = int(eid_str)
            self.entities[eid] = Entity.from_dict(entity_data)
        
        # Restore stats
        stats = data.get("stats", {})
        self.total_lookups = stats.get("total_lookups", 0)
        self.total_creations = stats.get("total_creations", 0)
        self.total_merges = stats.get("total_merges", 0)
        
        # Load embeddings
        try:
            self.entity_embeddings.load_state_dict(
                torch.load(filepath.replace('.json', '_embeddings.pt'))
            )
            # Restore embedding references
            for eid, entity in self.entities.items():
                entity.embedding = self.entity_embeddings(torch.tensor(eid))
        except FileNotFoundError:
            print("Warning: Embeddings file not found, using fresh embeddings")
    
    def get_stats(self) -> dict:
        """Get registry statistics."""
        return {
            "total_entities": len(self.entities),
            "total_lookups": self.total_lookups,
            "total_creations": self.total_creations,
            "total_merges": self.total_merges,
            "entities_by_type": {k: len(v) for k, v in self.type_index.items()}
        }
    
    def reset(self):
        """Clear all entities (for testing/new experiments)."""
        self.entities = {}
        self.next_id = 0
        self.name_to_id = {}
        self.type_index = defaultdict(set)
        self.total_lookups = 0
        self.total_creations = 0
        self.total_merges = 0
    
    def __len__(self):
        return len(self.entities)
    
    def __repr__(self):
        return f"EntityRegistry({len(self.entities)} entities, {len(self.type_index)} types)"


if __name__ == "__main__":
    # Demo usage
    print("=== Entity Registry Demo ===\n")
    
    registry = PersistentEntityRegistry(embedding_dim=64)
    
    # Create entities
    mary_id = registry.get_or_create_entity("Mary", entity_type="person")
    john_id = registry.get_or_create_entity("John", entity_type="person")
    bathroom_id = registry.get_or_create_entity("bathroom", entity_type="location")
    
    print(f"Created entities: Mary={mary_id}, John={john_id}, bathroom={bathroom_id}")
    
    # Add properties
    registry.add_property(mary_id, "gender", "female")
    registry.add_property(bathroom_id, "room_type", "bathroom")
    
    # Add relations
    registry.add_relation(mary_id, "located_at", bathroom_id)
    
    # Query
    mary_location = registry.query_relations(mary_id, "located_at")
    print(f"\nMary's location: {mary_location}")
    
    mary_gender = registry.query_property(mary_id, "gender")
    print(f"Mary's gender: {mary_gender}")
    
    # Get entity
    mary = registry.get_entity(mary_id)
    print(f"\nMary entity: {mary.to_dict()}")
    
    # Stats
    print(f"\nRegistry stats: {registry.get_stats()}")
    
    # Save/load
    registry.save("test_registry.json")
    print("\n✓ Registry saved to test_registry.json")
    
    # Load in new registry
    new_registry = PersistentEntityRegistry(embedding_dim=64)
    new_registry.load("test_registry.json")
    print(f"✓ Loaded registry: {new_registry}")
    print(f"  Stats: {new_registry.get_stats()}")
