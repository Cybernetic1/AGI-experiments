"""
Implicit Graph Working Memory

Key insight: We don't need explicit tree/graph structures.
Entity IDs create implicit links between propositions.

Example:
    WM = [
        [cat_1, type, cat],      # Entity cat_1 defined
        [cat_1, on, mat_1],      # Links cat_1 to mat_1
        [mat_1, type, mat],      # Entity mat_1 defined
        [mat_1, on, floor_1],    # Links mat_1 to floor_1
    ]
    
    Graph structure is implicit:
    cat_1 → on → mat_1 → on → floor_1
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional


class ImplicitGraphWM:
    """
    Working Memory with implicit graph structure via entity indexing.
    
    Propositions are stored as flat list: [entity, relation, value]
    Entity IDs create implicit links (no explicit graph needed).
    """
    
    def __init__(self):
        # Primary storage: flat list of propositions
        self.propositions = []  # List of [entity, relation, value]
        
        # Index for fast lookup (optional optimization)
        self.entity_index = {}  # entity_id → list of proposition indices
        self.relation_index = {}  # relation → list of proposition indices
    
    def add_proposition(self, proposition: List) -> int:
        """
        Add proposition to working memory.
        
        Args:
            proposition: [entity, relation, value] or tensor
        
        Returns:
            Index of added proposition
        """
        idx = len(self.propositions)
        self.propositions.append(proposition)
        
        # Update entity index
        entity = proposition[0]
        if entity not in self.entity_index:
            self.entity_index[entity] = []
        self.entity_index[entity].append(idx)
        
        # Update relation index
        relation = proposition[1]
        if relation not in self.relation_index:
            self.relation_index[relation] = []
        self.relation_index[relation].append(idx)
        
        return idx
    
    def get_entity_propositions(self, entity_id) -> List:
        """
        Get all propositions about an entity.
        
        This is like querying a node in a graph!
        """
        indices = self.entity_index.get(entity_id, [])
        return [self.propositions[i] for i in indices]
    
    def find_links(self, entity_id, relation, threshold: float = 0.8) -> List:
        """
        Soft graph traversal: Find entities linked via relation.
        
        Args:
            entity_id: Starting entity
            relation: Relation to follow
            threshold: Minimum match confidence
        
        Returns:
            List of (confidence, target_entity) tuples
        
        Example:
            find_links(cat_1, "on") → [(0.95, mat_1)]
        """
        matches = []
        
        # Get all propositions about this entity
        for idx in self.entity_index.get(entity_id, []):
            prop = self.propositions[idx]
            
            # Check if relation matches (soft)
            if self._fuzzy_match(prop[1], relation) > threshold:
                confidence = self._fuzzy_match(prop[1], relation)
                target = prop[2]
                matches.append((confidence, target))
        
        return matches
    
    def find_path(self, start_entity, end_entity, max_hops: int = 3) -> Optional[List]:
        """
        Find path between entities (graph search over implicit graph).
        
        Args:
            start_entity: Starting entity ID
            end_entity: Target entity ID
            max_hops: Maximum path length
        
        Returns:
            List of propositions forming the path, or None
        
        Example:
            find_path(cat_1, floor_1) → [
                [cat_1, on, mat_1],
                [mat_1, on, floor_1]
            ]
        """
        # BFS search
        queue = [(start_entity, [])]
        visited = set()
        
        while queue:
            current, path = queue.pop(0)
            
            if current == end_entity:
                return path
            
            if len(path) >= max_hops:
                continue
            
            if current in visited:
                continue
            visited.add(current)
            
            # Get all propositions about current entity
            for idx in self.entity_index.get(current, []):
                prop = self.propositions[idx]
                target = prop[2]
                
                # Skip if target is not an entity
                if not self._is_entity(target):
                    continue
                
                # Add to queue
                new_path = path + [prop]
                queue.append((target, new_path))
        
        return None  # No path found
    
    def to_tensor(self, device='cpu') -> torch.Tensor:
        """
        Convert propositions to tensor for neural network processing.
        
        Returns:
            Tensor of shape (N, 3) where N is number of propositions
        """
        if not self.propositions:
            return torch.zeros(0, 3, device=device)
        
        # Convert to tensor (assuming propositions are already numeric)
        return torch.tensor(self.propositions, device=device, dtype=torch.float32)
    
    def clear(self):
        """Clear working memory."""
        self.propositions.clear()
        self.entity_index.clear()
        self.relation_index.clear()
    
    def _fuzzy_match(self, a, b) -> float:
        """
        Soft matching between two values.
        
        For prototype: exact match only
        TODO: Replace with learned similarity
        """
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            return torch.cosine_similarity(a, b, dim=0).item()
        else:
            return 1.0 if a == b else 0.0
    
    def _is_entity(self, value) -> bool:
        """Check if value is an entity ID."""
        # Simple heuristic: entities end with _number
        if isinstance(value, str):
            return '_' in value and value.split('_')[-1].isdigit()
        return False
    
    def __len__(self) -> int:
        return len(self.propositions)
    
    def __repr__(self) -> str:
        return f"ImplicitGraphWM({len(self.propositions)} propositions, " \
               f"{len(self.entity_index)} entities)"


class SoftGraphMatcher(nn.Module):
    """
    Neural module for soft graph pattern matching over implicit graph.
    
    Performs O(N²) soft matching to find graph patterns.
    But only O(R) parameters (rule templates).
    """
    
    def __init__(self, pattern_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        self.pattern_dim = pattern_dim
        self.hidden_dim = hidden_dim
        
        # Learnable pattern embeddings
        self.pattern_embedder = nn.Linear(pattern_dim, hidden_dim)
        self.prop_embedder = nn.Linear(pattern_dim, hidden_dim)
        
        # Similarity scorer
        self.similarity = nn.CosineSimilarity(dim=-1)
    
    def match_pattern(self, pattern: torch.Tensor, working_memory: torch.Tensor) -> torch.Tensor:
        """
        Soft match pattern against all propositions in WM.
        
        Args:
            pattern: (pattern_dim,) - Pattern to match
            working_memory: (N, pattern_dim) - Propositions in WM
        
        Returns:
            match_scores: (N,) - Soft match score for each proposition
        """
        # Embed pattern and propositions
        pattern_emb = self.pattern_embedder(pattern)  # (hidden_dim,)
        prop_embs = self.prop_embedder(working_memory)  # (N, hidden_dim)
        
        # Compute similarity (differentiable!)
        match_scores = self.similarity(
            pattern_emb.unsqueeze(0),  # (1, hidden_dim)
            prop_embs                   # (N, hidden_dim)
        )  # (N,)
        
        return torch.sigmoid(match_scores)  # Normalize to (0, 1)
    
    def match_multi_hop(self, 
                       pattern1: torch.Tensor,
                       pattern2: torch.Tensor,
                       working_memory: torch.Tensor) -> torch.Tensor:
        """
        Match two-hop pattern: pattern1 → pattern2.
        
        This is O(N²) but finds implicit graph connections!
        
        Args:
            pattern1: First pattern (e.g., [X, on, Y])
            pattern2: Second pattern (e.g., [Y, on, Z])
            working_memory: (N, 3) propositions
        
        Returns:
            match_scores: (N, N) - match[i,j] = confidence that prop[i] and prop[j] form pattern
        """
        N = working_memory.size(0)
        
        # Match each pattern independently
        match1 = self.match_pattern(pattern1, working_memory)  # (N,)
        match2 = self.match_pattern(pattern2, working_memory)  # (N,)
        
        # Compute pairwise matching (O(N²))
        match_pairs = torch.zeros(N, N)
        
        for i in range(N):
            for j in range(N):
                if i != j:
                    # Check if prop[i][2] (Y in first pattern) matches prop[j][0] (Y in second pattern)
                    # This is the "link" condition!
                    link_match = self._check_entity_link(working_memory[i], working_memory[j])
                    
                    # Combined match score
                    match_pairs[i, j] = match1[i] * match2[j] * link_match
        
        return match_pairs
    
    def _check_entity_link(self, prop1: torch.Tensor, prop2: torch.Tensor) -> torch.Tensor:
        """
        Check if prop1[2] (object of first) matches prop2[0] (subject of second).
        
        This is the implicit graph link!
        """
        # For continuous representations, use similarity
        similarity = self.similarity(prop1[2:3], prop2[0:1])
        return torch.sigmoid(similarity)


def test_implicit_graph():
    """Test implicit graph working memory."""
    print("Testing Implicit Graph Working Memory")
    print("=" * 60)
    
    # Create WM
    wm = ImplicitGraphWM()
    
    # Add propositions (simulating: cat on mat, mat on floor)
    wm.add_proposition(["cat_1", "type", "cat"])
    wm.add_proposition(["cat_1", "on", "mat_1"])
    wm.add_proposition(["mat_1", "type", "mat"])
    wm.add_proposition(["mat_1", "on", "floor_1"])
    wm.add_proposition(["floor_1", "type", "floor"])
    
    print(f"\nWorking Memory: {wm}")
    
    # Test entity lookup
    print("\nPropositions about cat_1:")
    cat_props = wm.get_entity_propositions("cat_1")
    for prop in cat_props:
        print(f"  {prop}")
    
    # Test link finding
    print("\nWhat is cat_1 on?")
    links = wm.find_links("cat_1", "on")
    for conf, target in links:
        print(f"  {target} (confidence: {conf:.2f})")
    
    # Test path finding
    print("\nPath from cat_1 to floor_1:")
    path = wm.find_path("cat_1", "floor_1")
    if path:
        for prop in path:
            print(f"  {prop}")
    
    print("\n" + "=" * 60)
    print("✓ Implicit graph structure works!")
    print("  - No explicit graph data structure")
    print("  - Entity IDs create implicit links")
    print("  - Multi-hop reasoning via path finding")


if __name__ == "__main__":
    test_implicit_graph()
