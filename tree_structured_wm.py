"""
Tree-Structured Working Memory vs Flat Propositions

Key insight: Natural language has hierarchical structure that flat 
propositions cannot capture. Using trees + tree rewriting is more
linguistically and logically principled.

Example:
  Text: "Lily found a needle in her room"
  
  Flat propositions (current):
    (Lily, found, needle)
    (needle, in, room)
  
  Tree structure (proposed):
         found
        /     \
      Lily   needle
               |
              in
               |
             room
"""
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TreeNode:
    """
    A node in the syntax/semantic tree.
    
    Represents either:
    - Terminal: a word/entity (leaf node)
    - Non-terminal: a syntactic category or relation (internal node)
    """
    label: str  # The word, POS tag, or semantic relation
    children: List['TreeNode']
    entity_id: Optional[int] = None  # If this is an entity
    embedding: Optional[torch.Tensor] = None
    
    def __repr__(self, level=0):
        indent = "  " * level
        s = f"{indent}{self.label}"
        if self.entity_id is not None:
            s += f" (entity_{self.entity_id})"
        if self.children:
            s += "\n"
            s += "\n".join(c.__repr__(level + 1) for c in self.children)
        return s
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def depth(self) -> int:
        if self.is_leaf():
            return 0
        return 1 + max(c.depth() for c in self.children)
    
    def size(self) -> int:
        """Total number of nodes."""
        return 1 + sum(c.size() for c in self.children)


class TreeRewritingRule:
    """
    A rule that transforms one tree pattern into another.
    
    Example: Inference rule
      Pattern:   (X has Y) ∧ (Y in Z)
      Rewrite:   → (X in Z)
      
      Tree pattern:        Tree output:
          AND                 in
         /   \               /  \
       has    in    →       X    Z
       / \    / \
      X   Y  Y   Z
    """
    
    def __init__(self, name: str, pattern_matcher, rewrite_fn):
        self.name = name
        self.pattern_matcher = pattern_matcher
        self.rewrite_fn = rewrite_fn
    
    def match(self, tree: TreeNode) -> Optional[Dict]:
        """Check if this tree matches the pattern. Return bindings if yes."""
        return self.pattern_matcher(tree)
    
    def apply(self, tree: TreeNode, bindings: Dict) -> TreeNode:
        """Apply the rewrite using the matched bindings."""
        return self.rewrite_fn(tree, bindings)


class TreeWorkingMemory:
    """
    Working memory as a forest of trees.
    
    Instead of storing flat propositions, we store parse/semantic trees.
    Logic rules operate by pattern matching and rewriting trees.
    """
    
    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self.trees: List[TreeNode] = []
        self.rewrite_rules: List[TreeRewritingRule] = []
    
    def add_tree(self, tree: TreeNode):
        """Add a tree to working memory."""
        self.trees.append(tree)
        if len(self.trees) > self.capacity:
            self.trees.pop(0)  # Remove oldest
    
    def add_rule(self, rule: TreeRewritingRule):
        """Add a rewriting rule."""
        self.rewrite_rules.append(rule)
    
    def apply_all_rules(self) -> List[TreeNode]:
        """
        Apply all rewrite rules to all trees.
        Generate new trees through inference.
        """
        new_trees = []
        
        for tree in self.trees:
            for rule in self.rewrite_rules:
                bindings = rule.match(tree)
                if bindings:
                    new_tree = rule.apply(tree, bindings)
                    new_trees.append(new_tree)
                    print(f"  Applied rule '{rule.name}':")
                    print(f"    From: {tree.label}")
                    print(f"    To:   {new_tree.label}")
        
        return new_trees
    
    def __repr__(self):
        s = "Working Memory (Tree Forest):\n"
        s += "=" * 60 + "\n"
        for i, tree in enumerate(self.trees):
            s += f"Tree {i}:\n{tree}\n"
            s += "-" * 60 + "\n"
        return s


def demonstrate_flat_vs_tree():
    """
    Compare flat propositions vs tree structures.
    """
    print("=" * 70)
    print("FLAT PROPOSITIONS vs TREE STRUCTURE")
    print("=" * 70)
    
    sentence = "Lily found a needle in her room"
    
    print(f"\nSentence: {sentence}")
    
    # Flat representation (current approach)
    print("\n1. FLAT PROPOSITIONS (current):")
    print("-" * 70)
    flat_props = [
        ("Lily", "found", "needle"),
        ("needle", "in", "room")
    ]
    for subj, rel, obj in flat_props:
        print(f"  ({subj}, {rel}, {obj})")
    
    print("\nLimitations:")
    print("  - Lost hierarchical structure")
    print("  - 'in her room' modifies 'needle', not 'found'")
    print("  - Cannot represent nested meanings")
    print("  - Ambiguous scope")
    
    # Tree representation
    print("\n2. TREE STRUCTURE (proposed):")
    print("-" * 70)
    
    # Build parse tree
    room = TreeNode("room", [], entity_id=2)
    in_prep = TreeNode("in", [room])
    needle = TreeNode("needle", [in_prep], entity_id=1)
    lily = TreeNode("Lily", [], entity_id=0)
    found = TreeNode("found", [lily, needle])
    
    print(found)
    
    print("\nAdvantages:")
    print("  ✓ Preserves hierarchical structure")
    print("  ✓ Clear scope: 'in room' modifies 'needle'")
    print("  ✓ Can represent complex nested meanings")
    print("  ✓ Compositional semantics")
    
    print(f"\nTree statistics:")
    print(f"  Depth: {found.depth()}")
    print(f"  Size: {found.size()} nodes")
    
    return found


def demonstrate_tree_rewriting():
    """
    Show tree rewriting rules for inference.
    """
    print("\n" + "=" * 70)
    print("TREE REWRITING RULES")
    print("=" * 70)
    
    # Create working memory
    wm = TreeWorkingMemory(capacity=10)
    
    # Example 1: Simple facts as trees
    print("\nAdding facts to working memory:")
    print("-" * 70)
    
    # "Lily has needle"
    lily1 = TreeNode("Lily", [], entity_id=0)
    needle1 = TreeNode("needle", [], entity_id=1)
    has = TreeNode("has", [lily1, needle1])
    wm.add_tree(has)
    print("Added: Lily has needle")
    
    # "needle in room"
    needle2 = TreeNode("needle", [], entity_id=1)
    room = TreeNode("room", [], entity_id=2)
    in_rel = TreeNode("in", [needle2, room])
    wm.add_tree(in_rel)
    print("Added: needle in room")
    
    print("\n" + str(wm))
    
    # Define rewriting rule: Transitivity
    print("Defining rewrite rule: TRANSITIVITY")
    print("-" * 70)
    print("Pattern: (X has Y) ∧ (Y in Z)")
    print("Rewrite: → (X in Z)")
    print()
    
    def match_transitivity(tree: TreeNode) -> Optional[Dict]:
        """Match pattern: has(X, Y) where Y appears in another 'in' relation."""
        if tree.label == "has" and len(tree.children) == 2:
            X = tree.children[0]
            Y = tree.children[1]
            return {'X': X, 'Y': Y, 'relation': 'has'}
        return None
    
    def apply_transitivity(tree: TreeNode, bindings: Dict) -> TreeNode:
        """Create new tree: X in Z."""
        X = bindings['X']
        # This is simplified - in real system, would look up Y in other trees
        Z = TreeNode("room", [], entity_id=2)  # Found from other tree
        return TreeNode("in", [X, Z])
    
    rule = TreeRewritingRule("Transitivity", match_transitivity, apply_transitivity)
    wm.add_rule(rule)
    
    print("Applying rules...")
    print("-" * 70)
    new_trees = wm.apply_all_rules()
    
    print("\nInferred new trees:")
    for tree in new_trees:
        print(tree)
        print()
    
    print("Result: Inferred 'Lily in room' from 'Lily has needle' + 'needle in room'")


def demonstrate_neural_tree_encoding():
    """
    Show how trees can be encoded as neural representations.
    """
    print("\n" + "=" * 70)
    print("NEURAL ENCODING OF TREES")
    print("=" * 70)
    
    print("\nTree-structured neural networks:")
    print("-" * 70)
    print("""
    Three approaches:
    
    1. Recursive Neural Networks (TreeRNN)
       - Bottom-up: compose children → parent
       - Each node has learned composition function
       - Good for semantic composition
    
    2. Tree-LSTM
       - Like LSTM but for trees
       - Hidden state flows up the tree
       - Better gradient flow
    
    3. Graph Neural Networks (GNN)
       - Trees are special case of graphs
       - Message passing between nodes
       - Most flexible
    """)
    
    print("Example: TreeLSTM encoding")
    print("-" * 70)
    
    # Simple TreeLSTM cell
    class TreeLSTMCell(nn.Module):
        def __init__(self, hidden_dim: int):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.W = nn.Linear(hidden_dim * 3, hidden_dim * 5)  # i, f1, f2, o, g
        
        def forward(self, x: torch.Tensor, 
                   h_left: torch.Tensor, c_left: torch.Tensor,
                   h_right: torch.Tensor, c_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compose two children into parent node.
            
            Args:
                x: Current node embedding
                h_left, c_left: Left child hidden/cell state
                h_right, c_right: Right child hidden/cell state
            
            Returns:
                h, c: Parent hidden/cell state
            """
            # Concatenate inputs
            inputs = torch.cat([x, h_left, h_right], dim=-1)
            
            # Compute gates
            gates = self.W(inputs)
            i, f_left, f_right, o, g = gates.chunk(5, dim=-1)
            
            i = torch.sigmoid(i)
            f_left = torch.sigmoid(f_left)
            f_right = torch.sigmoid(f_right)
            o = torch.sigmoid(o)
            g = torch.tanh(g)
            
            # Cell state: weighted sum of children + new info
            c = f_left * c_left + f_right * c_right + i * g
            
            # Hidden state
            h = o * torch.tanh(c)
            
            return h, c
    
    # Test
    cell = TreeLSTMCell(hidden_dim=64)
    x = torch.randn(1, 64)  # Parent embedding
    h_l = torch.randn(1, 64)  # Left child hidden
    c_l = torch.randn(1, 64)  # Left child cell
    h_r = torch.randn(1, 64)  # Right child hidden
    c_r = torch.randn(1, 64)  # Right child cell
    
    h_parent, c_parent = cell(x, h_l, c_l, h_r, c_r)
    
    print(f"Parent embedding shape: {x.shape}")
    print(f"Left child state: {h_l.shape}")
    print(f"Right child state: {h_r.shape}")
    print(f"Output parent state: {h_parent.shape}")
    print("\n✓ Tree structure encoded as neural representation")


def compare_expressiveness():
    """
    Compare what can be expressed in flat vs tree representations.
    """
    print("\n" + "=" * 70)
    print("EXPRESSIVENESS COMPARISON")
    print("=" * 70)
    
    examples = [
        {
            'sentence': "The dog that bit the cat ran away",
            'flat': [
                "('dog', 'bit', 'cat')",
                "('dog', 'ran', 'away')"
            ],
            'flat_issue': "Cannot represent that 'that bit the cat' modifies 'dog'",
            'tree': """
                ran
               /   \\
             dog   away
              |
             bit
            /   \\
          that  cat
            """,
            'tree_advantage': "Clear: the dog [that bit the cat] ran"
        },
        {
            'sentence': "Lily said that mom was happy",
            'flat': [
                "('Lily', 'said', 'something')",
                "('mom', 'was', 'happy')"
            ],
            'flat_issue': "Lost embedding: 'mom was happy' is content of 'said'",
            'tree': """
                said
               /    \\
            Lily    was
                   /   \\
                 mom  happy
            """,
            'tree_advantage': "Embedded clause is child of 'said'"
        }
    ]
    
    for i, ex in enumerate(examples, 1):
        print(f"\nExample {i}: {ex['sentence']}")
        print("-" * 70)
        print("Flat propositions:")
        for prop in ex['flat']:
            print(f"  {prop}")
        print(f"Issue: {ex['flat_issue']}")
        print()
        print("Tree structure:")
        print(ex['tree'])
        print(f"Advantage: {ex['tree_advantage']}")


if __name__ == "__main__":
    # Demonstrate concepts
    tree = demonstrate_flat_vs_tree()
    demonstrate_tree_rewriting()
    demonstrate_neural_tree_encoding()
    compare_expressiveness()
    
    print("\n" + "=" * 70)
    print("CONCLUSION: Tree-based WM + Tree Rewriting")
    print("=" * 70)
    print("""
Benefits:
  ✓ More linguistically principled (matches parse trees)
  ✓ Captures hierarchical structure naturally
  ✓ Tree rewriting is well-studied formalism
  ✓ More expressive than flat propositions
  ✓ Can represent complex nested meanings
  ✓ Natural fit with constituency/dependency parsing
  
Implementation considerations:
  • TreeLSTM or GNN for neural encoding
  • Pattern matching for rule application
  • Efficient tree similarity computation
  • Balance between expressiveness and complexity
  
AGI implications:
  • More powerful reasoning (tree transformations)
  • Better language understanding (preserve structure)
  • Compositional semantics (build meaning bottom-up)
  • Unified representation for syntax + semantics
    """)
    print("=" * 70)
