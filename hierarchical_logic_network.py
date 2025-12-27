#!/usr/bin/env python3
"""
Hierarchical Logic Network: Combining Autoregressive and Reinforcement Learning

Architecture:
- Level 1: Concept Formation (Logic Rules) - learns game dynamics via AR
- Level 2: Concept Valuation (Value Network) - learns strategy via RL

Training Protocol:
- Phase 1: AR only - learn concepts from game data
- Phase 2: RL only - learn to value concepts (logic rules frozen)
- Phase 3: Joint refinement - both adapt together
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from neural_logic_core import LogicNetwork


class HierarchicalLogicNetwork(nn.Module):
    """
    Dual-head architecture with shared logic rules.
    
    Shared: Logic rules extract concepts (captured variables)
    Head 1: Autoregressive - predict next state from concepts
    Head 2: RL - evaluate actions based on concepts
    """
    
    def __init__(self, num_rules=8, num_premises=2, var_slots=3, 
                 value_hidden_dim=32):
        super().__init__()
        
        # Level 1: Concept formation (shared logic rules)
        # Input: 9 propositions [player, position]
        # Output: captured variables (concepts)
        self.logic_rules = LogicNetwork(
            prop_length=2,        # [player, position]
            num_props=9,          # 9 board positions
            output_dim=var_slots * num_rules,  # Combined captured vars from all rules
            num_rules=num_rules,
            num_premises=num_premises,
            var_slots=var_slots,
        )
        
        self.var_slots = var_slots
        self.num_rules = num_rules
        self.total_concept_dim = var_slots * num_rules
        
        # Head 1: Autoregressive prediction
        # Input: concepts → Output: next state (9 positions × 3 classes)
        self.ar_head = nn.Sequential(
            nn.Linear(self.total_concept_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 9 * 3),  # 9 positions, 3 states each
        )
        
        # Head 2: RL value function
        # Input: concepts → Output: Q-values for 9 actions
        self.value_network = nn.Sequential(
            nn.Linear(self.total_concept_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 9),  # Q-value per board position
        )
    
    def extract_concepts(self, working_memory):
        """
        Extract concepts from working memory using logic rules.
        
        Args:
            working_memory: (batch, 9, 2) - board as propositions
        
        Returns:
            concepts: (batch, total_concept_dim) - captured variables from all rules
            rule_details: list of per-rule information
        """
        # Process through logic rules
        output, rule_details = self.logic_rules(working_memory, return_details=True)
        
        # Extract captured variables from each rule as concepts
        batch_size = working_memory.shape[0]
        concepts = torch.zeros(batch_size, self.total_concept_dim, 
                              device=working_memory.device)
        
        for i, rule_info in enumerate(rule_details):
            start_idx = i * self.var_slots
            end_idx = start_idx + self.var_slots
            concepts[:, start_idx:end_idx] = rule_info['captured_vars']
        
        return concepts, rule_details
    
    def forward_ar(self, working_memory):
        """
        Autoregressive forward pass: predict next state.
        
        Args:
            working_memory: (batch, 9, 2)
        
        Returns:
            next_state_logits: (batch, 9, 3)
            concepts: (batch, concept_dim)
        """
        concepts, _ = self.extract_concepts(working_memory)
        next_state_logits = self.ar_head(concepts)
        next_state_logits = next_state_logits.view(-1, 9, 3)
        
        return next_state_logits, concepts
    
    def forward_rl(self, working_memory):
        """
        RL forward pass: evaluate actions.
        
        Args:
            working_memory: (batch, 9, 2)
        
        Returns:
            q_values: (batch, 9) - Q-value for each action
            concepts: (batch, concept_dim)
        """
        concepts, _ = self.extract_concepts(working_memory)
        q_values = self.value_network(concepts)
        
        return q_values, concepts
    
    def forward(self, working_memory, mode='both'):
        """
        Full forward pass with specified mode.
        
        Args:
            working_memory: (batch, 9, 2)
            mode: 'ar', 'rl', or 'both'
        
        Returns:
            Depending on mode:
            - 'ar': (next_state_logits, concepts)
            - 'rl': (q_values, concepts)
            - 'both': (next_state_logits, q_values, concepts)
        """
        concepts, rule_details = self.extract_concepts(working_memory)
        
        results = {}
        
        if mode in ['ar', 'both']:
            next_state_logits = self.ar_head(concepts)
            next_state_logits = next_state_logits.view(-1, 9, 3)
            results['next_state'] = next_state_logits
        
        if mode in ['rl', 'both']:
            q_values = self.value_network(concepts)
            results['q_values'] = q_values
        
        results['concepts'] = concepts
        results['rule_details'] = rule_details
        
        return results
    
    def choose_action(self, working_memory, epsilon=0.1, deterministic=False):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            working_memory: (1, 9, 2) - single board state
            epsilon: exploration rate
            deterministic: if True, always pick best action
        
        Returns:
            action: integer 0-8
        """
        with torch.no_grad():
            q_values, _ = self.forward_rl(working_memory)
            q_values = q_values.squeeze(0)  # (9,)
        
        # Mask invalid actions (non-empty squares)
        # working_memory[:, :, 0] is player: 0=empty
        valid_mask = (working_memory[0, :, 0].abs() < 0.1)  # Empty positions
        
        if not valid_mask.any():
            return torch.randint(0, 9, (1,)).item()
        
        # Set invalid actions to very low Q-value
        q_values[~valid_mask] = float('-inf')
        
        # Epsilon-greedy
        if not deterministic and torch.rand(1).item() < epsilon:
            valid_actions = torch.where(valid_mask)[0]
            return valid_actions[torch.randint(0, len(valid_actions), (1,))].item()
        
        return q_values.argmax().item()
    
    def freeze_logic_rules(self):
        """Freeze logic rules (for Phase 2)."""
        for param in self.logic_rules.parameters():
            param.requires_grad = False
    
    def unfreeze_logic_rules(self):
        """Unfreeze logic rules (for Phase 3)."""
        for param in self.logic_rules.parameters():
            param.requires_grad = True
    
    def get_ar_parameters(self):
        """Get parameters for AR training (logic + AR head)."""
        return list(self.logic_rules.parameters()) + list(self.ar_head.parameters())
    
    def get_rl_parameters(self):
        """Get parameters for RL training (value network only when frozen)."""
        return list(self.value_network.parameters())
    
    def get_all_parameters(self):
        """Get all parameters (for Phase 3)."""
        return self.parameters()


def board_to_working_memory(board, device='cpu'):
    """
    Convert board to working memory (propositions).
    
    Args:
        board: list/tensor of 9 values (0=empty, 1=X, -1=O)
    
    Returns:
        working_memory: (1, 9, 2) tensor [player, position]
    """
    wm = torch.zeros(1, 9, 2, device=device)
    
    if isinstance(board, list):
        board = torch.tensor(board, dtype=torch.float32)
    
    for i in range(9):
        wm[0, i, 0] = board[i]  # player
        wm[0, i, 1] = (i - 4) / 4.0  # normalized position
    
    return wm


def onehot_to_working_memory(onehot_state, device='cpu'):
    """
    Convert one-hot state to working memory.
    
    Args:
        onehot_state: (batch, 9, 3) - one-hot [empty, X, O]
    
    Returns:
        working_memory: (batch, 9, 2) - [player, position]
    """
    batch_size = onehot_state.shape[0]
    wm = torch.zeros(batch_size, 9, 2, device=device)
    
    for b in range(batch_size):
        for i in range(9):
            # Convert one-hot to player value
            if onehot_state[b, i, 0] > 0.5:  # empty
                player = 0.0
            elif onehot_state[b, i, 1] > 0.5:  # X
                player = 1.0
            else:  # O
                player = -1.0
            
            wm[b, i, 0] = player
            wm[b, i, 1] = (i - 4) / 4.0
    
    return wm


def test_architecture():
    """Test the hierarchical architecture."""
    print("Testing Hierarchical Logic Network")
    print("=" * 60)
    
    # Create model
    model = HierarchicalLogicNetwork(
        num_rules=4,
        num_premises=2,
        var_slots=3,
        value_hidden_dim=32
    )
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Logic rules parameters: {sum(p.numel() for p in model.logic_rules.parameters())}")
    print(f"AR head parameters: {sum(p.numel() for p in model.ar_head.parameters())}")
    print(f"RL head parameters: {sum(p.numel() for p in model.value_network.parameters())}")
    
    # Test with sample board
    board = [1, 1, 0, 0, -1, 0, 0, 0, 0]
    wm = board_to_working_memory(board)
    
    print(f"\nInput board: {board}")
    print(f"Working memory shape: {wm.shape}")
    
    # Test AR mode
    results = model(wm, mode='ar')
    print(f"\nAR mode:")
    print(f"  Next state logits shape: {results['next_state'].shape}")
    print(f"  Concepts shape: {results['concepts'].shape}")
    
    # Test RL mode
    results = model(wm, mode='rl')
    print(f"\nRL mode:")
    print(f"  Q-values shape: {results['q_values'].shape}")
    print(f"  Q-values: {results['q_values'].squeeze().detach().numpy()}")
    
    # Test both mode
    results = model(wm, mode='both')
    print(f"\nBoth mode:")
    print(f"  Has next_state: {'next_state' in results}")
    print(f"  Has q_values: {'q_values' in results}")
    print(f"  Has concepts: {'concepts' in results}")
    
    # Test action selection
    action = model.choose_action(wm, epsilon=0.0, deterministic=True)
    print(f"\nChosen action: {action}")
    
    # Test freezing/unfreezing
    print("\nTesting freeze/unfreeze:")
    model.freeze_logic_rules()
    print(f"  Logic rules frozen: {not next(model.logic_rules.parameters()).requires_grad}")
    model.unfreeze_logic_rules()
    print(f"  Logic rules unfrozen: {next(model.logic_rules.parameters()).requires_grad}")
    
    print("\nArchitecture test complete!")


if __name__ == "__main__":
    test_architecture()
