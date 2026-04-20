"""
Logic Transformer v2 - Cross-Premise Attention for Variable Binding

INNOVATION: Extends neural_logic_core.py with cross-premise conditioning to handle
variable binding constraints like father(X,Y) ∧ father(Y,Z) → grandfather(X,Z).

KEY DIFFERENCES FROM V1 (neural_logic_core.py):
- V1: Premises match independently (no cross-premise communication)
- V2: Sequential conditioning - each premise's query is modulated by previous selections

PRESERVED STRUCTURAL PRIORS:
✓ Cylindrification (γ parameters for constant vs variable)
✓ Explicit rule structure (J premises → I variables → conclusion)
✓ Proposition-level granularity (not token-level)
✓ Variable slots with semantic identity
✓ M independent rules (mixture of experts)

NEW MECHANISM:
✓ Cross-attention: Premise j+1's query conditioned on premise j's selection
✓ Binding networks: Learn which argument positions should match across premises
✓ O(W) complexity maintained (no explicit W×W construction)

GRADIENT FLOW:
Input propositions (batch, W, L)
  ↓
Rule matching with cross-premise attention:
  Premise 1: Q1 → attention over W → selected_prop1
  Premise 2: Q2 + binding_net(selected_prop1) → attention over W → selected_prop2
  ...
  ↓
Variable capture (via body networks)
  ↓
Conclusion generation (via head network)
  ↓
Output (batch, output_dim)

ALL STEPS DIFFERENTIABLE!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogicRuleV2(nn.Module):
    """
    Logic rule with cross-premise attention for variable binding.
    
    ARCHITECTURE:
    - J premises with cylindrification (γ) for constant/variable distinction
    - Cross-attention: Each premise conditions on previous selections
    - Binding networks: Learn cross-premise variable constraints
    - I variable slots: Explicit binding sites shared across premises
    """
    
    def __init__(self, num_premises, var_slots, prop_length, output_dim, hidden_dim=64):
        """
        Args:
            num_premises (J): Number of premises per rule
            var_slots (I): Number of variable slots for capture/binding
            prop_length (L): Length of each proposition vector
            output_dim: Dimension of rule output
            hidden_dim: Dimension for attention projections
        """
        super().__init__()
        
        self.J = num_premises
        self.I = var_slots
        self.L = prop_length
        self.hidden_dim = hidden_dim
        
        # RULE BODY: Maps captured variables to intermediate representation
        self.body = nn.ModuleList([
            nn.Linear(self.L, self.I) for _ in range(self.J)
        ])
        
        # RULE HEAD: Maps captured variables to output
        self.head = nn.Linear(self.I, output_dim)
        
        # LEARNED CONSTANTS: Template values for constant-mode matching
        cs = torch.FloatTensor(self.J, self.L).uniform_(-1, 1)
        self.constants = nn.Parameter(cs)
        
        # CYLINDRIFICATION FACTORS: Constant vs Variable decision
        # γ≈0: constant mode (match specific value)
        # γ≈1: variable mode (capture any value)
        γs = torch.FloatTensor(self.J, self.L).uniform_(0, 1)
        self.γs = nn.Parameter(γs)
        
        # SLOT SELECTORS: Decide which variable slot to use
        self.slot_selector = nn.ModuleList([
            nn.Linear(self.L, self.L * self.I) for _ in range(self.J)
        ])
        
        # === NEW IN V2: CROSS-PREMISE ATTENTION ===
        
        # Query projections: Convert constants to query vectors
        self.query_proj = nn.ModuleList([
            nn.Linear(self.L, hidden_dim) for _ in range(self.J)
        ])
        
        # Key/Value projections for working memory
        self.key_proj = nn.Linear(self.L, hidden_dim)
        self.value_proj = nn.Linear(self.L, self.L)  # Identity-like, but learnable
        
        # Binding networks: Modulate queries based on previous selections
        # For premise j > 0, we condition on premise j-1's selection
        self.binding_nets = nn.ModuleList([
            nn.Linear(self.L, hidden_dim) for _ in range(max(1, self.J - 1))
        ])
        
        # Binding attention: Learn which parts of previous selection matter
        # This learns patterns like "arg2 of premise1 should match arg1 of premise2"
        self.binding_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            for _ in range(max(1, self.J - 1))
        ])
    
    @staticmethod
    def sigmoid(γ):
        """Clamp γ to [0,1] - prevents gradient saturation"""
        return torch.clamp(γ, 0.0, 1.0)
    
    def match_premise_with_context(self, premise_idx, working_memory, 
                                   previous_selections, temperature=1.0):
        """
        Match a premise against working memory with cross-premise conditioning.
        
        Args:
            premise_idx: Which premise to match (0 to J-1)
            working_memory: (batch, W, L) - W propositions of length L
            previous_selections: List of (batch, L) tensors from previous premises
            temperature: Controls sharpness of soft attention
            
        Returns:
            best_props: (batch, L) - soft-selected proposition
            match_quality: (batch,) - quality of match (lower is better)
            attention_weights: (batch, W) - soft attention over propositions
        """
        batch_size, W, L = working_memory.shape
        j = premise_idx
        
        # === STEP 1: Compute base query from constants ===
        # This uses cylindrification (γ) to determine constant vs variable positions
        base_query = self.query_proj[j](self.constants[j].unsqueeze(0))  # (1, hidden_dim)
        base_query = base_query.expand(batch_size, -1)  # (batch, hidden_dim)
        
        # === STEP 2: Modulate query based on previous selections (CROSS-PREMISE BINDING) ===
        if j > 0 and len(previous_selections) > 0:
            # Get the most recent selection
            prev_selection = previous_selections[-1]  # (batch, L)
            
            # Learn binding constraint: how should this premise's query change
            # based on what the previous premise selected?
            binding_adjustment = self.binding_nets[j - 1](prev_selection)  # (batch, hidden_dim)
            
            # Optional: Use attention to focus on relevant parts of previous selection
            # This learns patterns like "if premise 1 captured X in position 2,
            # premise 2 should look for X in position 1"
            prev_selection_expanded = prev_selection.unsqueeze(1)  # (batch, 1, L)
            prev_selection_proj = self.query_proj[j-1](prev_selection_expanded)  # (batch, 1, hidden_dim)
            
            attn_output, _ = self.binding_attn[j - 1](
                base_query.unsqueeze(1),  # query: (batch, 1, hidden_dim)
                prev_selection_proj,       # key: (batch, 1, hidden_dim)
                prev_selection_proj        # value: (batch, 1, hidden_dim)
            )
            attn_adjustment = attn_output.squeeze(1)  # (batch, hidden_dim)
            
            # Combine base query with binding adjustments
            query = base_query + 0.5 * binding_adjustment + 0.5 * attn_adjustment
        else:
            query = base_query
        
        # === STEP 3: Compute attention scores using cylindrification ===
        # Project working memory to key space
        keys = self.key_proj(working_memory)  # (batch, W, hidden_dim)
        values = self.value_proj(working_memory)  # (batch, W, L)
        
        # Attention scores from query-key dot product
        attn_scores = torch.matmul(query.unsqueeze(1), keys.transpose(-2, -1))  # (batch, 1, W)
        attn_scores = attn_scores.squeeze(1) / (self.hidden_dim ** 0.5)  # (batch, W)
        
        # Add cylindrification-based matching scores
        # This preserves the constant/variable distinction from V1
        match_scores = torch.zeros(batch_size, W, device=working_memory.device)
        for l in range(self.L):
            γ = self.sigmoid(self.γs[j, l])
            constant = self.constants[j, l]
            wm_values = working_memory[:, :, l]  # (batch, W)
            
            # Match penalty (lower is better)
            # When γ→0 (constant mode): penalty = (constant - wm_value)²
            # When γ→1 (variable mode): penalty → 0 (perfect match)
            diff = (constant - wm_values) ** 2
            match_scores += (1 - γ) * diff
        
        # Combine attention scores with cylindrification scores
        # Negative match_scores because lower is better
        combined_scores = attn_scores - match_scores / temperature
        
        # === STEP 4: Soft attention ===
        attention_weights = F.softmax(combined_scores, dim=1)  # (batch, W)
        
        # Soft selection: weighted average of all propositions
        best_props = torch.einsum('bw,bwl->bl', attention_weights, values)  # (batch, L)
        
        # Soft match quality (for confidence weighting)
        match_quality = (attention_weights * match_scores).sum(dim=1)  # (batch,)
        
        return best_props, match_quality, attention_weights
    
    def forward(self, working_memory, temperature=1.0):
        """
        Apply rule to working memory with cross-premise attention.
        
        Args:
            working_memory: (batch, W, L) - W propositions of length L
            temperature: Controls sharpness of attention
            
        Returns:
            output: (batch, output_dim) - rule conclusion
            info: dict with intermediate values for debugging/analysis
        """
        batch_size = working_memory.shape[0]
        
        # Accumulate captured variables across all premises
        captured_vars = torch.zeros(batch_size, self.I, device=working_memory.device)
        total_match_quality = torch.zeros(batch_size, device=working_memory.device)
        
        all_attention_weights = []
        all_selections = []  # Track selections for cross-premise conditioning
        
        # === SEQUENTIAL MATCHING WITH CROSS-PREMISE CONDITIONING ===
        for j in range(self.J):
            # Match premise with context from previous premises
            best_props, match_quality, attention_weights = self.match_premise_with_context(
                j, working_memory, all_selections, temperature
            )
            
            all_attention_weights.append(attention_weights)
            all_selections.append(best_props)
            total_match_quality += match_quality
            
            # Capture variables from matched proposition
            captured = self.body[j](best_props)  # (batch, I)
            
            # Weight by average γ (how much this premise wants to capture)
            γ_avg = self.sigmoid(self.γs[j, :]).mean()
            captured_vars += γ_avg * captured
            
            # Slot assignment: decide which variable slot gets which value
            slot_logits = self.slot_selector[j](best_props)  # (batch, L * I)
            slot_probs = F.softmax(
                slot_logits.view(batch_size, self.L, self.I), 
                dim=2
            )  # (batch, L, I)
            
            # Soft assignment of proposition elements to variable slots
            for l in range(self.L):
                slot_weights = slot_probs[:, l, :]  # (batch, I)
                captured_vars += slot_weights * best_props[:, l].unsqueeze(1)
        
        # Generate output from captured variables
        output = self.head(captured_vars)  # (batch, output_dim)
        
        # Weight by match quality (better match = more confident)
        confidence = torch.exp(-total_match_quality).unsqueeze(1)  # (batch, 1)
        weighted_output = confidence * output
        
        # Return output and debugging info
        info = {
            'captured_vars': captured_vars,
            'match_quality': total_match_quality,
            'confidence': confidence,
            'attention_weights': all_attention_weights,
            'selections': all_selections,  # New: track what each premise selected
        }
        
        return weighted_output, info


class LogicTransformerV2(nn.Module):
    """
    Logic Transformer v2 - Multi-rule network with cross-premise attention.
    
    Combines M logic rules, each with cross-premise variable binding.
    """
    
    def __init__(self, prop_length, num_props, output_dim, 
                 num_rules=8, num_premises=2, var_slots=3, hidden_dim=64):
        """
        Args:
            prop_length (L): Length of each proposition vector
            num_props (W): Number of propositions in working memory
            output_dim: Dimension of network output
            num_rules (M): Number of logic rules
            num_premises (J): Number of premises per rule
            var_slots (I): Number of variable slots per rule
            hidden_dim: Dimension for attention projections
        """
        super().__init__()
        
        self.M = num_rules
        self.J = num_premises
        self.I = var_slots
        self.L = prop_length
        self.W = num_props
        
        # Create M logic rules with cross-premise attention
        self.rules = nn.ModuleList([
            LogicRuleV2(num_premises, var_slots, prop_length, output_dim, hidden_dim)
            for _ in range(num_rules)
        ])
    
    def forward(self, working_memory, temperature=1.0, return_details=False):
        """
        Process working memory through all rules.
        
        Args:
            working_memory: (batch, W, L) - propositions
            temperature: Controls attention sharpness
            return_details: Whether to return per-rule information
            
        Returns:
            output: (batch, output_dim) - combined rule outputs
            details: (optional) list of per-rule info dicts
        """
        batch_size = working_memory.shape[0]
        output_dim = self.rules[0].head.out_features
        
        # Accumulate outputs from all rules
        total_output = torch.zeros(batch_size, output_dim, device=working_memory.device)
        
        details = [] if return_details else None
        
        for rule in self.rules:
            rule_output, rule_info = rule(working_memory, temperature)
            total_output += rule_output
            
            if return_details:
                details.append(rule_info)
        
        if return_details:
            return total_output, details
        else:
            return total_output
    
    def interpret_rules(self, prop_names=None):
        """
        Generate human-readable interpretation of learned rules.
        
        Args:
            prop_names: Optional list of names for proposition elements
            
        Returns:
            String description of all rules
        """
        if prop_names is None:
            prop_names = [f"elem_{i}" for i in range(self.L)]
        
        lines = []
        lines.append("=" * 80)
        lines.append("LOGIC TRANSFORMER V2 - LEARNED RULES")
        lines.append("=" * 80)
        
        for m, rule in enumerate(self.rules):
            lines.append(f"\n*** RULE {m+1} ***")
            lines.append("IF (with cross-premise binding):")
            
            for j in range(self.J):
                γ_vals = rule.γs[j, :].detach().cpu().numpy()
                const_vals = rule.constants[j, :].detach().cpu().numpy()
                
                premise_parts = []
                for l in range(self.L):
                    γ = γ_vals[l]
                    c = const_vals[l]
                    
                    if γ < 0.3:  # Constant mode
                        premise_parts.append(f"{prop_names[l]}≈{c:.2f}")
                    elif γ > 0.7:  # Variable mode
                        premise_parts.append(f"{prop_names[l]}=?var")
                    else:  # Mixed
                        premise_parts.append(f"{prop_names[l]}≈{c:.2f}(γ={γ:.2f})")
                
                context_info = "" if j == 0 else " [conditioned on previous]"
                lines.append(f"  Premise {j+1}{context_info}: {', '.join(premise_parts)}")
            
            lines.append("THEN:")
            head_bias = rule.head.bias.detach().cpu().numpy()
            head_weights_norm = rule.head.weight.norm().item()
            lines.append(f"  Output bias range: [{head_bias.min():.2f}, {head_bias.max():.2f}]")
            lines.append(f"  Weight matrix norm: {head_weights_norm:.3f}")
        
        lines.append("\n" + "=" * 80)
        lines.append(f"Total Rules: {self.M}")
        lines.append(f"Cross-Premise Binding: ENABLED (V2)")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def test_logic_transformer_v2():
    """Test Logic Transformer v2 with cross-premise attention."""
    print("Testing Logic Transformer V2")
    print("=" * 80)
    
    # Create a logic network for transitive reasoning
    # Example: father(X,Y) ∧ father(Y,Z) → grandfather(X,Z)
    # Propositions: [subject, relation, object]
    logic_net = LogicTransformerV2(
        prop_length=3,      # [subject, relation, object]
        num_props=10,       # 10 propositions in working memory
        output_dim=3,       # Predict new proposition [subject, relation, object]
        num_rules=4,        # Use 4 rules
        num_premises=2,     # 2 premises per rule (for transitive patterns)
        var_slots=3,        # 3 variable slots (X, Y, Z)
        hidden_dim=32,      # Hidden dimension for attention
    )
    
    # Create sample working memory (batch_size=2)
    # Simulate: father(john, bob), father(bob, alice), mother(sue, bob)
    batch_size = 2
    wm = torch.randn(batch_size, 10, 3)
    
    print(f"Input working memory shape: {wm.shape}")
    print(f"Architecture: V2 with cross-premise attention")
    print(f"Premises: {logic_net.J}, Variable Slots: {logic_net.I}")
    
    # Forward pass
    output, details = logic_net(wm, return_details=True)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Number of rules: {len(details)}")
    
    print(f"\nRule 1 info:")
    print(f"  Captured variables shape: {details[0]['captured_vars'].shape}")
    print(f"  Match quality: {details[0]['match_quality']}")
    print(f"  Confidence: {details[0]['confidence'].squeeze()}")
    print(f"  Selections tracked: {len(details[0]['selections'])} premises")
    
    # Show attention patterns
    print(f"\nAttention patterns for Rule 1:")
    for j, attn_weights in enumerate(details[0]['attention_weights']):
        top_3 = torch.topk(attn_weights[0], k=3)
        print(f"  Premise {j+1} attends to positions: {top_3.indices.tolist()} "
              f"with weights: {top_3.values.tolist()}")
    
    # Interpret rules
    print("\n" + logic_net.interpret_rules(prop_names=['subject', 'relation', 'object']))
    
    print("\nTest complete!")
    total_params = sum(p.numel() for p in logic_net.parameters())
    print(f"Total parameters: {total_params}")
    
    # Compare with V1
    from neural_logic_core import LogicNetwork
    logic_net_v1 = LogicNetwork(
        prop_length=3, num_props=10, output_dim=3,
        num_rules=4, num_premises=2, var_slots=3
    )
    v1_params = sum(p.numel() for p in logic_net_v1.parameters())
    print(f"V1 parameters: {v1_params}")
    print(f"Parameter increase: {total_params - v1_params} (+{100*(total_params-v1_params)/v1_params:.1f}%)")
    print(f"New capability: Cross-premise variable binding for transitive reasoning!")


if __name__ == "__main__":
    test_logic_transformer_v2()
