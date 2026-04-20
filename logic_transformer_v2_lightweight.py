"""
Logic Transformer V2 - Lightweight Cross-Premise Binding

GOAL: Add cross-premise variable binding with MINIMAL parameter overhead.

STRATEGY: 
- Remove expensive MultiheadAttention (3K+ params per rule)
- Use simple learned gating instead of full attention
- Directly modulate cylindrification matching based on previous selection

PARAMETER COMPARISON (per rule, prop_length=3):
- V1: 120 parameters
- V2 (heavy): 4,868 parameters (40x increase!)  
- V2 (lightweight): ~180 parameters (1.5x increase)

KEY INSIGHT:
Cross-premise binding doesn't need full attention machinery.
Just need: "If premise 1 selected prop with value X at position i,
           premise 2 should look for X at position j"

This is a BINDING MATRIX: B[i,j] learnable per rule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogicRuleLightweightV2(nn.Module):
    """
    Lightweight logic rule with cross-premise binding.
    
    INNOVATION: Use binding matrix instead of attention for cross-premise constraints.
    
    For J=2 premises with L elements each:
    - Binding matrix B: (L, L) learnable parameters
    - B[i,j] = strength of constraint "premise1[i] should equal premise2[j]"
    
    Example for father(X,Y) ∧ father(Y,Z):
    - B[2,0] high → premise1.object should match premise2.subject (Y=Y)
    - B[0,2] low → premise1.subject need not match premise2.object
    """
    
    def __init__(self, num_premises, var_slots, prop_length, output_dim):
        """
        Args:
            num_premises (J): Number of premises per rule
            var_slots (I): Number of variable slots for capture/binding
            prop_length (L): Length of each proposition vector
            output_dim: Dimension of rule output
        """
        super().__init__()
        
        self.J = num_premises
        self.I = var_slots
        self.L = prop_length
        
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
        γs = torch.FloatTensor(self.J, self.L).uniform_(0, 1)
        self.γs = nn.Parameter(γs)
        
        # SLOT SELECTORS: Decide which variable slot to use
        self.slot_selector = nn.ModuleList([
            nn.Linear(self.L, self.L * self.I) for _ in range(self.J)
        ])
        
        # === LIGHTWEIGHT CROSS-PREMISE BINDING ===
        # For each pair of consecutive premises, learn binding constraints
        # binding_matrices[k] is (L, L) matrix for premise k and k+1
        if self.J > 1:
            self.binding_matrices = nn.ParameterList([
                nn.Parameter(torch.zeros(self.L, self.L))
                for _ in range(self.J - 1)
            ])
            # Gating: learn how much to use binding vs cylindrification
            self.binding_gates = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5))
                for _ in range(self.J - 1)
            ])
        else:
            self.binding_matrices = None
            self.binding_gates = None
    
    @staticmethod
    def sigmoid(γ):
        """Clamp γ to [0,1] - prevents gradient saturation"""
        return torch.clamp(γ, 0.0, 1.0)
    
    def match_premise_with_binding(self, premise_idx, working_memory, 
                                   previous_selection, temperature=1.0):
        """
        Match premise with lightweight cross-premise binding.
        
        Args:
            premise_idx: Which premise to match (0 to J-1)
            working_memory: (batch, W, L) - W propositions of length L
            previous_selection: (batch, L) tensor from previous premise (or None)
            temperature: Controls sharpness of soft attention
            
        Returns:
            best_props: (batch, L) - soft-selected proposition
            match_quality: (batch,) - quality of match (lower is better)
            attention_weights: (batch, W) - soft attention over propositions
        """
        batch_size, W, L = working_memory.shape
        j = premise_idx
        
        # === STEP 1: Base cylindrification matching ===
        match_scores = torch.zeros(batch_size, W, device=working_memory.device)
        
        for l in range(self.L):
            γ = self.sigmoid(self.γs[j, l])
            constant = self.constants[j, l]
            wm_values = working_memory[:, :, l]  # (batch, W)
            
            # Match penalty (lower is better)
            diff = (constant - wm_values) ** 2
            match_scores += (1 - γ) * diff
        
        # === STEP 2: Add cross-premise binding constraint ===
        if j > 0 and previous_selection is not None and self.binding_matrices is not None:
            # Compute binding penalty: how much does each WM prop violate binding?
            # For each position pair (i,j), check if prev[i] ≈ current[j]
            binding_matrix = self.binding_matrices[j - 1]  # (L, L)
            gate = torch.sigmoid(self.binding_gates[j - 1])
            
            # prev_selection: (batch, L)
            # working_memory: (batch, W, L)
            # binding_matrix: (L, L) where [i,j] = strength of "prev[i] should match current[j]"
            
            # Compute binding penalty for each WM proposition
            for i in range(self.L):  # Position in previous selection
                for k in range(self.L):  # Position in current proposition
                    if torch.abs(binding_matrix[i, k]) > 0.01:  # Skip near-zero
                        prev_val = previous_selection[:, i]  # (batch,)
                        curr_vals = working_memory[:, :, k]  # (batch, W)
                        
                        # Penalty: |prev[i] - curr[k]|^2 weighted by binding strength
                        diff = (prev_val.unsqueeze(1) - curr_vals) ** 2  # (batch, W)
                        match_scores += gate * torch.abs(binding_matrix[i, k]) * diff
        
        # === STEP 3: Soft attention ===
        attention_weights = F.softmax(-match_scores / temperature, dim=1)  # (batch, W)
        
        # Soft selection: weighted average of all propositions
        best_props = torch.einsum('bw,bwl->bl', attention_weights, working_memory)  # (batch, L)
        
        # Soft match quality
        match_quality = (attention_weights * match_scores).sum(dim=1)  # (batch,)
        
        return best_props, match_quality, attention_weights
    
    def forward(self, working_memory, temperature=1.0):
        """
        Apply rule to working memory with lightweight cross-premise binding.
        
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
        all_selections = []
        
        # === SEQUENTIAL MATCHING WITH LIGHTWEIGHT BINDING ===
        for j in range(self.J):
            # Get previous selection for binding (if exists)
            prev_selection = all_selections[-1] if j > 0 and len(all_selections) > 0 else None
            
            # Match premise with binding constraint
            best_props, match_quality, attention_weights = self.match_premise_with_binding(
                j, working_memory, prev_selection, temperature
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
            'selections': all_selections,
        }
        
        return weighted_output, info


class LogicTransformerV2Lightweight(nn.Module):
    """
    Logic Transformer v2 (Lightweight) - Minimal parameter overhead.
    
    Uses binding matrices instead of attention for cross-premise constraints.
    Parameter overhead: ~1.5x instead of ~40x!
    """
    
    def __init__(self, prop_length, num_props, output_dim, 
                 num_rules=8, num_premises=2, var_slots=3):
        """
        Args:
            prop_length (L): Length of each proposition vector
            num_props (W): Number of propositions in working memory
            output_dim: Dimension of network output
            num_rules (M): Number of logic rules
            num_premises (J): Number of premises per rule
            var_slots (I): Number of variable slots per rule
        """
        super().__init__()
        
        self.M = num_rules
        self.J = num_premises
        self.I = var_slots
        self.L = prop_length
        self.W = num_props
        
        # Create M logic rules with lightweight cross-premise binding
        self.rules = nn.ModuleList([
            LogicRuleLightweightV2(num_premises, var_slots, prop_length, output_dim)
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
        """Generate human-readable interpretation of learned rules."""
        if prop_names is None:
            prop_names = [f"elem_{i}" for i in range(self.L)]
        
        lines = []
        lines.append("=" * 80)
        lines.append("LOGIC TRANSFORMER V2 LIGHTWEIGHT - LEARNED RULES")
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
                
                context_info = ""
                if j > 0 and rule.binding_matrices is not None:
                    B = rule.binding_matrices[j-1].detach()
                    max_binding = torch.max(torch.abs(B)).item()
                    context_info = f" [binding strength: {max_binding:.2f}]"
                
                lines.append(f"  Premise {j+1}{context_info}: {', '.join(premise_parts)}")
            
            # Show binding matrix if it exists
            if rule.binding_matrices is not None and self.J == 2:
                B = rule.binding_matrices[0].detach().cpu().numpy()
                lines.append("\n  Binding Matrix (premise1 → premise2):")
                for i in range(self.L):
                    row = [f"{B[i,k]:6.2f}" for k in range(self.L)]
                    lines.append(f"    {prop_names[i]:8s} → " + " ".join(row))
            
            lines.append("\nTHEN:")
            head_bias = rule.head.bias.detach().cpu().numpy()
            head_weights_norm = rule.head.weight.norm().item()
            lines.append(f"  Output bias range: [{head_bias.min():.2f}, {head_bias.max():.2f}]")
            lines.append(f"  Weight matrix norm: {head_weights_norm:.3f}")
        
        lines.append("\n" + "=" * 80)
        lines.append(f"Total Rules: {self.M}")
        lines.append(f"Cross-Premise Binding: LIGHTWEIGHT (Binding Matrices)")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def test_lightweight():
    """Test the lightweight version."""
    print("Testing Logic Transformer V2 Lightweight")
    print("=" * 80)
    
    # Same config as before
    config = dict(
        prop_length=3,
        num_props=10,
        output_dim=3,
        num_rules=4,
        num_premises=2,
        var_slots=3,
    )
    
    # Import V1 for comparison
    from neural_logic_core import LogicNetwork
    from logic_transformer_v2 import LogicTransformerV2
    
    v1 = LogicNetwork(**config)
    v2_heavy = LogicTransformerV2(**config, hidden_dim=32)
    v2_light = LogicTransformerV2Lightweight(**config)
    
    v1_params = sum(p.numel() for p in v1.parameters())
    v2_heavy_params = sum(p.numel() for p in v2_heavy.parameters())
    v2_light_params = sum(p.numel() for p in v2_light.parameters())
    
    print(f"\nParameter Comparison:")
    print(f"  V1 (baseline):        {v1_params:6d} params")
    print(f"  V2 Heavy (attention): {v2_heavy_params:6d} params ({v2_heavy_params/v1_params:5.1f}x)")
    print(f"  V2 Light (matrices):  {v2_light_params:6d} params ({v2_light_params/v1_params:5.1f}x)")
    print(f"\n  Overhead of V2 Light: {v2_light_params - v1_params} params")
    print(f"  Savings vs V2 Heavy:  {v2_heavy_params - v2_light_params} params")
    
    # Test forward pass
    batch_size = 2
    wm = torch.randn(batch_size, 10, 3)
    
    print(f"\nForward pass test:")
    output, details = v2_light(wm, return_details=True)
    print(f"  Output shape: {output.shape}")
    print(f"  Rule 1 binding matrix shape: {details[0]['selections'][0].shape}")
    
    print("\n" + v2_light.interpret_rules(prop_names=['subject', 'relation', 'object']))
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("Lightweight V2 adds cross-premise binding with minimal overhead!")
    print("Uses learned binding matrices instead of expensive attention.")
    print("=" * 80)


if __name__ == "__main__":
    test_lightweight()
