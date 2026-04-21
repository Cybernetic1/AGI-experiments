"""
Logic Transformer V2 - Simplified (Up+Down Model)

SIMPLIFICATION: Remove redundant variable slot machinery.

ORIGINAL V2 Lightweight had THREE components:
  1. body[j]: J separate (L → I) networks
  2. slot_selector[j]: J separate (L → L×I) routing networks  
  3. head: (I → output)

Both body and slot_selector were trying to map propositions to variable slots!

NEW SIMPLIFIED MODEL has only TWO components:
  1. Up: Single (J×L → I) matrix - combines all selected premises
  2. Down: Single (I → output) matrix - maps variables to conclusion

This matches the clean conceptual model:
  - Binding matrix ensures cross-premise consistency (during matching)
  - Up+Down provides learned transformation (after matching)

PARAMETER SAVINGS:
  - Original: 108 params per rule (for J=2, L=3, I=3, output=3)
  - Simplified: 33 params per rule (69% reduction!)
  - For 4 rules: 132 vs 432 params (300 params saved)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogicRuleSimplified(nn.Module):
    """
    Simplified logic rule with Up+Down variable slot model.
    
    ARCHITECTURE:
    - J premises with cylindrification (γ) and binding matrix
    - Up matrix: (J×L) → I (combine all selected premises)
    - Down matrix: I → output (generate conclusion)
    
    Much simpler than original which had redundant body + slot_selector.
    """
    
    def __init__(self, num_premises, var_slots, prop_length, output_dim):
        """
        Args:
            num_premises (J): Number of premises per rule
            var_slots (I): Number of variable slots
            prop_length (L): Length of each proposition vector
            output_dim: Dimension of rule output
        """
        super().__init__()
        
        self.J = num_premises
        self.I = var_slots
        self.L = prop_length
        
        # === UP MATRIX: Combine all selected premises into variable slots ===
        # Input: concatenation of all J selected propositions (J×L values)
        # Output: I variable slots
        self.up = nn.Linear(self.J * self.L, self.I)
        
        # === DOWN MATRIX: Map variable slots to conclusion ===
        self.down = nn.Linear(self.I, output_dim)
        
        # === LEARNED CONSTANTS: Template values for matching ===
        cs = torch.FloatTensor(self.J, self.L).uniform_(-1, 1)
        self.constants = nn.Parameter(cs)
        
        # === CYLINDRIFICATION FACTORS: Constant vs Variable ===
        γs = torch.FloatTensor(self.J, self.L).uniform_(0, 1)
        self.γs = nn.Parameter(γs)
        
        # === BINDING MATRICES: Cross-premise constraints ===
        if self.J > 1:
            self.binding_matrices = nn.ParameterList([
                nn.Parameter(torch.zeros(self.L, self.L))
                for _ in range(self.J - 1)
            ])
            self.binding_gates = nn.ParameterList([
                nn.Parameter(torch.tensor(0.5))
                for _ in range(self.J - 1)
            ])
        else:
            self.binding_matrices = None
            self.binding_gates = None
    
    @staticmethod
    def sigmoid(γ):
        """Clamp γ to [0,1]"""
        return torch.clamp(γ, 0.0, 1.0)
    
    def match_premise_with_binding(self, premise_idx, working_memory, 
                                   previous_selection, temperature=1.0):
        """
        Match a premise against working memory with binding constraints.
        
        Same as V2 Lightweight - this part is unchanged.
        """
        batch_size, W, L = working_memory.shape
        j = premise_idx
        
        # === STEP 1: Base cylindrification matching ===
        match_scores = torch.zeros(batch_size, W, device=working_memory.device)
        
        for l in range(self.L):
            γ = self.sigmoid(self.γs[j, l])
            constant = self.constants[j, l]
            wm_values = working_memory[:, :, l]  # (batch, W)
            
            diff = (constant - wm_values) ** 2
            match_scores += (1 - γ) * diff
        
        # === STEP 2: Add binding constraint ===
        if j > 0 and previous_selection is not None and self.binding_matrices is not None:
            binding_matrix = self.binding_matrices[j - 1]  # (L, L)
            gate = torch.sigmoid(self.binding_gates[j - 1])
            
            for i in range(self.L):
                for k in range(self.L):
                    if torch.abs(binding_matrix[i, k]) > 0.01:
                        prev_val = previous_selection[:, i]  # (batch,)
                        curr_vals = working_memory[:, :, k]  # (batch, W)
                        
                        diff = (prev_val.unsqueeze(1) - curr_vals) ** 2
                        match_scores += gate * torch.abs(binding_matrix[i, k]) * diff
        
        # === STEP 3: Soft attention ===
        attention_weights = F.softmax(-match_scores / temperature, dim=1)
        best_props = torch.einsum('bw,bwl->bl', attention_weights, working_memory)
        match_quality = (attention_weights * match_scores).sum(dim=1)
        
        return best_props, match_quality, attention_weights
    
    def forward(self, working_memory, temperature=1.0):
        """
        Apply rule to working memory.
        
        SIMPLIFIED FLOW:
        1. Match each premise (with binding) → get J selected propositions
        2. Concatenate all J×L values
        3. Up matrix: (J×L) → I variable slots
        4. Down matrix: I → output
        """
        batch_size = working_memory.shape[0]
        
        total_match_quality = torch.zeros(batch_size, device=working_memory.device)
        all_attention_weights = []
        all_selections = []
        
        # === STEP 1: Match all premises ===
        for j in range(self.J):
            prev_selection = all_selections[-1] if j > 0 and len(all_selections) > 0 else None
            
            best_props, match_quality, attention_weights = self.match_premise_with_binding(
                j, working_memory, prev_selection, temperature
            )
            
            all_attention_weights.append(attention_weights)
            all_selections.append(best_props)
            total_match_quality += match_quality
        
        # === STEP 2: Concatenate all selected propositions ===
        # all_selections is a list of J tensors, each (batch, L)
        # Concatenate to (batch, J×L)
        all_selected = torch.cat(all_selections, dim=-1)  # (batch, J×L)
        
        # === STEP 3: Up matrix - combine into variable slots ===
        captured_vars = self.up(all_selected)  # (batch, I)
        
        # === STEP 4: Down matrix - generate output ===
        output = self.down(captured_vars)  # (batch, output_dim)
        
        # Weight by match quality (confidence)
        confidence = torch.exp(-total_match_quality).unsqueeze(1)
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


class LogicTransformerV2Simplified(nn.Module):
    """
    Logic Transformer V2 - Simplified with Up+Down model.
    
    Much fewer parameters than original V2 Lightweight.
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
        
        # Create M logic rules with simplified Up+Down architecture
        self.rules = nn.ModuleList([
            LogicRuleSimplified(num_premises, var_slots, prop_length, output_dim)
            for _ in range(num_rules)
        ])
    
    def forward(self, working_memory, temperature=1.0, return_details=False):
        """Process working memory through all rules."""
        batch_size = working_memory.shape[0]
        output_dim = self.rules[0].down.out_features
        
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
        lines.append("LOGIC TRANSFORMER V2 SIMPLIFIED - LEARNED RULES")
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
                    
                    if γ < 0.3:
                        premise_parts.append(f"{prop_names[l]}≈{c:.2f}")
                    elif γ > 0.7:
                        premise_parts.append(f"{prop_names[l]}=?var")
                    else:
                        premise_parts.append(f"{prop_names[l]}≈{c:.2f}(γ={γ:.2f})")
                
                context_info = ""
                if j > 0 and rule.binding_matrices is not None:
                    B = rule.binding_matrices[j-1].detach()
                    max_binding = torch.max(torch.abs(B)).item()
                    context_info = f" [binding: {max_binding:.2f}]"
                
                lines.append(f"  Premise {j+1}{context_info}: {', '.join(premise_parts)}")
            
            # Show binding matrix
            if rule.binding_matrices is not None and self.J == 2:
                B = rule.binding_matrices[0].detach().cpu().numpy()
                lines.append("\n  Binding Matrix (premise1 → premise2):")
                for i in range(self.L):
                    row = [f"{B[i,k]:6.2f}" for k in range(self.L)]
                    lines.append(f"    {prop_names[i]:8s} → " + " ".join(row))
            
            lines.append("\nTHEN:")
            lines.append(f"  Up matrix: ({self.J}×{self.L}={self.J*self.L}) → {self.I} slots")
            lines.append(f"  Down matrix: {self.I} slots → {rule.down.out_features} output")
            
            up_norm = rule.up.weight.norm().item()
            down_norm = rule.down.weight.norm().item()
            lines.append(f"  Up weight norm: {up_norm:.3f}")
            lines.append(f"  Down weight norm: {down_norm:.3f}")
        
        lines.append("\n" + "=" * 80)
        lines.append(f"Total Rules: {self.M}")
        lines.append(f"Architecture: SIMPLIFIED (Up+Down model)")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def test_simplified():
    """Test the simplified version."""
    print("Testing Logic Transformer V2 Simplified")
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
    
    # Import for comparison
    from neural_logic_core import LogicNetwork
    from logic_transformer_v2_lightweight import LogicTransformerV2Lightweight
    
    v1 = LogicNetwork(**config)
    v2_light = LogicTransformerV2Lightweight(**config)
    v2_simplified = LogicTransformerV2Simplified(**config)
    
    v1_params = sum(p.numel() for p in v1.parameters())
    v2_light_params = sum(p.numel() for p in v2_light.parameters())
    v2_simplified_params = sum(p.numel() for p in v2_simplified.parameters())
    
    print(f"\nParameter Comparison:")
    print(f"  V1 (baseline):             {v1_params:6d} params")
    print(f"  V2 Lightweight:            {v2_light_params:6d} params ({v2_light_params/v1_params:5.1f}x)")
    print(f"  V2 Simplified (Up+Down):   {v2_simplified_params:6d} params ({v2_simplified_params/v1_params:5.1f}x)")
    print(f"\n  Savings vs V2 Lightweight: {v2_light_params - v2_simplified_params} params")
    print(f"  Reduction: {100*(v2_light_params - v2_simplified_params)/v2_light_params:.1f}%")
    
    # Test forward pass
    batch_size = 2
    wm = torch.randn(batch_size, 10, 3)
    
    print(f"\nForward pass test:")
    output, details = v2_simplified(wm, return_details=True)
    print(f"  Output shape: {output.shape}")
    print(f"  Captured vars shape: {details[0]['captured_vars'].shape}")
    
    print("\n" + v2_simplified.interpret_rules(prop_names=['subject', 'relation', 'object']))
    
    print("\n" + "=" * 80)
    print("ARCHITECTURE COMPARISON:")
    print("=" * 80)
    print("\nV2 Lightweight (per rule):")
    print("  - body networks: 2 × Linear(3→3) = 24 params")
    print("  - slot_selector: 2 × Linear(3→9) = 72 params")
    print("  - head: Linear(3→3) = 12 params")
    print("  - binding: 9 + 1 = 10 params")
    print("  - Total: 118 params/rule")
    
    print("\nV2 Simplified (per rule):")
    print("  - Up: Linear(6→3) = 21 params")
    print("  - Down: Linear(3→3) = 12 params")
    print("  - binding: 9 + 1 = 10 params")
    print("  - Total: 43 params/rule")
    
    print("\nSavings: 75 params/rule (64% reduction!)")
    print("=" * 80)


if __name__ == "__main__":
    test_simplified()
