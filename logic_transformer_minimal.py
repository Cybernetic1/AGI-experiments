"""
Logic Transformer V2 - Minimal (Up+Down ONLY, NO Binding Matrix)

HYPOTHESIS: The Up+Down variable slot mechanism may already handle cross-premise
binding implicitly by learning to combine the selected propositions correctly.

TEST: Remove the binding matrix entirely and see if it can still learn
transitive reasoning (father → grandfather).

If this works, the binding matrix may be unnecessary!

ARCHITECTURE:
- Cylindrification (γ) for matching
- Up matrix: (J×L) → I (combine selected premises)
- Down matrix: I → output
- NO binding matrix

This is the ABSOLUTE SIMPLEST possible architecture with cross-premise reasoning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogicRuleMinimal(nn.Module):
    """
    Minimal logic rule: Just cylindrification + Up+Down.
    NO binding matrix!
    """
    
    def __init__(self, num_premises, var_slots, prop_length, output_dim):
        super().__init__()
        
        self.J = num_premises
        self.I = var_slots
        self.L = prop_length
        
        # === UP+DOWN ONLY ===
        self.up = nn.Linear(self.J * self.L, self.I)
        self.down = nn.Linear(self.I, output_dim)
        
        # === CYLINDRIFICATION ===
        cs = torch.FloatTensor(self.J, self.L).uniform_(-1, 1)
        self.constants = nn.Parameter(cs)
        
        γs = torch.FloatTensor(self.J, self.L).uniform_(0, 1)
        self.γs = nn.Parameter(γs)
    
    @staticmethod
    def sigmoid(γ):
        return torch.clamp(γ, 0.0, 1.0)
    
    def match_premise(self, premise_idx, working_memory, temperature=1.0):
        """
        Match premise using only cylindrification - NO binding!
        (Same as V1)
        """
        batch_size, W, L = working_memory.shape
        j = premise_idx
        
        # Cylindrification matching
        match_scores = torch.zeros(batch_size, W, device=working_memory.device)
        
        for l in range(self.L):
            γ = self.sigmoid(self.γs[j, l])
            constant = self.constants[j, l]
            wm_values = working_memory[:, :, l]
            
            diff = (constant - wm_values) ** 2
            match_scores += (1 - γ) * diff
        
        # Soft attention
        attention_weights = F.softmax(-match_scores / temperature, dim=1)
        best_props = torch.einsum('bw,bwl->bl', attention_weights, working_memory)
        match_quality = (attention_weights * match_scores).sum(dim=1)
        
        return best_props, match_quality, attention_weights
    
    def forward(self, working_memory, temperature=1.0):
        """
        Apply rule: match premises independently (NO binding),
        then use Up+Down to combine.
        """
        batch_size = working_memory.shape[0]
        
        total_match_quality = torch.zeros(batch_size, device=working_memory.device)
        all_attention_weights = []
        all_selections = []
        
        # Match all premises INDEPENDENTLY (like V1)
        for j in range(self.J):
            best_props, match_quality, attention_weights = self.match_premise(
                j, working_memory, temperature
            )
            
            all_attention_weights.append(attention_weights)
            all_selections.append(best_props)
            total_match_quality += match_quality
        
        # Concatenate and transform via Up+Down
        all_selected = torch.cat(all_selections, dim=-1)
        captured_vars = self.up(all_selected)
        output = self.down(captured_vars)
        
        confidence = torch.exp(-total_match_quality).unsqueeze(1)
        weighted_output = confidence * output
        
        info = {
            'captured_vars': captured_vars,
            'match_quality': total_match_quality,
            'confidence': confidence,
            'attention_weights': all_attention_weights,
            'selections': all_selections,
        }
        
        return weighted_output, info


class LogicTransformerMinimal(nn.Module):
    """Minimal Logic Transformer: Cylindrification + Up+Down only."""
    
    def __init__(self, prop_length, num_props, output_dim, 
                 num_rules=8, num_premises=2, var_slots=3):
        super().__init__()
        
        self.M = num_rules
        self.J = num_premises
        self.I = var_slots
        self.L = prop_length
        self.W = num_props
        
        self.rules = nn.ModuleList([
            LogicRuleMinimal(num_premises, var_slots, prop_length, output_dim)
            for _ in range(num_rules)
        ])
    
    def forward(self, working_memory, temperature=1.0, return_details=False):
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
        if prop_names is None:
            prop_names = [f"elem_{i}" for i in range(self.L)]
        
        lines = []
        lines.append("=" * 80)
        lines.append("LOGIC TRANSFORMER MINIMAL - NO BINDING MATRIX")
        lines.append("=" * 80)
        
        for m, rule in enumerate(self.rules):
            lines.append(f"\n*** RULE {m+1} ***")
            lines.append("IF (independent premise matching):")
            
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
                
                lines.append(f"  Premise {j+1}: {', '.join(premise_parts)}")
            
            lines.append("\nTHEN:")
            lines.append(f"  Up: ({self.J}×{self.L}) → {self.I} slots")
            lines.append(f"  Down: {self.I} → {rule.down.out_features} output")
        
        lines.append("\n" + "=" * 80)
        lines.append(f"Total Rules: {self.M}")
        lines.append("Architecture: MINIMAL (No binding matrix!)")
        lines.append("=" * 80)
        
        return "\n".join(lines)


# Quick test on transitive reasoning
if __name__ == "__main__":
    print("Testing MINIMAL Logic Transformer (No Binding Matrix)")
    print("=" * 80)
    
    # Create model
    model = LogicTransformerMinimal(
        prop_length=3,
        num_props=10,
        output_dim=3,
        num_rules=4,
        num_premises=2,
        var_slots=3,
    )
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params}")
    print(f"Per rule: {params // 4}")
    
    # Compare with previous versions
    print("\nParameter Comparison:")
    print(f"  V1:             480 params")
    print(f"  V2 Simplified:  220 params (with binding matrix)")
    print(f"  V2 Minimal:     {params} params (NO binding matrix)")
    print(f"  Savings:        {220 - params} params")
    
    # Test forward pass
    wm = torch.randn(2, 10, 3)
    output = model(wm)
    print(f"\nForward pass successful: output shape {output.shape}")
    
    print("\n" + "=" * 80)
    print("Ready to test on father→grandfather task!")
    print("If this works, the binding matrix may be unnecessary...")
    print("=" * 80)
