"""
Neural Logic Network - Core implementation extracted from DQN_logic_with_vars.py

CORE INNOVATION: Logic rules with learnable parameters using fuzzy unification.

KEY CONCEPTS:
- Rules are parameterized "if-then" statements with fuzzy matching
- Variables in rules get "unified" with propositions through learned similarity
- Truth values propagate from premises to conclusions via differentiable logic
- Each rule learns to recognize patterns through gradient descent

ARCHITECTURE:
- Propositions: Working Memory contains propositions (e.g., board state)
- Rules: M learnable rules of the form "premise → conclusion"
- Matching: Fuzzy pattern matching with learned constants
- Unification: Variable capture through soft attention
- Cylindrification: γ ∈ [0,1] determines constant vs variable mode

GRADIENT FLOW:
Input propositions
  ↓ for each rule:
Match against rule.constants (adjusted by rule.γs)
  ↓ softmax (differentiable!)
Soft attention weights
  ↓ einsum
Selected propositions
  ↓ rule.body (Linear layer)
Captured variables
  ↓ rule.head (Linear layer)
Output (e.g., next state predictions)

ALL STEPS ARE DIFFERENTIABLE - gradients flow from loss back to all rule parameters!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogicRule(nn.Module):
    """
    Single learnable logic rule with fuzzy unification.
    
    STRUCTURE:
    - J premises: Patterns to match against working memory
    - I variable slots: For capturing/binding values
    - Constants: Template values for constant-mode matching
    - γs (gammas): Cylindrification factors (0=constant, 1=variable)
    - Body: Maps matched propositions to captured variables
    - Head: Maps captured variables to conclusions
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
        # For each premise, we have a linear layer: L → I
        self.body = nn.ModuleList([
            nn.Linear(self.L, self.I) for _ in range(self.J)
        ])
        
        # RULE HEAD: Maps captured variables to output
        self.head = nn.Linear(self.I, output_dim)
        
        # LEARNED CONSTANTS: Template values for constant-mode matching
        # Initialized uniformly in [-1,1]
        cs = torch.FloatTensor(self.J, self.L).uniform_(-1, 1)
        self.constants = nn.Parameter(cs)
        
        # CYLINDRIFICATION FACTORS: Constant vs Variable decision
        # γ[j][l] ∈ [0,1] controls whether position (j,l) acts as constant or variable
        # γ≈0: constant mode (match specific value)
        # γ≈1: variable mode (capture any value)
        γs = torch.FloatTensor(self.J, self.L).uniform_(0, 1)
        self.γs = nn.Parameter(γs)
        
        # SLOT SELECTORS: Decide which variable slot to use
        self.slot_selector = nn.ModuleList([
            nn.Linear(self.L, self.L * self.I) for _ in range(self.J)
        ])
    
    @staticmethod
    def sigmoid(γ):
        """Clamp γ to [0,1] - prevents gradient saturation"""
        return torch.clamp(γ, 0.0, 1.0)
    
    def match_premise(self, premise_idx, working_memory, temperature=1.0):
        """
        Match a single premise against working memory using fuzzy unification.
        
        Args:
            premise_idx: Which premise to match (0 to J-1)
            working_memory: (batch, W, L) - W propositions of length L
            temperature: Controls sharpness of soft attention
            
        Returns:
            best_props: (batch, L) - soft-selected proposition
            match_quality: (batch,) - quality of match (lower is better)
            attention_weights: (batch, W) - soft attention over propositions
        """
        batch_size, W, L = working_memory.shape
        j = premise_idx
        
        # Compute match scores for all propositions in working memory
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
        
        # DIFFERENTIABLE soft attention instead of argmin
        # This allows gradients to flow!
        attention_weights = F.softmax(-match_scores / temperature, dim=1)  # (batch, W)
        
        # Soft selection: weighted average of all propositions
        best_props = torch.einsum('bw,bwl->bl', attention_weights, working_memory)  # (batch, L)
        
        # Soft match quality
        match_quality = (attention_weights * match_scores).sum(dim=1)  # (batch,)
        
        return best_props, match_quality, attention_weights
    
    def forward(self, working_memory, temperature=1.0):
        """
        Apply rule to working memory.
        
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
        
        for j in range(self.J):
            # Match premise against working memory
            best_props, match_quality, attention_weights = self.match_premise(
                j, working_memory, temperature
            )
            
            all_attention_weights.append(attention_weights)
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
        }
        
        return weighted_output, info


class LogicNetwork(nn.Module):
    """
    Multi-rule logic network combining multiple learnable rules.
    
    Each rule independently processes the input and produces an output.
    Outputs are combined (typically summed) to produce final result.
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
        
        # Create M logic rules
        self.rules = nn.ModuleList([
            LogicRule(num_premises, var_slots, prop_length, output_dim)
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
        lines.append("LEARNED LOGIC RULES")
        lines.append("=" * 80)
        
        for m, rule in enumerate(self.rules):
            lines.append(f"\n*** RULE {m+1} ***")
            lines.append("IF:")
            
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
                
                lines.append(f"  Premise {j+1}: {', '.join(premise_parts)}")
            
            lines.append("THEN:")
            head_bias = rule.head.bias.detach().cpu().numpy()
            head_weights_norm = rule.head.weight.norm().item()
            lines.append(f"  Output bias range: [{head_bias.min():.2f}, {head_bias.max():.2f}]")
            lines.append(f"  Weight matrix norm: {head_weights_norm:.3f}")
        
        lines.append("\n" + "=" * 80)
        lines.append(f"Total Rules: {self.M}")
        lines.append("=" * 80)
        
        return "\n".join(lines)


def test_logic_network():
    """Test the logic network with simple examples."""
    print("Testing Neural Logic Network")
    print("=" * 60)
    
    # Create a simple logic network
    # Propositions: 9 board positions, each with 2 elements [player, position]
    logic_net = LogicNetwork(
        prop_length=2,      # [player, position]
        num_props=9,        # 9 board positions
        output_dim=9,       # Predict for each position
        num_rules=4,        # Use 4 rules
        num_premises=2,     # 2 premises per rule
        var_slots=3,        # 3 variable slots
    )
    
    # Create sample working memory (batch_size=2)
    # Representing two board states
    batch_size = 2
    wm = torch.randn(batch_size, 9, 2)
    
    print(f"Input working memory shape: {wm.shape}")
    
    # Forward pass
    output, details = logic_net(wm, return_details=True)
    
    print(f"Output shape: {output.shape}")
    print(f"Number of rules: {len(details)}")
    print(f"\nRule 1 info:")
    print(f"  Captured variables shape: {details[0]['captured_vars'].shape}")
    print(f"  Match quality: {details[0]['match_quality']}")
    print(f"  Confidence: {details[0]['confidence'].squeeze()}")
    
    # Interpret rules
    print("\n" + logic_net.interpret_rules(prop_names=['player', 'position']))
    
    print("\nTest complete!")
    print(f"Total parameters: {sum(p.numel() for p in logic_net.parameters())}")


if __name__ == "__main__":
    test_logic_network()
