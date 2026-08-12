import torch
import torch.nn as nn
import torch.nn.functional as F

class LogicTransformer(nn.Module):
    """
    Logic Transformer (LT) / Differentiable Logic Network (DLN)
    
    Acts as System 1 in a Tri-System Neurosymbolic Architecture.
    This module applies fuzzy, differentiable logic rules in parallel over 
    a continuous working memory using tensor operations (unification & cylindrification).
    """
    def __init__(self, prop_length, max_props, output_dim, num_rules=8, num_premises=2, var_slots=3):
        """
        Args:
            prop_length (L): Dimensionality of each semantic proposition.
            max_props (W): Maximum number of propositions in working memory.
            output_dim (D): Output dimension for the logical conclusions.
            num_rules (M): Number of parallel learned logic rules.
            num_premises (J): Number of premises per rule.
            var_slots (I): Number of logic variables available per rule.
        """
        super().__init__()
        self.M = num_rules
        self.J = num_premises
        self.I = var_slots
        self.L = prop_length
        self.W = max_props
        
        # Rule Constants: The latent fuzzy templates each premise matches against.
        self.premise_constants = nn.Parameter(torch.randn(num_rules, num_premises, prop_length) * 0.1)
        
        # Slot Selectors (Cylindrification): Attention logits to bind working memory propositions to variable slots.
        self.slot_selectors = nn.Parameter(torch.randn(num_rules, num_premises, var_slots, max_props) * 0.1)
        
        # Consequent Projections (Rule Heads): Maps the unified variables to the final conclusion vector.
        self.rule_weights = nn.Parameter(torch.randn(num_rules, num_premises * prop_length, output_dim) / (prop_length ** 0.5))
        self.rule_bias = nn.Parameter(torch.zeros(num_rules, output_dim))
        
    def forward(self, working_memory, temperature=1.0):
        """
        Args:
            working_memory: (Batch, W, L) tensor of semantic propositions.
            temperature: Controls the sharpness of fuzzy unification.
            
        Returns:
            outputs: (Batch, output_dim) tensor representing the summed logical conclusions.
        """
        B, W_in, L_in = working_memory.shape
        assert L_in == self.L, f"Expected proposition length {self.L}, got {L_in}"
        
        # Truncate or pad slot selectors if actual W differs from max_props
        W = min(W_in, self.W)
        wm_subset = working_memory[:, :W, :]
        
        # 1. Variable Binding / Cylindrification
        # Compute soft attention over the working memory propositions for each variable slot
        # slot_attention: (M, J, I, W)
        slot_attention = F.softmax(self.slot_selectors[..., :W] / temperature, dim=-1)
        
        # 2. Extract values for variables
        # Soft-select propositions from working memory: (B, M, J, I, L)
        selected_slots = torch.einsum('bwl,mjiw->bmjil', wm_subset, slot_attention)
        
        # Average over variable slots to form the unified premise representation
        # selected_avg: (B, M, J, L)
        selected_avg = selected_slots.mean(dim=3)
        
        # 3. Fuzzy Matching (L2 distance to rule constants)
        # diff: (B, M, J, L)
        diff = selected_avg - self.premise_constants.unsqueeze(0)
        # match_scores: (B, M, J)
        match_scores = -(diff ** 2).sum(dim=-1) 
        
        # 4. Premise Attention
        # premise_attention: (B, M, J)
        premise_attention = F.softmax(match_scores / temperature, dim=-1)
        
        # 5. Modulate and project to conclusion
        # Weight the representations by how well they matched the rule conditions
        # weighted_selected: (B, M, J, L)
        weighted_selected = selected_avg * premise_attention.unsqueeze(-1)
        
        # Flatten premises to feed into the rule head: (B, M, J*L)
        rule_inputs = weighted_selected.view(B, self.M, self.J * self.L)
        
        # Rule outputs: (B, M, output_dim)
        rule_outputs = torch.einsum('bmk,mkd->bmd', rule_inputs, self.rule_weights) + self.rule_bias
        
        # 6. Aggregation
        # Sum rule outputs (OR-like aggregation) to form the final logical inference
        return rule_outputs.sum(dim=1)
