"""
Vectorized Logic Network - eliminates Python loops for GPU efficiency.

Key optimization: Process all rules and premises in parallel using batched tensor operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorizedLogicNetwork(nn.Module):
    """
    Fully vectorized logic network - no Python loops over rules or premises.
    
    All rules and premises are processed in parallel using batched operations.
    Expected speedup: 5-10× over sequential version.
    """
    
    def __init__(self, prop_length, num_props, output_dim, 
                 num_rules=8, num_premises=3, var_slots=2):
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
        self.output_dim = output_dim
        
        # Premise constants: [M, J, L] - all premises for all rules
        self.premise_constants = nn.Parameter(torch.randn(num_rules, num_premises, prop_length))
        
        # Slot selectors: [M, J, I, W] - attention over working memory slots
        # Much smaller than full cylindrification!
        self.slot_selectors = nn.Parameter(
            torch.randn(num_rules, num_premises, var_slots, num_props) * 0.1
        )
        
        # Rule heads: [M, J*L, output_dim] - batched weights for all rules
        self.rule_weights = nn.Parameter(torch.randn(num_rules, num_premises * prop_length, output_dim))
        self.rule_bias = nn.Parameter(torch.zeros(num_rules, output_dim))
        
        # Initialize premise constants uniformly
        nn.init.uniform_(self.premise_constants, -0.5, 0.5)
    
    def forward(self, working_memory, temperature=1.0):
        """
        Vectorized forward pass - no Python loops!
        
        Args:
            working_memory: (B, W, L) - batch of working memories
            temperature: Controls attention sharpness
            
        Returns:
            output: (B, output_dim) - combined outputs from all rules
        """
        B, W, L = working_memory.shape
        M, J, I = self.M, self.J, self.I
        
        # Step 1: Compute attention using slot selectors
        # selectors: (M, J, I, W) → attention weights over working memory
        # Apply softmax: (M, J, I, W)
        slot_attention = F.softmax(self.slot_selectors, dim=-1)
        
        # Step 2: Select from working memory using slot attention
        # wm: (B, W, L), attention: (M, J, I, W)
        # Result: (B, M, J, I, L) - selected content for each slot
        selected_slots = torch.einsum('bwl,mjiw->bmjil', working_memory, slot_attention)
        
        # Step 3: Average over variable slots: (B, M, J, L)
        selected_avg = selected_slots.mean(dim=3)
        
        # Step 4: Compute match scores with premise constants
        # premises: (M, J, L), selected: (B, M, J, L)
        # Distance: (B, M, J)
        diff = selected_avg - self.premise_constants.unsqueeze(0)
        match_scores = -(diff ** 2).sum(dim=-1)  # (B, M, J) - negative distance
        
        # Step 5: Softmax over premises: (B, M, J)
        premise_attention = F.softmax(match_scores / temperature, dim=-1)
        
        # Step 6: Weight selected content by premise attention
        # (B, M, J, L) weighted by (B, M, J)
        weighted_selected = selected_avg * premise_attention.unsqueeze(-1)  # (B, M, J, L)
        
        
        # Step 7: Flatten premises and apply rule heads
        # weighted: (B, M, J, L) → (B, M, J*L)
        rule_inputs = weighted_selected.view(B, M, J * L)
        
        # Apply rule heads: (M, J*L, output_dim)
        rule_outputs = torch.einsum('bmk,mkd->bmd', rule_inputs, self.rule_weights) + self.rule_bias
        
        # Step 8: Sum over all rules: (B, output_dim)
        total_output = rule_outputs.sum(dim=1)
        
        return total_output
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class VectorizedDLNWrapper(nn.Module):
    """
    Wrapper to use vectorized DLN for bAbI QA task.
    Matches the interface of the original DLN.
    """
    
    def __init__(self, vocab_size, embed_dim=48, num_rules=5, num_premises=3, var_slots=2):
        super().__init__()
        
        self.prop_length = embed_dim
        self.num_props = 10  # Fixed working memory size
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Vectorized DLN core
        self.dln = VectorizedLogicNetwork(
            prop_length=embed_dim,
            num_props=self.num_props,
            output_dim=embed_dim * 2,
            num_rules=num_rules,
            num_premises=num_premises,
            var_slots=var_slots
        )
        
        # Output head
        self.output_head = nn.Linear(embed_dim * 2, vocab_size)
        
    def forward(self, facts_idx, question_idx):
        """
        Args:
            facts_idx: (B, max_facts) - token indices
            question_idx: (B, max_question) - token indices
            
        Returns:
            logits: (B, vocab_size) - answer predictions
        """
        # Embed facts
        facts_emb = self.embedding(facts_idx)  # (B, max_facts, embed_dim)
        
        # Embed question
        question_emb = self.embedding(question_idx).mean(dim=1, keepdim=True)  # (B, 1, embed_dim)
        
        # Concatenate and pad to num_props
        combined = torch.cat([facts_emb, question_emb], dim=1)  # (B, max_facts+1, embed_dim)
        
        B = combined.shape[0]
        if combined.shape[1] < self.num_props:
            # Pad with zeros
            padding = torch.zeros(B, self.num_props - combined.shape[1], self.prop_length,
                                device=combined.device)
            working_memory = torch.cat([combined, padding], dim=1)
        else:
            # Truncate
            working_memory = combined[:, :self.num_props, :]
        
        # Run DLN
        dln_output = self.dln(working_memory)
        
        # Predict answer
        logits = self.output_head(dln_output)
        
        return logits
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
