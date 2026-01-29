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
        
        # Cylindrification matrices: [M, J, I, L, L] - one per premise
        self.cylindrification = nn.Parameter(
            torch.randn(num_rules, num_premises, var_slots, prop_length, prop_length) * 0.01
        )
        
        # Rule heads: [M, J*L, output_dim] - one MLP per rule
        self.rule_heads = nn.Parameter(torch.randn(num_rules, num_premises * prop_length, output_dim))
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
        
        # Step 1: Cylindrify working memory for all rules and premises
        # wm: (B, W, L) → (B, 1, 1, W, L)
        wm_expanded = working_memory.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, W, L)
        
        # cyl: (M, J, I, L, L)
        # For each premise, apply cylindrification matrices to working memory
        # Result: (B, M, J, W, I, L)
        wm_cylindrified = torch.einsum('bwl,mjikl->bmjwik', 
                                       working_memory, 
                                       self.cylindrification)
        
        # Step 2: Compute distances to premise constants
        # premises: (M, J, L) → (1, M, J, 1, 1, L)
        premises = self.premise_constants.view(1, M, J, 1, 1, L)
        
        # Broadcast and compute squared distances: (B, M, J, W, I)
        diff = wm_cylindrified - premises
        distances = (diff ** 2).sum(dim=-1)  # Sum over L dimension
        
        # Step 3: Softmax over working memory and variables
        # Flatten W and I: (B, M, J, W*I)
        distances_flat = distances.view(B, M, J, W * I)
        attention = F.softmax(-distances_flat / temperature, dim=-1)  # (B, M, J, W*I)
        
        # Step 4: Select working memory using attention
        # Reshape: (B, M, J, W, I)
        attention_reshaped = attention.view(B, M, J, W, I)
        
        # Average over variable slots: (B, M, J, W)
        attention_avg = attention_reshaped.mean(dim=-1)
        
        # Apply to working memory: (B, M, J, L)
        selected_wm = torch.einsum('bmjw,bwl->bmjl', attention_avg, working_memory)
        
        # Step 5: Flatten premises and apply rule heads
        # selected: (B, M, J*L)
        selected_flat = selected_wm.view(B, M, J * L)
        
        # heads: (M, J*L, output_dim) → (1, M, J*L, output_dim)
        heads = self.rule_heads.unsqueeze(0)
        
        # Apply: (B, M, output_dim)
        rule_outputs = torch.einsum('bmk,mkd->bmd', selected_flat, heads) + self.rule_bias
        
        # Step 6: Sum over all rules: (B, output_dim)
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
