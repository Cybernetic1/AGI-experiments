#!/usr/bin/env python3
"""
Debug Real DLN: Why isn't it learning on TinyStories task?

Compare with TTT setup to understand differences.
"""

import torch
import torch.nn as nn
from neural_logic_core import LogicNetwork, LogicRule
from logic_core import Proposition
import json
from pathlib import Path
import random


def inspect_dln_architecture():
    """Check basic DLN functionality."""
    
    print("="*70)
    print("DLN ARCHITECTURE INSPECTION")
    print("="*70)
    
    # Create a small DLN
    prop_length = 48  # 16 * 3 (embed_dim * 3 for [pred, arg1, arg2])
    num_props = 4     # 3 premises + 1 conclusion
    output_dim = 1
    num_rules = 2
    
    dln = LogicNetwork(
        prop_length=prop_length,
        num_props=num_props,
        output_dim=output_dim,
        num_rules=num_rules,
        num_premises=3,
        var_slots=4
    )
    
    print(f"\nDLN Configuration:")
    print(f"  Proposition length: {prop_length}")
    print(f"  Number of propositions: {num_props}")
    print(f"  Number of rules: {num_rules}")
    print(f"  Premises per rule: 3")
    print(f"  Variable slots: 4")
    
    # Count parameters
    total_params = sum(p.numel() for p in dln.parameters())
    print(f"  Total parameters: {total_params:,}")
    
    # Check parameter breakdown per rule
    rule = dln.rules[0]
    print(f"\nPer-rule parameters:")
    print(f"  Constants: {rule.constants.numel()}")
    print(f"  Gammas: {rule.γs.numel()}")
    for i, body_layer in enumerate(rule.body):
        print(f"  Body layer {i}: {sum(p.numel() for p in body_layer.parameters())}")
    print(f"  Head: {sum(p.numel() for p in rule.head.parameters())}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    working_memory = torch.randn(batch_size, num_props, prop_length)
    
    output = dln(working_memory)
    print(f"  Input shape: {working_memory.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Check gradient flow
    print("\nTesting gradient flow...")
    loss = output.mean()
    loss.backward()
    
    has_grad = sum(1 for p in dln.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total_params_count = sum(1 for p in dln.parameters())
    print(f"  Parameters with gradient: {has_grad}/{total_params_count}")
    
    if has_grad < total_params_count:
        print("  ⚠ WARNING: Some parameters not receiving gradients!")
    else:
        print("  ✓ All parameters receiving gradients")
    
    return dln


def check_learning_on_simple_task():
    """Test if DLN can learn a trivial task."""
    
    print("\n" + "="*70)
    print("SIMPLE LEARNING TEST")
    print("="*70)
    
    print("\nTask: Learn to output 1.0 when first proposition has high values")
    
    prop_length = 16
    num_props = 3
    
    dln = LogicNetwork(
        prop_length=prop_length,
        num_props=num_props,
        output_dim=1,
        num_rules=2,
        num_premises=2,
        var_slots=2
    )
    
    optimizer = torch.optim.Adam(dln.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print("\nTraining for 50 iterations...")
    
    losses = []
    for iteration in range(50):
        # Generate simple data
        batch_size = 32
        working_memory = torch.randn(batch_size, num_props, prop_length)
        
        # Simple rule: if first prop has mean > 0, output 1, else output 0
        target = (working_memory[:, 0, :].mean(dim=1) > 0).float().unsqueeze(1)
        
        optimizer.zero_grad()
        output = dln(working_memory)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if iteration % 10 == 0:
            print(f"  Iteration {iteration:2d}: Loss = {loss.item():.4f}")
    
    print(f"\nFinal loss: {losses[-1]:.4f}")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Improvement: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
    
    if losses[-1] < losses[0] * 0.5:
        print("✓ DLN CAN learn simple patterns")
    else:
        print("✗ DLN struggling to learn even simple patterns")
        print("  Possible issues:")
        print("  - Learning rate too low")
        print("  - Architecture mismatch")
        print("  - Initialization problems")


def compare_with_ttt_setup():
    """Compare our setup with working TTT setup."""
    
    print("\n" + "="*70)
    print("TTT vs TinyStories SETUP COMPARISON")
    print("="*70)
    
    print("\n" + "-"*70)
    print("TTT (WORKING):")
    print("-"*70)
    print("  Task: Predict next board state")
    print("  Input: Board state (9 positions × 3 states = 27 dim)")
    print("  Output: Next board state (27 dim)")
    print("  Training: AR (predict next state) + RL (win/loss reward)")
    print("  Data: Game trajectories")
    print("  DLN config: 6 rules, 2 premises, 3 var_slots")
    print("  Total params: 4,680")
    print("  Result: 100% win rate")
    
    print("\n" + "-"*70)
    print("TinyStories Logical Inference (NOT WORKING):")
    print("-"*70)
    print("  Task: Binary classification (fact true/false)")
    print("  Input: 3 premises + 1 conclusion (4 props × 48 dim)")
    print("  Output: Probability (1 dim)")
    print("  Training: Binary cross-entropy on labels")
    print("  Data: Random fact triples")
    print("  DLN config: 2+ rules, 3 premises, 4 var_slots")
    print("  Total params: 59K+ (10× larger than TTT!)")
    print("  Result: Stuck at 50% (random guessing)")
    
    print("\n" + "-"*70)
    print("KEY DIFFERENCES:")
    print("-"*70)
    print("  1. Task complexity:")
    print("     TTT: Sequential, grounded in physical board")
    print("     TinyStories: Abstract logical inference")
    
    print("\n  2. Input representation:")
    print("     TTT: Direct board encoding (hand-crafted features)")
    print("     TinyStories: Learned embeddings → might be poor")
    
    print("\n  3. Training signal:")
    print("     TTT: Strong (win/loss + AR reconstruction)")
    print("     TinyStories: Weak (single binary label)")
    
    print("\n  4. Data quality:")
    print("     TTT: Clean game trajectories")
    print("     TinyStories: Noisy fact triples + synthetic negatives")
    
    print("\n  5. Parameter scale:")
    print("     TTT: 4.7K params (appropriate for task)")
    print("     TinyStories: 59K+ params (might be over-parameterized)")


def diagnose_learning_failure():
    """Specific diagnostics for why learning fails."""
    
    print("\n" + "="*70)
    print("LEARNING FAILURE DIAGNOSTICS")
    print("="*70)
    
    print("\nPossible causes:")
    print("\n1. POOR EMBEDDINGS")
    print("   - TinyStories uses learned embeddings (16-dim)")
    print("   - Embeddings might not capture semantic meaning")
    print("   - DLN tries to learn logic over meaningless vectors")
    print("   → Test: Use pre-trained embeddings or one-hot (but smaller vocab)")
    
    print("\n2. WRONG TASK FORMULATION")
    print("   - Binary classification too simple for DLN")
    print("   - DLN designed for sequential/structured reasoning")
    print("   - No temporal or causal structure in our data")
    print("   → Test: Use autoregressive task like TTT")
    
    print("\n3. INSUFFICIENT TRAINING SIGNAL")
    print("   - Each example only provides 1 bit of information (true/false)")
    print("   - Synthetic negatives might be too random")
    print("   - No reward shaping or intermediate supervision")
    print("   → Test: Add auxiliary tasks or multi-task learning")
    
    print("\n4. ARCHITECTURE MISMATCH")
    print("   - Proposition format might not match what DLN expects")
    print("   - 3 premises + 1 conclusion might not provide enough context")
    print("   - Variable binding might not be needed for this task")
    print("   → Test: Vary num_premises, var_slots")
    
    print("\n5. HYPERPARAMETER ISSUES")
    print("   - Learning rate (0.001 might be wrong)")
    print("   - Batch size (1 example at a time vs batches)")
    print("   - Loss function (BCE might not work well)")
    print("   → Test: Grid search hyperparameters")


def propose_fixes():
    """Concrete next steps to fix the issue."""
    
    print("\n" + "="*70)
    print("PROPOSED FIXES (Priority Order)")
    print("="*70)
    
    print("\n1. REPLICATE TTT SUCCESS (HIGHEST PRIORITY)")
    print("   Action: Adapt TTT training methodology to TinyStories")
    print("   How:")
    print("   - Use AR task: Given story facts 1..N-1, predict fact N")
    print("   - Add RL: Reward model for generating coherent stories")
    print("   - Use same 3-phase training: AR → RL → Joint")
    print("   Time: 2-3 hours to implement and test")
    
    print("\n2. FIX REPRESENTATION (MEDIUM PRIORITY)")
    print("   Action: Improve input representation")
    print("   How:")
    print("   - Use pre-trained word embeddings (GloVe/Word2Vec)")
    print("   - Or use larger embed_dim (32 → 64)")
    print("   - Or use graph structure encoding")
    print("   Time: 1-2 hours")
    
    print("\n3. SIMPLIFY TASK (QUICK WIN)")
    print("   Action: Start with easier logical inference")
    print("   How:")
    print("   - Single-hop inference only")
    print("   - Deterministic rules (no ambiguity)")
    print("   - More structured data (bAbI style)")
    print("   Time: 1 hour")
    
    print("\n4. DEBUG MODE (DETAILED ANALYSIS)")
    print("   Action: Instrument DLN to see what it's learning")
    print("   How:")
    print("   - Log γ (cylindrification) values over time")
    print("   - Visualize rule constants")
    print("   - Check which rules fire on which examples")
    print("   - Monitor gradient magnitudes")
    print("   Time: 2-3 hours")
    
    print("\n5. HYPERPARAMETER SWEEP (SYSTEMATIC)")
    print("   Action: Grid search key hyperparameters")
    print("   Parameters:")
    print("   - Learning rate: [0.0001, 0.001, 0.01]")
    print("   - Num rules: [2, 4, 8]")
    print("   - Var slots: [2, 4, 8]")
    print("   - Embed dim: [8, 16, 32]")
    print("   Time: 4-6 hours (can parallelize)")


def main():
    print("\n" + "="*70)
    print("DLN DEBUGGING SESSION")
    print("Goal: Understand why real DLN works on TTT but not TinyStories")
    print("="*70)
    
    # 1. Basic architecture check
    inspect_dln_architecture()
    
    # 2. Simple learning test
    check_learning_on_simple_task()
    
    # 3. Compare setups
    compare_with_ttt_setup()
    
    # 4. Diagnose specific issues
    diagnose_learning_failure()
    
    # 5. Propose solutions
    propose_fixes()
    
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("\nFor your mid-February presentation:")
    print("\n1. SAFE OPTION: Present TTT result")
    print("   - Real DLN (4.7K params)")
    print("   - 100% win rate vs optimal opponent")
    print("   - Clear success story")
    print("   - Limitation: Only one task demonstrated")
    
    print("\n2. AMBITIOUS OPTION: Fix TinyStories comparison")
    print("   - Requires 1-2 weeks of debugging/experimentation")
    print("   - Higher risk but stronger claim")
    print("   - Shows generalization across tasks")
    
    print("\n3. MIDDLE GROUND: Multiple simple tasks")
    print("   - Get DLN working on 2-3 simple tasks")
    print("   - Show consistent parameter efficiency")
    print("   - More convincing than single task")
    
    print("\nYou have ~2 weeks until mid-Feb. Recommended path:")
    print("  Week 1: Fix #1 (replicate TTT methodology on new task)")
    print("  Week 2: If successful, run parameter sweep for presentation")
    print("  Backup: Always have TTT result as fallback")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
