"""Profile DLN performance to identify bottlenecks"""
import torch
import time
import sys
from neural_logic_core import LogicNetwork

def profile_forward_pass():
    """Profile a single forward pass through DLN"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create a small DLN
    model = LogicNetwork(
        prop_length=48,
        num_props=10,
        output_dim=128,
        num_rules=5,
        num_premises=3,
        var_slots=2
    ).to(device)
    
    # Create sample input
    batch_size = 32
    num_props = 10
    prop_length = 48
    
    inputs = torch.randn(batch_size, num_props, prop_length).to(device)
    
    # Warm up
    for _ in range(3):
        _ = model(inputs)
    
    # Profile forward pass
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    num_iterations = 100
    for _ in range(num_iterations):
        output = model(inputs)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    avg_time = (end - start) / num_iterations
    print(f"\nForward pass timing:")
    print(f"  Average time: {avg_time*1000:.2f}ms")
    print(f"  Throughput: {batch_size/avg_time:.1f} examples/sec")
    
    return avg_time

def profile_components():
    """Profile individual DLN components"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LogicNetwork(
        prop_length=48,
        num_props=10,
        output_dim=128,
        num_rules=5,
        num_premises=3,
        var_slots=2
    ).to(device)
    
    batch_size = 32
    inputs = torch.randn(batch_size, 10, 48).to(device)
    
    # Time each component
    print("\nComponent timing:")
    
    # 1. Match premise (all premises for each rule)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(100):
        for rule in model.rules:
            for j in range(rule.J):  # For each premise
                best_props, match_quality, attention = rule.match_premise(j, inputs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"  Premise matching: {(time.time()-start)*10:.2f}ms per iteration")
    
    # 2. Apply rule head
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(100):
        for rule in model.rules:
            test_vars = torch.randn(batch_size, rule.I).to(device)
            output = rule.head(test_vars)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"  Rule heads: {(time.time()-start)*10:.2f}ms per iteration")
    
    # 3. Full forward
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(100):
        output = model(inputs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    print(f"  Full forward: {(time.time()-start)*10:.2f}ms per iteration")

def estimate_training_time():
    """Estimate full training time"""
    
    avg_forward_time = profile_forward_pass()
    
    # Estimate backward pass (typically 2-3x forward)
    avg_backward_time = avg_forward_time * 2.5
    
    # Estimate per-batch time
    per_batch_time = avg_forward_time + avg_backward_time
    
    # For bAbI task
    num_train_examples = 2000
    batch_size = 32
    epochs = 20
    
    batches_per_epoch = num_train_examples // batch_size
    total_batches = batches_per_epoch * epochs
    
    estimated_time = total_batches * per_batch_time
    
    print(f"\nTraining time estimate:")
    print(f"  Forward pass: {avg_forward_time*1000:.2f}ms")
    print(f"  Backward pass (est): {avg_backward_time*1000:.2f}ms")
    print(f"  Per batch: {per_batch_time*1000:.2f}ms")
    print(f"  Batches per epoch: {batches_per_epoch}")
    print(f"  Total batches: {total_batches}")
    print(f"  Estimated training time: {estimated_time:.1f}s ({estimated_time/60:.1f} min)")

if __name__ == "__main__":
    print("="*60)
    print("DLN Performance Profiling")
    print("="*60)
    
    profile_forward_pass()
    profile_components()
    estimate_training_time()
    
    print("\n" + "="*60)
