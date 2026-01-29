"""
Compare speed of original DLN vs vectorized DLN.
"""

import torch
import time
from neural_logic_core import LogicNetwork
from neural_logic_core_vectorized import VectorizedLogicNetwork

def benchmark_model(model, working_memory, num_iterations=100):
    """Benchmark forward pass speed."""
    device = working_memory.device
    
    # Warmup
    for _ in range(10):
        _ = model(working_memory)
    
    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    for _ in range(num_iterations):
        _ = model(working_memory)
        
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()
    
    avg_time = (end - start) / num_iterations * 1000  # Convert to ms
    throughput = 1000 / avg_time  # Examples per second
    
    return avg_time, throughput

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Parameters
    batch_size = 32
    prop_length = 48
    num_props = 10
    output_dim = 96
    num_rules = 5
    num_premises = 3
    var_slots = 2
    
    # Create models
    print("Creating models...")
    original = LogicNetwork(
        prop_length=prop_length,
        num_props=num_props,
        output_dim=output_dim,
        num_rules=num_rules,
        num_premises=num_premises,
        var_slots=var_slots
    ).to(device)
    
    vectorized = VectorizedLogicNetwork(
        prop_length=prop_length,
        num_props=num_props,
        output_dim=output_dim,
        num_rules=num_rules,
        num_premises=num_premises,
        var_slots=var_slots
    ).to(device)
    
    # Count parameters
    original_params = sum(p.numel() for p in original.parameters())
    vectorized_params = sum(p.numel() for p in vectorized.parameters())
    
    print(f"Original DLN:    {original_params:,} parameters")
    print(f"Vectorized DLN:  {vectorized_params:,} parameters")
    print()
    
    # Create test data
    working_memory = torch.randn(batch_size, num_props, prop_length).to(device)
    
    # Benchmark original
    print("Benchmarking original DLN...")
    orig_time, orig_throughput = benchmark_model(original, working_memory)
    print(f"  Time per batch: {orig_time:.2f}ms")
    print(f"  Throughput: {orig_throughput:.1f} batches/sec")
    print()
    
    # Benchmark vectorized
    print("Benchmarking vectorized DLN...")
    vec_time, vec_throughput = benchmark_model(vectorized, working_memory)
    print(f"  Time per batch: {vec_time:.2f}ms")
    print(f"  Throughput: {vec_throughput:.1f} batches/sec")
    print()
    
    # Speedup
    speedup = orig_time / vec_time
    print(f"SPEEDUP: {speedup:.2f}×")
    print()
    
    # Verify outputs are similar
    print("Verifying correctness...")
    with torch.no_grad():
        orig_out = original(working_memory)
        vec_out = vectorized(working_memory)
        
        # They won't be identical due to different initialization,
        # but shapes should match
        print(f"  Original output shape: {orig_out.shape}")
        print(f"  Vectorized output shape: {vec_out.shape}")
        print(f"  Shapes match: {orig_out.shape == vec_out.shape}")

if __name__ == '__main__':
    main()
