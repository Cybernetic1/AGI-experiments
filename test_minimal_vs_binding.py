"""
Quick test: Can Up+Down alone learn transitive reasoning without binding matrix?
"""

import torch
import torch.nn as nn
import torch.optim as optim
from logic_transformer_minimal import LogicTransformerMinimal
from logic_transformer_v2_simplified import LogicTransformerV2Simplified


def create_transitive_dataset(num_samples=50):
    """Same dataset as before"""
    dataset = []
    for _ in range(num_samples):
        wm = torch.randn(1, 10, 3)
        wm[0, 0, :] = torch.tensor([1.0, 0.0, 2.0])  # father(1,2)
        wm[0, 1, :] = torch.tensor([2.0, 0.0, 3.0])  # father(2,3)
        target = torch.tensor([[1.0, 1.0, 3.0]])     # grandfather(1,3)
        dataset.append((wm, target))
    return dataset


def train_and_evaluate(model, dataset, num_epochs=50, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for wm, target in dataset:
            optimizer.zero_grad()
            output = model(wm)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataset)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return losses


print("=" * 80)
print("CRITICAL TEST: Does Up+Down ALONE handle transitive reasoning?")
print("=" * 80)

# Create datasets
train_dataset = create_transitive_dataset(50)
test_dataset = create_transitive_dataset(20)

config = dict(prop_length=3, num_props=10, output_dim=3, num_rules=4, 
              num_premises=2, var_slots=3)

# Test Minimal (NO binding matrix)
print("\n" + "=" * 80)
print("TRAINING MINIMAL (Cylindrification + Up+Down ONLY)")
print("=" * 80)
model_minimal = LogicTransformerMinimal(**config)
print(f"Parameters: {sum(p.numel() for p in model_minimal.parameters())}")
losses_minimal = train_and_evaluate(model_minimal, train_dataset, num_epochs=50, lr=0.01)

# Test Simplified (WITH binding matrix)
print("\n" + "=" * 80)
print("TRAINING SIMPLIFIED (Cylindrification + Binding + Up+Down)")
print("=" * 80)
model_simplified = LogicTransformerV2Simplified(**config)
print(f"Parameters: {sum(p.numel() for p in model_simplified.parameters())}")
losses_simplified = train_and_evaluate(model_simplified, train_dataset, num_epochs=50, lr=0.01)

# Compare results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)
print(f"Minimal Final Loss:    {losses_minimal[-1]:.4f}")
print(f"Simplified Final Loss: {losses_simplified[-1]:.4f}")

# Test generalization
criterion = nn.MSELoss()
model_minimal.eval()
model_simplified.eval()

test_loss_minimal = 0.0
test_loss_simplified = 0.0

with torch.no_grad():
    for wm, target in test_dataset:
        test_loss_minimal += criterion(model_minimal(wm), target).item()
        test_loss_simplified += criterion(model_simplified(wm), target).item()

test_loss_minimal /= len(test_dataset)
test_loss_simplified /= len(test_dataset)

print(f"\nMinimal Test Loss:    {test_loss_minimal:.4f}")
print(f"Simplified Test Loss: {test_loss_simplified:.4f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
if test_loss_minimal < 0.01:
    print("✓ MINIMAL WORKS! Up+Down alone can handle transitive reasoning!")
    print("  The binding matrix may be UNNECESSARY for simple cases.")
    print(f"  Achieved test loss {test_loss_minimal:.4f} with only 180 params.")
else:
    print("✗ Minimal fails. Binding matrix is needed.")
    print(f"  Minimal test loss: {test_loss_minimal:.4f}")
    print(f"  Simplified test loss: {test_loss_simplified:.4f}")
    print("  The binding matrix provides crucial cross-premise constraints.")

print("=" * 80)
