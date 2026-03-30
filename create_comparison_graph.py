import matplotlib.pyplot as plt
import numpy as np

# Transformer results (from transformer-babi-results.txt)
transformer_data = [
    {"layers": 2, "params": 70814, "accuracy": 8.9, "time": 80},
    {"layers": 3, "params": 405150, "accuracy": 18.2, "time": 104},
    {"layers": 4, "params": 2123806, "accuracy": 0.0, "time": 124},
]

# DLN results (from sweep)
dln_data = [
    {"rules": 3, "params": 44526, "accuracy": 25.1, "time": 60.5},
    {"rules": 5, "params": 72894, "accuracy": 25.0, "time": 59.5},
    {"rules": 7, "params": 101262, "accuracy": 25.7, "time": 61.7},
    {"rules": 10, "params": 143814, "accuracy": 25.4, "time": 60.4},
    {"rules": 15, "params": 214734, "accuracy": 25.6, "time": 59.8},
    {"rules": 20, "params": 285654, "accuracy": 24.0, "time": 61.0},
]

# Calculate effective compression ratios
dln_best = {"params": 44526, "accuracy": 25.1}
transformer_comparable = {"params": 405150, "accuracy": 18.2}
compression_ratio = transformer_comparable["params"] / dln_best["params"]

# Create Plot 1: Parameters vs Accuracy
fig1, ax1 = plt.subplots(figsize=(10, 6))
transformer_params = [d["params"] for d in transformer_data]
transformer_acc = [d["accuracy"] for d in transformer_data]
dln_params = [d["params"] for d in dln_data]
dln_acc = [d["accuracy"] for d in dln_data]

ax1.plot(transformer_params, transformer_acc, 'o-', linewidth=2, markersize=8, 
         label='Transformer', color='#E74C3C')
ax1.plot(dln_params, dln_acc, 's-', linewidth=2, markersize=8, 
         label='DLN (Vectorized)', color='#3498DB')

ax1.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_ylim(0, 50)  # Set y-axis to 50%
ax1.set_title('Parameter Efficiency: DLN vs Transformer)', # \n(using bAbI tests)', 
              fontsize=13, fontweight='bold')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)

# Annotate key points
ax1.annotate('DLN: 25% accuracy with as few as 44K params', 
             xy=(72894, 25.0), xytext=(50000, 28),
             # arrowprops=dict(arrowstyle='->', color='#3498DB', lw=1.5),
             fontsize=10, color='#3498DB', fontweight='bold')
ax1.annotate('Transformer: 18% accuracy with 405K params', 
             xy=(405150, 18.2), xytext=(200000, 19.8),
             # arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5),
             fontsize=10, color='#E74C3C', fontweight='bold')

plt.tight_layout()
plt.savefig('docs/parameter_efficiency_comparison.png', dpi=300, bbox_inches='tight')
print("Graph 1 saved to docs/parameter_efficiency_comparison.png")

# Create Plot 2: Compression Ratio Comparison (Two panels side by side)
fig2, (ax_params, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

width = 0.4  # Increased from 0.1 to make bars wider

# LEFT PANEL: Parameters comparison
x_pos_params = [0.5, 1.5]  # Centered positions
bars_params = ax_params.bar(x_pos_params, 
                            [transformer_comparable["params"], dln_best["params"]], 
                            width,
                            color=['#E74C3C', '#3498DB'], alpha=0.3)
ax_params.set_ylabel('Parameters', fontsize=12, fontweight='bold')
ax_params.set_title('Model Size\n(number of parameters)', fontsize=11, fontweight='bold')
ax_params.set_xticks(x_pos_params)
ax_params.set_xticklabels(['Transformer', 'DLN'], fontsize=10)
ax_params.set_xlim(0, 2.2)  # Center the bars
ax_params.grid(axis='y', alpha=0.3, linestyle='--')
ax_params.set_ylim(0, transformer_comparable["params"] * 1.2)

# Add value labels on parameter bars
for i, bar in enumerate(bars_params):
    height = bar.get_height()
    ax_params.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

# RIGHT PANEL: Accuracy comparison
x_pos_acc = [0.5, 1.5]  # Centered positions
bars_acc = ax_acc.bar(x_pos_acc, 
                      [transformer_comparable["accuracy"], dln_best["accuracy"]], 
                      width,
                      color=['#E74C3C', '#3498DB'], alpha=0.3)
ax_acc.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax_acc.set_title('Test Accuracy\n(%)', fontsize=11, fontweight='bold')
ax_acc.set_xticks(x_pos_acc)
ax_acc.set_xticklabels(['Transformer', 'DLN'], fontsize=10)
ax_acc.set_xlim(0, 2.1)  # Center the bars
ax_acc.grid(axis='y', alpha=0.3, linestyle='--')
ax_acc.set_ylim(0, 50)

# Add value labels on accuracy bars
for bar in bars_acc:
    height = bar.get_height()
    ax_acc.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

# Overall title
fig2.suptitle(f'Preliminary tests show DLN can achieve similar accuracy with up to {compression_ratio:1.0f}× fewer parameters', 
              fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('docs/compression_ratio_comparison.png', dpi=300, bbox_inches='tight')
print("Graph 2 saved to docs/compression_ratio_comparison.png")
print(f"\nKEY FINDING: DLN achieves {compression_ratio:1.0f}× compression with better accuracy!")
