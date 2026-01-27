#!/usr/bin/env python3
"""Generate performance comparison graph for presentation"""
import matplotlib.pyplot as plt
import numpy as np

# Data from test results
algorithms = ['Frequency', 'FOIL', 'Confidence']
small_scale = [0.112, 0.061, 0.043]  # 10 stories - eval MSE
large_scale = [0.029, 0.043, 0.112]  # 500 stories - eval MSE

x = np.arange(len(algorithms))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, small_scale, width, label='10 Stories', color='#6495ED', alpha=0.8)
bars2 = ax.bar(x + width/2, large_scale, width, label='500 Stories', color='#FF6B6B', alpha=0.8)

ax.set_ylabel('Evaluation MSE (lower is better)', fontsize=12)
ax.set_xlabel('Algorithm', fontsize=12)
ax.set_title('Rule Learning Algorithm Performance at Different Scales', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(algorithms)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Annotate bars with values
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('docs/performance_comparison.png', dpi=150, bbox_inches='tight')
print("Graph saved to docs/performance_comparison.png")
