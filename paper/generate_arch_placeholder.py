import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(6, 3))
ax.text(0.5, 0.5, 'Architecture Diagram Placeholder\n(Please replace with your actual diagram)', 
        fontsize=14, ha='center', va='center', color='gray')
ax.axis('off')

plt.tight_layout()
plt.savefig('figures/architecture.pdf')
print("Architecture placeholder generated successfully at figures/architecture.pdf")
