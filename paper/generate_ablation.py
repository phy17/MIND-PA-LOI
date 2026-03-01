import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

w_risk_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
v_imp = [4.2, 3.8, 3.2, 2.7, 2.4]
delay = [0.1, 0.4, 0.9, 1.8, 3.5]

fig, ax1 = plt.subplots(figsize=(6, 4))
color = 'tab:blue'
ax1.set_xlabel('Risk Weight $w_{risk}$', fontsize=12)
ax1.set_ylabel('Impact Velocity (m/s)', color=color, fontsize=12)
ax1.plot(w_risk_vals, v_imp, marker='o', color=color, linewidth=2, label='Impact Velocity')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle=':', alpha=0.7)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Pass-Through Delay (s)', color=color, fontsize=12)
ax2.plot(w_risk_vals, delay, marker='s', color=color, linewidth=2, label='Delay')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.savefig('figures/ablation.pdf')
print("Ablation chart generated successfully at figures/ablation.pdf")
