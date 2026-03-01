import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

configs = ['Vanilla MIND', 'AEB Only', 'PA-LOI+AEB', 'Conservative WCA']
impact_vel = [4.55, 2.88, 0.0, 0.0]  # Impact velocity at 4.3m trigger distance
delay = [0.0, 0.0, 0.9, 4.5]        # Delay in false-positive scenario

fig, ax1 = plt.subplots(figsize=(7, 4.5))

color1 = 'tab:red'
ax1.set_ylabel('Impact Velocity $v_{imp}$ (m/s) [Lower is Safer]', color=color1, fontsize=11, weight='bold')
bars1 = ax1.bar([x - 0.2 for x in range(len(configs))], impact_vel, width=0.4, color=color1, alpha=0.8, label='Safety Risk (Impact Vel.)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 5)

ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('False-Positive Delay (s) [Lower is more Efficient]', color=color2, fontsize=11, weight='bold')
bars2 = ax2.bar([x + 0.2 for x in range(len(configs))], delay, width=0.4, color=color2, alpha=0.8, label='Efficiency Cost (Delay)')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 5)

ax1.set_xticks(range(len(configs)))
ax1.set_xticklabels(configs, fontsize=11, weight='bold')

plt.title('Safety-Efficiency Trade-off vs. Baselines', fontsize=13, weight='bold')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

fig.tight_layout()
plt.subplots_adjust(bottom=0.2)
plt.savefig('figures/pareto_tradeoff.pdf')
print("Trade-off plot generated at figures/pareto_tradeoff.pdf")
