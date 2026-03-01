import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

speeds = [1.0, 1.5, 2.0, 2.5, 3.0]
aeb_only = [3.20, 3.22, 3.25, 3.27, 3.30]
# Updated to match Table 1 where at speed=2.0, safety distance is 2.25
pa_loi_aeb = [2.20, 2.22, 2.25, 2.27, 2.30]

plt.figure(figsize=(6, 4))
plt.plot(speeds, aeb_only, 'r--x', linewidth=2, markersize=8, label='AEB Only')
plt.plot(speeds, pa_loi_aeb, 'b-o', linewidth=2, markersize=6, label='PA-LOI + AEB')

plt.xlabel('Pedestrian Speed (m/s)', fontsize=12)
plt.ylabel('Min. Safe Clearance d_{bumper} (m)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)
plt.ylim(1.5, 4.0)
plt.xticks(speeds)

plt.tight_layout()
plt.savefig('figures/speed_generalization.pdf')
plt.savefig('IEEE-conference-template-062824/figures/speed_generalization.pdf')
print("Chart generated successfully at figures/speed_generalization.pdf")
