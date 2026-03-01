import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# IEEE Academic Font Formatting
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
})

def compute_sigmoid(clearance, ghost_lateral, k_steep=2.0):
    exp_arg = k_steep * (clearance - ghost_lateral)
    return 1.0 / (1.0 + np.exp(exp_arg))

x = np.linspace(0, 50, 500)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

ghost_x = 30.0
ghost_y = -1.8  
ghost_lateral = 1.5
ego_half_width = 1.0
w_base = 25.0
v_safe = 2.5
v_ego = 8.0

Z = np.zeros_like(X)
excess_vel = max(0.0, v_ego - v_safe)
kinetic_energy = excess_vel ** 2

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pos_x, pos_y = X[i, j], Y[i, j]
        longitudinal_dist = ghost_x - pos_x
        # 风险场在碰撞点前方无限生效（图上延展到绘图边界），而在超越点后无惩罚
        if longitudinal_dist > 0:
            lateral_dist = abs(pos_y - ghost_y)
            clearance = max(lateral_dist - ego_half_width, 0.0)
            sig = compute_sigmoid(clearance, ghost_lateral, k_steep=2.0)
            Z[i, j] = w_base * sig * kinetic_energy

fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
cmap = plt.get_cmap('Reds')
Z_masked = np.ma.masked_where(Z < 0.1, Z)
c = ax.pcolormesh(X, Y, Z_masked, cmap=cmap, shading='auto', alpha=0.85)

# 画出车道线
ax.axhline(0, color='#555555', linestyle='--', linewidth=1.5)
ax.axhline(3.5, color='black', linestyle='-', linewidth=2)
ax.axhline(-3.5, color='black', linestyle='-', linewidth=2)

bus_width = 2.5
bus_length = 10

base_x = ghost_x - bus_length 
base_y = ghost_y - bus_width  

bus_rect = patches.Rectangle((base_x, base_y), bus_length, bus_width, 
                             linewidth=1.5, edgecolor='black', facecolor='#E6E6E6', hatch='//', label='Occluding Vehicle')
ax.add_patch(bus_rect)

# 目标点在 (30, -1.8)
ax.plot(ghost_x, ghost_y, 'ro', markersize=8, markeredgecolor='black', label='Target Point', zorder=5)

# 自车的位置
ego_rect = patches.Rectangle((10, -ego_half_width), 4.5, ego_half_width*2, 
                             linewidth=1.5, edgecolor='black', facecolor='#4A90E2', alpha=0.9, label='Ego Vehicle')
ax.add_patch(ego_rect)

# 给自车画一个向右行驶的箭头
ax.arrow(12.25, 0, 1.5, 0, head_width=0.5, head_length=0.7, fc='black', ec='black', zorder=6)

cbar = plt.colorbar(c)
# LaTeX formatting for colorbar
cbar.set_label(r'Risk Potential Cost $\mathcal{C}_{\mathrm{risk}}$', fontsize=14)

plt.xlabel(r'Longitudinal Distance $x$ (m)', fontsize=14, weight='bold')
plt.ylabel(r'Lateral Distance $y$ (m)', fontsize=14, weight='bold')
plt.ylim(-4.8, 4)
plt.xlim(5, 45)
plt.legend(loc='lower left', framealpha=0.95)
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('/Users/phy/Desktop/MIND/paper/IEEE-conference-template-062824/figures/risk_field_academic.pdf', bbox_inches='tight', dpi=300)
plt.savefig('/Users/phy/Desktop/MIND/paper/figures/risk_field_academic.pdf', bbox_inches='tight', dpi=300)
print("Risk Field chart generated successfully")
