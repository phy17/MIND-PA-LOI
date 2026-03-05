import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

# ======== IEEE-compliant formatting: Type 42 fonts, no Type 3 ========
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'text.usetex': False,
})

# ======== Physical parameters (matching paper) ========
ghost_x = 30.0        # Target point longitudinal position
ghost_y = -1.8        # Target point lateral position (nearside lane)
ghost_lateral = 1.5   # Sigmoid lateral threshold
ego_half_width = 1.0  # Half-width of ego vehicle
w_base = 25.0         # Base weight
v_safe = 2.5          # Safe speed threshold (m/s)
v_ego = 8.0           # Ego cruising speed (m/s)
k_steep = 2.0         # Sigmoid steepness

# ======== Compute risk field ========
x = np.linspace(0, 50, 600)
y = np.linspace(-5.5, 5.5, 300)
X, Y = np.meshgrid(x, y)

def compute_sigmoid(clearance, ghost_lat, k=2.0):
    return 1.0 / (1.0 + np.exp(k * (clearance - ghost_lat)))

excess_vel = max(0.0, v_ego - v_safe)
kinetic_energy = excess_vel ** 2

# Vectorized computation for speed
lateral_dist = np.abs(Y - ghost_y)
clearance = np.maximum(lateral_dist - ego_half_width, 0.0)
sig = compute_sigmoid(clearance, ghost_lateral, k_steep)

# No longitudinal decay: the penalty field extends infinitely along
# the driving direction (paper Section III-B, Fig.1 caption).
# External logic removes the risk source only after the ego passes the target.
Z = w_base * sig * kinetic_energy

# ======== Create figure ========
fig, ax = plt.subplots(figsize=(7, 3.5), dpi=300)

# Risk field heatmap with better contrast
Z_masked = np.ma.masked_where(Z < 0.5, Z)
c = ax.pcolormesh(X, Y, Z_masked, cmap='OrRd', shading='gouraud',
                  alpha=0.9, vmin=0, vmax=w_base * kinetic_energy * 1.0)

# ======== Lane markings ========
lane_width = 3.5
# Solid lane boundaries
ax.axhline(lane_width, color='#2C2C2C', linestyle='-', linewidth=2.0)
ax.axhline(-lane_width, color='#2C2C2C', linestyle='-', linewidth=2.0)
# Center dashed line
ax.axhline(0, color='#666666', linestyle='--', linewidth=1.2, dashes=(6, 4))

# ======== Occluding Vehicle (parked bus/large vehicle) ========
# Target point is at the front-side corner (top-right) of the occluding vehicle
bus_length = 10.0
bus_width = 2.2
bus_x = ghost_x - bus_length   # bus front edge aligns with target x
bus_y = ghost_y - bus_width     # bus top edge aligns with target y

bus_rect = patches.FancyBboxPatch(
    (bus_x, bus_y), bus_length, bus_width,
    boxstyle="round,pad=0.1",
    linewidth=1.5, edgecolor='#333333', facecolor='#D5D5D5',
    hatch='///', zorder=4, label='Occluding Vehicle'
)
ax.add_patch(bus_rect)

# ======== Target Point ========
ax.plot(ghost_x, ghost_y, 'o', color='#E63946', markersize=9,
        markeredgecolor='#1D1D1D', markeredgewidth=1.5,
        zorder=6, label='Target Point')

# ======== Ego Vehicle ========
ego_length = 4.5
ego_x = 9.0
ego_rect = patches.FancyBboxPatch(
    (ego_x, -ego_half_width), ego_length, ego_half_width * 2,
    boxstyle="round,pad=0.08",
    linewidth=1.5, edgecolor='#1D3557', facecolor='#457B9D',
    alpha=0.95, zorder=5, label='Ego Vehicle'
)
ax.add_patch(ego_rect)

# Direction arrow on ego vehicle
ax.annotate('', xy=(ego_x + ego_length + 1.8, 0),
            xytext=(ego_x + ego_length + 0.3, 0),
            arrowprops=dict(arrowstyle='->', color='#1D3557', lw=2.0),
            zorder=7)

# ======== Colorbar ========
cbar = plt.colorbar(c, ax=ax, pad=0.02, aspect=25)
cbar.set_label(r'$C_{\mathrm{risk}}$  (Risk Potential Cost)', fontsize=11)
cbar.ax.tick_params(labelsize=9)

# ======== Axis labels (matching paper notation) ========
ax.set_xlabel(r'Longitudinal Position $x$ (m)', fontsize=11)
ax.set_ylabel(r'Lateral Position $y$ (m)', fontsize=11)
ax.set_xlim(5, 45)
ax.set_ylim(-4.8, 4.2)

# ======== Legend ========
leg = ax.legend(loc='upper left', framealpha=0.92, edgecolor='#999999',
                fontsize=8.5, handlelength=1.5)
leg.get_frame().set_linewidth(0.8)

# ======== Grid ========
ax.grid(True, linestyle=':', alpha=0.35, color='#AAAAAA')

# ======== Annotation: indicate infinite extension ========
ax.annotate(r'$\longleftarrow$ Field extends to $-\infty$',
            xy=(6, ghost_y + 0.3), fontsize=8, color='#8B0000',
            fontstyle='italic',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

plt.tight_layout()

# Save to both locations
out1 = '/Users/phy/Desktop/MIND/paper/IEEE-conference-template-062824/figures/risk_field_academic.pdf'
out2 = '/Users/phy/Desktop/MIND/paper/figures/risk_field_academic.pdf'
plt.savefig(out1, bbox_inches='tight', dpi=300)
plt.savefig(out2, bbox_inches='tight', dpi=300)
print(f"Risk field figure saved to:\n  {out1}\n  {out2}")
print("Font types: pdf.fonttype=42, ps.fonttype=42 (Type 42 / TrueType, NO Type 3)")
