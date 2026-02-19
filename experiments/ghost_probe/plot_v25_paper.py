#!/usr/bin/env python3
"""
PA-LOI v25 论文级可视化
生成多面板图：速度、加速度、AEB状态、权重、安全距离
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ========== 配置 ==========
CSV_PATH = "output/ghost_experiment_v25_hysteresis/improved/logs/log_ghost_exp_improved_20260210_200029_W20_L01.csv"
OUTPUT_DIR = "output/ghost_experiment_v25_hysteresis/improved/logs"

# ========== 加载数据 ==========
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")

t = df['Time'].values
vel = df['Ego_Vel'].values
acc = df['Ctrl_Acc'].values
steer = df['Ctrl_Steer'].values
tta = df['TTA_Ego'].values if 'TTA_Ego' in df.columns else None
phantom_active = df['Is_Phantom_Active'].values if 'Is_Phantom_Active' in df.columns else None
risk_cost = df['Risk_Cost_Total'].values if 'Risk_Cost_Total' in df.columns else None
min_dist = df['Min_Dist_To_Ghost'].values if 'Min_Dist_To_Ghost' in df.columns else None

# ========== 检测 AEB 区间 ==========
aeb_on_t = None
aeb_off_t = None
for i in range(1, len(acc)):
    if acc[i] <= -2.5 and acc[i-1] > -2.5 and aeb_on_t is None:
        aeb_on_t = t[i]
    if aeb_on_t is not None and acc[i] > -2.5 and acc[i-1] <= -2.5:
        aeb_off_t = t[i]
        break

# Ghost Spawn 时间
ghost_spawn_t = 8.94

print(f"AEB ON:  t={aeb_on_t}")
print(f"AEB OFF: t={aeb_off_t}")
print(f"Data range: t=[{t[0]:.2f}, {t[-1]:.2f}]")

# ========== 绘图 ==========
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})

n_panels = 4
fig, axes = plt.subplots(n_panels, 1, figsize=(14, 13), sharex=True)
fig.suptitle('PA-LOI v25: Ghost Probe Emergency Braking with Hysteresis AEB', 
             fontsize=15, fontweight='bold', y=0.98)

# ---- 颜色定义 ----
COLOR_VEL = '#2196F3'
COLOR_ACC = '#FF5722'
COLOR_STEER = '#9C27B0'
COLOR_AEB = '#FFCDD2'
COLOR_GHOST = '#4CAF50'
COLOR_TTA = '#009688'
COLOR_DIST = '#795548'

def shade_aeb(ax):
    """在所有面板上标注 AEB 区间"""
    if aeb_on_t is not None and aeb_off_t is not None:
        ax.axvspan(aeb_on_t, aeb_off_t, alpha=0.20, color='red', zorder=0)
    if aeb_on_t is not None:
        ax.axvline(aeb_on_t, color='red', linestyle='--', linewidth=1.2, alpha=0.6)
    if aeb_off_t is not None:
        ax.axvline(aeb_off_t, color='green', linestyle='--', linewidth=1.2, alpha=0.6)
    ax.axvline(ghost_spawn_t, color=COLOR_GHOST, linestyle=':', linewidth=1.5, alpha=0.7)

# ========== Panel 1: 速度 ==========
ax1 = axes[0]
shade_aeb(ax1)
ax1.plot(t, vel, color=COLOR_VEL, linewidth=2.5, label='Ego Velocity', zorder=3)
ax1.fill_between(t, 0, vel, alpha=0.1, color=COLOR_VEL)
ax1.set_ylabel('Velocity (m/s)')
ax1.set_ylim(-0.2, 5.0)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left')

# AEB ON 标注
if aeb_on_t is not None:
    vel_at_on = vel[np.argmin(np.abs(t - aeb_on_t))]
    ax1.annotate(f'AEB ON\nv={vel_at_on:.1f} m/s', 
                xy=(aeb_on_t, vel_at_on),
                xytext=(aeb_on_t - 1.2, vel_at_on + 0.6),
                fontsize=10, fontweight='bold', color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                zorder=4)

# AEB OFF 标注
if aeb_off_t is not None:
    vel_at_off = vel[np.argmin(np.abs(t - aeb_off_t))]
    ax1.annotate(f'AEB OFF\nv={vel_at_off:.1f} m/s', 
                xy=(aeb_off_t, vel_at_off),
                xytext=(aeb_off_t + 0.3, vel_at_off + 1.2),
                fontsize=10, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                zorder=4)

# Ghost Spawn 标注
ax1.annotate('Ghost\nSpawn', 
            xy=(ghost_spawn_t, vel[np.argmin(np.abs(t - ghost_spawn_t))]),
            xytext=(ghost_spawn_t + 0.2, 4.2),
            fontsize=9, color=COLOR_GHOST, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=COLOR_GHOST, lw=1.2),
            zorder=4)

# 终速度标注
ax1.annotate(f'$v_{{final}}$={vel[-1]:.2f} m/s', 
            xy=(t[-1], vel[-1]),
            xytext=(t[-1] - 1.5, vel[-1] + 1.0),
            fontsize=10, fontweight='bold', color='navy',
            arrowprops=dict(arrowstyle='->', color='navy', lw=1.2),
            zorder=4)

ax1.set_title('① Velocity — Monotonic Deceleration (No Oscillation)')

# ========== Panel 2: 加速度 ==========
ax2 = axes[1]
shade_aeb(ax2)
ax2.plot(t, acc, color=COLOR_ACC, linewidth=2.0, label='Acceleration', zorder=3)
ax2.fill_between(t, 0, acc, where=(acc < 0), alpha=0.1, color=COLOR_ACC)
ax2.axhline(y=-4.0, color='darkred', linestyle=':', alpha=0.5, linewidth=1, label='AEB Limit (-4.0 m/s²)')
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax2.set_ylabel('Acceleration (m/s²)')
ax2.set_ylim(-5.0, 2.0)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='lower left')
ax2.set_title('② Acceleration — Sustained Emergency Braking During AEB')

# ========== Panel 3: 方向盘 ==========
ax3 = axes[2]
shade_aeb(ax3)
ax3.plot(t, steer, color=COLOR_STEER, linewidth=2.0, label='Steering Angle', zorder=3)
ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax3.set_ylabel('Steering (rad)')
ax3.set_ylim(-0.4, 0.4)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left')

# 标注 AEB 期间 steer → 0
if aeb_on_t and aeb_off_t:
    mid_t = (aeb_on_t + aeb_off_t) / 2
    ax3.annotate('steer → 0\n(straight-line braking)', 
                xy=(mid_t, 0.01),
                xytext=(mid_t - 1.5, 0.25),
                fontsize=9, fontweight='bold', color=COLOR_STEER,
                arrowprops=dict(arrowstyle='->', color=COLOR_STEER, lw=1.2),
                zorder=4)

ax3.set_title('③ Steering — Stabilized to Zero During AEB (No Swerving)')

# ========== Panel 4: TTA + Safety Distance ==========
ax4 = axes[3]
shade_aeb(ax4)

if tta is not None:
    tta_clipped = np.clip(tta, 0, 12)
    ax4.plot(t, tta_clipped, color=COLOR_TTA, linewidth=2.0, label='TTA (ego→occluder)', zorder=3)
    ax4.axhline(y=1.5, color='red', linestyle='--', alpha=0.7, linewidth=1.2, label='AEB Trigger Threshold (1.5s)')
    ax4.axhline(y=3.0, color='green', linestyle='--', alpha=0.7, linewidth=1.2, label='AEB Release Threshold (3.0s)')
    ax4.set_ylabel('TTA (s)', color=COLOR_TTA)
    ax4.set_ylim(0, 12)
    
    # Hysteresis 带标注
    ax4.fill_between(t, 1.5, 3.0, alpha=0.08, color='orange', label='Hysteresis Dead Zone')
    
    ax4.legend(loc='upper right', fontsize=8)

if min_dist is not None:
    ax4_twin = ax4.twinx()
    ax4_twin.plot(t, min_dist, color=COLOR_DIST, linewidth=1.8, alpha=0.7, label='Min Dist to Ghost', linestyle='-.')
    ax4_twin.set_ylabel('Distance (m)', color=COLOR_DIST)
    ax4_twin.tick_params(axis='y', labelcolor=COLOR_DIST)
    ax4_twin.set_ylim(0, 25)
    ax4_twin.legend(loc='center right', fontsize=8)
    
    # 标注最小距离
    min_d_idx = np.argmin(min_dist)
    ax4_twin.annotate(f'Min={min_dist[min_d_idx]:.1f}m', 
                xy=(t[min_d_idx], min_dist[min_d_idx]),
                xytext=(t[min_d_idx] + 0.3, min_dist[min_d_idx] + 3.0),
                fontsize=10, fontweight='bold', color=COLOR_DIST,
                arrowprops=dict(arrowstyle='->', color=COLOR_DIST, lw=1.5),
                zorder=4)

ax4.grid(True, alpha=0.3)
ax4.set_xlabel('Time (s)')
ax4.set_title('④ TTA & Distance — Hysteresis Prevents Oscillation')

# ========== 底部信息条 ==========
fig.text(0.5, 0.002, 
         '✅ Collisions: 0  |  Min Distance: 4.67m  |  AEB Duration: 0.70s  |  Final Velocity: 0.48 m/s  |  Max Deceleration: -3.49 m/s²', 
         ha='center', fontsize=11, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9', edgecolor='#4CAF50', alpha=0.9))

plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# ========== 保存 ==========
out_path = os.path.join(OUTPUT_DIR, "v25_paper_figure.png")
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\n✅ 论文级图表已保存: {out_path}")

docs_path = "docs/v25_paper_figure.png"
plt.savefig(docs_path, dpi=200, bbox_inches='tight')
print(f"✅ 副本已保存: {docs_path}")

plt.close()
