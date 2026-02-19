
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sys

# 文件路径
log_file = "output/ghost_experiment_v32_finalest/improved/logs/log_ghost_exp_improved_20260210_222242_W20_L01.csv"
output_file = "output/ghost_experiment_v32_finalest/v32_control_paper_figure.png"

try:
    df = pd.read_csv(log_file)
except Exception as e:
    print(f"Error reading CSV: {e}")
    sys.exit(1)

# 数据预处理
time = df['Time']
vel = df['Ego_Vel']
acc = df['Ctrl_Acc']
steer = df['Ctrl_Steer']
min_dist = df['Risk_Source_Dist']
# risk_weight = df['Max_Risk_Weight'] if 'Max_Risk_Weight' in df.columns else np.zeros_like(time)
# Cost 也是一个很好的指标
cost = df['Risk_Cost_Total']

# 创建图表
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 1])

# 1. Velocity
ax0 = plt.subplot(gs[0])
ax0.plot(time, vel, 'b-', linewidth=2, label='Velocity')
ax0.set_ylabel('Velocity (m/s)', fontsize=12)
ax0.set_title('Control Experiment (No Ghost): PA-LOI Performance', fontsize=16)
ax0.grid(True, linestyle='--', alpha=0.6)
ax0.legend(loc='upper right')

# 2. Acceleration (Key for AEB check)
ax1 = plt.subplot(gs[1], sharex=ax0)
ax1.plot(time, acc, 'r-', linewidth=2, label='Acceleration')
ax1.axhline(y=-4.0, color='k', linestyle='--', alpha=0.5, label='AEB Threshold (-4.0)')
ax1.set_ylabel('Acc (m/s²)', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='upper right')

# 3. Steering (Lateral Control)
ax2 = plt.subplot(gs[2], sharex=ax0)
ax2.plot(time, steer, 'g-', linewidth=2, label='Steering')
ax2.set_ylabel('Steering (rad)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(loc='upper right')

# 4. Distance to Obstacles
ax3 = plt.subplot(gs[3], sharex=ax0)
ax3.plot(time, min_dist, 'm-', linewidth=2, label='Min Distance')
# 标记遮挡区域(大概的时间)
ax3.axvspan(3.0, 6.0, color='gray', alpha=0.1, label='Occlusion Zone')
ax3.set_ylabel('Distance (m)', fontsize=12)
ax3.set_xlabel('Time (s)', fontsize=12)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.legend(loc='upper right')

plt.tight_layout()
plt.savefig(output_file, dpi=300)
print(f"Figure saved to {output_file}")
