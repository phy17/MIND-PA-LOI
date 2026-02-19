import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def plot_experiment_log(csv_file):
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} not found.")
        return

    try:
        # Load Data
        df = pd.read_csv(csv_file)
        
        # Setup Figure
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        plt.subplots_adjust(hspace=0.1)
        
        # 1. Velocity Profile
        ax1 = axes[0]
        ax1.plot(df['Time'], df['Ego_Vel'], 'b-', linewidth=2.5, label='Ego Speed (m/s)')
        # Draw v_safe line
        ax1.axhline(y=2.5, color='g', linestyle='--', alpha=0.7, label='Defensive Speed (2.5 m/s)')
        ax1.set_ylabel('Speed (m/s)', fontsize=12)
        ax1.set_title(f'Figure 3: Defensive Approach Dynamics (File: {os.path.basename(csv_file)})', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Risk Dynamics
        ax2 = axes[1]
        ax2.plot(df['Time'], df['Risk_Source_Dist'], 'r-', linewidth=2, label='Dist to Blind Spot (m)')
        ax2.set_ylabel('Distance (m)', fontsize=12)
        
        # Dual risk cost (on same plot)
        ax2b = ax2.twinx()
        ax2b.fill_between(df['Time'], df['Risk_Cost_Total'], color='orange', alpha=0.3, label='Hinge-Loss Cost')
        ax2b.set_ylabel('Potential Field Cost', color='orange', fontsize=12)
        
        # Combine legends
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Control Inputs (AEB Check)
        ax3 = axes[2]
        ax3.plot(df['Time'], df['Ctrl_Acc'], 'k-', linewidth=2, label='Acceleration Command')
        ax3.axhline(y=-4.0, color='r', linestyle=':', label='Max Braking (-4.0)')
        ax3.set_ylabel('Acc Cmd (m/s²)', fontsize=12)
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.legend(loc='lower right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Highlight trigger point if ghost appeared
        ghost_active = df[df['Is_Phantom_Active'] == 1]
        if not ghost_active.empty:
            trigger_time = ghost_active.iloc[0]['Time']
            for ax in axes:
                ax.axvline(x=trigger_time, color='r', linestyle='--', alpha=0.8, linewidth=1.5)
                # Add text only on top plot
                if ax == axes[0]:
                    ax.text(trigger_time, ax.get_ylim()[1]*0.9, ' GHOST TRIGGER', color='red', fontweight='bold')

        # Save
        output_png = csv_file.replace('.csv', '.png')
        plt.savefig(output_png, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_png}")
        
    except Exception as e:
        print(f"Error plotting: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default test file (the one user asked for)
        default_file = "output/ghost_experiment/improved/logs/log_ghost_exp_improved_20260214_080235_W20_L01.csv"
        plot_experiment_log(default_file)
    else:
        plot_experiment_log(sys.argv[1])
