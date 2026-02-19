
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import sys

def plot_results(log_file=None):
    # Find latest log if not provided
    if log_file is None:
        list_of_files = glob.glob('output/ghost_experiment/improved/logs/*.csv')
        if not list_of_files:
            print("No log files found!")
            return
        log_file = max(list_of_files, key=os.path.getctime)
    
    print(f"Plotting: {log_file}")
    
    try:
        df = pd.read_csv(log_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Create figure
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    time = df['Time']
    
    # 1. Velocity & Risk Cost
    ax1 = axs[0]
    ax1.plot(time, df['Ego_Vel'], label='Velocity (m/s)', color='blue', linewidth=2)
    ax1.set_ylabel('Velocity (m/s)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    
    # Dual axis for Risk Cost
    if 'Risk_Cost_Total' in df.columns:
        ax1b = ax1.twinx()
        ax1b.plot(time, df['Risk_Cost_Total'], label='Risk Cost', color='red', linestyle='--', alpha=0.5)
        ax1b.set_ylabel('Risk Cost', color='red')
        ax1b.tick_params(axis='y', labelcolor='red')
    
    ax1.set_title(f'Simulation Analysis: {os.path.basename(log_file)}')
    # Combined legend
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1b.legend(lines + lines2, labels + labels2, loc='upper left')

    # 2. Acceleration
    ax2 = axs[1]
    ax2.plot(time, df['Ctrl_Acc'], label='Command Acc (m/s^2)', color='orange')
    ax2.plot(time, df['Ego_Acc'], label='Actual Acc (m/s^2)', color='brown', linestyle=':')
    ax2.axhline(y=-2.0, color='r', linestyle=':', label='Commfort Limit')
    ax2.set_ylabel('Acc (m/s^2)')
    ax2.grid(True)
    ax2.legend()
    
    # 3. Steering
    ax3 = axs[2]
    ax3.plot(time, df['Ctrl_Steer'], label='Steering (rad)', color='green')
    ax3.set_ylabel('Steering (rad)')
    ax3.grid(True)
    ax3.legend()
    
    # 4. Distance to Risk Source
    ax4 = axs[3]
    # Check column names
    dist_col = 'Risk_Source_Dist'
    if dist_col in df.columns:
        ax4.plot(time, df[dist_col], label='Dist to Risk (m)', color='purple')
        ax4.axhline(y=0.0, color='black', linewidth=1)
        ax4.set_ylim(-5, 50)
    
    # Check for Ghost Dist
    if 'Ghost_Dist' in df.columns:
         ax4.plot(time, df['Ghost_Dist'], label='Ghost Dist (m)', color='red', linestyle='--')
         
    ax4.set_ylabel('Distance (m)')
    ax4.set_xlabel('Time (s)')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    output_png = log_file.replace('.csv', '.png')
    plt.savefig(output_png)
    print(f"Saved plot to: {output_png}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_results(sys.argv[1])
    else:
        plot_results()
