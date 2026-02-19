import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from planners.mind.data_logger import plot_experiment_results

# Point to the LATEST log (Vanilla MIND)
# Point to the LATEST log (Baseline, Upper Spawn)
log_file = "output/ghost_experiment/baseline/logs/log_02_upper_spawn.csv"

# Plot!
output_png = plot_experiment_results(log_file)
print(f"Plot saved to: {output_png}")
