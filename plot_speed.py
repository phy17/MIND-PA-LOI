import csv
import matplotlib.pyplot as plt

v1 = [float(r['ego_vel']) for r in list(csv.DictReader(open('output/no_ghost_efficiency_test/vanilla_mind_frame_log.csv')))]
v2 = [float(r['ego_vel']) for r in list(csv.DictReader(open('output/no_ghost_efficiency_test/paloi_aeb_no_ghost_frame_log.csv')))]

plt.figure(figsize=(10, 5))
plt.plot(v1, label='Vanilla MIND', color='blue')
plt.plot(v2, label='PA-LOI + AEB', color='red', linestyle='dashed')
plt.title('Speed Profile Comparison (No Pedestrian)')
plt.xlabel('Simulation Step')
plt.ylabel('Ego Velocity (m/s)')
plt.legend()
plt.grid(True)
plt.savefig('output/no_ghost_efficiency_test/speed_compare.png')
print("Plot saved to output/no_ghost_efficiency_test/speed_compare.png")
