"""
MEEN 700 - Robotic Perception
Program 2: Step Counter + Distance Estimator (Accelerometer + Barometer + KF)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from filterpy.kalman import KalmanFilter

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("=" * 55)
print("  PROGRAM 2: Step Counter (Accelerometer + Barometer + KF)")
print("=" * 55)

acc  = pd.read_csv('Linear Acceleration.csv')
pres = pd.read_csv('Pressure.csv')

acc.columns  = ['time', 'ax', 'ay', 'az']
pres.columns = ['time', 'pressure']

STEP_LENGTH_M = 0.40   # meters per step

print(f"\n[1] Data Loaded:")
print(f"    Accelerometer : {len(acc)} samples over {acc['time'].max():.1f} seconds")
print(f"    Barometer     : {len(pres)} samples over {pres['time'].max():.1f} seconds")

# ─────────────────────────────────────────
# 2. ACCELERATION MAGNITUDE + FILTERING
# ─────────────────────────────────────────
acc['magnitude'] = np.sqrt(acc['ax']**2 + acc['ay']**2 + acc['az']**2)

sample_rate = len(acc) / acc['time'].max()
b, a = butter(N=4, Wn=3.0 / (sample_rate / 2), btype='low')
acc['mag_filtered'] = filtfilt(b, a, acc['magnitude'])

print(f"\n[2] Signal Processing:")
print(f"    Sample rate     : {sample_rate:.1f} Hz")
print(f"    Low-pass cutoff : 3.0 Hz")

# ─────────────────────────────────────────
# 3. STEP DETECTION
# ─────────────────────────────────────────
min_samples = int(0.4 * sample_rate)

peaks, _ = find_peaks(
    acc['mag_filtered'],
    height=0.15,
    distance=min_samples,
    prominence=0.08
)

detected_steps    = len(peaks)
estimated_dist_m  = detected_steps * STEP_LENGTH_M
step_times        = acc['time'].iloc[peaks].values

print(f"\n[3] Step Detection:")
print(f"    Steps detected    : {detected_steps}")
print(f"    Estimated distance: {estimated_dist_m:.1f} m")
print(f"    Strava reference  : {0.05 * 1609.34:.1f} m")

# ─────────────────────────────────────────
# 4. KALMAN FILTER ON POSITION
# ─────────────────────────────────────────
kf = KalmanFilter(dim_x=2, dim_z=1)
kf.F = np.array([[1, 1], [0, 1]])
kf.H = np.array([[1, 0]])
kf.x = np.array([[0.], [0.]])
kf.P = np.eye(2) * 10.0
kf.Q = np.array([[0.01, 0.00], [0.00, 0.01]])
kf.R = np.array([[0.05]])

raw_positions = np.arange(1, detected_steps + 1) * STEP_LENGTH_M
kf_positions  = []

for i in range(detected_steps):
    kf.predict()
    kf.update(np.array([[raw_positions[i]]]))
    kf_positions.append(kf.x[0, 0])

kf_positions = np.array(kf_positions)

print(f"\n[4] Kalman Filter:")
print(f"    Raw position estimate : {raw_positions[-1]:.2f} m")
print(f"    KF position estimate  : {kf_positions[-1]:.2f} m")

# ─────────────────────────────────────────
# 5. BAROMETER KALMAN FILTER
# ─────────────────────────────────────────
kf_p = KalmanFilter(dim_x=1, dim_z=1)
kf_p.F = np.array([[1.]])
kf_p.H = np.array([[1.]])
kf_p.x = np.array([[pres['pressure'].iloc[0]]])
kf_p.P = np.array([[1.]])
kf_p.Q = np.array([[0.0001]])
kf_p.R = np.array([[0.01]])

kf_pressure = []
for p in pres['pressure']:
    kf_p.predict()
    kf_p.update(np.array([[p]]))
    kf_pressure.append(kf_p.x[0, 0])

kf_pressure = np.array(kf_pressure)
pressure_var = pres['pressure'].max() - pres['pressure'].min()

print(f"\n[5] Barometer:")
print(f"    Pressure variation : {pressure_var:.4f} hPa")
print(f"    Altitude change    : ~{pressure_var * 8.5:.2f} m")

# ─────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#1a1a2e')

for ax in axes.flat:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

# ── Plot 1: Raw vs filtered acceleration ──
ax1 = axes[0, 0]
ax1.plot(acc['time'], acc['magnitude'],
         color='#4488ff', linewidth=0.6, alpha=0.5, label='Raw')
ax1.plot(acc['time'], acc['mag_filtered'],
         color='#00ccff', linewidth=1.5, label='Low-Pass Filtered')
ax1.set_title('Acceleration Magnitude — Raw vs Filtered', fontweight='bold')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Acceleration (m/s²)')
ax1.legend(facecolor='#1a1a2e', labelcolor='white')

# ── Plot 2: Step detection ──
ax2 = axes[0, 1]
ax2.plot(acc['time'], acc['mag_filtered'],
         color='#00ccff', linewidth=1.2, label='Filtered Signal')
ax2.plot(acc['time'].iloc[peaks], acc['mag_filtered'].iloc[peaks],
         'v', color='#ff6644', markersize=8,
         label=f'Steps Detected: {detected_steps}', zorder=5)
ax2.set_title('Step Detection via Peak Finding', fontweight='bold')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Acceleration (m/s²)')
ax2.legend(facecolor='#1a1a2e', labelcolor='white')

# ── Plot 3: Kalman filter position ──
ax3 = axes[1, 0]
ax3.step(step_times, raw_positions, where='post',
         color='#ffaa00', linewidth=1.5, linestyle='--', label='Raw Step Estimate')
ax3.step(step_times, kf_positions, where='post',
         color='#00ff88', linewidth=2.0, label='Kalman Filter Estimate')
ax3.axhline(y=0.05*1609.34, color='#ff4444', linestyle=':',
            linewidth=1.5, label=f'Strava Ref ({0.05*1609.34:.0f} m)')
ax3.set_title('Distance Estimate — Raw vs Kalman Filter', fontweight='bold')
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Distance (m)')
ax3.legend(facecolor='#1a1a2e', labelcolor='white')

# ── Plot 4: Barometer ──
ax4 = axes[1, 1]
ax4.plot(pres['time'], pres['pressure'],
         color='#ff88aa', linewidth=1.0, alpha=0.7, label='Raw Pressure')
ax4.plot(pres['time'], kf_pressure,
         color='#ff2266', linewidth=2.0, label='KF Smoothed')
ax4.set_title('Barometer — Raw vs Kalman Filtered', fontweight='bold')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Pressure (hPa)')
ax4.legend(facecolor='#1a1a2e', labelcolor='white')

plt.suptitle('MEEN 700 — Program 2: Step Counter (Accelerometer + Barometer + KF)',
             color='white', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/program2_steps.png',
            dpi=150, bbox_inches='tight', facecolor='#1a1a2e')

print("\n[6] Plot saved: program2_steps.png")
print(f"\n{'='*55}")
print(f"  PROGRAM 2 RESULTS")
print(f"{'='*55}")
print(f"  Steps Detected       : {detected_steps}")
print(f"  Distance (raw)       : {raw_positions[-1]:.1f} m")
print(f"  Distance (KF)        : {kf_positions[-1]:.1f} m")
print(f"  Strava Reference     : {0.05*1609.34:.1f} m")
print(f"  Barometer variation  : {pressure_var:.4f} hPa (stable)")
print(f"{'='*55}")

# Save results for combined dashboard
import json
results = {
    'steps': int(detected_steps),
    'distance_kf': float(kf_positions[-1]),
    'distance_raw': float(raw_positions[-1]),
    'step_times': step_times.tolist(),
    'kf_positions': kf_positions.tolist(),
    'raw_positions': raw_positions.tolist(),
    'pressure_var': float(pressure_var)
}
with open('/home/claude/program2_results.json', 'w') as f:
    json.dump(results, f)
