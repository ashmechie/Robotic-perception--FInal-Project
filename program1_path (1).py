"""
MEEN 700 - Robotic Perception
Program 1: GPS + Gyroscope Path Reconstruction using EKF
Strava-style path visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from scipy.signal import butter, filtfilt

# ─────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────
print("=" * 55)
print("  PROGRAM 1: Path Reconstruction (GPS + Gyroscope + EKF)")
print("=" * 55)

gyro = pd.read_csv('Gyroscope.csv')
gps  = pd.read_csv('Location.csv')

gyro.columns = ['time', 'gx', 'gy', 'gz']
gps.columns  = ['time', 'lat', 'lon', 'height', 'velocity', 'direction', 'h_acc', 'v_acc']

# Drop first GPS row (bad accuracy) and rows with NaN lat/lon
gps = gps[gps['h_acc'] < 20].copy().reset_index(drop=True)
gps = gps.dropna(subset=['lat', 'lon']).reset_index(drop=True)

print(f"\n[1] Data Loaded:")
print(f"    Gyroscope : {len(gyro)} samples @ ~499 Hz")
print(f"    GPS       : {len(gps)} fixes (after quality filter)")

# ─────────────────────────────────────────
# 2. CONVERT GPS TO LOCAL X/Y COORDINATES (meters)
# ─────────────────────────────────────────
# Use first GPS fix as origin
lat0 = gps['lat'].iloc[0]
lon0 = gps['lon'].iloc[0]

R_EARTH = 6371000.0  # meters

def latlon_to_xy(lat, lon, lat0, lon0):
    """Convert lat/lon to local x/y in meters relative to origin"""
    x = (lon - lon0) * np.pi/180 * R_EARTH * np.cos(np.radians(lat0))
    y = (lat - lat0) * np.pi/180 * R_EARTH
    return x, y

gps['x'], gps['y'] = latlon_to_xy(gps['lat'].values, gps['lon'].values, lat0, lon0)

# Compute GPS total distance
dists = np.sqrt(np.diff(gps['x'])**2 + np.diff(gps['y'])**2)
gps_total_dist = dists.sum()
print(f"\n[2] GPS Path:")
print(f"    Total GPS distance : {gps_total_dist:.1f} m")
print(f"    Strava reference   : {0.05 * 1609.34:.1f} m (0.05 miles)")

# ─────────────────────────────────────────
# 3. GYROSCOPE PREPROCESSING
# ─────────────────────────────────────────
# Low-pass filter gyro z-axis (yaw rate) — this controls heading changes
sample_rate_gyro = 499.0
b, a = butter(N=4, Wn=2.0 / (sample_rate_gyro / 2), btype='low')
gyro['gz_filt'] = filtfilt(b, a, gyro['gz'])

# Integrate gz to get heading angle (yaw)
dt_gyro = np.diff(gyro['time'].values)
gyro_yaw = np.zeros(len(gyro))
for i in range(1, len(gyro)):
    gyro_yaw[i] = gyro_yaw[i-1] + gyro['gz_filt'].iloc[i] * dt_gyro[i-1]

gyro['yaw'] = gyro_yaw

print(f"\n[3] Gyroscope:")
print(f"    Yaw range : {np.degrees(gyro_yaw.min()):.1f}° to {np.degrees(gyro_yaw.max()):.1f}°")

# ─────────────────────────────────────────
# 4. EKF — FUSE GPS + GYROSCOPE
# ─────────────────────────────────────────
# State: [x, y, heading, velocity]
# Prediction: use gyro yaw rate to update heading
# Update: correct x, y, heading from GPS fixes

print(f"\n[4] Running Extended Kalman Filter...")

# EKF state: [x, y, heading]
x_est = np.array([gps['x'].iloc[0], gps['y'].iloc[0], 0.0])  # initial state
P = np.diag([5.0, 5.0, 0.5])   # initial covariance

# Noise matrices
Q = np.diag([0.01, 0.01, 0.001])   # process noise (gyro drift)
R_gps = np.diag([2.0, 2.0])        # measurement noise (GPS ~2m accuracy)

# Interpolate gyro yaw to GPS timestamps for EKF updates
gyro_yaw_at_gps = np.interp(gps['time'].values, gyro['time'].values, gyro['yaw'].values)
gyro_rate_at_gps = np.interp(gps['time'].values, gyro['time'].values, gyro['gz_filt'].values)

ekf_path_x = [x_est[0]]
ekf_path_y = [x_est[1]]

gps_times = gps['time'].values
prev_time = gps_times[0]

for i in range(1, len(gps)):
    dt = gps_times[i] - prev_time
    prev_time = gps_times[i]

    # ── PREDICT step ──
    # Use GPS velocity for motion magnitude
    v = gps['velocity'].iloc[i] if not np.isnan(gps['velocity'].iloc[i]) else 0.8
    heading = x_est[2]
    yaw_rate = gyro_rate_at_gps[i]

    # Predicted state
    x_pred = np.array([
        x_est[0] + v * np.cos(heading) * dt,
        x_est[1] + v * np.sin(heading) * dt,
        x_est[2] + yaw_rate * dt
    ])

    # Jacobian of motion model
    F = np.array([
        [1, 0, -v * np.sin(heading) * dt],
        [0, 1,  v * np.cos(heading) * dt],
        [0, 0,  1]
    ])

    P_pred = F @ P @ F.T + Q

    # ── UPDATE step ──
    # Measurement: GPS x, y
    z = np.array([gps['x'].iloc[i], gps['y'].iloc[i]])

    H = np.array([
        [1, 0, 0],
        [0, 1, 0]
    ])

    y_innov = z - H @ x_pred             # innovation
    S = H @ P_pred @ H.T + R_gps         # innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

    x_est = x_pred + K @ y_innov
    P = (np.eye(3) - K @ H) @ P_pred

    ekf_path_x.append(x_est[0])
    ekf_path_y.append(x_est[1])

ekf_path_x = np.array(ekf_path_x)
ekf_path_y = np.array(ekf_path_y)

ekf_dist = np.sqrt(np.diff(ekf_path_x)**2 + np.diff(ekf_path_y)**2).sum()
print(f"    EKF path distance  : {ekf_dist:.1f} m")

# ─────────────────────────────────────────
# 5. PLOT — Strava-style path
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor('#1a1a2e')

for ax in axes:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

# ── Plot A: Raw GPS path ──
ax1 = axes[0]
ax1.plot(gps['x'], gps['y'], 'o-', color='#aaaaaa',
         linewidth=1.5, markersize=3, alpha=0.7, label='Raw GPS')
ax1.plot(gps['x'].iloc[0],  gps['y'].iloc[0],
         'o', color='#00ff88', markersize=12, zorder=5, label='Start')
ax1.plot(gps['x'].iloc[-1], gps['y'].iloc[-1],
         's', color='#ff4444', markersize=12, zorder=5, label='End')
ax1.set_title('Raw GPS Path', fontweight='bold', fontsize=11)
ax1.set_xlabel('East (m)')
ax1.set_ylabel('North (m)')
ax1.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
ax1.set_aspect('equal')

# ── Plot B: EKF fused path (Strava style) ──
ax2 = axes[1]

# Color the path by speed (Strava-style gradient)
speeds = gps['velocity'].ffill().values
speeds_norm = (speeds - speeds.min()) / (speeds.max() - speeds.min() + 1e-6)

points = np.array([ekf_path_x, ekf_path_y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
lc = LineCollection(segments, cmap='cool', linewidth=3, alpha=0.9)
lc.set_array(speeds_norm)
ax2.add_collection(lc)

ax2.plot(ekf_path_x[0],  ekf_path_y[0],
         'o', color='#00ff88', markersize=14, zorder=5, label='Start')
ax2.plot(ekf_path_x[-1], ekf_path_y[-1],
         's', color='#ff4444', markersize=14, zorder=5, label='End')

ax2.set_xlim(ekf_path_x.min()-5, ekf_path_x.max()+5)
ax2.set_ylim(ekf_path_y.min()-5, ekf_path_y.max()+5)
ax2.set_title('EKF Fused Path\n(GPS + Gyroscope)', fontweight='bold', fontsize=11)
ax2.set_xlabel('East (m)')
ax2.set_ylabel('North (m)')
ax2.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)
ax2.set_aspect('equal')

cbar = plt.colorbar(lc, ax=ax2)
cbar.set_label('Speed (normalized)', color='white', fontsize=8)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

# ── Plot C: Gyroscope yaw over time ──
ax3 = axes[2]
ax3.plot(gyro['time'], np.degrees(gyro['yaw']),
         color='#00aaff', linewidth=1.2, label='Integrated Yaw (°)')
ax3.set_title('Gyroscope — Heading Over Time', fontweight='bold', fontsize=11)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Heading (degrees)')
ax3.legend(facecolor='#1a1a2e', labelcolor='white', fontsize=8)

plt.suptitle('MEEN 700 — Program 1: Path Reconstruction (GPS + Gyroscope + EKF)',
             color='white', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/program1_path.png',
            dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
print("\n[5] Plot saved: program1_path.png")

print(f"\n{'='*55}")
print(f"  PROGRAM 1 RESULTS")
print(f"{'='*55}")
print(f"  Raw GPS distance  : {gps_total_dist:.1f} m")
print(f"  EKF path distance : {ekf_dist:.1f} m")
print(f"  Strava reference  : {0.05*1609.34:.1f} m")
print(f"  Walk duration     : {gps['time'].max():.0f} seconds")
print(f"{'='*55}")
