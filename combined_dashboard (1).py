"""
MEEN 700 - Robotic Perception
Combined Dashboard: Path + Step Counter Results
Runs both programs and displays unified output
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from scipy.signal import find_peaks, butter, filtfilt
from filterpy.kalman import KalmanFilter

print("=" * 60)
print("  MEEN 700 — Combined Dashboard")
print("  Running Program 1 (GPS+Gyro) + Program 2 (Acc+Baro)...")
print("=" * 60)

# ═══════════════════════════════════════════════════════
# LOAD ALL DATA
# ═══════════════════════════════════════════════════════
gyro = pd.read_csv('Gyroscope.csv')
gps  = pd.read_csv('Location.csv')
acc  = pd.read_csv('Linear Acceleration.csv')
pres = pd.read_csv('Pressure.csv')

gyro.columns = ['time','gx','gy','gz']
gps.columns  = ['time','lat','lon','height','velocity','direction','h_acc','v_acc']
acc.columns  = ['time','ax','ay','az']
pres.columns = ['time','pressure']

STEP_LENGTH_M = 0.40
STRAVA_DIST_M = 0.05 * 1609.34

# ═══════════════════════════════════════════════════════
# PROGRAM 1: GPS + GYROSCOPE EKF PATH
# ═══════════════════════════════════════════════════════
print("\n[Program 1] Running GPS + Gyroscope EKF...")

gps_clean = gps[gps['h_acc'] < 20].dropna(subset=['lat','lon']).reset_index(drop=True)

lat0, lon0 = gps_clean['lat'].iloc[0], gps_clean['lon'].iloc[0]
R_EARTH = 6371000.0

gps_clean['x'] = (gps_clean['lon'] - lon0) * np.pi/180 * R_EARTH * np.cos(np.radians(lat0))
gps_clean['y'] = (gps_clean['lat'] - lat0) * np.pi/180 * R_EARTH

# Gyro filter + bias removal
sr_gyro = 499.0
b, a = butter(N=4, Wn=2.0/(sr_gyro/2), btype='low')
gyro['gz_filt'] = filtfilt(b, a, gyro['gz'])
bias_samples = int(5.0 * sr_gyro)
gyro_bias = gyro['gz_filt'].iloc[:bias_samples].mean()
gyro['gz_filt'] = gyro['gz_filt'] - gyro_bias

# Interpolate gyro rate at GPS timestamps
gyro_rate_at_gps = np.interp(gps_clean['time'].values,
                              gyro['time'].values,
                              gyro['gz_filt'].values)

# EKF initial state
x_est = np.array([gps_clean['x'].iloc[0], gps_clean['y'].iloc[0], 0.0])
P     = np.diag([5.0, 5.0, 0.5])
Q     = np.diag([0.01, 0.01, 0.001])
R_gps = np.diag([2.0, 2.0])

ekf_x, ekf_y = [x_est[0]], [x_est[1]]
prev_time = gps_clean['time'].values[0]

for i in range(1, len(gps_clean)):
    dt        = gps_clean['time'].values[i] - prev_time
    prev_time = gps_clean['time'].values[i]
    v = gps_clean['velocity'].iloc[i] if not np.isnan(gps_clean['velocity'].iloc[i]) else 0.8

    # ── Key fix: derive heading from GPS positions, not gyroscope ──
    # GPS tells us exactly which direction we actually moved
    dx = gps_clean['x'].iloc[i] - gps_clean['x'].iloc[i-1]
    dy = gps_clean['y'].iloc[i] - gps_clean['y'].iloc[i-1]
    gps_heading = np.arctan2(dy, dx)

    # Gyroscope adds a small correction on top of GPS heading
    yr = gyro_rate_at_gps[i] * 0.1   # scale down heavily — just smoothing
    heading = gps_heading + yr * dt

    x_pred = np.array([
        x_est[0] + v * np.cos(heading) * dt,
        x_est[1] + v * np.sin(heading) * dt,
        heading
    ])

    F = np.array([[1, 0, -v * np.sin(heading) * dt],
                  [0, 1,  v * np.cos(heading) * dt],
                  [0, 0,  1]])

    P_pred = F @ P @ F.T + Q

    z     = np.array([gps_clean['x'].iloc[i], gps_clean['y'].iloc[i]])
    H     = np.array([[1, 0, 0], [0, 1, 0]])
    y_inn = z - H @ x_pred
    S     = H @ P_pred @ H.T + R_gps
    K     = P_pred @ H.T @ np.linalg.inv(S)

    x_est = x_pred + K @ y_inn
    P     = (np.eye(3) - K @ H) @ P_pred

    ekf_x.append(x_est[0])
    ekf_y.append(x_est[1])

ekf_x    = np.array(ekf_x)
ekf_y    = np.array(ekf_y)
ekf_dist = np.sqrt(np.diff(ekf_x)**2 + np.diff(ekf_y)**2).sum()
gps_dist = np.sqrt(np.diff(gps_clean['x'].values)**2 + np.diff(gps_clean['y'].values)**2).sum()
speeds   = gps_clean['velocity'].ffill().values

print(f"    EKF distance: {ekf_dist:.1f} m | GPS distance: {gps_dist:.1f} m")

# ═══════════════════════════════════════════════════════
# PROGRAM 2: ACCELEROMETER + BAROMETER STEP COUNTER
# ═══════════════════════════════════════════════════════
print("\n[Program 2] Running Step Counter + KF...")

acc['magnitude'] = np.sqrt(acc['ax']**2 + acc['ay']**2 + acc['az']**2)
sr_acc = len(acc) / acc['time'].max()
b2, a2 = butter(N=4, Wn=3.0/(sr_acc/2), btype='low')
acc['mag_filt'] = filtfilt(b2, a2, acc['magnitude'])

peaks, _ = find_peaks(acc['mag_filt'], height=0.15,
                      distance=int(0.4*sr_acc), prominence=0.08)
n_steps   = len(peaks)
step_times = acc['time'].iloc[peaks].values
raw_pos   = np.arange(1, n_steps+1) * STEP_LENGTH_M

kf = KalmanFilter(dim_x=2, dim_z=1)
kf.F = np.array([[1, 1], [0, 1]])
kf.H = np.array([[1, 0]])
kf.x = np.array([[0.], [0.]])
kf.P = np.eye(2) * 10
kf.Q = np.array([[0.01, 0], [0, 0.01]])
kf.R = np.array([[0.05]])

kf_pos = []
for i in range(n_steps):
    kf.predict()
    kf.update(np.array([[raw_pos[i]]]))
    kf_pos.append(kf.x[0, 0])
kf_pos = np.array(kf_pos)

# Barometer KF
kf_b = KalmanFilter(dim_x=1, dim_z=1)
kf_b.F = np.array([[1.]])
kf_b.H = np.array([[1.]])
kf_b.x = np.array([[pres['pressure'].iloc[0]]])
kf_b.P = np.array([[1.]])
kf_b.Q = np.array([[0.0001]])
kf_b.R = np.array([[0.01]])

kf_baro = []
for p in pres['pressure']:
    kf_b.predict()
    kf_b.update(np.array([[p]]))
    kf_baro.append(kf_b.x[0, 0])
kf_baro = np.array(kf_baro)

walk_duration = acc['time'].max()
cadence       = n_steps / (walk_duration / 60)

print(f"    Steps: {n_steps} | Distance (KF): {kf_pos[-1]:.1f} m")

# ═══════════════════════════════════════════════════════
# COMBINED DASHBOARD
# ═══════════════════════════════════════════════════════
print("\n[Dashboard] Building combined figure...")

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor('#0d1117')
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# ── Main path plot ──
ax_map = fig.add_subplot(gs[0:2, 0:2])
ax_map.set_facecolor('#161b22')

ax_map.plot(gps_clean['x'], gps_clean['y'], 'o',
            color='#444', markersize=4, alpha=0.5, label='Raw GPS fixes')

points     = np.array([ekf_x, ekf_y]).T.reshape(-1, 1, 2)
segments   = np.concatenate([points[:-1], points[1:]], axis=1)
speeds_norm = (speeds - speeds.min()) / (speeds.max() - speeds.min() + 1e-6)
lc = LineCollection(segments, cmap='plasma', linewidth=4, alpha=0.95)
lc.set_array(speeds_norm)
ax_map.add_collection(lc)

ax_map.plot(ekf_x[0],  ekf_y[0],  'o', color='#00ff88', markersize=16, zorder=10, label='Start')
ax_map.plot(ekf_x[-1], ekf_y[-1], 's', color='#ff4444', markersize=16, zorder=10, label='End')

mid = len(ekf_x) // 2
ax_map.annotate('', xy=(ekf_x[mid+2], ekf_y[mid+2]),
                xytext=(ekf_x[mid], ekf_y[mid]),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))

ax_map.set_xlim(ekf_x.min()-8, ekf_x.max()+8)
ax_map.set_ylim(ekf_y.min()-8, ekf_y.max()+8)
ax_map.set_title('EKF Path Reconstruction\n(GPS + Gyroscope)',
                 color='white', fontweight='bold', fontsize=12)
ax_map.set_xlabel('East (m)', color='white')
ax_map.set_ylabel('North (m)', color='white')
ax_map.tick_params(colors='white')
ax_map.set_aspect('equal')
ax_map.legend(facecolor='#0d1117', labelcolor='white', fontsize=9, loc='upper right')
for sp in ax_map.spines.values(): sp.set_edgecolor('#30363d')

cbar = plt.colorbar(lc, ax=ax_map, fraction=0.03, pad=0.04)
cbar.set_label('Speed →', color='white', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

# ── Stats cards ──
def stat_card(ax, label, value, unit, color):
    ax.set_facecolor('#161b22')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, 0.72, value, ha='center', va='center',
            fontsize=28, fontweight='bold', color=color)
    ax.text(0.5, 0.35, unit,  ha='center', va='center',
            fontsize=11, color='#8b949e')
    ax.text(0.5, 0.10, label, ha='center', va='center',
            fontsize=10, color='#8b949e')
    for sp in ax.spines.values():
        sp.set_edgecolor(color); sp.set_linewidth(2)

ax_s1 = fig.add_subplot(gs[0, 2])
ax_s2 = fig.add_subplot(gs[0, 3])
ax_s3 = fig.add_subplot(gs[1, 2])
ax_s4 = fig.add_subplot(gs[1, 3])

stat_card(ax_s1, 'STEPS DETECTED', str(n_steps),            'steps',      '#00ff88')
stat_card(ax_s2, 'KF DISTANCE',    f'{kf_pos[-1]:.0f}',     'meters',     '#00aaff')
stat_card(ax_s3, 'DURATION',       f'{walk_duration:.0f}',  'seconds',    '#ffaa00')
stat_card(ax_s4, 'CADENCE',        f'{cadence:.0f}',        'steps/min',  '#ff6644')

# ── Step detection plot ──
ax_steps = fig.add_subplot(gs[2, 0:2])
ax_steps.set_facecolor('#161b22')
ax_steps.plot(acc['time'], acc['mag_filt'],
              color='#4488ff', linewidth=1.0, alpha=0.8, label='Filtered Acc.')
ax_steps.plot(acc['time'].iloc[peaks], acc['mag_filt'].iloc[peaks],
              'v', color='#ff6644', markersize=6,
              label=f'{n_steps} Steps Detected', zorder=5)
ax_steps.set_title('Step Detection (Accelerometer)', color='white', fontweight='bold')
ax_steps.set_xlabel('Time (s)', color='white')
ax_steps.set_ylabel('Acceleration (m/s²)', color='white')
ax_steps.tick_params(colors='white')
ax_steps.legend(facecolor='#0d1117', labelcolor='white', fontsize=9)
for sp in ax_steps.spines.values(): sp.set_edgecolor('#30363d')

# ── KF position estimate plot ──
ax_kf = fig.add_subplot(gs[2, 2])
ax_kf.set_facecolor('#161b22')
ax_kf.step(step_times, raw_pos, where='post',
           color='#ffaa00', linewidth=1.2, linestyle='--', alpha=0.7, label='Raw')
ax_kf.step(step_times, kf_pos, where='post',
           color='#00ff88', linewidth=2.0, label='KF Estimate')
ax_kf.axhline(y=STRAVA_DIST_M, color='#ff4444', linestyle=':',
              linewidth=1.5, label=f'Strava ({STRAVA_DIST_M:.0f}m)')
ax_kf.set_title('KF Distance Estimate', color='white', fontweight='bold')
ax_kf.set_xlabel('Time (s)', color='white')
ax_kf.set_ylabel('Distance (m)', color='white')
ax_kf.tick_params(colors='white')
ax_kf.legend(facecolor='#0d1117', labelcolor='white', fontsize=8)
for sp in ax_kf.spines.values(): sp.set_edgecolor('#30363d')

# ── Barometer plot ──
ax_baro = fig.add_subplot(gs[2, 3])
ax_baro.set_facecolor('#161b22')
ax_baro.plot(pres['time'], pres['pressure'],
             color='#ff88aa', linewidth=0.8, alpha=0.6, label='Raw')
ax_baro.plot(pres['time'], kf_baro,
             color='#ff2266', linewidth=2.0, label='KF Smoothed')
ax_baro.set_title('Barometer (KF Filtered)', color='white', fontweight='bold')
ax_baro.set_xlabel('Time (s)', color='white')
ax_baro.set_ylabel('Pressure (hPa)', color='white')
ax_baro.tick_params(colors='white')
ax_baro.legend(facecolor='#0d1117', labelcolor='white', fontsize=8)
for sp in ax_baro.spines.values(): sp.set_edgecolor('#30363d')

plt.suptitle('MEEN 700 — Robotic Perception  |  Indoor Localization & Path Reconstruction',
             color='white', fontsize=15, fontweight='bold', y=1.01)

plt.savefig('combined_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')

print("\n[Dashboard] Saved: combined_dashboard.png")
print(f"\n{'='*60}")
print(f"  FINAL COMBINED RESULTS")
print(f"{'='*60}")
print(f"  ── PATH (GPS + Gyroscope + EKF) ──")
print(f"  Raw GPS Distance    : {gps_dist:.1f} m")
print(f"  EKF Path Distance   : {ekf_dist:.1f} m")
print(f"  Strava Reference    : {STRAVA_DIST_M:.1f} m")
print(f"")
print(f"  ── STEPS (Accelerometer + Barometer + KF) ──")
print(f"  Steps Detected      : {n_steps}")
print(f"  KF Distance         : {kf_pos[-1]:.1f} m")
print(f"  Walk Duration       : {walk_duration:.0f} s")
print(f"  Cadence             : {cadence:.0f} steps/min")
print(f"{'='*60}")

plt.show()
