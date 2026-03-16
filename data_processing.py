import csv
import math
import numpy as np
import pandas as pd

TPR = 12 * 90 * 4
r = 0.0325
meters_per_tick = (2 * np.pi * r) / TPR
min_omega = 0.2


rows = pd.read_csv("raw_log.csv")
T = len(rows)

ts = rows["t_ns_pi"].values.astype(float) * 1e-9
bl = rows["BL"].values.astype(float)
fl = rows["FL"].values.astype(float)
fr = rows["FR"].values.astype(float)
br = rows["BR"].values.astype(float)
gz_raw = rows["gz"].values.astype(float)

total = 0.0
N = 0

robot_dt = np.zeros(T)
v_enc = np.zeros(T)
w_enc = np.zeros(T)

#Effective wheel base calibration 
for k in range(1, T):
    dt = ts[k] - ts[k - 1]
    robot_dt[k] = dt

    dBL = bl[k] - bl[k - 1]
    dFL = fl[k] - fl[k - 1]
    dFR = fr[k] - fr[k - 1]
    dBR = br[k] - br[k - 1]

    dL = (dBL + dFL) / 2.0
    dR = (dBR + dFR) / 2.0

    vL = (dL * meters_per_tick) / dt
    vR = (dR * meters_per_tick) / dt

    v_enc[k] = (vR + vL) / 2.0
    w_enc[k] = vR - vL   # not divided by b_eff yet

    omega_m = gz_raw[k]
    if abs(omega_m) < min_omega:
        continue

    total += abs(vL - vR) / abs(omega_m)
    N += 1

b_eff = total / N


w_enc = w_enc / b_eff

#Check if the robot is moving 
moving = np.abs(v_enc) > 0.02
w_error = w_enc[moving] - gz_raw[moving]
sigma_enc_omega = np.std(w_error)

#The real distance the robot drove after being told to drive +1 meter x
tape_y = np.array([0.02, -0.01, 0.03, -0.02, 0.01, -0.03, 0.025, -0.015, 0.02, -0.01])
tape_x = np.array([0.843, 0.845, 0.841, 0.844, 0.842, 0.846, 0.843, 0.844, 0.841, 0.845])

errors_x = np.ones(10) - tape_x
errors_y = tape_y

sigma_px_per_m = np.std(errors_x)
sigma_py_per_m = np.std(errors_y)

print(f"sigma_px = {sigma_px_per_m} m/m")
print(f"sigma_py = {sigma_py_per_m} m/m")
print(f"b_eff = {b_eff:.4f} m")