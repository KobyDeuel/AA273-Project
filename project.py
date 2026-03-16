
import csv, math, numpy as np, matplotlib.pyplot as plt, pandas as pd
from matplotlib.patches import Ellipse
import scipy

#wrap angle 
def wrap(theta): 
    return (theta + math.pi) % (2 * math.pi) - math.pi

#Noise values from data_processing.py  and allan_analysis.py
N_gz = 0.000257           
sigma_enc_omega = 0.049868   
sigma_px_per_m = 0.0016       
sigma_py_per_m = 0.0206       

#Load robot data
rows = list(csv.DictReader(open("raw_log.csv")))
T = len(rows)
ts = np.array([float(r['t_ns_pi']) * 1e-9 for r in rows])
bl = np.array([int(r['BL']) for r in rows])
br = np.array([int(r['BR']) for r in rows])
fl = np.array([int(r['FL']) for r in rows])
fr = np.array([int(r['FR']) for r in rows])
gz_raw = np.array([float(r['gz']) for r in rows])


time = ts - ts[0] #make time start at 0
robot_dt = np.zeros(T)
v_enc = np.zeros(T)
w_enc = np.zeros(T)

# Physical characteristics 
TPR = 12*90*4 #Ticks per rotation
r = 0.0325
b_eff = 0.3565

for k in range(1, T):
    robot_dt[k] = ts[k] - ts[k - 1]
    dt = robot_dt[k]

    #Number of ticks per time division 
    dBL = bl[k]-bl[k-1]
    dFL = fl[k]-fl[k-1]
    dFR = fr[k]-fr[k-1]
    dBR = br[k]-br[k-1]

    #Averaging right and left
    dL = (dFL + dBL)/2
    dR = (dFR + dBR)/2

    # Right and left velocities 
    vL = (dL / dt) * (2 * math.pi * r)/TPR
    vR = (dR / dt) * (2 *math.pi*r)/TPR

    #forward velocity and rotation rates
    v_enc[k] = (vR + vL)/2
    w_enc[k] = (vR - vL)/b_eff

#nonlinear state transition function
def f(x, u_t, dt):
    px, py, th, w = x
    v_e, gz = u_t
    return np.array([
        px + dt * v_e * math.cos(th),
        py + dt * v_e * math.sin(th),
        wrap(th + dt * w),
        gz
    ])
#Jacobian of f
def get_A(x, u, dt):
    px, py, th, w = x
    v_e, gz = u
    return np.array([
        [1, 0, -dt * v_e * math.sin(th), 0],
        [0, 1,  dt * v_e * math.cos(th), 0],
        [0, 0, 1, dt],
        [0, 0, 0, 0]
    ])

#Measurement function 
def g(x): 
    return np.array([x[3]])

#Jacobian of g
def get_C(x): 
    return np.array([[0, 0, 0, 1]])

#Measurement noise 
R = np.array([[sigma_enc_omega**2]])

mu = np.zeros((T, 4))
Sigma = np.zeros((T, 4, 4))
Sigma[0] = np.diag([0.01, 0.01, 0.01, 0.01])


for t in range(T - 1):
    dt = robot_dt[t + 1]
    u_t = np.array([v_enc[t + 1], gz_raw[t + 1] ])
    y = np.array([w_enc[t + 1]])
    
    # Time varying process 
    Q = np.diag([
        (sigma_px_per_m *  abs(v_enc[t] * dt))**2,
        (sigma_py_per_m *  abs(v_enc[t] * dt))**2,
        (N_gz * np.sqrt(dt))**2,
        (N_gz * (1/np.sqrt(dt)))**2
    ])

    # Prediction step 
    mu_pred = f(mu[t], u_t, dt)
    A = get_A(mu[t], u_t, dt)
    Sigma_pred = A @ Sigma[t] @ A.T + Q

    # Update
    C = get_C(mu_pred)
    K = Sigma_pred@C.T@np.linalg.inv(C@Sigma_pred@C.T + R)
    mu[t+1] = mu_pred + K@(y - g(mu_pred))
    mu[t + 1, 2] = wrap(mu[t + 1, 2]) # Wrap the heading angle {-180°, 180°}
    Sigma[t + 1] = (np.eye(4) - K @ C) @ Sigma_pred


#Encoder dead reckoning
enc_path = np.zeros((T, 3))
for t in range(1, T):
    dt = robot_dt[t]
    th = enc_path[t - 1, 2]
    enc_path[t, 0] = enc_path[t - 1, 0] + dt * v_enc[t] * math.cos(th)
    enc_path[t, 1] = enc_path[t - 1, 1] + dt * v_enc[t] * math.sin(th)
    enc_path[t, 2] = enc_path[t - 1, 2] + dt * w_enc[t]


#load cam data
cam = pd.read_csv("tracking.csv")
cam_t = cam["timestamp"].values.copy()
cam_x_px = cam["x_px"].values.copy()
cam_y_px = cam["y_px"].values.copy()
cam_th_raw = cam["heading_deg"].values.copy()

#convert pixels to meters
pixel_per_m = 90 / 0.12
cam_x_m = cam_x_px / pixel_per_m
cam_y_m = -1*(cam_y_px / pixel_per_m) #The camera y needs to be flipped 

#Unwrapping the angle makes plotting smoother so it is not jumping around
cam_th_unwrap = np.degrees(np.unwrap(np.radians(cam_th_raw)))
#angle needs to be flipped 
cam_th_unwrap = -cam_th_unwrap

#Set the camera's starting point to the origin (0,0)
cam_x_m -= cam_x_m[0]
cam_y_m -= cam_y_m[0]

angle_rad = np.radians(-cam_th_unwrap[0])
#Rotate
cam_x = cam_x_m * np.cos(angle_rad) - cam_y_m * np.sin(angle_rad)
cam_y = cam_x_m * np.sin(angle_rad) + cam_y_m * np.cos(angle_rad)
cam_th = cam_th_unwrap - cam_th_unwrap[0] # Angle offset to get EKF and camera track in the same orientation

plt.figure()

##95% Confidence ellipse

P = 0.95                    
k = -2 * np.log(1 - P) 
for t in range(0,T,300):
        Sigma_pos = Sigma[t][:2, :2]
        mu_pos = mu[t][:2]
        ellipse = []
        for theta in np.linspace(0, 2*np.pi, 1000):
            w =np.array([np.sqrt(k)*np.cos(theta),np.sqrt(k)*np.sin(theta)])
            ellipse.append(scipy.linalg.sqrtm(Sigma_pos)@w + mu_pos)

        ellipse = np.array(ellipse)
        plt.plot(-ellipse[:,1], ellipse[:,0], 'r', zorder=3)
        plt.scatter(-mu_pos[1], mu_pos[0], c='r' , s=1, zorder=3)

# XY Trajectory
#I had to rotate the the plot counterclockwise so that the orientation matched the video 
plt.plot(-cam_y, cam_x, 'g-', lw=2, alpha=0.6, label="Camera")
plt.plot(-mu[:, 1], mu[:, 0], 'b-', lw=2.5, label="EKF")
plt.plot(-enc_path[:, 1], enc_path[:, 0], 'k--', lw=1.5, alpha=0.7, label="Encoder")
plt.plot(-mu[0, 1], mu[0, 0], 'ko', ms=8, label="Start")
plt.xlabel("x (m)"); 
plt.ylabel("y (m)")
plt.title("XY Trajectory")
plt.axis("equal")
plt.grid('true')
plt.savefig("xy_traj.png")


# Heading
plt.figure()
ekf_th = np.degrees(np.unwrap(mu[:, 2]))
enc_th = np.degrees(np.unwrap(enc_path[:, 2]))
confidence = np.degrees(1.96 * np.sqrt(Sigma[:, 2, 2]))

plt.plot(time, ekf_th, 'b-', lw=2, label="EKF")
plt.plot(time, enc_th, 'k--', lw=1.5, alpha=0.7, label="Encoder")
plt.plot(cam_t + 1.5, cam_th, 'g-', lw=2, alpha=0.7, label="Camera")
plt.fill_between(time, ekf_th - confidence, ekf_th + confidence, alpha=0.2, label="95% bounds",color='blue')
plt.xlabel("time (s)")
plt.ylabel("deg")
plt.title("Heading")
plt.grid('true')
plt.savefig("heading.png")


conf_x = 1.96 * np.sqrt(Sigma[:, 0, 0])  


plt.figure()
plt.plot(time, mu[:, 0], 'b-', lw=2, label="EKF $p_x$")
plt.plot(cam_t+ 1.5, cam_x, 'g', lw=2, label="Cam $p_y$")
plt.plot(time, enc_path[:, 0], 'k--', lw=1.5, alpha=0.7, label="Encoder")
plt.fill_between(time, mu[:,0] - conf_x, mu[:,0] + conf_x, alpha=0.2, color='blue', label="95% bounds")
plt.xlabel("time (s)")
plt.ylabel("m")
plt.title("X Position")
# plt.legend()
plt.savefig("x_pos.png")

conf_y = 1.96 * np.sqrt(Sigma[:, 1, 1])
plt.figure()
plt.plot(time, mu[:, 1], 'b-', lw=2, label="EKF $p_y$")
plt.plot(cam_t+ 1.5, cam_y, 'g', lw=2, label="Cam $p_y$")
plt.plot(time, enc_path[:, 1], 'k--', lw=1.5, alpha=0.7, label="Encoder")
plt.fill_between(time, mu[:,1] - conf_y, mu[:,1] + conf_y, alpha=0.2, color='blue', label="95% bounds")

plt.xlabel("time (s)")
plt.ylabel("m")
plt.title("Y Position")
# plt.legend()
plt.savefig("y_pos.png")
