import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def allan_variance(omega, t0, max_clusters=100):
    #set up log space
    N = len(omega)
    n_values = np.unique(np.logspace(0, np.log10(N//2), max_clusters).astype(int))
    T = n_values * t0
    allan_var = np.zeros(len(n_values))

    for i, n in enumerate(n_values):
        num_clusters = N//n
        
        #average gyro output over each cluster
        omega_bar = []
        for k in range(num_clusters):
            cluster_sum = 0
            for j in range(n):
                cluster_sum += omega[k*n +j]
            omega_bar.append(cluster_sum/n)

        #allan variance
        total = 0
        for k in range(len(omega_bar) -1):
            omega_bar_next = omega_bar[k+1]
            omega_bar_k = omega_bar[k]
            total += (omega_bar_next - omega_bar_k)**2
        allan_var[i] = 0.5*total/(len(omega_bar) -1)

    return T, allan_var


# Load IMU data
rows = pd.read_csv("allan_static_log.csv")
gz_raw = rows["gz"].values.astype(float)

t0 = 1.0 /50 # 50 Hz

T, allan_var = allan_variance(gz_raw, t0)
sigma_T = np.sqrt(allan_var)

plt.loglog(T, sigma_T)
plt.xlabel(r"$\tau$ (s)")
plt.ylabel(r"$\sigma(\tau)$ [rad/s]")
plt.title(r"$g_z$")
plt.grid(True, which="both")

#get value at t=1
index = np.argmin(np.abs(T - 1.0))
N_gz = sigma_T[index]
print(f"N_gz = {N_gz:.6f}")

plt.savefig("allan_variance.png")
