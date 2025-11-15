import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import time
from matplotlib.animation import FuncAnimation
from random import randrange, seed as py_seed

seed = 100 
np.random.seed(seed)
py_seed(seed)


Ns = 100
L = 100
v0 = 600
sigma_ang = 5*np.pi / Ns
diff_neurons_angle = 2*np.pi / Ns
h0 = 0.051
hb = 0.0122

target_pos = np.array([-50.0, -50.0])  

# for each neuron group what is the angles 
thetas = np.linspace(-np.pi, np.pi, Ns, endpoint=False)

# list of beta values to sweep (low -> high order)
beta_list = [400.0] #1.0,5.0,

# number of attempted spin updates per motion step
t0 = 5
updates_per_step = int(round(Ns * t0))

# Compute dynamic external field depending on i
def compute_h_ext(theta_diff):
    difference_target = (thetas - theta_diff + np.pi) % (2 * np.pi) - np.pi
    return (h0 / np.sqrt(2 * np.pi * (sigma_ang**2))) * np.exp(- (difference_target ** 2) / (2 * (sigma_ang**2) ) )

# synaptic connectivity of the network, J with parameter v
v = 0.5  # shape parameter
def compute_J(i):
    kernel = None
    alpha_i = copy.deepcopy(thetas[i])
    angle_diffs = np.abs((copy.deepcopy(thetas) - alpha_i + np.pi) % (2 * np.pi) - np.pi)
    kernel = np.cos(np.pi * ((angle_diffs / np.pi) ** v))
    return kernel

# compute Hamiltonian for the system
def compute_delta_H(spins, h_ext, i):
    H = np.zeros(2)
    computing_spins = copy.deepcopy(spins)
    j_curr = compute_J(i)
    j_curr[i] = 0  # zero out self-connection

    interaction_sum = np.dot(j_curr, spins)
   
    spin_flip = np.array([spins[i], 1-spins[i]])

    interaction_energy = (interaction_sum * spin_flip) / (Ns - 1)

    input_energy = h_ext[i] * spin_flip
    leak_energy = hb * spin_flip
    H = -1 * (interaction_energy + input_energy - leak_energy)
    return (H[1] - H[0])

def euclidean_distance(p1, p2):
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

hext_for_time = []
bump_for_time = []
target_for_time = []
distance_between_target = []    

def run_sim(beta, mode='allocentric'):
    # initial random spins with equal number of 1 and 0
    spins = np.random.choice([1,0], size=Ns)

    spins_history = np.zeros((L, Ns), dtype=int)
    pop_angles = np.zeros(L)

    # agent init
    pos = np.zeros((L, 2))
    heading = 0.0
    pos[0] = np.array([0.0, 0.0])

    for t in range(L):
        theta_target = np.arctan2( target_pos[1] - pos[t][1],target_pos[0] - pos[t][0])
        distance_between_target.append(euclidean_distance(pos[t], target_pos))
        if mode == 'egocentric':
            theta_diff = (theta_target-heading) % (2 * np.pi) 
        else:
            theta_diff = theta_target

        deltaH_list = []
        n_accepted = 0
        acc_uphill = 0
        acc_downhill = 0

        h_ext = compute_h_ext(theta_diff)

        hext_for_time.append(h_ext)
        target_for_time.append(theta_target)
        for _ in range(updates_per_step):
            i = np.random.randint(0,Ns) 
            delta_H = compute_delta_H(spins, h_ext, i)
            deltaH_list.append(delta_H)

            if delta_H < 0:
                # Energy difference negative — always accept
                spins[i] = 1 - spins[i]

                n_accepted += 1
                acc_downhill += 1 
            else:
                # Energy difference positive — accept with probability exp(-beta * delta_H)
                p = np.exp(-beta * delta_H)
                if np.random.rand() < p:
                    spins[i] = 1 - spins[i]

                    n_accepted += 1
                    acc_uphill += 1 

        active_indices = np.where(spins == 1)[0]
        n_active = len(active_indices)
        if n_active > 0:
            phi = np.angle(np.sum(np.exp(1j * thetas[spins == 1])))
            thethaDegree = math.degrees(phi)
            print("At ", t, " ,Angle phi (degrees):", thethaDegree, " with n_active:", n_active," -- Accepted:", n_accepted,"-  Downhill accepted:", acc_downhill, " -   Uphill accepted:", acc_uphill, " -   Delta_H mean: ",np.mean(deltaH_list))
        else:
            phi = 0.0  
        pop_angles[t] = phi
        bump_for_time.append(phi)

        if mode == 'egocentric':
            # egocentric motion
            heading = (phi + heading) % (2 * np.pi) 
            if t < L-1:
                pos[t+1, 0] = pos[t, 0] + v0 * np.cos(heading)/Ns
                pos[t+1, 1] = pos[t, 1] + v0 * np.sin(heading)/Ns
        else:
            # allocentric motion
            if t < L-1:
                pos[t+1, 0] = pos[t, 0] + v0 * np.cos(phi)/Ns
                pos[t+1, 1] = pos[t, 1] + v0 * np.sin(phi)/Ns

        spins_history[t] = copy.deepcopy(spins)

    return {
        "spins_history": spins_history,
        "pop_angles": pop_angles,
        "pos": pos
    }

results_alloc = [run_sim(beta, mode='allocentric') for beta in beta_list]

# results_alloc = [run_sim(beta, mode='allocentric') for beta in beta_list]
# results_ego   = [run_sim(beta, mode='egocentric')  for beta in beta_list]

# # Plotting - I took it from chatGPT the plotting code 
# fig = plt.figure(figsize=(16, 9))
# n_beta = len(beta_list)

# for i, beta in enumerate(beta_list):
#     res_ego = results_ego[i]

#     # (i) network activity
#     ax_raster = fig.add_subplot(n_beta, 3, i*3 + 1)
#     ax_raster.imshow(res_ego["spins_history"].T, aspect='auto', origin='lower',
#                      cmap='gray_r', interpolation='nearest', extent=[0, L, 0, Ns])
#     ax_raster.set_title(f"(A{i+1}) β={beta:.2f} — network activity ego")
#     ax_raster.set_xlabel("time")
#     ax_raster.set_ylabel("unit index")

#     # (ii) egocentric motion
#     ax_ego = fig.add_subplot(n_beta, 3, i*3 + 2)
#     posE = res_ego["pos"]
#     ax_ego.plot(posE[:, 0], posE[:, 1])
#     ax_ego.scatter(posE[0, 0], posE[0, 1], color='green', s=20, label='start')
#     ax_ego.scatter(posE[-1, 0], posE[-1, 1], color='red', s=20, label='end')
#     ax_ego.scatter(*target_pos, color='black', marker='x', s=40, label='target')
#     ax_ego.set_title(f"(B{i+1}) Egocentric, β={beta:.2f}")
#     ax_ego.set_aspect('equal')
#     ax_ego.legend()


# plt.tight_layout()
# plt.show()

fig = plt.figure(figsize=(16, 9))
n_beta = len(beta_list)

for i, beta in enumerate(beta_list):
    res_alloc = results_alloc[i]

    # (iii) network activity
    ax_raster = fig.add_subplot(n_beta, 3, i*3 + 1)
    ax_raster.imshow(res_alloc["spins_history"].T, aspect='auto', origin='lower',
                     cmap='gray_r', interpolation='nearest', extent=[0, L, 0, Ns])
    ax_raster.set_title(f"(A{i+1}) β={beta:.2f} — network activity alloc")
    ax_raster.set_xlabel("time")
    ax_raster.set_ylabel("unit index")

    # (iv) allocentric motion
    ax_alloc = fig.add_subplot(n_beta, 3, i*3 + 2)
    posA = res_alloc["pos"]
    ax_alloc.plot(posA[:, 0], posA[:, 1])
    ax_alloc.scatter(posA[0, 0], posA[0, 1], color='green', s=20, label='start')
    ax_alloc.scatter(posA[-1, 0], posA[-1, 1], color='red', s=20, label='end')
    ax_alloc.scatter(*target_pos, color='black', marker='x', s=40, label='target')
    ax_alloc.set_title(f"(C{i+1}) Allocentric, β={beta:.2f}")
    ax_alloc.set_aspect('equal')
    # ax_alloc.legend()

plt.tight_layout()
plt.show()

plt.plot(range(100), distance_between_target)
plt.title("distance_between_target")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()

hext_for_time = np.array(hext_for_time)

# If hext_for_time accidentally came in as (N_s,) for a single timepoint, force a 2D shape
if hext_for_time.ndim == 1:
    hext_for_time = hext_for_time[np.newaxis, :]

T, N_s = hext_for_time.shape

# convert radians -> degrees only if inputs are in radians; if already degrees skip np.degrees
alpha_deg = np.degrees(thetas)
target_angles = np.degrees(target_for_time)
bump_angles = np.degrees(bump_for_time)

# (optional) ensure arrays have expected length
assert alpha_deg.shape[0] == N_s, "alpha_deg length must match number of sensors N_s"
assert target_angles.shape[0] == T, "target_for_time length must match time T"
assert bump_angles.shape[0] == T, "bump_for_time length must match time T"

fig, ax = plt.subplots(figsize=(8, 4))

# width chosen so bars tile the circle portion; for a linear plot this is fine
width = 360.0 / N_s
bar_container = ax.bar(alpha_deg, hext_for_time[0], width=width, color='skyblue')

# Create vertical lines; axvline returns a Line2D with two x-points internally
target_line = ax.axvline(x=target_angles[0], color='orange', linestyle='--', label='Target angle')
bump_line = ax.axvline(x=bump_angles[0], color='green', linestyle='-', label='Bump angle')

ax.set_xlim(-180, 180)
ax.set_ylim(0, np.max(hext_for_time) * 1.1)
ax.set_xlabel("Angle (°)")
ax.set_ylabel("h_i (External Input)")
ax.set_title("External Input h_i vs. Angle Over Time")
ax.legend()

def update(frame):
    # update bar heights
    for rect, h in zip(bar_container, hext_for_time[frame]):
        rect.set_height(h)

    # set_xdata must receive a sequence; use two identical points for a vertical line
    x_target = target_angles[frame]
    x_bump = bump_angles[frame]
    target_line.set_xdata([x_target, x_target])
    bump_line.set_xdata([x_bump, x_bump])

    ax.set_title(f"External Input h_i (t={frame})")
    # DO NOT use time.sleep() here — FuncAnimation controls the timing via 'interval'
    return list(bar_container) + [target_line, bump_line]

# blit=True can be faster, but requires returning artists correctly; keep blit=False if unsure
ani = FuncAnimation(fig, update, frames=T, blit=False, interval=200)  # interval in ms
plt.tight_layout()
plt.show()
