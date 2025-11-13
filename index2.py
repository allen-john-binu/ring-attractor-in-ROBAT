import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import time
from matplotlib.animation import FuncAnimation

np.random.seed(1)

Ns = 100
L = 100   
v0 = 6.0              
sigma_ang = 5*np.pi / Ns
diff_neurons_angle = 2*np.pi / Ns
h0 = 0.0051
hb = 0.00122

target_pos = np.array([50.0, 0.0])  

# for each neuron group what is the angles 
thetas = np.arange(Ns) * diff_neurons_angle - np.pi 

# list of beta values to sweep (low -> high order)
beta_list = [400.0] #1.0,5.0,

# number of attempted spin updates per motion step
t0 = 10
updates_per_step = int(round(Ns * t0))

def circ_dist(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 2*np.pi - d)

# Compute dynamic external field depending on i
def compute_h_ext_for_i(theta_diff):
    difference_target = circ_dist(thetas, theta_diff)
    return (h0 / np.sqrt(2 * np.pi * (sigma_ang**2))) * np.exp(- (difference_target ** 2) / (2 * (sigma_ang**2) ) )

# synaptic connectivity of the network, J with parameter v
v = 0.5  # shape parameter
def compute_J(i):
    considering_alhpa_i = copy.deepcopy(thetas[i])
    considering_alphas = copy.deepcopy(thetas)
    
    angle_diffs = np.abs((considering_alphas - considering_alhpa_i + np.pi) % (2 * np.pi) - np.pi)
    angle_diffs = angle_diffs.squeeze()
    normalized = angle_diffs / np.pi 
    powered = normalized ** v
    return np.cos((np.pi * powered))

dist = circ_dist(thetas[:, None], thetas[None, :])
J_full = np.cos(np.pi * (dist / np.pi) ** v)

# print("J_full shape:", J_full.shape)

# J_for_test = compute_J(25)
# print(J_for_test.size)
# plt.plot(range(100), J_for_test)
# plt.title("J value with neuron at index 50")
# plt.xlabel("Index")
# plt.ylabel("Value")
# plt.show()

# compute Hamiltonian for the system
def compute_delta_H(spins, h_ext, i):
    computing_spins = copy.deepcopy(spins)
    j_curr = compute_J(i)
    j_curr[i] = 0  # zero out self-connection

    spins_energy = np.dot(j_curr, computing_spins)
    before_spin = copy.deepcopy(computing_spins[i])
    after_spin = 1 - before_spin

    external_energy_before = (h_ext[i] * before_spin) - (hb * before_spin)
    spins_energy_before = (1.0 / (Ns-1)) * spins_energy * before_spin
    H_before = - (spins_energy_before + external_energy_before)

    external_energy_after = (h_ext[i] * after_spin) - (hb * after_spin)
    spins_energy_after = (1.0 / (Ns-1)) * spins_energy * after_spin
    H_after = - (spins_energy_after + external_energy_after)

    return (H_after - H_before)

def compute_total_H(spins, h_ext):
    computing_spins = copy.deepcopy(spins)
    temp = np.dot((J_full @ computing_spins), computing_spins)
    external_energy = np.dot(h_ext, computing_spins)
    return - ((1.0/Ns)*temp - external_energy)

hext_for_time = []
bump_for_time = []
target_for_time = []

def run_sim(beta, mode='allocentric'):
    # initial random spins with equal number of -1 and 1
    spins = np.concatenate((np.ones(Ns // 2, dtype=int), np.zeros(Ns // 2, dtype=int)))
    np.random.shuffle(spins)
    spins_history = np.zeros((L, Ns), dtype=int)
    pop_angles = np.zeros(L)

    # agent init
    pos = np.zeros((L, 2))
    heading = 0.0
    pos[0] = np.array([0.0, 0.0])
    total_H = []

    for t in range(L):
        theta_target = np.arctan2( target_pos[1] - pos[t][1],target_pos[0] - pos[t][0])
        if mode == 'egocentric':
            theta_diff = (theta_target-heading) % (2 * np.pi) 
        else:
            theta_diff = theta_target

        deltaH_list = []
        n_accepted = 0
        acc_uphill = 0
        acc_downhill = 0

        h_ext = compute_h_ext_for_i(theta_diff)
        hext_for_time.append(h_ext)
        target_for_time.append(theta_target)
        total_H.append(compute_total_H(spins, h_ext))
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

        # plt.plot(np.degrees(thetas), h_ext)
        # plt.title("h_ext for neurons with target")
        # plt.xlabel("Index")
        # plt.ylabel("Value")
        # plt.show()

        active_indices = np.where(spins == 1)[0]
        n_active = len(active_indices)
        if n_active > 0:
            phi = np.angle(np.sum(np.exp(1j * thetas[active_indices])))
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
                pos[t+1, 0] = pos[t, 0] + v0 * np.cos(heading)
                pos[t+1, 1] = pos[t, 1] + v0 * np.sin(heading)
        else:
            # allocentric motion
            if t < L-1:
                pos[t+1, 0] = pos[t, 0] + v0 * np.cos(phi)
                pos[t+1, 1] = pos[t, 1] + v0 * np.sin(phi)

        spins_history[t] = copy.deepcopy(spins)

    plt.plot(range(L), total_H)
    plt.title("total_H for neurons")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.show()
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

hext_for_time = []
bump_for_time = []
target_for_time = []

T, N_s = hext_for_time.shape
alpha_deg = np.degrees(thetas) 
target_angles = np.degrees(target_for_time) 
bump_angles = np.degrees(bump_for_time)
fig, ax = plt.subplots(figsize=(8, 4))
bar_container = ax.bar(alpha_deg, hext_for_time[0], width=360 / N_s, color='skyblue')
target_line = ax.axvline(x=0, color='orange', linestyle='--', label='Target angle')
bump_line = ax.axvline(x=0, color='green', linestyle='-', label='Bump angle')

ax.set_xlim(-180, 180)
ax.set_ylim(0, np.max(hext_for_time) * 1.1)
ax.set_xlabel("Angle (°)")
ax.set_ylabel("h_i (External Input)")
ax.set_title("External Input h_i vs. Angle Over Time")

def update(frame):
    for rect, h in zip(bar_container, hext_for_time[frame]):
        rect.set_height(h)
    target_line.set_xdata(target_angles[frame])
    bump_line.set_xdata(bump_angles[frame])
    ax.set_title(f"External Input h_i (t={frame})")
    time.sleep(0.2)
    return list(bar_container) + [target_line] + [bump_line]

ani = FuncAnimation(fig, update, frames=T, blit=False, interval=50)
plt.tight_layout()
plt.show()
#functionAnimation