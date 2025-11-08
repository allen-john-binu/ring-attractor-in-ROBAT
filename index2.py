import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(1)

Ns = 100
L = 500     
v0 = 10.0              
sigma_ang = 2*np.pi / Ns
h0 = 0.0025
hb = 0.0

target_pos = np.array([500.0, 500.0])  


# for each neuron group what is the angles ????
thetas = np.arange(Ns) * sigma_ang

# So without wrapping, neurons on opposite ends of the ring look maximally far apart, breaking the ring’s continuity
def circ_dist(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 2*np.pi - d)

# synaptic connectivity of the network, J with parameter v
v = 0.5  # shape parameter
def compute_J(theta_i):
    dist = circ_dist(thetas[:, None], theta_i)
    dist = dist.squeeze()
    return np.cos((np.pi * ((dist / np.pi) ** v)))

# list of beta values to sweep (low -> high order)
beta_list = [400.0] #1.0,5.0,

# number of attempted spin updates per motion step
t0 = 50
updates_per_step = int(round(Ns * t0))

# wrap to [0,2*pi]
def wrap_pi(x):
    return (x % (2*np.pi))

# compute Hamiltonian for the system
def compute_H(spins, h_ext, hb, Ns,i):
    j_curr = compute_J(thetas[i])
    quad =(1.0 / Ns) * np.dot(j_curr, spins) * spins[i]
    linear = ((h_ext[i] * spins[i]) - (hb * spins[i]))
    H = - (quad + linear)
    return H

def compute_h_ext(mode,pos_now,heading_ego):
    # --- Compute dynamic external field depending on mode ---
    if mode == 'allocentric':
        # absolute direction from agent to target
        dx, dy = target_pos - pos_now
        theta_target = np.arctan2(dy, dx)
        if(theta_target < 0):
            theta_target = theta_target + 2*np.pi
        difference_target = circ_dist(thetas, theta_target)
        hext = (h0 / np.sqrt(2 * np.pi * sigma_ang**2)) * np.exp(-((difference_target) ** 2) / (2 * sigma_ang**2))
        return hext
    else:
        # relative direction from current heading to target
        dx, dy = target_pos - pos_now
        theta_abs = np.arctan2(dy, dx)
        theta_target = theta_abs - heading_ego
        return (h0 / np.sqrt(2 * np.pi * sigma_ang**2)) * np.exp(-((thetas - theta_target) ** 2) / (2 * sigma_ang**2))

def run_sim(beta, mode='allocentric'):
    # initial random spins
    spins = np.random.choice([-1, 1], size=(Ns,))
    spins_history = np.zeros((L, Ns), dtype=int)

    pop_angles = np.zeros(L)

    # agent init
    pos = np.zeros((L, 2))
    heading_ego = np.random.uniform(0, 2*np.pi) 
    heading_alloc = 0.0

    x0, y0 = 0.0, 0.0
    pos[0] = np.array([x0, y0])

    for t in range(L):
        h_ext = compute_h_ext(mode,pos[t],heading_ego)
        if t == 0:
            spins_proposed = spins.copy()
        else:
            spins_proposed = spins_history[t-1].copy()
        deltaH_list = []
        n_accepted = 0
        acc_uphill = 0
        acc_downhill = 0
        for _ in range(updates_per_step):
            spins_current = spins_proposed.copy()
            i = np.random.randint(Ns)  
            # compute current energy
            H_before = compute_H(spins_current, h_ext, hb, Ns,i)

            # flip of spin i
            spins_current[i] = -spins_current[i]

            # compute new energy
            H_after = compute_H(spins_current, h_ext, hb, Ns,i)

            # delta H = H(after) - H(before)
            delta_H = H_after - H_before
            deltaH_list.append(delta_H)
            if delta_H < 0:
                spins_proposed[i] = -spins_proposed[i] # Energy difference negative then flip
                n_accepted += 1
                acc_downhill += 1 
            else:
                # Energy difference positive — accept with probability exp(-beta * delta_H)
                p = np.exp(-beta * delta_H)
                # safety clamp
                if p > 1.0:
                    p = 1.0
                if np.random.rand() < p:
                    spins_proposed[i] = -spins_proposed[i]
                    n_accepted += 1
                    acc_uphill += 1

        spins_history[t] = spins_proposed.copy()
        print("Accepted:", n_accepted," -  Downhill accepted:", acc_downhill, " -   Uphill accepted:", acc_uphill, " -   Delta_H mean: ",np.mean(deltaH_list))

        active = spins_history[t] > 0
        n_active = np.count_nonzero(active)
        if n_active > 0:
            phi = (thetas[active].sum())/n_active
            thethaDegree = math.degrees(phi)
            print("At ", t, " ,Angle phi (degrees):", thethaDegree, " with n_active:", n_active )
        else:
            phi = 0.0  
        pop_angles[t] = phi

        if mode == 'egocentric':
            # egocentric motion
            heading_ego = wrap_pi(phi + heading_ego)
            if t < L-1:
                pos[t+1, 0] = pos[t, 0] + v0 * np.cos(heading_ego)
                pos[t+1, 1] = pos[t, 1] + v0 * np.sin(heading_ego)
        else:
            # allocentric motion
            heading_alloc = phi
            if t < L-1:
                pos[t+1, 0] = pos[t, 0] + v0 * np.cos(heading_alloc)
                pos[t+1, 1] = pos[t, 1] + v0 * np.sin(heading_alloc)

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
