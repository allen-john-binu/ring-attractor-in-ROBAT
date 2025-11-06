import numpy as np
import math
import matplotlib.pyplot as plt

np.random.seed(1)

Ns = 100
L = 1000              
v0 = 10.0              
sigma_ang = 2*np.pi / Ns
h0 = 0.0
hb = 0.0


# for each neuron group what is the angles ????
thetas = np.arange(Ns) * sigma_ang

# So without wrapping, neurons on opposite ends of the ring look maximally far apart, breaking the ring’s continuity
def circ_dist(a, b):
    d = np.abs(a - b)
    return np.minimum(d, 2*np.pi - d)

# synaptic connectivity of the network, J with parameter v
v = 0.5  # shape parameter
dist = circ_dist(thetas[:, None], thetas[None, :])
J = np.cos(np.pi * (dist / np.pi) ** v)

# external field
theta_target = 0.0 
h_ext = (h0 / np.sqrt(2 * np.pi * sigma_ang**2)) * np.exp(-((thetas - theta_target) ** 2 ) / (2 * sigma_ang**2))

# list of beta values to sweep (low -> high order)
beta_list = [200.0] #1.0,5.0,

# number of attempted spin updates per motion step
t0 = 0.5
updates_per_step = int(round(Ns * t0))

# wrap to [-pi,pi]
def wrap_pi(x):
    return (x % (2*np.pi))

# compute Hamiltonian for the system
def compute_H(spins, J, h_ext, hb, Ns):
    quad = (1.0 / Ns) * (spins @ (J @ spins))
    linear = ((h_ext - hb) * spins).sum()
    H = - (quad + linear)
    return H

def run_sim(beta):
    # initial random spins
    spins = np.random.choice([-1, 1], size=(Ns,))
    spins_history = np.zeros((L, Ns), dtype=int)

    pop_angles = np.zeros(L)

    # agent init
    pos_alloc = np.zeros((L, 2))
    pos_ego = np.zeros((L, 2))
    heading_ego = 0.0
    heading_alloc = 0.0

    x0, y0 = 0.0, 0.0
    pos_alloc[0] = np.array([x0, y0])
    pos_ego[0] = np.array([x0, y0])

    for t in range(L):
        for _ in range(updates_per_step):
            i = np.random.randint(Ns)    
            # compute current energy
            H_before = compute_H(spins, J, h_ext, hb, Ns)

            # flip of spin i
            spins_proposed = spins.copy()
            spins_proposed[i] = -spins_proposed[i]

            # compute new energy
            H_after = compute_H(spins_proposed, J, h_ext, hb, Ns)

            # delta H = H(after) - H(before)
            delta_H = H_after - H_before
            if delta_H < 0:
                spins[i] = -spins[i] # Energy difference negative then flip
            else:
                # Energy difference positive — accept with probability exp(-beta * delta_H)
                if np.random.rand() < np.exp(-beta * delta_H):
                    spins[i] = -spins[i]

        spins_history[t] = spins.copy()

        active = spins > 0
        n_active = np.count_nonzero(active)
        if n_active > 0:
            temp = (thetas[active].sum())/n_active
            phi = temp - np.pi
            thethaDegree = math.degrees(phi)
            print("Angle (degrees):", thethaDegree)
        else:
            phi = 0.0  
        pop_angles[t] = phi

        # allocentric motion
        heading_alloc = phi
        if t < L-1:
            pos_alloc[t+1, 0] = pos_alloc[t, 0] + v0 * np.cos(heading_alloc)
            pos_alloc[t+1, 1] = pos_alloc[t, 1] + v0 * np.sin(heading_alloc)

        # egocentric motion
        heading_ego = wrap_pi(phi + heading_ego)
        if t < L-1:
            pos_ego[t+1, 0] = pos_ego[t, 0] + v0 * np.cos(heading_ego)
            pos_ego[t+1, 1] = pos_ego[t, 1] + v0 * np.sin(heading_ego)

    return {
        "spins_history": spins_history,
        "pop_angles": pop_angles,
        "pos_alloc": pos_alloc,
        "pos_ego": pos_ego
    }

results = []
for beta in beta_list:
    results.append(run_sim(beta))

# Plotting - I took it from chatGPT the plotting code 
fig = plt.figure(figsize=(16, 9))
n_beta = len(beta_list)
for i, beta in enumerate(beta_list):
    res = results[i]

    # Panel (i): raster plot (network activity as a function of time)
    ax_raster = fig.add_subplot(3, 3, i*3 + 1)
    # show spins_history as image: rows=time, cols=units; show +1 as white, -1 as black
    im = ax_raster.imshow(res["spins_history"].T, aspect='auto', origin='lower',
                          cmap='gray_r', interpolation='nearest', extent=[0, L, 0, Ns])
    ax_raster.set_title(f"(A{i+1}) beta={beta:.2f}  — network spins over time")
    ax_raster.set_ylabel("unit index")
    ax_raster.set_xlabel("time step")

    # Panel (ii): egocentric trajectory
    ax_ego = fig.add_subplot(3, 3, i*3 + 2)
    pos = res["pos_ego"]
    ax_ego.plot(pos[:, 0], pos[:, 1], linewidth=1)
    ax_ego.scatter(pos[0, 0], pos[0, 1], color='green', s=20, label='start')
    ax_ego.scatter(pos[-1, 0], pos[-1, 1], color='red', s=20, label='end')
    ax_ego.set_title(f"(B{i+1}) Egocentric (beta={beta:.2f})")
    ax_ego.set_aspect('equal')
    ax_ego.set_xlabel("x")
    ax_ego.set_ylabel("y")
    ax_ego.legend()

    # Panel (iii): allocentric trajectory
    ax_alloc = fig.add_subplot(3, 3, i*3 + 3)
    pos = res["pos_alloc"]
    ax_alloc.plot(pos[:, 0], pos[:, 1], linewidth=1)
    ax_alloc.scatter(pos[0, 0], pos[0, 1], color='green', s=20, label='start')
    ax_alloc.scatter(pos[-1, 0], pos[-1, 1], color='red', s=20, label='end')
    ax_alloc.set_title(f"(C{i+1}) Allocentric (beta={beta:.2f})")
    ax_alloc.set_aspect('equal')
    ax_alloc.set_xlabel("x")
    ax_alloc.set_ylabel("y")
    ax_alloc.legend()

plt.tight_layout()
plt.show()

