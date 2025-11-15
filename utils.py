import numpy as np
import copy
import matplotlib.pyplot as plt

def compute_h_ext(ring, theta_diff):
    difference_target = (ring.thetas - theta_diff + np.pi) % (2 * np.pi) - np.pi
    ring.h_ext = (ring.h_0 / np.sqrt(2 * np.pi * (ring.sigma_ang**2))) * np.exp(- (difference_target ** 2) / (2 * (ring.sigma_ang**2) ) )

def compute_J(ring,i):
    kernel = None
    alpha_i = copy.deepcopy(ring.thetas[i])
    angle_diffs = np.abs((copy.deepcopy(ring.thetas) - alpha_i + np.pi) % (2 * np.pi) - np.pi)
    kernel = np.cos(np.pi * ((angle_diffs / np.pi) ** ring.v))
    return kernel

def compute_delta_H(ring, i):
    H = np.zeros(2)
    j_curr = compute_J(ring,i)
    j_curr[i] = 0 

    interaction_sum = np.dot(j_curr, ring.spins)
   
    spin_flip = np.array([ring.spins[i], 1-ring.spins[i]])

    interaction_energy = (interaction_sum * spin_flip) / (ring.Ns - 1)

    if ring.h_ext is None:
        input_energy = 0
    else:
        input_energy = ring.h_ext[i] * spin_flip

    leak_energy = ring.h_b * spin_flip
    H = -1 * (interaction_energy + input_energy - leak_energy)
    return (H[1] - H[0])

def plot_summary_no_target(ring,pos_alloc,pos_ego,spins_history,bump_angles,L):
    fig, axes = plt.subplots(1, 3, figsize=(15, 9))

    # network activity as a function of time
    ax_raster = axes[0] 
    im = ax_raster.imshow(spins_history.T, aspect='auto', origin='lower',
                        cmap='gray', interpolation='nearest', extent=[0, L, -180, 180])
    ax_raster.set_title(f" beta={ring.beta:.2f}  — network spins over time")
    ax_raster.set_ylabel("unit index")
    ax_raster.set_xlabel("time step")
    time_axis = np.arange(L) 
    ax_raster.plot(time_axis, bump_angles, color='red', linewidth=2, label='bump angle')

    # Add legend (colorbar)
    cbar = plt.colorbar(im, ax=ax_raster, ticks=[0, 1])
    cbar.ax.set_yticklabels(['0 (black)', '1 (white)'])
    cbar.set_label("Spin value")

    # egocentric trajectory
    ax_ego = axes[1] 
    pos = pos_ego
    ax_ego.plot(pos[:, 0], pos[:, 1], linewidth=1)
    ax_ego.scatter(pos[0, 0], pos[0, 1], color='green', s=20, label='start')
    ax_ego.scatter(pos[-1, 0], pos[-1, 1], color='red', s=20, label='end')
    ax_ego.set_title(f" Egocentric (beta={ring.beta:.2f})")
    ax_ego.set_aspect('equal')
    ax_ego.set_xlabel("x")
    ax_ego.set_ylabel("y")
    ax_ego.legend()

    # allocentric trajectory
    ax_alloc = axes[2] 
    pos = pos_alloc
    ax_alloc.plot(pos[:, 0], pos[:, 1], linewidth=1)
    ax_alloc.scatter(pos[0, 0], pos[0, 1], color='green', s=20, label='start')
    ax_alloc.scatter(pos[-1, 0], pos[-1, 1], color='red', s=20, label='end')
    ax_alloc.set_title(f" Allocentric (beta={ring.beta:.2f})")
    ax_alloc.set_aspect('equal')
    ax_alloc.set_xlabel("x")
    ax_alloc.set_ylabel("y")
    ax_alloc.legend()

    plt.tight_layout()
    plt.show()

def plot_summary_with_target(ring,pos,spins_history,bump_angles,target_pos,L,end_needed=True):
    fig, axes = plt.subplots(1, 2, figsize=(15, 9))

    # network activity as a function of time
    ax_raster = axes[0] 
    im = ax_raster.imshow(spins_history.T, aspect='auto', origin='lower',
                        cmap='gray', interpolation='nearest', extent=[0, L, -180, 180])
    ax_raster.set_title(f" beta={ring.beta:.2f}  — network spins over time")
    ax_raster.set_ylabel("unit index")
    ax_raster.set_xlabel("time step")
    time_axis = np.arange(L) 
    ax_raster.plot(time_axis, bump_angles[0:L], color='red', linewidth=2, label='bump angle')

    # Add legend (colorbar)
    cbar = plt.colorbar(im, ax=ax_raster, ticks=[0, 1])
    cbar.ax.set_yticklabels(['0 (black)', '1 (white)'])
    cbar.set_label("Spin value")

    #  trajectory
    ax_alloc = axes[1] 
    ax_alloc.plot(pos[:L, 0], pos[:L, 1], linewidth=1)
    ax_alloc.scatter(pos[0, 0], pos[0, 1], color='green', s=20, label='start')
    if end_needed:
        ax_alloc.scatter(pos[-1, 0], pos[-1, 1], color='red', s=20, label='end')
    ax_alloc.scatter(*target_pos, color='black', marker='x', s=40, label='target')
    ax_alloc.set_title(f" Allocentric (beta={ring.beta:.2f})")
    ax_alloc.set_aspect('equal')
    ax_alloc.set_xlabel("x")
    ax_alloc.set_ylabel("y")

    plt.tight_layout()
    plt.show()

def plot_any_line(x, y, xlabel, ylabel, title):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.show()