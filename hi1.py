import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load CSV file
# -----------------------------
# Change filename as needed
filename = "one_target.csv"
filename = "two_target.csv"

data = np.genfromtxt(filename, delimiter=",", dtype=float, filling_values=-1)

# -----------------------------
# Extract columns
# -----------------------------
# Columns 1–100: spins_history
spins_history = data[:, 0:100]

# Column 101: bump_angles
bump_angles = data[:, 100]

angles_we = data[:, 101]

# Number of time steps
L = spins_history.shape[0]

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 1, figsize=(10, 6))

# Network activity as a function of time
ax_raster = axes

im = ax_raster.imshow(
    spins_history.T,
    aspect='auto',
    origin='lower',
    cmap='gray',
    interpolation='nearest',
    extent=[0, L, -180, 180]
)

ax_raster.set_title("network spins over time")
ax_raster.set_ylabel("unit index")
ax_raster.set_xlabel("time step")

# Time axis
time_axis = np.arange(L)

# Overlay bump angles
ax_raster.plot(time_axis, bump_angles[0:L], color='red', linewidth=2, label='bump angle')

# detected angles
# ax_raster.plot(time_axis, angles_we[0:L], color='blue', linewidth=2, label='detected angle')

# Colorbar (legend)
cbar = plt.colorbar(im, ax=ax_raster, ticks=[0, 1])
cbar.ax.set_yticklabels(['0 (black)', '1 (white)'])
cbar.set_label("Spin value")

ax_raster.legend()

plt.tight_layout()
plt.show()
