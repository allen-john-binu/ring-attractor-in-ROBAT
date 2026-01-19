import numpy as np
import matplotlib.pyplot as plt
import math

# ---------------------------
# USER INPUTS
# ---------------------------
# filename = "one_target.csv"
filename = "two_target.csv"


data = np.loadtxt(filename, delimiter=",")

filename2 = "fromring.csv"


data2 = np.loadtxt(filename2, delimiter=",")

# -----------------------------
# Extract columns
# -----------------------------
# Columns 1–100: spins_history
bump_angles = data[:, 100]
# Turn angles at each timestep (radians)
phi = bump_angles

bump_angles2 = data2
# Turn angles at each timestep (radians)
phi2 = bump_angles2

# Initial position
x0, y0 = 0.0, 0.0

# Initial heading (radians)
heading0 = 0.0

# Constant speed
v0 = 10.0

# Number of substeps per unit time (as in your code)
Ns = 50

# ---------------------------
# PATH SIMULATION
# ---------------------------

L = len(phi)
pos = np.zeros((L + 1, 2))
pos[0] = [x0, y0]
pos2 = np.zeros((L + 1, 2))
pos2[0] = [x0, y0]

heading = heading0
heading2 = heading0

for t in range(L):
    # Update heading

    heading = (math.radians(phi[t]) + heading) % (2 * np.pi)

    # Update position
    pos[t + 1, 0] = pos[t, 0] + v0 * np.cos(heading) / 100
    pos[t + 1, 1] = pos[t, 1] + v0 * np.sin(heading) / 100

    heading2 = (math.radians(phi2[t]) + heading2) % (2 * np.pi)

    # Update position
    pos2[t + 1, 0] = pos2[t, 0] + v0 * np.cos(heading2) / 100
    pos2[t + 1, 1] = pos2[t, 1] + v0 * np.sin(heading2) / 100

# ---------------------------
# PLOTTING
# ---------------------------

plt.figure()
plt.plot(pos[:, 0], pos[:, 1], marker='o')
plt.plot(pos2[:, 0], pos2[:, 1], marker='x')

plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Robot Path")
plt.axis("equal")
plt.grid(True)
plt.show()
