import numpy as np
import matplotlib.pyplot as plt
import utils

class RA():
    def __init__(self):
        self.Ns = 100
        self.v = 0.5
        self.h_0 = 0.051
        self.h_b = 0.0122
        self.v_0 = 10
        self.h_ext = None
        self.beta = 4000
        self.sigma_ang = 5*np.pi / self.Ns
        self.thetas = np.linspace(-np.pi, np.pi, self.Ns, endpoint=False)
        self.spins = np.random.choice([1,0], size=self.Ns)
        self.pos = np.zeros(2)
        self.allocentric = True
        self.heading = 0
        self.updates_per_step = int(round(self.Ns * 10))

ring = RA()

utils.compute_h_ext_multiple(ring,[-0.785398]) #check this line


# Plot
plt.figure()
plt.plot(ring.h_ext)
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("h_ext Plot")
plt.show()
