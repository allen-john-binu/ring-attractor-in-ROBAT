import numpy as np
import matplotlib.pyplot as plt
import utils

thetas = np.linspace(-np.pi, np.pi, 100, endpoint=False)

# Plot
plt.figure()
plt.plot(thetas)
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Converted String to Float Plot")
plt.show()
