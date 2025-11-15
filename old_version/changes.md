- thetas spacing
    - thetas = np.arange(Ns) * diff_neurons_angle - np.pi 
    - thetas = np.linspace(-np.pi, np.pi, Ns, endpoint=False)
    -> The two methods produce the same theta values (except for tiny floating-point rounding differences).

- circular distance function
    - def circ_dist(a, b):
        d = np.abs(a - b)
        return np.minimum(d, 2*np.pi - d)
    - (thetas - theta_diff + np.pi) % (2 * np.pi) - np.pi
    -> Yes, they differ: They solve related but different problems.
        - circ_dist(a, b) → returns a non-negative minimal angular separation.
        - (a - b + π) % (2π) − π → returns a signed angular difference in (−π, π].

- step calculation [MAYBE]
    - pos[t+1, 0] = pos[t, 0] + v0 * np.cos(phi)
      pos[t+1, 1] = pos[t, 1] + v0 * np.sin(phi)
    - pos[t+1, 0] = pos[t, 0] + v0 * np.cos(phi)/Ns
      pos[t+1, 1] = pos[t, 1] + v0 * np.sin(phi)/Ns
    -> Tiny movement: step divided by Ns

- spin randomness increased [NO]
