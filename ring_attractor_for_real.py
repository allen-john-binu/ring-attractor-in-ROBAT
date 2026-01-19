import numpy as np
from random import seed as py_seed
import math
import utils
import copy
import csv

values = [10, 20, 30, 40, 50]

with open("fromring.csv", mode="w", newline="") as file:
    writer = csv.writer(file)


    seed = 100 
    np.random.seed(seed)
    py_seed(seed)

    filename = "two_target.csv"
    data = np.loadtxt(filename, delimiter=",")

    array1 = data[:, 101]
    array2 = data[:, 102]

# do one thing, first check if the robot able to hear the other sound, then wait for us to get two values and we start a counter when it is greater than some threshold we hear only one sound

    class RA():
        def __init__(self):
            self.Ns = 100
            self.v = 0.5 # so far best .065
            self.h_0 = 0.051
            self.h_b = 0.0122
            self.v_0 = 1
            self.h_ext = None
            self.beta = 4000
            self.sigma_ang = 5*np.pi / self.Ns
            self.thetas = np.linspace(-np.pi, np.pi, self.Ns, endpoint=False)
            self.spins = np.random.choice([1,0], size=self.Ns)
            self.pos = np.zeros(2)
            self.allocentric = False
            self.heading = 0
            self.updates_per_step = int(round(self.Ns * 4))

    ring = RA()
    L = len(array1)


    pos = np.zeros((L, 2))
    bump_angles = np.zeros(L)
    spins_history = np.zeros((L, ring.Ns), dtype=int)
    heading = 0.0

    # for target tracking
    distance_between_target = np.zeros(L)

    # for animation
    hext_for_time = []
    distance_between_target = np.zeros(L)

    for t in range (L):
        thetas = []
        theta_diff1 = (math.radians(array1[t])-heading) % (2 * np.pi) 
        thetas.append(theta_diff1)
        if(array2[t] < 200):
            theta_diff2 = (math.radians(array2[t])-heading) % (2 * np.pi)
            thetas.append(theta_diff2)

        utils.compute_h_ext_multiple(ring,thetas) #check this line

        #for animation
        hext_for_time.append(ring.h_ext)
            
        for _ in range (ring.updates_per_step):
            i = np.random.randint(0,ring.Ns)
            delta_H = utils.compute_delta_H(ring,i)

            if delta_H < 0:
                # Energy difference negative — always accept
                ring.spins[i] = 1 - ring.spins[i]
            else:
                # Energy difference positive — accept with probability exp(-beta * delta_H)
                p = np.exp(-ring.beta * delta_H)
                if np.random.rand() < p:
                    ring.spins[i] = 1 - ring.spins[i]
                    
        active_indices = np.where(ring.spins == 1)[0]
        n_active = len(active_indices)
        if n_active > 0:
            phi = np.angle(np.sum(np.exp(1j * ring.thetas[ring.spins == 1])))
        else:
            phi = 0.0  
        bump_angles[t] = math.degrees(phi)
        writer.writerow([bump_angles[t]])

        print(f'Time step {t+1}/{L}, Position: {pos[t]}, theta_targets: {[math.degrees(entho) for entho in thetas]} degrees, Bump angle: {bump_angles[t]:.2f} degrees, num_step: {ring.updates_per_step}')

        
        if ring.allocentric:
            # allocentric motion
            if t < L-1:
                pos[t+1, 0] = pos[t, 0] + ring.v_0 * np.cos(phi)/ring.Ns
                pos[t+1, 1] = pos[t, 1] + ring.v_0 * np.sin(phi)/ring.Ns
        else:  
            # egocentric motion
            heading = (phi + heading) % (2 * np.pi) 
            if t < L-1:
                pos[t+1, 0] = pos[t, 0] + ring.v_0 * np.cos(heading)/ring.Ns
                pos[t+1, 1] = pos[t, 1] + ring.v_0 * np.sin(heading)/ring.Ns

        spins_history[t] = copy.deepcopy(ring.spins)

    print("Simulation complete. ")

    utils.plot_summary_with_multiple_targets(ring,pos,spins_history,bump_angles,[],L)