import argparse
import time
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import datetime
from numba import njit

@njit
def _glauber_loop(sigma, beta, h, n, max_iter, check_interval, tol, window):
    '''
    This helper function executes the glauber algorithm to output a physically consistent spin configuration for a given set of temperature and external magnetic field values.
    The algorithm consists of:
    1. Start at a random configuration (say all up spin - does not matter)
    2. At the ith step pick a uniformly random site
    3. Resample the site using the gibbs measure under the condition that the rest of the lattice is fixed,
       i.e. flip to +1 with probability 1 / (1 + exp(-2 * beta * delta)) where delta is the local field
       (sum of the 4 neighbours plus h) - this is just the conditional Gibbs distribution for a single spin
    4. Repeat for max_iter steps, periodically checking the energy to see if we've reached equilibrium

    Energy here is computed inline (not via self.energy()) since this is a plain njit function and can't call
    methods on the Ising class - has to be a vectorized numpy version instead.
    '''
    energy_history = []
    for i in range(max_iter):

        # Picking a uniform random site
        x = np.random.randint(1, n-1)
        y = np.random.randint(1, n-1)
        neighbours = sigma[x+1,y] + sigma[x,y+1] + sigma[x-1,y] + sigma[x,y-1]
        delta = neighbours + h
        measure = 1 / (1 + np.exp(-2 * beta * delta))
        if np.random.random() < measure:
            sigma[x, y] = 1
        else:
            sigma[x, y] = -1

        if (i + 1) % check_interval == 0:
            # vectorized energy inline since we can't call self.energy() here
            h_field = -h * np.sum(sigma)
            h_neighbors = -np.sum(sigma[:-1, :] * sigma[1:, :]) \
                          -np.sum(sigma[:, :-1] * sigma[:, 1:])
            e = (h_field + h_neighbors) / n ** 2
            energy_history.append(e)

            # An expression to check if the algorithm should stop.
            if len(energy_history) >= window:
                half = window // 2
                recent = energy_history[-half:]
                older = energy_history[-window:-half]
                mean_recent = sum(recent) / len(recent)
                mean_older = sum(older) / len(older)
                if abs(mean_recent - mean_older) < tol:
                    break
    return sigma

class Ising:
    def __init__(self, h, beta, n):
        self.sigma = np.ones((n,n))
        self.sigma[n//2:, :] = -1 # Initial condition: half the lattice starts up, half down
        self.h = h
        self.beta = beta
        self.n = n
        self.up_spin_num = (n//2) * n
        self.down_spin_num = (n - n//2) * n

    def energy(self):
        '''
        This function used to be how energy was computed, but was too slow, nevertheless it is easier to see how energy is computed with this.

        '''
        H = 0
        for i in range(self.n):
            for j in range(self.n):
                H -= self.h * self.sigma[i, j]
                if j < self.n - 1:
                    H -= self.sigma[i, j] * self.sigma[i, j + 1]
                if i < self.n - 1:
                    H -= self.sigma[i, j] * self.sigma[i + 1, j]
        return H

    def generate(self, base_coeff=500, tol=0.01, window=20):
        '''
        This function generates a consistent spin configuration with variable beta, with the help of the njit powered helper function.

        '''
        beta_c = np.log(1 + np.sqrt(2)) / 2
        distance = max(abs(self.beta - beta_c), 0.05)
        max_iter = int(np.floor(self.n * np.log(self.n) * base_coeff / distance))
        check_interval = self.n ** 2
        self.sigma = _glauber_loop(
            self.sigma, self.beta, self.h, self.n,
            max_iter, check_interval, tol, window
        )

    def above_criticality(self):
        '''
        Using the analytically verified critical temperature, the function outputs whether the spin configuration happens to be above or below the phase transition.

        '''
        beta_c = np.log(1 + np.sqrt(2)) / 2
        return self.beta < beta_c

    def visualize(self):
        '''
        Just a visualization function for the spins.

        '''
        plt.imshow(self.sigma, cmap='bwr', vmin=-1, vmax=1)
        plt.colorbar(label='Spin')
        plt.title('Ising Model')
        plt.show()

    def save(self, directory="data"):
        '''
        Just a function to save the spin configuration.

        '''

        phase = "above" if self.above_criticality() else "below"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        np.savez(f"{directory}/{self.beta}_{self.h}_{phase}_{timestamp}", sigma=self.sigma, phase=phase)


if __name__ == "__main__":
    # Generating Ising visualizations with user-defined temperature and external magnetic field.
    parser = argparse.ArgumentParser(description="Generate and visualize a single Ising configuration.")
    parser.add_argument("--beta", type=float, default=0.46, help="Inverse temperature (default: 0.46)")
    parser.add_argument("--h", type=float, default=0.0, help="External field (default: 0.0)")
    parser.add_argument("-n", type=int, default=200, help="Lattice size n x n (default: 200)")
    args = parser.parse_args()

    ising = Ising(args.h, args.beta, args.n)
    ising.generate()
    ising.visualize()



