import time
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import datetime
from numba import njit

@njit
def _metropolis_loop(sigma, beta, h, n, max_iter, check_interval, tol, window):
    energy_history = []
    for i in range(max_iter):
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
            # vectorized energy inline since we cant call self.energy() here
            h_field = -h * np.sum(sigma)
            h_neighbors = -np.sum(sigma[:-1, :] * sigma[1:, :]) \
                          -np.sum(sigma[:, :-1] * sigma[:, 1:])
            e = (h_field + h_neighbors) / n ** 2
            energy_history.append(e)
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
        self.sigma[n//2:, :] = -1
        self.h = h
        self.beta = beta
        self.n = n
        self.up_spin_num = (n//2) * n
        self.down_spin_num = (n - n//2) * n

    def energy(self):
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
        beta_c = np.log(1 + np.sqrt(2)) / 2
        distance = max(abs(self.beta - beta_c), 0.05)
        max_iter = int(np.floor(self.n * np.log(self.n) * base_coeff / distance))
        check_interval = self.n ** 2
        self.sigma = _metropolis_loop(
            self.sigma, self.beta, self.h, self.n,
            max_iter, check_interval, tol, window
        )

    def above_criticality(self):
        beta_c = np.log(1 + np.sqrt(2)) / 2
        return self.beta < beta_c

    def visualize(self):
        plt.imshow(self.sigma, cmap='bwr', vmin=-1, vmax=1)
        plt.colorbar(label='Spin')
        plt.title('Ising Model')
        plt.show()

    def save(self, directory="data"):
        phase = "above" if self.above_criticality() else "below"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        np.savez(f"{directory}/{self.beta}_{self.h}_{phase}_{timestamp}", sigma=self.sigma, phase=phase)


#ising = Ising(0,0.3,200)
#ising.generate()
#ising.visualize()



