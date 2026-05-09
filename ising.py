import time
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import datetime


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
        energy_history = []

        for i in range(max_iter):
            x = rd.randint(1, self.n-2)
            y = rd.randint(1, self.n-2)

            neighbours = [self.sigma[x+1, y], self.sigma[x, y+1], self.sigma[x-1, y], self.sigma[x, y-1]]
            number_of_up_neighbours = np.sum([1 for s in neighbours if s == 1])
            number_of_down_neighbours = np.sum([1 for s in neighbours if s == -1])

            delta = number_of_up_neighbours - number_of_down_neighbours + self.h
            measure = 1 / (1 + np.exp(-2 * self.beta * delta))

            uniform = rd.random()

            if uniform < measure:
                if self.sigma[x,y] == -1:
                    self.up_spin_num += 1
                    self.down_spin_num -= 1
                self.sigma[x,y] = 1
            else:
                if self.sigma[x,y] == 1:
                    self.up_spin_num -= 1
                    self.down_spin_num += 1
                self.sigma[x,y] = -1

            if (i + 1) % check_interval == 0:
                e = self.energy() / self.n ** 2
                energy_history.append(e)
                if len(energy_history) >= window:
                    half = window // 2
                    if abs(np.mean(energy_history[-window:-half]) - np.mean(energy_history[-half:])) < tol:
                        break
                
                
    def above_criticality(self):
        beta_c = np.log(1 + np.sqrt(2)) / 2
        return self.beta < beta_c



    def visualize(self):
        plt.imshow(self.sigma, cmap='bwr', vmin=-1, vmax=1)
        plt.colorbar(label='Spin')
        plt.title('Ising Model')
        plt.show()

    def save(self):
        phase = "above" if self.above_criticality() else "below"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        np.savez(f"data/{self.beta}_{self.h}_{phase}_{timestamp}", sigma=self.sigma, phase=phase)



#ising = Ising(0, 0.4447, 150)
#ising.generate()
#ising.visualize()



