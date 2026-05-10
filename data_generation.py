import ising
import numpy as np
from tqdm import tqdm

n = 200
repeat = 10

betas = np.concatenate([
    np.linspace(0.001, 0.35, 10),
    np.linspace(0.35,  0.53, 60),
    np.linspace(0.53,  0.8,  10),
])

hs = np.linspace(0,1,10)

for beta in tqdm(betas, desc="Generating"):
    for h in tqdm(hs):
        for _ in tqdm(range(repeat)):
            model = ising.Ising(h, beta, n)
            model.generate()
            model.save(directory="data/non_zero_h")



