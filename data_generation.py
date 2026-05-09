import ising
import numpy as np
from tqdm import tqdm

n = 200
h = 0

betas = np.concatenate([
    np.linspace(0.001, 0.35, 80),
    np.linspace(0.35,  0.53, 140),
    np.linspace(0.53,  0.8,  80),
])

for beta in tqdm(betas, desc="Generating"):
    model = ising.Ising(h, beta, n)
    model.generate()
    model.save()



