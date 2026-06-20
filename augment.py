import numpy as np

def random_symmetry(sigma, h):
    sigma = np.rot90(sigma, np.random.randint(4))
    if np.random.rand() < 0.5:
        sigma = np.fliplr(sigma)
    if h == 0 and np.random.rand() < 0.5:
        sigma = -sigma
    return np.ascontiguousarray(sigma)
