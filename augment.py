import numpy as np

def random_symmetry(sigma, h):
    '''
    This function applies a random symmetry of the Ising model to a spin configuration, used to augment the
    training data (free extra samples that are physically just as valid as the original, since the energy/
    Hamiltonian doesn't care which one we picked).
    The transformations applied are:
    1. A random rotation by 0, 90, 180 or 270 degrees - the square lattice is invariant under these,
       so the energy is unchanged
    2. With 50% chance, a left-right flip - same reasoning, reflection is also a lattice symmetry
    3. If h == 0, with 50% chance flip every spin (sigma -> -sigma) - this is the global Z2 symmetry of the
       Ising model, but it's only an actual symmetry when there's no external field, since the field term
       -h * sum(sigma) flips sign under sigma -> -sigma and is no longer the same configuration energetically.
       So we only do this when h == 0, otherwise we'd be feeding the network physically inconsistent data.

    Returns a contiguous copy at the end since rot90/fliplr give back strided views, not actual contiguous
    arrays, which can cause issues downstream.
    '''
    sigma = np.rot90(sigma, np.random.randint(4))
    if np.random.rand() < 0.5:
        sigma = np.fliplr(sigma) # mirror array left - to - right
    if h == 0 and np.random.rand() < 0.5:
        sigma = -sigma
    return np.ascontiguousarray(sigma) # forces actual copy
