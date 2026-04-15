# Ising Model


## Problem Statement
Given G = (V, E), with nodes corresponding to spins sigma = {-1, +1}, there exists a critical temperature Tc where a phase transition takes place — that is, a discontinuity in the derivative of the partition function. Below Tc the system exhibits long-range magnetic order (ferromagnetic phase), while above Tc thermal fluctuations dominate and the system becomes disordered (paramagnetic phase). The 2D Ising model is one of the simplest statistical mechanics models that admits an exact analytical solution for Tc, making it a canonical benchmark for numerical methods.


## Solution

The phase transition can be modeled by a Markov chain Monte Carlo (MCMC) method: a random walk beginning at some lattice site where, applying the Metropolis criterion, the spin is flipped with a probability determined by the resulting energy change. If the energy decreases the change is always accepted; if not, it is accepted with probability P = exp(-beta * dE), where beta = 1/kT. Near Tc this algorithm becomes computationally expensive due to critical slowing down, where correlation lengths diverge and the chain mixes slowly. To address this, we use a binary classification model trained on MCMC-generated spin configurations to predict whether the system is in the low-temperature (T < Tc) or high-temperature (T > Tc) phase, bypassing the need for costly simulation near the critical point.






