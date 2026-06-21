# Ising Model

## Problem Statement

Given G = (V, E), with nodes corresponding to spins σ = {-1, +1}, there exists a critical temperature Tc where a phase transition takes place — that is, a discontinuity in the derivative of the partition function. Below Tc the system exhibits long-range magnetic order (ferromagnetic phase), while above Tc thermal fluctuations dominate and the system becomes disordered (paramagnetic phase). The 2D Ising model is one of the simplest statistical mechanics models that admits an exact analytical solution for Tc, making it a canonical benchmark for numerical methods.

Direct Markov Chain Monte Carlo (MCMC) sampling becomes computationally expensive near Tc, due to critical slowing down, where correlation lengths diverge and the chain mixes slowly. In this repository three machine learning approaches are investigated, each aimed at either characterizing the phase transition without costly simulation, or bypassing the simulation altogether.

## Approach

Sample configurations are generated with Glauber dynamics (the heat-bath algorithm): at each step a lattice site is resampled directly from its equilibrium conditional distribution given its neighbours, namely P(σᵢ = +1 | neighbours) = 1 / (1 + exp(-2β f)), where f is the local field (the sum of the four neighbouring spins, plus the external field h) and β = 1/kT. This generation procedure, implemented in `ising.py` and swept over a (β, h) grid in `data_generation.py`, underlies all three phases below.

### Phase 1 — Unsupervised Phase Detection

A feature vector (magnetization, energy, and the correlation function C(r) at several r) is computed for every h = 0 configuration, embedded with UMAP, and clustered with a two-component Gaussian Mixture Model, namely without any access to the true β labels. Two clusters are recovered below Tc, corresponding to the positive- and negative-magnetization Gibbs states, collapsing to a single cluster above Tc; the transition in cluster assignment occurs almost exactly at the analytically known βc ≈ 0.4407, indicating that spontaneous symmetry breaking is recovered from the configurations alone.

### Phase 2 — Conditional Generation

A conditional VAE is trained to map (β, h) directly to a spin configuration, bypassing Glauber dynamics at generation time. The encoder and decoder are conditioned on (β, h) by concatenation, and the reconstruction term uses a per-pixel binary cross-entropy, due to the ±1 nature of the spin data. Training converges without overfitting, with train and validation loss tracking each other closely over 150 epochs.

Validating generated configurations against real MCMC samples at matching (β, h), it is found that magnetization and energy are reproduced well, but the spatial correlation C(r) and domain wall density are systematically underestimated, to roughly 27–38% of the real magnitude, across all tested β regimes. Two explanations were considered and ruled out in turn, namely model capacity (doubling the bottleneck channel count barely changes the gap) and training data imbalance (the same gap is observed whether β is densely or sparsely represented in the training set). The remaining, and most likely, explanation is a property of the loss function itself: per-pixel binary cross-entropy rewards getting each pixel's own marginal probability right, but does not directly reward neighbouring pixels covarying correctly, such that a model can drive the loss low while still under-representing correlation length. This is taken as the headline finding of Phase 2, rather than as an unresolved defect.

### Phase 3 — Feature Importance

An XGBoost regressor is trained to predict β directly from the physical feature vector (magnetization, |magnetization|, energy, C(r) at r = {1, 2, 5, 10, 20}, and domain wall density), achieving a train RMSE of 0.0017 and a test RMSE of 0.0076 over a β range of roughly 0.001–0.8. SHAP analysis indicates that energy and C(1) jointly dominate the prediction across the whole β range, near Tc and deep in the ordered phase alike, with magnetization comparatively redundant once energy and C(1) are known — to be expected, given that the two are nearly the same physical quantity at h = 0.

## Repository Structure

| File | Description |
|---|---|
| `ising.py` | `Ising` class and the Glauber-dynamics (heat-bath) update loop (`_glauber_loop`) that generates a single spin configuration for a given (β, h). Run directly to generate and visualize one configuration. |
| `data_generation.py` | Sweeps β (denser near βc) and h, generating configurations via `ising.py` and writing them to `data/non_zero_h`. |
| `data_extraction.py` | Loads the generated `.npz` files into the `dataset` list consumed by every downstream script. |
| `mgibbs.py` | Physical feature extractors (`energy`, `C`, `features`, `domain_wall_density`); run directly for Phase 1 (UMAP embedding + GMM clustering). |
| `augment.py` | `random_symmetry`: applies an exact symmetry of the Ising Hamiltonian (lattice rotation/reflection, and global spin flip when h = 0) to a configuration, used as data augmentation for the CVAE. |
| `vae.py` | Phase 2 conditional VAE: encoder/decoder architecture, training loop, and per-run logging to `experiments_log.csv`. |
| `vae_validation.py` | Generates samples from a trained checkpoint at a chosen (β, h) and compares their physical features against real configurations. |
| `decision_tree.py` | Phase 3: XGBoost regression of β from physical features, with SHAP analysis. |
| `TODO.md` | Live phase/task checklist. |

## Usage

The project depends on `numpy`, `numba`, `matplotlib`, `tqdm`, `torch`, `scikit-learn`, `umap-learn`, `xgboost`, and `shap`, installable via pip:

```bash
pip install numpy numba matplotlib tqdm torch scikit-learn umap-learn xgboost shap
```

Below the commands required to reproduce each phase are given. Data generation has to be run first, as every other script depends on its output.

```bash
# Generate the dataset (β × h grid, 10 repeats per point) into data/non_zero_h
python data_generation.py

# Phase 1: UMAP embedding + GMM clustering
python mgibbs.py

# Phase 2: train the conditional VAE (checkpoints to models/, log to experiments_log.csv)
python vae.py
# then, after pointing model_path at a checkpoint, validate against real physics
python vae_validation.py

# Phase 3: XGBoost regression + SHAP analysis
python decision_tree.py
```

A pre-generated mirror of the dataset is additionally hosted at [huggingface.co/datasets/zalboss/2dIsingData](https://huggingface.co/datasets/zalboss/2dIsingData), as local generation of the full grid takes a non-trivial amount of time near βc.
