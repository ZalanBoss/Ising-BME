# Project TODO

Completed: binary FCN classifier for T < Tc / T > Tc, finite-size (submatrix) investigation, critical beta extraction via sigmoid fit.

---

## Phase 1 — Multiple Gibbs States via Unsupervised Clustering

Goal: detect spontaneous symmetry breaking without any labels. Below Tc two pure Gibbs states (m > 0, m < 0) should appear as two separable clusters; above Tc only one.

- [x] Compute a feature vector per configuration: (m, E, χ=⟨m²⟩−⟨m⟩², C(r) at r=1,2,5,10,20)
- [x] Run UMAP on raw configs (200×200 flattened) colored by β — visual sanity check
- [x] Fit a GMM (k=2) to the UMAP embedding; plot cluster assignment vs β
- [x] Show that the GMM switches from 2 clusters to 1 cluster at β ≈ βc

---

## Phase 2 — Conditional VAE: (β, h) → configuration

Goal: replace MCMC sampling near Tc with a CVAE that, given (β, h), generates a physically plausible spin configuration directly — skipping Glauber dynamics.

- [x] Build VAE skeleton: encoder (Conv → μ, logσ), decoder (deconv → 200×200 spin grid), reparameterization trick
- [x] Dataset/DataLoader wrapping `dataset` dicts, with normalized (β, h) labels
- [x] Wire (β, h) into the encoder (concat to flattened features) and decoder (concat to z) — actual CVAE conditioning
- [x] Switch reconstruction loss from Tanh+MSE to logits + BCE (better fit for ±1 spin data)
- [x] Include `data/non_zero_h` in `data_extraction.py` so h actually varies in training
- [x] Train and check loss curves (train/val split, watch for under/overfitting)
- [x] Validate "physically plausible": compare M, E, χ, C(r) of generated samples vs real MCMC samples at matching (β, h)
- [x] Generate samples across a β sweep at fixed h=0 and visualize the transition

---

## Phase 3 — Decision Trees + SHAP

Goal: use gradient-boosted trees on hand-crafted physical features and recover the known order parameter hierarchy from SHAP values.

- [x] Build feature extractor: m, |m|, E, C(r) at several r, domain wall density (χ/specific heat skipped — ensemble quantities, deemed not essential given C(r)/domain wall density as proxies)
- [x] Train XGBoost to regress β (given h=0 configs); evaluate RMSE, check for overfitting (train vs test) — train RMSE 0.0017, test RMSE 0.0076 (range ~0.001-0.8), mild overfit, no concern
- [x] Compute SHAP values — finding: E and C(1) jointly dominate everywhere (near βc and deep in ordered phase alike), magnetization (m, |m|) is comparatively redundant once E/C(1) are known. Differs from the original "χ near βc, |m| far below" hypothesis but is a real, defensible result (E and C(1) are nearly the same physical quantity at h=0, and that signal is informative across the whole β range, not just one regime).

---

## Documentation & polish 

- [x] Docstrings on every function (input/output/purpose) across `ising.py`, `data_extraction.py`, `data_generation.py`, `mgibbs.py`, `vae.py`
- [ ] README.md: add usage/setup section (how to generate data, how to run each phase), short description of each file
- [x] Light hyperparameter tuning pass on FCN/CVAE/XGBoost (learning rate, latent dim, tree depth) with a brief note on what was tried and why the final choice was made 
- [ ] Final pass on plots: consistent labels/titles/units across all figures for the presentation


