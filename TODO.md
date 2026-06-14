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

## Phase 2 — Latent Space = Order Parameter (VAE)

Goal: train a VAE with no labels and show that the 2D latent space organizes itself by magnetization, bifurcating at Tc — a data-driven discovery of the order parameter.

- [ ] Build a VAE: encoder (Conv layers → μ, logσ), decoder (deconv → 200×200 spin grid), KL + reconstruction loss
- [ ] Train on configurations spanning β ∈ [0.1, 0.8], h = 0
- [ ] Plot 2D latent space colored by β — expect a line/curve ordered by temperature
- [ ] Plot 2D latent space colored by sign(m) — expect two branches below βc, one above
- [ ] Show the bifurcation point aligns with βc
- [ ] Extract the leading latent dimension as a function of β; fit to Onsager magnetization m ~ (βc − β)^{1/8} and check exponent
- [ ] Interpolate latent vectors between a below-Tc and above-Tc config; decode and visualize the "phase transition walk"
- [ ] Repeat with h ≠ 0 data; check that h lifts the degeneracy (one cluster even below Tc)

---

## Phase 3 — Renormalization Group via CNN + Finite-Size Scaling

Goal: replace the flat FCN with a CNN that coarse-grains spatially (like Kadanoff block-spin RG), then use finite-size scaling collapse to extract the critical exponent ν.

- [ ] Build a CNN classifier (Conv + stride-2 pooling blocks → Dense → sigmoid), input = full 200×200 grid
- [ ] Train on same binary labels (above/below Tc) and compare accuracy to FCN baseline
- [ ] Train separate CNN classifiers on crops of size L = 200, 150, 100, 50, 25
- [ ] For each L, record the model's output probability P(β) and fit sigmoid to extract apparent βc(L)
- [ ] Plot βc(L) vs L; theory predicts βc(L) = βc + aL^{-1/ν}, fit to extract ν (expect ν ≈ 1)
- [ ] Perform scaling collapse: plot P vs (β − βc) · L^{1/ν} for all L on a single axes — curves should overlap
- [ ] Visualize CNN filter weights at each conv layer; interpret as learned coarse-graining kernels
- [ ] (Stretch) test transfer: apply the CNN trained on 2D Ising to q=2 Potts model configurations — does it locate the right Tc?

---

## Phase 4 — Decision Trees + SHAP (time permitting)

Goal: use gradient-boosted trees on hand-crafted physical features and recover the known order parameter hierarchy from SHAP values.

- [ ] Build feature extractor: m, |m|, E, χ, specific heat C, C(r) at several r, domain wall density
- [ ] Train XGBoost to regress β (given h=0 configs); evaluate RMSE
- [ ] Compute SHAP values; check that χ dominates near βc, |m| dominates far below
- [ ] Train XGBoost to jointly regress (β, h) using non-zero h data
- [ ] (Stretch) fit a GP on the (β, h) → ⟨m⟩ surface; plot GP uncertainty — should peak along the critical line
