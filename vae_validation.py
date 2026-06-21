from numpy import float32
from vae import *
from mgibbs import *

with torch.no_grad():

    model_path = "models/vae_loss10665.5988_valloss10752.5495_20260620_172049.pt"
    model = VAE(16)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    unique_betas = sorted(set(d["beta"] for d in dataset if d["h"] == 0))
    beta_val = min(unique_betas, key=lambda b: abs(b - 0.44))
    h_val = 0.0
    sample_size = 32

    z = torch.randn( sample_size, 16)
    beta_norm = (beta_val - ds.beta_mean) / ds.beta_std
    h_norm = (h_val - ds.h_mean) / ds.h_std

    y_cond = torch.tensor([[beta_norm, h_norm] for _ in range(sample_size)], dtype=torch.float32) # this is just a conditioning label
    logits = model.decoder(z, y_cond)

    probs = torch.bernoulli(torch.sigmoid(logits))
    spins = 2*probs - 1 # reconstruct spins, based on decoder probability

    generated_features = []
    for i in range(sample_size):
        sigma_i = spins[i, 0].numpy()
        f = np.concat((features(sigma_i, h_val), [domain_wall_density(sigma_i)]))
        generated_features.append(f)
    generated_features = np.array(generated_features)

    real_data = [d for d in dataset if d["h"] == 0 and d["beta"] == beta_val]
    real_features = []
    for d in real_data:
        f = features(d["sigma"], 0)
        real_features.append(np.concatenate((f, [domain_wall_density(d["sigma"])])))
    real_features = np.array(real_features)

    feature_names = ["M", "E", "C1", "C2", "C5", "C10", "C20", "domain_wall_density"]
    print(f"{'feature':<20}{'generated':>12}{'real':>12}")
    for name, gen_val, real_val in zip(feature_names, generated_features.mean(axis=0), real_features.mean(axis=0)):
        print(f"{name:<20}{gen_val:>12.4f}{real_val:>12.4f}")




    pass
