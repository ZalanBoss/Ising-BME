import os
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_extraction import dataset
from augment import random_symmetry

'''
Given x in {+-}^N with measure p(x), learn a generative model via a latent z in R^d:
    p(x) = Integral p_theta(x|z) p(z) dz
True posterior p_theta(z|x) is intractable, so we approximate it with a probe
distribution q_phi(z|x) = N(mu_phi(x), diag(sigma_phi(x)^2)) and maximize the ELBO:
    ELBO(x; phi, theta) = E_q_phi(z|x)[log p_theta(x|z)] - KL(q_phi(z|x) || p(z))
                            \\___ reconstruction ___/        \\___ regularization ___/
phi = encoder params, theta = decoder params. We minimize -ELBO ("vae_loss" below).
'''
class IsingDataset(Dataset):
    def __init__(self, dataset):
        self.sigma = [d["sigma"] for d in dataset]

        betas = np.array([d["beta"] for d in dataset], dtype=np.float32)
        hs = np.array([d["h"] for d in dataset], dtype=np.float32)

        # normalize conditioning labels so beta and h sit on comparable scales
        self.beta_mean, self.beta_std = betas.mean(), betas.std() or 1.0
        self.h_mean, self.h_std = hs.mean(), hs.std() or 1.0

        self.betas = (betas - self.beta_mean) / self.beta_std
        self.hs = (hs - self.h_mean) / self.h_std
        self.hs_raw = hs

    def __len__(self):
        return len(self.sigma)

    def __getitem__(self, idx):
        sigma = random_symmetry(self.sigma[idx], self.hs_raw[idx])
        x = torch.tensor(sigma, dtype=torch.float32).unsqueeze(0)  # (1, 200, 200)
        y = torch.tensor([self.betas[idx], self.hs[idx]], dtype=torch.float32)
        return x, y

batch_size = 16
ds = IsingDataset(dataset)
train_, test = torch.utils.data.random_split(ds, [0.8, 0.2]) #type: ignore
train_loader = DataLoader(train_, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

class Encoder(nn.Module):
    '''
    Recognition network implementing the probe distribution q_phi(z|x).
    Input: x, a (batch, 1, 200, 200) spin configuration.
    Output: (mu, log_sigma), the parameters of q_phi(z|x) = N(mu, diag(exp(log_sigma)^2)).
    '''
    def __init__(self, latent_dim):
        super().__init__()

        # dimensional reduction
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 200x200 -> 100x100
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # 100x100 -> 50x50
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=4, stride=2, padding=1),# 50x50 -> 25x25, bottleneck channel count (bumped back up, augmentation now shares the regularization load)
            nn.ReLU(),
        )


        self.fc_mu = nn.Linear(16*25*25 + 2, latent_dim)
        self.fc_log_sigma = nn.Linear(16*25*25 + 2, latent_dim) # we want to output log sigma, because sigma has to be positive

    def forward(self, x, y):
        h = self.conv(x)
        h = h.view(h.size(0), -1) # flatten the shape
        h = torch.cat([h, y], dim=1)
        mu = self.fc_mu(h)
        log_sigma = self.fc_log_sigma(h)
        return mu, log_sigma

def reparameterize(mu, log_sigma):
    '''
    This function reparameterizes z. It is the case that for gradient descent, the gradient with respect to phi
    needs to be known, however since z is a random number that is not possible to be computed. Instead the linearity of
    normal distribution with respect to the standard normal distribution is used.
    Epsilon is a random noise source which does not depend on phi.

    '''
    epsilon = torch.randn_like(mu)
    z = mu + torch.exp(log_sigma) * epsilon
    return z

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        
        # First expand z back to a spatial feature map
        self.fc = nn.Linear(latent_dim + 2, 16 * 25 * 25)

        # Then upsample back to 200x200, "deconvolve"
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(16, 64, kernel_size=4, stride=2, padding=1), # 25x25 -> 50x50
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 50x50 -> 100x100
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),   # 100x100 -> 200x200
        )
    
    def forward(self, z, y):

        z = torch.cat([z, y], dim=1)
        h = self.fc(z)
        h = h.view(h.size(0), 16, 25, 25)  # unflatten
        x_reconstructed = self.deconv(h)
        return x_reconstructed


class VAE(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()

        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)


    def forward(self, x, y):
        # Encode
        mu, log_sigma = self.encoder(x, y)

        # Reparameterize
        z = reparameterize(mu, log_sigma)

        # Decode
        x_reconstructed = self.decoder(z, y)

        return x_reconstructed, mu, log_sigma



def vae_loss(x, x_reconstructed, mu, log_sigma):
    '''
    This function computes the VAE loss, i.e. the negative ELBO: a binary cross entropy reconstruction term
    (how well x_reconstructed matches x) plus a KL regularization term (how close q_phi(z|x) is to the prior).

    '''
    target = (x + 1) / 2
    # Reconstruction term — how well did we reconstruct x?
    reconstruction = nn.functional.binary_cross_entropy_with_logits(x_reconstructed, target, reduction='sum') / x.size(0)

    # KL term — closed form for two Gaussians
    kl = -0.5 * torch.sum(1 + 2*log_sigma - mu**2 - torch.exp(2*log_sigma)) / x.size(0)

    return reconstruction + kl


def train():
    '''
    This function trains the autoencoder.

    '''

    latent_dim = 16
    epochs = 150

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    model = VAE(latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_loss = []
    test_loss = []


    for epoch in range(epochs):
        total_loss = 0.0

        for x, y in train_loader:   # train_loader
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()   # clear gradients from previous step

            x_reconstructed, mu, log_sigma = model(x, y)  # forward pass

            loss = vae_loss(x, x_reconstructed, mu, log_sigma)  # compute loss

            loss.backward()         # backpropagation
            optimizer.step()        # updating weights

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(train_loader):.4f}")
        train_loss.append(total_loss / len(train_loader))

        total_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x,y in test_loader:
                x, y = x.to(device), y.to(device)
                x_reconstructed, mu, log_sigma = model(x, y)  # forward pass
                loss = vae_loss(x, x_reconstructed, mu, log_sigma)  # compute loss
                total_loss += loss.item()
            test_loss.append(total_loss / len(test_loader))
        model.train()
        print(f"Epoch {epoch+1}/{epochs} | Val loss: {test_loss[-1]:.4f}")

    plt.figure()
    plt.plot(range(1, epochs + 1), train_loss, label="train")
    plt.plot(range(1, epochs + 1), test_loss, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss (per batch avg)")
    plt.title("VAE loss curve")
    plt.legend()
    plt.savefig("vae_loss_curve.png", dpi=150)
    plt.show()

    os.makedirs("models", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/vae_loss{train_loss[-1]:.4f}_valloss{test_loss[-1]:.4f}_{timestamp}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    log_path = "experiments_log.csv"
    log_is_new = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if log_is_new:
            writer.writerow(["timestamp", "model_path", "latent_dim", "bottleneck_channels", "epochs", "batch_size", "lr", "weight_decay", "train_loss", "val_loss"])
        writer.writerow([timestamp, model_path, latent_dim, 16, epochs, batch_size, optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["weight_decay"], train_loss[-1], test_loss[-1]])

if __name__ == "__main__":
    train()
