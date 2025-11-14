import torch
import torch.nn as nn


class VAEDecoupler(nn.Module):

    def __init__(self, input_dim, latent_dim=64):
        super(VAEDecoupler, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_usr = self.decoder(z)
        return x_usr, mu, logvar


class FeatureDecoupling(nn.Module):

    def __init__(self, input_dim, latent_dim=64, rho=0.7):
        super(FeatureDecoupling, self).__init__()
        self.vae = VAEDecoupler(input_dim, latent_dim)
        self.rho = rho

    def forward(self, x):
        x_usr, mu, logvar = self.vae(x)
        x_rep = x - x_usr

        # L2 norm clipping
        norm = torch.norm(x_rep, p=2, dim=1, keepdim=True)
        scale = torch.clamp(self.rho / (norm + 1e-8), max=1.0)
        x_rep = x_rep * scale

        return x_rep, x_usr, mu, logvar