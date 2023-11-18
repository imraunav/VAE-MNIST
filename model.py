import torch
from torch import nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self, in_dim=28 * 28, hidden_dim=200, latent_dim=20) -> None:
        super().__init__()
        # encoder
        self.in2hid = nn.Linear(in_dim, hidden_dim)
        self.hid2mu = nn.Linear(hidden_dim, latent_dim)
        self.hid2sigma = nn.Linear(hidden_dim, latent_dim)

        # decoder
        self.latent2hid = nn.Linear(latent_dim, hidden_dim)
        self.hid2out = nn.Linear(hidden_dim, in_dim)

        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h = self.relu(self.in2hid(x))
        mu = self.hid2mu(h)
        sigma = self.hid2sigma(h)
        return mu, sigma

    def decode(self, x):
        h = self.relu(self.latent2hid(x))
        return self.sigmoid(self.hid2out(h))

    def forward(self, x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        recon = self.decode(z)
        return recon, mu, sigma

# if __name__ == "__main__":
#     vae = VariationalAutoEncoder()
#     intensor = torch.randn(7, 28*28)
#     recon, mu, sigma = vae(intensor)
#     print(recon.shape, mu.shape, sigma.shape)