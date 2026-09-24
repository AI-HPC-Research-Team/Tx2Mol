"""Checkpoint-compatible GeneVAE used for training and molecule conditioning.

Adapted from the study's GxVAEs ``GeneVAE.py``. Parameter names, layer order,
stochastic reparameterization, and the summed training loss are preserved.
"""

from pathlib import Path

import torch
from torch import nn


class GeneEncoder(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, activation_fn, dropout):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.latent_size = latent_size
        self.activation_fn = activation_fn
        self.dropout = [dropout] * len(self.hidden_sizes)
        units = [input_size] + self.hidden_sizes
        layers = []
        for index in range(1, len(units)):
            layers.extend([nn.Linear(units[index - 1], units[index]), activation_fn])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.encoding = nn.Sequential(*layers)
        self.encoding_to_mu = nn.Linear(self.hidden_sizes[-1], latent_size)
        self.encoding_to_logvar = nn.Linear(self.hidden_sizes[-1], latent_size)

    def forward(self, inputs):
        projection = self.encoding(inputs)
        return self.encoding_to_mu(projection), self.encoding_to_logvar(projection)


class GeneDecoder(nn.Module):
    def __init__(self, latent_size, hidden_sizes, output_size, activation_fn, dropout):
        super().__init__()
        self.latent_size = latent_size
        # Copy before reversing: the original implementation mutated caller args.
        self.hidden_sizes = list(reversed(hidden_sizes))
        self.output_size = output_size
        self.activation_fn = activation_fn
        self.dropout = [dropout] * len(self.hidden_sizes)
        units = [latent_size] + self.hidden_sizes + [output_size]
        layers = []
        for index in range(1, len(units) - 1):
            layers.extend([nn.Linear(units[index - 1], units[index]), activation_fn])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(units[-2], units[-1]))
        self.decoding = nn.Sequential(*layers)

    def forward(self, latent_z):
        return self.decoding(latent_z)


def kld_loss(mu, logvar):
    """Summed KL(q(z|x) || N(0,I)), calculated in float64 as in the source."""
    mu, logvar = mu.double(), logvar.double()
    return -0.5 * torch.sum(1 + logvar - mu.square() - logvar.exp())


class GeneVAE(nn.Module):
    def __init__(self, input_size=978, hidden_sizes=(512, 256, 128), latent_size=64,
                 output_size=978, activation_fn=None, dropout=0.2):
        super().__init__()
        if not hidden_sizes or any(int(size) <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive layer widths")
        activation_fn = nn.ReLU() if activation_fn is None else activation_fn
        self.encoder = GeneEncoder(input_size, hidden_sizes, latent_size, activation_fn, dropout)
        self.decoder = GeneDecoder(latent_size, hidden_sizes, output_size, activation_fn, dropout)
        self.reconstruction_loss = nn.MSELoss(reduction="sum")
        self.kld_loss = kld_loss

    def reparameterize(self, mu, logvar):
        # Sampling remains enabled in eval mode, matching generation in the study.
        return torch.randn_like(mu).mul_(torch.exp(0.5 * logvar)).add_(mu)

    def forward(self, inputs):
        self.mu, self.logvar = self.encoder(inputs)
        latent_z = self.reparameterize(self.mu, self.logvar)
        return latent_z, self.mu, self.decoder(latent_z)

    def joint_loss(self, outputs, targets, alpha=0.5, beta=1):
        reconstruction = self.reconstruction_loss(outputs, targets).double()
        kl = self.kld_loss(self.mu, self.logvar)
        joint = alpha * reconstruction + (1 - alpha) * beta * kl
        return joint, reconstruction, kl

    def load_model(self, path, map_location=None, device=None):
        """Load a bare state_dict; optional wrapped ``state_dict`` is accepted."""
        location = device if device is not None else map_location
        if location is None:
            location = next(self.parameters()).device
        weights = torch.load(path, map_location=location, weights_only=True)
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
        self.load_state_dict(weights, strict=True)
        if device is not None:
            self.to(device)
        return self

    def save_model(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)


def load_gene_vae(checkpoint, device="cpu", **architecture):
    """Load a frozen inference model with the study architecture by default."""
    model = GeneVAE(**architecture).to(device)
    model.load_model(checkpoint, map_location=device)
    model.eval()
    model.requires_grad_(False)
    return model
