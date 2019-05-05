import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class MLP(nn.Module):
    def __init__(self, sizes):
        super(MLP, self).__init__()
        layers = []

        for i in range(0, len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < (len(sizes) - 2):
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)
        self.net.apply(Xavier)

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    """ VAE decoder """
    def __init__(self, obs_size, action_size, latent_size):
        super(Decoder, self).__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.net = MLP([(obs_size+action_size+latent_size), 128, 128, obs_size])

    def forward(self, o, a, z):
        x = torch.cat((o, a, z), dim=2)
        o_next = self.net(x)
        return o_next

class Encoder(nn.Module):
    """ VAE encoder """
    def __init__(self, obs_size, action_size, hidden_size, latent_size):
        super(Encoder, self).__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.latent_size = latent_size

        self.rnn = nn.GRU(input_size=(obs_size * 2)+action_size,
                          hidden_size=hidden_size,
                          batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_logsigma = nn.Linear(hidden_size, latent_size)


    def forward(self, o, a, o_next):
        x = torch.cat((o, a, o_next), dim=2)
        output, hidden = self.rnn(x)
        mu = self.fc_mu(output)
        logsigma = self.fc_logsigma(output)

        return mu, logsigma

class VAE(nn.Module):
    """ Variational Autoencoder """
    def __init__(self, obs_size, action_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(obs_size, action_size, hidden_size, latent_size)
        self.decoder = Decoder(obs_size, action_size, latent_size)

    def forward(self, o, a, o_next):
        mu, logsigma = self.encoder(o, a, o_next)
        sigma = logsigma.exp()
        eps = torch.randn_like(sigma)
        z = eps.mul(sigma).add_(mu)

        recon_x = self.decoder(o, a, z)
        return recon_x, mu, logsigma
