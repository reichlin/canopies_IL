import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils import MLP
import os
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily



class Model(nn.Module):

    def __init__(self, input_size, goal_size, action_size, N_gaussians, device='cpu'):
        super(Model, self).__init__()

        self.N_gaussians = N_gaussians
        self.encoder = MLP(input_size, action_size).to(device)
        self.policy = MLP(input_size + goal_size, action_size).to(device)
        self.MDN = MLP(goal_size, action_size * N_gaussians).to(device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = device

    def forward(self, x, g=None):
        z = self.encoder(x)
        if g is not None:
            a = self.policy(torch.cat([x, g], -1))
            rho = self.MDN(g).view(-1, self.N_gaussians, a.shape[-1])
            return z, a, rho
        return z, None, None

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def training_step(self, batch):

        s, g, a, s1 = batch
        a_dim = a.shape[-1]
        sigma=0.01

        z0 = self.encoder(s)
        z1 = self.encoder(s1)
        a_hat = self.policy(s, g)
        rho = self.MDN(g).view(-1, self.N_gaussians, a_dim)

        loss_equi = torch.mean(torch.sum((z1 - z0 - a) ** 2, -1)) #+ torch.mean(z0 ** 2)
        loss_policy = torch.mean(torch.sum((a_hat - a) ** 2, -1))

        # neg. log. like.
        comp = MultivariateNormal(rho, torch.eye(a_dim).to(z0.device) * sigma)
        mix = Categorical(torch.ones_like(rho[:,:,0]) / self.N_gaussians)
        gmm = MixtureSameFamily(mix, comp)
        nll = - torch.mean(gmm.log_prob(z0.detach()))

        tot_loss = loss_equi + loss_policy + nll
        loss_dict = dict(loss_equi=loss_equi.item(), loss_policy=loss_policy.item(), loss_nll=nll.item())

        self.optimizer.zero_grad()
        tot_loss.backward()
        self.optimizer.step()

        return tot_loss.item(), loss_dict


