import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils_new import MLP
import os
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily



class Model(nn.Module):

    def __init__(self, input_size, goal_size, action_size, N_gaussians, device=None):
        super(Model, self).__init__()

        self.N_gaussians = N_gaussians

        self.encoder = MLP(input_size, action_size)
        self.policy = MLP(input_size + goal_size, action_size)
        self.MDN = MLP(goal_size, action_size * N_gaussians)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = device

    def forward(self, x, g=None):
        z = self.encoder(x)
        if g is not None:
            a = self.policy(torch.cat([x, g], -1))
            rho = self.MDN(z).view(-1, self.N_gaussians, a.shape[-1])
            return z, a, rho
        return z, None, None

    def select_action(self, s, g):
        return

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def training_step(self, batch):

        s, g, a, s1 = batch
        s, g, a, s1 = s.to(self.device), g.to(self.device), a.to(self.device), s1.to(self.device)
        a_dim = a.shape[-1]

        z0 = self.encoder(torch.zeros_like(s))
        z = self.encoder(s)
        z1 = self.encoder(s1)
        a_hat = self.policy(s, g)
        rho = self.MDN(g).view(-1, self.N_gaussians, a_dim)

        loss_equi = torch.mean(torch.sum((z1 - z - a) ** 2, -1)) + torch.mean(z0 ** 2)
        loss_policy = torch.mean(torch.sum((a_hat - a) ** 2, -1))

        comp = MultivariateNormal(rho, torch.eye(a_dim).to(z.device) * 0.01)
        mix = Categorical(torch.ones_like(rho[:,:,0]) / self.N_gaussians)
        gmm = MixtureSameFamily(mix, comp)
        nll = - torch.mean(gmm.log_prob(z))

        tot_loss = loss_equi + loss_policy + nll

        self.optimizer.zero_grad()
        tot_loss.backward()
        self.optimizer.step()

        return tot_loss.item()

#
# class KDE():
#
#     def __init__(self, states):
#         super(KDE, self).__init__()
#         self.states = states
#         self.N = self.states.shape[0]
#         self.d = self.states.shape[1]
#         self.h = 0.1
#
#     def K(self, x, x_data):
#         diff = torch.cdist(x, x_data, p=2)
#         delta = - 0.5 * (diff / self.h) ** 2
#         return torch.exp(delta)
#
#     def compute_p(self, z):
#         p = (2 * np.pi * (self.h ** self.d)) ** (-0.5) * self.K(z, self.states)
#         return torch.tanh(torch.sum(p, -1) * 0.0005)
#
#     def get_gradient(self, z):
#         z.requires_grad_(True)
#         p = (2 * np.pi * (self.h ** self.d)) ** (-0.5) * self.K(z, self.states)
#
#         # density
#         rho = torch.sum(p)
#         rho.requires_grad_ = True
#
#         # gradient
#         grad = torch.autograd.grad(outputs=rho, inputs=z)[0]
#         if torch.any(torch.isnan(grad)):
#             grad = torch.nan_to_num(grad, nan=0.)
#
#         return rho, grad

