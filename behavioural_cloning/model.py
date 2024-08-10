import numpy as np
import torch
import torch.nn as nn
from utils import MLPDual, MLP
import os
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily



class Model(nn.Module):

    def __init__(self, input_size, goal_size, action_size, N_gaussians, device='cpu'):
        super(Model, self).__init__()

        self.N_gaussians = N_gaussians
        self.encoder = MLP(input_size, action_size[0]).to(device)
        self.policy = MLPDual(input_size + goal_size, action_size).to(device)
        self.MDN = MLP(goal_size, action_size[0] * N_gaussians).to(device)

        self.criterion = nn.MSELoss()
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



    def training_step_IL(self, batch):

        s, g, a, s1 = batch

        a_hat = self.policy(s, g)
        a_pos, a_rot, a_hat_pos, a_hat_rot = a[:,:3], a[:,3:], a_hat[:,:3], a_hat[:,3:]
        loss_pos = torch.mean(torch.sum((a_hat_pos - a_pos) ** 2, -1))
        loss_rot = torch.mean(torch.sum((a_hat_rot - a_rot) ** 2, -1))
        #loss_rot = torch.mean(torch.minimum((torch.sum((a_hat_rot - a_rot)**2, -1)),(torch.sum(((-a_hat_rot) - a_rot)**2, -1))))
        loss = loss_pos + loss_rot * 2
        return loss, (loss_pos, loss_rot)

    def training_step_equi(self, batch):
        s, g, a, s1 = batch

        #equivariance
        z0 = self.encoder(s)
        z1 = self.encoder(s1)
        loss = torch.mean(torch.sum((z1 - z0 - a) ** 2, -1))

        return loss

    def training_step_NLL(self, batch):
        s, g, a, s1 = batch
        a_dim = a.shape[-1]
        sigma = 0.001

        z0 = self.encoder(s)
        rho = self.MDN(g).view(-1, self.N_gaussians, a_dim)
        comp = MultivariateNormal(rho, torch.eye(a_dim).to(z0.device) * sigma)
        mix = Categorical(torch.ones_like(rho[:, :, 0]) / self.N_gaussians)
        gmm = MixtureSameFamily(mix, comp)
        loss = - torch.mean(gmm.log_prob(z0.detach()))

        return loss


