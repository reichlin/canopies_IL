import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily

class MLP(nn.Module):

    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        hidden = 128
        self.f = nn.Sequential(nn.Linear(input_size, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, output_size))

    def forward(self, x, g=None):
        if g is not None:
            return self.f(torch.cat([x, g], -1))
        return self.f(x)



class Agent(nn.Module):

    def __init__(self, input_size, goal_size, action_size, N_gaussians, stable=True,  device=None):
        super(Agent, self).__init__()
        self.stable = stable
        self.N_gaussians = N_gaussians

        self.encoder = MLP(input_size, action_size)
        self.policy = MLP(input_size + goal_size, action_size)
        self.MDN = MLP(goal_size, action_size * N_gaussians)

        self.density_estimator = KDE()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = device

    def select_action(self, s, g):

        z = self.encoder(s)
        mu = self.MDN(g).reshape(25, 3)
        rho, grad = self.density_estimator.get_gradient(z, mu)
        a_IL = self.policy(s, g)
        p = torch.tanh(rho * 0.002)

        if self.stable:
            return p*a_IL + (1-p)*grad
        else:
            return a_IL

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        print("Loading model from {path}".format(path=path))
        self.load_state_dict(torch.load(path))



class KDE():

    def __init__(self):
        super(KDE, self).__init__()
        self.N = 25
        self.d = 3
        self.h = 1.0 #0.1
        self.ni = 0.0005

    def K(self, x, mu):
        diff = torch.cdist(x, mu, p=2)
        delta = - 0.5 * (diff / self.h) ** 2
        return torch.exp(delta)

    def get_gradient(self, z, mu):
        z.requires_grad_(True)

        p = (2 * np.pi * (self.h ** self.d)) ** (-0.5) * self.K(z, mu)

        # density
        rho = torch.sum(p)
        rho.requires_grad_ = True

        # gradient
        grad = torch.autograd.grad(outputs=rho, inputs=z)[0]
        # grad = torch.clamp(grad, min=-0.1, max=0.1)

        if torch.any(torch.isnan(grad)):
            grad = torch.nan_to_num(grad, nan=0.)

        return rho, grad / torch.linalg.norm(grad) * self.ni

