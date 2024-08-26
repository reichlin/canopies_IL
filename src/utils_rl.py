import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily

import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_size, output_size, hidden: int = 32, big=False):
        super(MLP, self).__init__()
        if big:
            self.f = nn.Sequential(nn.Linear(input_size, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, output_size))
        else:
            self.f = nn.Sequential(nn.Linear(input_size, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, output_size))
    def forward(self, x, g=None):
        if g is not None:
            return self.f(torch.cat([x, g], -1))
        return self.f(x)

    def load_model(self, path):
        print('Loading: {}'.format(path))
        self.load_state_dict(torch.load(path))

class MLPDual(nn.Module):

    def __init__(self, input_size, output_size:tuple, hidden:int = 64):
        super(MLPDual, self).__init__()
        self.f = nn.Sequential(nn.Linear(input_size, hidden),
                               nn.BatchNorm1d(hidden),
                               nn.ReLU(),
                               # nn.Dropout(0.2),
                               nn.Linear(hidden, hidden),
                               nn.BatchNorm1d(hidden),
                               nn.ReLU(),
                               # nn.Dropout(0.2),
                               nn.Linear(hidden, hidden),
                               nn.BatchNorm1d(hidden),
                               nn.ReLU(),
                               # nn.Dropout(0.2)
                               )
        self.head1 = nn.Sequential(nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, output_size[0])
                                   )

        self.head2 = nn.Sequential(nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, output_size[1])
                                   )

    def forward(self, x, g):
        h = self.f(torch.cat([x, g], -1))
        return torch.cat([self.head1(h), F.normalize(self.head2(h), dim=-1)], dim=-1)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


class Agent(nn.Module):

    def __init__(self, input_size, goal_size, action_size, N_gaussians, sigma=0.001, stable=True,  device=None):
        super(Agent, self).__init__()
        self.stable = stable
        self.N_gaussians = N_gaussians
        self.phi = 0.1 #0.002

        self.encoder = MLP(input_size, action_size[0], hidden=32).to(device)
        self.policy = MLPDual(input_size + goal_size, action_size).to(device)
        self.MDN = MLP(goal_size, action_size[0] * N_gaussians, hidden=64, big=True).to(device)

        self.density_estimator = KDE(h=sigma)

        self.collision_avoidance_estimator = KDE(h=1.)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = device

    def select_action(self, s, g):
        a_IL = self.policy(s, g)

        if not self.stable:
            return a_IL
        else:
            #compute the back-to-the-manifold action and probability
            z = self.encoder(s)
            mu = self.MDN(g).reshape(25, 3)
            rho, a_GRAD = self.density_estimator.get_gradient(z, mu)
            #p = torch.tanh(rho * self.phi)
            p = torch.sigmoid(rho - 0.5)

            a = torch.zeros_like(a_IL)
            a[:,:3] = (1 - p) * a_GRAD + p * a_IL[:,:3]
            a[:,3:] = a_IL[:,3:]
            return a

    def compute_CA(self, s, z_obs, sigma=0.5):
        z = self.encoder(s)
        z.requires_grad_(True)
        diff = torch.cdist(z, z_obs, p=2)
        delta = - 0.5 * (diff / sigma) ** 2
        p = (2 * np.pi * (sigma ** 3)) ** (-0.5) * torch.exp(delta)
        rho = torch.sum(p)
        rho.requires_grad_ = True
        grad = torch.autograd.grad(outputs=rho, inputs=z)[0]
        return -grad, torch.sigmoid(rho - 0.5)

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_models(self, path):
        print("Loading models from {path}".format(path=path))
        self.policy.load_state_dict(torch.load(os.path.join(path, 'model_policy.pth')))
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'model_encoder.pth')))
        self.MDN.load_state_dict(torch.load(os.path.join(path, 'model_mdn.pth')))
        #self.load_state_dict(torch.load(path))

class KDE():

    def __init__(self, d=3, h=0.001, ni=0.005):
        super(KDE, self).__init__()
        self.d = d
        self.h = h
        self.ni = ni

    def K(self, x, mu):
        diff = torch.cdist(x, mu, p=2)
        delta = - 0.5 * (diff / self.h) ** 2
        return torch.exp(delta)

    def get_gradient(self, z, mu):
        z.requires_grad_(True)

        p = (2 * np.pi * (self.h ** self.d)) ** (-0.5) * self.K(z, mu)

        # density
        rho = torch.mean(p)
        rho.requires_grad_ = True

        # gradient
        grad = torch.autograd.grad(outputs=rho, inputs=z)[0]

        # grad = torch.clamp(grad, min=-0.1, max=0.1)

        if torch.any(torch.isnan(grad)):
            grad = torch.nan_to_num(grad, nan=0.)

        return rho, grad / (torch.linalg.norm(grad) + 10**-6) * self.ni

