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

    def __init__(self, input_size, output_size,hidden:int=32):
        super(MLP, self).__init__()
        hidden = 32
        self.f = nn.Sequential(nn.Linear(input_size, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, hidden),
                               nn.ReLU(),
                               #nn.Linear(hidden, hidden),
                               #nn.ReLU(),
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

    def __init__(self, input_size, goal_size, action_size, N_gaussians, stable=True,  device=None):
        super(Agent, self).__init__()
        self.stable = stable
        self.N_gaussians = N_gaussians
        self.phi = 0.1 #0.002

        self.encoder = MLP(input_size, action_size[0]).to(device)
        self.policy = MLPDual(input_size + goal_size, action_size).to(device)
        self.MDN = MLP(goal_size, action_size[0] * N_gaussians).to(device)

        self.density_estimator = KDE()

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.device = device

    '''def select_action(self, s, g):
        z = self.encoder(s)
        mu = self.MDN(g).reshape(25, 3)
        rho, grad = self.density_estimator.get_gradient(z, mu)
        a_IL = self.policy(s, g)
        p = torch.tanh(rho * self.phi)

        if self.stable:
            print(f'p = {p}, grad = {grad} ')
            return p*a_IL + (1-p)*grad
        else:
            return a_IL'''

    def select_action(self, s, g, a_last):

        z = self.encoder(s)
        mu = self.MDN(g).reshape(25, 3)
        rho, grad = self.density_estimator.get_gradient(z, mu)
        a_IL = self.policy(s, g)
        p = torch.tanh(rho * self.phi)
        if self.stable:
            a_pos = a_last + p * (a_IL[:,:3] - a_last) + (1 - p) * grad
            a_rot = a_IL[:,3:]
            print(f'p = {p.item()}, d_aIL = {[round(x.item(),2) for x in a_IL.detach().cpu().squeeze()]}, grad = {[round(x.item(),2) for x in grad.detach().cpu().squeeze()]}')
            return torch.cat((a_pos, a_rot), dim=-1)
        else:
            return a_IL

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        print("Loading models from {path}".format(path=path))
        self.policy.load_state_dict(torch.load(os.path.join(path, 'model_policy.pth')))
        self.encoder.load_state_dict(torch.load(os.path.join(path, 'model_encoder.pth')))
        self.MDN.load_state_dict(torch.load(os.path.join(path, 'model_mdn.pth')))
        #self.load_state_dict(torch.load(path))



class KDE():

    def __init__(self):
        super(KDE, self).__init__()
        self.N = 25
        self.d = 3
        self.h = 0.005 #1.0
        self.ni = 0.005 #0.0005

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

        return rho, grad / (torch.linalg.norm(grad) + 10**-6) * self.ni

