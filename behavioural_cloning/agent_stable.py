import torch
import torch.nn as nn
import torch.optim as optim
from model import MLP
import os
from encoder import EquivariantEncoder
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.distributions.mixture_same_family import MixtureSameFamily


class Agent(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, encoder_hidden_size, output_size, device='cpu'):
        super(Agent, self).__init__()
        self.encoder = EquivariantEncoder(input_size-3, output_size, encoder_hidden_size, device=device).to(device)
        self.policy = MLP(output_size+3, hidden_size1, hidden_size2, output_size).to(device)
        self.MDN = MLP(3, hidden_size1, hidden_size2, 3*25).to(device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.policy.parameters())+list(self.MDN.parameters()), lr=0.001)
        self.device = device

    def training_step(self, batch):
        observations, target_actions = batch

        g = observations[:, :3].to(self.device)
        states = observations[:, 3:].to(self.device)
        z = self.encoder(states)

        actions = self.forward(z, g)
        N_g = self.MDN(g).view(-1, 25, 3)

        comp = MultivariateNormal(N_g, torch.eye(3).to(z.device) * 0.01)
        mix = Categorical(torch.ones(N_g.shape[:2]).to(z.device) / N_g.shape[1])
        gmm = MixtureSameFamily(mix, comp)
        nll = -torch.mean(gmm.log_prob(z))

        # rho = MultivariateNormal(N_g, torch.eye(3).to(z.device)*0.1)
        #
        # nll = torch.mean(-rho.log_prob(z.view(-1, 1, 3).repeat(1, 25, 1)).sum(-1))

        loss_policy = self.compute_loss(actions, target_actions)

        loss = loss_policy + nll

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), nll.detach().cpu().item()

    def compute_loss(self, outputs, targets):
        return self.criterion(outputs, targets.to(self.device))

    def forward(self, z, g):
        return self.policy(torch.cat([z, g], -1))

    def select_action(self, x):
        g = x[:, :3]
        z = self.encoder(x[:,3:])
        #a = self.policy(z)
        # rho, grad = self.density_estimator.get_gradient(z)
        a_IL = self.policy(torch.cat([z, g], -1))
        # p = torch.sigmoid(rho)
        # a = p*a_IL + (1-p)*grad
        return a_IL


    def save_model(self, path_policy, path_mdn):
        torch.save(self.policy.state_dict(), path_policy)
        torch.save(self.MDN.state_dict(), path_mdn)
        #print(f"Model parameters saved to {path}")

    def load_model(self, path_policy, path_mdn):
        self.policy.load_state_dict(torch.load(path_policy))
        self.MDN.load_state_dict(torch.load(path_mdn))
        print(f"Model parameters loaded from {path_policy}")

    def process_obs(box_poses, joint_pos, joint_vel):
        #find the closest obj
        g = box_poses[0]
        obj_pos = [(g[0,0]+g[0,1])/2, (g[1,0]+g[1,1])/2, (g[2,0]+g[2,1])/2]
        obs = np.concatenate([obj_pos, joint_pos, joint_vel])
        obs_tsr = np.array(obs, dtype=torch.float32)
        return obs_tsr



class KDE():

    def __init__(self, states):
        super(KDE, self).__init__()
        self.states = states
        self.N = self.states.shape[0]
        self.d = self.states.shape[1]
        self.h = 0.1

    def K(self, x, x_data):
        
        diff = torch.cdist(x, x_data, p=2)
        delta = - 0.5 * (diff / self.h) ** 2
        return torch.exp(delta)

    def compute_p(self,z):
        p = (2*np.pi*(self.h**self.d))**(-0.5) * self.K(z, self.states)
        return torch.tanh(torch.sum(p, -1)*0.0005)

    def get_gradient(self, z):
        z.requires_grad_(True)
        p = (2*np.pi*(self.h**self.d))**(-0.5) * self.K(z, self.states)

        #density
        rho = torch.sum(p)
        rho.requires_grad_=True

        #gradient
        grad = torch.autograd.grad(outputs=rho, inputs=z)[0]
        if torch.any(torch.isnan(grad)):
            grad = torch.nan_to_num(grad, nan=0.)

        return rho, grad