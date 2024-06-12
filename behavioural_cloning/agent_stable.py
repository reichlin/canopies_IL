import torch
import torch.nn as nn
import torch.optim as optim
from model import MLP
import os
from encoder import EquivariantEncoder
import numpy as np

class Agent(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, encoder_hidden_size, output_size, device='cpu'):
        super(Agent, self).__init__()
        self.encoder = EquivariantEncoder(input_size, output_size, encoder_hidden_size, device=device).to(device)
        self.policy = MLP(output_size, hidden_size1, hidden_size2, output_size).to(device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.device = device

    def training_step(self, batch):
        observations, target_actions = batch

        actions = self.forward(observations.to(self.device))

        self.optimizer.zero_grad()
        loss = self.compute_loss(actions, target_actions)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_loss(self, outputs, targets):
        return self.criterion(outputs, targets.to(self.device))

    def forward(self,x):
        z = self.encoder(x)
        return self.policy(z)

    def select_action(self, x):
        z = self.encoder(x)
        #a = self.policy(z)
        rho, grad = self.density_estimator.get_gradient(z)
        a_IL = self.policy(z)
        p = torch.sigmoid(rho)
        a = p*a_IL + (1-p)*grad
        return a


    def save_model(self, path):
        torch.save(self.policy.state_dict(), path)
        #print(f"Model parameters saved to {path}")

    def load_model(self, path):
        self.policy.load_state_dict(torch.load(path))
        print(f"Model parameters loaded from {path}")

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