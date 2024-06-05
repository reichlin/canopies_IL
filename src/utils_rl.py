import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Agent(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, device='cpu'):
        super(Agent, self).__init__()
        self.model = MLP(input_size, hidden_size1, hidden_size2, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = device
    
    def select_action(self,x):
        return self.model(x)

    def to_device(self):
        self.to(self.device)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Model parameters loaded from {path}")



class AgentStable(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, encoder_hidden_size, output_size, device='cpu'):
        super(AgentStable, self).__init__()
        self.encoder = EquivariantEncoder(input_size, output_size, encoder_hidden_size, device=device).to(device)
        self.policy = MLP(output_size, hidden_size1, hidden_size2, output_size).to(device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=0.001)
        self.step = 0
        self.device = device

    def forward(self,x):
        z = self.encoder(x)
        return self.policy(z)

    def select_action(self, x):
        z = self.encoder(x)
        rho, grad = self.density_estimator.get_gradient(z)
        a_IL = self.policy(z)
        p = torch.tanh(rho)
        a = p*a_IL + (1-p)*grad
        self.step +=1
        return a

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

    def init_KDE(self, obs):
        obs_tsr = torch.stack(obs).float().to(self.device)
        z_states = self.encoder(obs_tsr).detach()
        self.density_estimator = KDE(z_states)
        print(f'KDE configured with {obs_tsr.shape[0]} states')


class KDE():

    def __init__(self, states):
        super(KDE, self).__init__()
        self.states = states
        self.N = self.states.shape[0]
        self.d = self.states.shape[1]
        self.h = 50.
        self.ni = 0.05


    def K(self, x, x_data):
        
        diff = torch.cdist(x, x_data, p=2)
        delta = - 0.5 * (diff / self.h) ** 2
        return torch.exp(delta)

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

        return rho, grad*self.ni



class EquivariantEncoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, device='cpu'):
        super(EquivariantEncoder, self).__init__()
        self.model = MLP(input_size, hidden_size, hidden_size, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = device

    def forward(self, x):
        return self.model(x)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        #print(f"Model parameters saved in {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Model parameters loaded from {path}")

    def training_step(self, batch):
        s, a, ns = batch 
        z_s = self.forward(s)
        z_ns = self.forward(ns)
        loss = self.criterion(z_ns,z_s+a)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

