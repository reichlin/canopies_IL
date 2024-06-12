import torch
import torch.nn as nn
import torch.optim as optim
from model import MLP
import os



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
        s, dp, ns = batch  
        z_s = self.forward(s)
        z_ns = self.forward(ns)
        loss = torch.mean(torch.sum((z_ns - z_s - dp)**2, -1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()



