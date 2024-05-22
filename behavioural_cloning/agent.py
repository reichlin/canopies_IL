import torch
import torch.nn as nn
import torch.optim as optim
from model import MLP
import os

class Agent(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, device='cpu'):
        super(Agent, self).__init__()
        self.model = MLP(input_size, hidden_size1, hidden_size2, output_size)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.device = device

    def training_step(self, batch):
        inputs, targets = batch

        self.optimizer.zero_grad()
        outputs = self.model(inputs.to(self.device))
        loss = self.criterion(outputs, targets.to(self.device))
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def forward(self,x):
        return self.model(x)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        #print(f"Model parameters saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print(f"Model parameters loaded from {path}")

    def process_obs(box_poses, joint_pos, joint_vel):
        #find the closest obj
        g = box_poses[0]
        obj_pos = [(g[0,0]+g[0,1])/2, (g[1,0]+g[1,1])/2, (g[2,0]+g[2,1])/2]
        obs = np.concatenate([obj_pos, joint_pos, joint_vel])
        obs_tsr = np.array(obs, dtype=torch.float32)
        return obs_tsr
