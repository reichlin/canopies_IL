import os
import torch
import numpy as np
from model import Model
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
import sys
from utils import fit_spline, load_data

module_path = '/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning'
sys.path.append(module_path)

from src.utils_rl import Agent

def test_equivariance():
    agent = Agent(
        input_size=7,
        goal_size=3,
        action_size=3,
        N_gaussians=25,
        device='cpu'
    )
    agent.load_model(os.path.join(os.getcwd(), 'params', f'model_{task}.pth'))

    states, goals, actions, positions, _ = load_data(
        '/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/data/grasping',
        j_only=True,
        convolving=True,
        frequency=5
    )
    states_tsr = torch.from_numpy(states).float()
    z_ee = agent.encoder(states_tsr).detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1, color='green')
    ax.scatter(z_ee[:, 0], z_ee[:, 1], z_ee[:, 2], s=1, color='red')

    ax.set_title('Dataset positions')
    fig.show()


def test_policy_actions():
    agent = Agent(
        input_size=7,
        goal_size=3,
        action_size=3,
        N_gaussians=25,
        device='cpu',
        stable=False
    )
    agent.load_model(os.path.join(os.getcwd(), 'params', f'model_grasping_J_only_5f.pth'))

    states, goals, actions, positions,_ = load_data(
        '/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/data/grasping',
        j_only=True,
        convolving=True,
        frequency=5
    )


    j = 0
    for s, g, a, p,n  in zip(states, goals, actions, positions):
        acts = []

        for s_j, g_j in zip(torch.from_numpy(s).float(), torch.from_numpy(g).float()):
            a_j = agent.select_action(s_j.unsqueeze(0).to(agent.device), g_j.unsqueeze(0).to(agent.device))
            acts.append(a_j.detach().numpy())
        acts = np.concatenate(acts,0)

        i = 0
        plt.scatter(range(acts.shape[0]), a[:, i], s=1, color='red', zorder=1)
        plt.scatter(range(acts.shape[0]), acts[:, i], s=1, color='blue', zorder=2)

        #for i in range(3):
        #    plt.plot(range(acts.shape[0]), a[:, i], linewidth=1, color='red', zorder=1)
        plt.title(f'actions {j},{i}')
        plt.show()
        j+=1




def test_distributions():
    states, goals, actions, next_states, positions = load_data(
        data_path,
        j_only=True,
        convolving=True,
        single_traj=True
    )
    agent = Model(
        input_size=7,
        goal_size=3,
        action_size=3,
        N_gaussians=25,
        device='cpu'
    )
    agent.load_model(os.path.join(os.getcwd(), 'params', f'model_grasping_J_only_50hz_grasping.pth'))

    for s, g, a in zip(states, goals, actions):

        noise = np.zeros(3) #np.random.uniform([-0.2] * 3, [0.2] * 3, size=(3,))

        s_tsr = torch.from_numpy(s).float()
        g_tsr = torch.from_numpy(g+noise).float()
        z_ee , _, rho = agent(s_tsr, g_tsr)

        z_grape = agent.encoder(s_tsr[-1]).detach().numpy()
        z_ee = z_ee.detach().cpu().numpy()
        rho = rho.detach().cpu()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(rho.shape[1]):
            rho_i = rho[-1, i, :]

            #generate the distribution and plot 100 samples
            distribution = MultivariateNormal(rho_i, torch.eye(3) * 0.01)
            samples = distribution.rsample((1000,))
            c = np.ones(samples.shape[0])*i
            ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=10, c=c, alpha=0.01, zorder=2)

        ax.scatter(z_ee[:, 0], z_ee[:, 1], z_ee[:, 2], s=10, color='red', zorder=1)
        #ax.scatter(*z_grape, s=100, color='green', zorder=1)
        fig.show()
        dio = 0

def inspect_trajectories():
    states, goals, actions, next_states, positions, names = load_data(
        data_path,
        j_only=True,
        convolving=True,
        single_traj=True
    )
    j=0
    for s, g, a, p, n in zip(states, goals, actions, positions, names):

        plt.scatter(range(a.shape[0]), np.stack(a)[:,0], s=1)
        plt.title(f'actions {n}')
        plt.show()


if __name__ == "__main__":
    task = 'grasping'
    data_path = os.path.join(os.path.dirname(os.getcwd()), f'data/{task}')
    test_distributions()
    #inspect_trajectories()
    #test_policy_actions()