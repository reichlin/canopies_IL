import os
import torch
import numpy as np
from model import Model
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
import sys
from utils import fit_spline, load_data, load_data_representation

module_path = '/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning'
sys.path.append(module_path)

from src.utils_rl import Agent

def test_equivariance(agent, states, positions):

    z_ee = agent.encoder(states).detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1, color='green')
    ax.scatter(z_ee[:, 0], z_ee[:, 1], z_ee[:, 2], s=1, color='red')

    ax.set_title('Dataset positions')
    fig.show()


def test_policy_actions(agent, states, goals, positions):

    j = 0
    for s, g, a, p, n in zip(states, goals, actions, positions):
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




def test_distributions(agent, states, goals, actions, positions):

    for s, g, a, p in zip(states, goals, actions, positions):

        s_tsr = torch.from_numpy(s).float()
        g_tsr = torch.from_numpy(g).float()
        z_ee = agent.encoder(s_tsr).detach().cpu().numpy()
        rho = agent.MDN(g_tsr).view(-1, agent.N_gaussians, a.shape[-1]).detach().cpu()

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(rho.shape[1]):
            rho_i = rho[-1, i, :]

            #generate the distribution and plot 100 samples
            distribution = MultivariateNormal(rho_i, torch.eye(3) * 0.001)
            samples = distribution.rsample((1000,))
            c = np.ones(samples.shape[0])*i
            ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=10, c=c, alpha=0.01, zorder=2)
            ax.scatter(*rho_i, s=10, color='green', alpha=1, zorder=2)

        ax.scatter(z_ee[:, 0], z_ee[:, 1], z_ee[:, 2], s=10, color='red', zorder=1)
        #ax.scatter(*z_grape, s=100, color='green', zorder=1)
        fig.show()


def inspect_integral_traj(agent, states, goals, actions):


    for s, g, acts in zip(states, goals, actions):

        acts_ = agent.policy(torch.from_numpy(s).float(), torch.from_numpy(g).float()).detach().cpu().numpy()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        cnt = 0
        cur_pos = np.zeros(3)
        cur_pos_ = np.zeros(3)
        for a, a_ in zip(acts, acts_):
            cur_pos += a[:3]
            cur_pos_ += a_[:3]
            ax.scatter(*cur_pos, color='red', s=1)
            ax.scatter(*cur_pos_, color='blue', s=0.1)
            cnt+=1
        ax.set_title(f'traj. {cnt}')
        plt.show()



def inspect_trajectories(agent, states, goals, actions):

    for i in range(3):
        cur_pos = np.zeros(3)
        cur_pos_ = np.zeros(3)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for s, g, a in zip(states[i], goals[i], actions[i]):
            a_ = agent.policy(torch.from_numpy(s).float().unsqueeze(0), torch.from_numpy(g[:2]).float().unsqueeze(0))
            cur_pos += a[:3]
            cur_pos_ += a_.detach().cpu().numpy().squeeze()[:3]
            ax.scatter(*cur_pos, color='red',  s=1)
            ax.scatter(*cur_pos_, color='blue',  s=0.1)

        plt.show()


if __name__ == "__main__":
    task = 'grasp_last'
    data_path = os.path.join(os.path.dirname(os.getcwd()), f'data/{task}')

    agent = Agent(
        input_size=7,
        goal_size=2,
        action_size=(3, 4),
        N_gaussians=25,
        sigma=0.001,
        device='cpu'
    )
    agent.encoder.load_model(
        os.path.join(os.getcwd(), 'params', f'grasping_J_only_grasp_last_equivariance/model_encoder.pth'))
    agent.MDN.load_model(os.path.join(os.getcwd(), 'params', f'grasping_J_only_grasp_last_equivariance/model_mdn.pth'))

    agent.policy.load_model(os.path.join(os.getcwd(), 'params', f'grasping_J_only_10hz_grasp_last/model_policy.pth'))

    '''states, goals, actions, next_states, positions = load_data(
        data_path,
        j_only=True,
        convolving=True,
        single_traj=True,
        frequency=10
    )'''
    states, goals, actions, next_states, positions = load_data_representation(
        data_path,
        j_only=True,
        single_traj=True,
        frequency=10
    )
    goals = [g[:, :2] for g in goals]
    actions = [a[:, :3] for a in actions]
    idx = [1, 6, 11, 16, 20]  # number of trajectories
    test_distributions(agent, [states[i] for i in idx], [goals[i] for i in idx],
                       [actions[i] for i in idx],
                       [positions[i] for i in idx])
    test_equivariance(agent, torch.from_numpy(np.concatenate([states[i] for i in idx])).float(),
                      np.concatenate([positions[i] for i in idx]),)
    # inspect_integral_trajtest_distributions(agent, states[:n], goals[:n], actions[:n])
    # inspect_integral_traj(agent, np.array(states)[idx], np.array(goals)[idx], np.array(actions)[idx])
