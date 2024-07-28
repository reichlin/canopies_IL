import os
from utils import load_data, get_config
import torch
import numpy as np
from model import Model
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
import sys
from utils import fit_spline, load_data_new

module_path = '/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning'
sys.path.append(module_path)

from src.utils_rl import Agent

def test_equivariance():
    agent = Agent(
        input_size=14,
        goal_size=3,
        action_size=3,
        N_gaussians=25,
        device='cpu'
    )
    agent.load_model(os.path.join(os.getcwd(), 'params', f'model_{task}.pth'))

    states, positions, actions, goals, next_states = get_data(data_path)

    states_tsr = torch.from_numpy(states).float()
    z_ee = agent.encoder(states_tsr).detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1, color='green')
    ax.scatter(z_ee[:, 0], z_ee[:, 1], z_ee[:, 2], s=1, color='red')

    ax.set_title('Dataset positions')
    fig.show()

def get_data(data_path, indices=None):

    data_files = os.listdir(data_path)

    positions, states, goals, actions, next_states = [], [],[], [],[]

    n_traj = 0

    for file_name in data_files:
        file_path = os.path.join(data_path, file_name)
        dataset = np.load(file_path, allow_pickle=True)

        # process the data
        ee_pos_ = dataset['ee_pos']
        ee_pos, idx = fit_spline(ee_pos_)

        obj_poses = dataset['obj_poses'][idx]
        J = dataset['joints_pos'][idx]
        J_dot = dataset['joints_vel'][idx]
        a = dataset['vr_act'][idx]
        state = np.concatenate((J, J_dot), -1)[:-1]
        next_state = np.concatenate((J, J_dot), -1)[1:]

        #get the goal
        grape_idx = np.argmin(np.linalg.norm(obj_poses[-1] - ee_pos[-1], axis=1))
        goal = np.tile(obj_poses[-1, grape_idx], (next_state.shape[0], 1))

        #get actions
        dee_pos = ee_pos[1:]-ee_pos[:-1]

        if (indices is None) or (indices is not None and n_traj in indices):
            positions.append(ee_pos)
            states.append(state)
            next_states.append(next_state)
            goals.append(goal)
            actions.append(dee_pos)
        n_traj +=1

    positions = np.concatenate(positions, axis=0)
    states = np.concatenate(states, axis=0)
    next_states = np.concatenate(next_states, axis=0)
    goals = np.concatenate(goals, axis=0)
    actions = np.concatenate(actions, axis=0)

    return states, positions, actions, goals, next_states


def test_policy_actions():
    agent = Agent(
        input_size=7,
        goal_size=3,
        action_size=3,
        N_gaussians=25,
        device='cpu',
        stable=False
    )
    agent.load_model(os.path.join(os.getcwd(), 'params', 'model_grasping_50.pth') )#f'model_{task}.pth'))

    (states, goals, actions, next_states), positions = load_data_new(data_path)

    acts = []

    for s, g in zip(states, goals):
        a = agent.select_action(s.unsqueeze(0).to(agent.device), g.unsqueeze(0).to(agent.device))
        acts.append(a.detach().numpy())
    acts = np.concatenate(acts,0)


    plt.scatter(acts[:, 0], acts[:, 2], s=1, color='blue', zorder=2)
    plt.scatter(actions[:, 0], actions[:, 2], s=10, color='red', zorder=1)
    plt.title('action')
    plt.show()


def test_distributions(t_n=None, grape_noise=False):
    agent = Model(
        input_size=14,
        goal_size=3,
        action_size=3,
        N_gaussians=25,
        device='cpu'
    )
    agent.load_model(os.path.join(os.getcwd(), 'params', f'model_{task}.pth'))

    states, positions, actions, goals, next_states = get_data(data_path, t_n)
    if grape_noise:
        noise = np.random.uniform([-0.2] * 3, [0.2] * 3, size=(3,))
    else:
        noise = np.zeros(3)
    s_tsr = torch.from_numpy(states).float()
    g_tsr = torch.from_numpy(goals + noise).float()
    idx = -1

    z_ee , _, rho = agent(s_tsr, g_tsr)
    z_grape = agent.encoder(s_tsr[idx]).detach().numpy() + noise
    z_ee = z_ee.detach().cpu().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    rho = rho.detach().cpu()
    for i in range(rho.shape[1]):
        rho_i = rho[idx, i, :]

        #generate the distribution and plot 100 samples
        distribution = MultivariateNormal(rho_i, torch.eye(3) * 0.01)
        samples = distribution.rsample((1000,))
        c=np.ones(samples.shape[0])*i
        ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=10, c=c, alpha=0.01, zorder=2)

    ax.scatter(z_ee[:, 0], z_ee[:, 1], z_ee[:, 2], s=10, color='red', zorder=1)
    ax.scatter(*z_grape, s=100, color='green', zorder=1)
    ax.set_title('traj. number {n}'.format(n=t_n))
    fig.show()



if __name__ == "__main__":
    task = 'grasping'
    data_path = os.path.join(os.path.dirname(os.getcwd()), f'data/{task}')
    #test_equivariance()
    #for _ in range(0, 10):
    #    test_distributions([0,1,2],True)
    test_policy_actions()