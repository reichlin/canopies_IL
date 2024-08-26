import time

import torch
import os
import numpy as np
import yaml
from model import Model
from utils import load_data, load_data_representation
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import wandb
import matplotlib.pyplot as plt
import argparse
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.optim as optim


def train(agent, loader):
    pbar = tqdm(total=config.epochs)
    best_scores = dict(mdn=np.inf, encoder=np.inf)
    optimizer = optim.Adam(agent.parameters(), lr=config.lr)
    agent.encoder.load_model(os.path.join(os.getcwd(), 'params', f'grasping_J_only_grasp_last_equivariance/model_encoder.pth'))
    #agent.MDN.load_model(os.path.join(os.getcwd(), 'params', f'grasping_J_only_grasp_last_equivariance/model_mdn.pth'))

    for epoch in range(config.epochs):
        agent.train()
        total_loss, tot_loss_equi, tot_loss_nll = [0.0]*3

        for (s, g, a, s1) in loader:

            loss_nll = agent.training_step_NLL((
                s.to(agent.device),
                g.to(agent.device),
                a.to(agent.device),
                s1.to(agent.device)
            ))

            loss_equi = agent.training_step_equi((
                s.to(agent.device),
                g.to(agent.device),
                a[:,:3].to(agent.device),
                s1.to(agent.device)
            ))

            #loss = loss_equi + loss_nll
            loss = loss_nll
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tot_loss_equi += loss_equi
            tot_loss_nll += loss_nll

        log_dict = dict(
            total_loss=total_loss / len(loader),
            loss_equi=tot_loss_equi / len(loader),
            loss_nll=tot_loss_nll / len(loader),
        )

        wandb.log(log_dict)

        #save best params
        if (epoch +1 ) % 10 == 0:

            if log_dict['loss_equi'] < best_scores['encoder']:
                best_scores['encoder'] = log_dict['loss_equi']
                #agent.encoder.save_model(os.path.join(config.save_dir, f'model_encoder.pth'))

            if log_dict['loss_nll'] < best_scores['mdn']:
                best_scores['mdn'] = log_dict['loss_nll']
                agent.MDN.save_model(os.path.join(config.save_dir, f'model_mdn.pth'))

            agent.eval()

            # test equivariance and distributions
            all_indices = [x for x in range(states.shape[0])]
            idx_1 = all_indices[:367]
            idx_2 = all_indices[1833:2070]
            idx = idx_1 + idx_2
            s, g, a = states.to(agent.device), goals.to(agent.device), actions.to(agent.device)

            z = agent.encoder(s).detach().cpu().numpy()
            rho = agent.MDN(g).view(-1, agent.N_gaussians, a.shape[-1]).detach().cpu()

            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(projection='3d')
            ax.scatter(positions[idx, 0], positions[idx, 1], positions[idx, 2], s=1, color='green')
            ax.scatter(z[idx, 0], z[idx, 1], z[idx, 2], s=1, color='red')
            plt.savefig(os.path.join(os.getcwd(), 'save/figures', f'{tag}.png'))
            plt.close()

            fig = plt.figure(figsize=(10,10))
            ax_1 = fig.add_subplot(projection='3d')
            for i in range(rho.shape[1]):
                rho_i = rho[idx_1[-1], i, :]
                distribution = MultivariateNormal(rho_i, torch.eye(3) * agent.sigma)
                samples = distribution.rsample((1000,))
                c = np.ones(samples.shape[0]) * i
                ax_1.scatter(samples[:, 0], samples[:, 1], samples[:, 2], s=5, c=c, alpha=0.01, zorder=2)
                ax_1.scatter(*rho_i, s=10, color='green', alpha=1, zorder=2)

            ax_1.scatter(z[idx_1, 0], z[idx_1, 1], z[idx_1, 2], s=10, color='red', zorder=1)


            plt.savefig(os.path.join(os.getcwd(), 'save/figures', f'{tag}_distribution.png'))
            plt.close()
        pbar.update()

    pbar.close()

def main():
    dataloader = DataLoader(TensorDataset(*data), batch_size=config.batch_size, shuffle=True)

    agent = Model(
        input_size=config.input_size,
        goal_size=2, #config.goal_size,
        action_size=(3, 4),
        N_gaussians=config.N_gaussians,
        sigma_min=0.001,
        device=config.device
    )
    best_score = train(agent, dataloader)

    print(f'Training finished with best score of {best_score}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Behavioural Clone')
    parser.add_argument("-wb", "--wb_mode", type=str, default="online", choices=["online", "offline"],help="online or offline?")
    parser.add_argument( "--wb_group", type=str, default="Null",help="Why group does the run belongs to?")
    parser.add_argument( "-task" "--task", type=str,help="What task to perform?")
    parser.add_argument( "-device", "--device", default='cuda' if torch.cuda.is_available() else 'cpu',
                         type=str, choices=["cuda", "cpu"])

    config = parser.parse_args()
    #args = get_config('config/hyperparameters.yaml')
    with open(os.path.join(os.getcwd(), 'config/hyperparameters.yaml'), "r") as f:
        hyperparams = yaml.safe_load(f)
    for item in hyperparams.keys():
        if not hasattr(config, item) or (hasattr(config, item) and getattr(config, item) is None):
            setattr(config, item, hyperparams[item])

    config.task = 'grasp_last'

    for name, value in sorted(vars(config).items()):
        print(f"{name}: {value}")
    print(f'{"-" * 20}\n')

    J_only = True
    testing_frequency = 10
    tag = f'J_only_{config.task}_equivariance' if J_only else f'J_J_dot_{config.task}_equivariance'
    config.save_dir = os.path.join(os.getcwd(), config.save_dir, f'grasping_{tag}')
    os.makedirs(config.save_dir, exist_ok=True)

    ## LOAD DATASET
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    #get the dataset for equivariant representation
    states, goals, actions, next_states, positions = load_data_representation(
        data_path=os.path.join(parent_dir, f'data/{config.task}'),
        j_only=J_only,
        single_traj=False,
        frequency=testing_frequency
    )
    goals = goals[:,:2]
    data = (states, goals, actions[:,:3], next_states)

    print(f'Dataset of {states.shape[0] } samples')

    wb_run = f'{config.task}_{tag}_only_equivariance'
    wandb.init(project="Behavioural_cloning", config=config, name=wb_run,
               mode=config.wb_mode,
               group=config.wb_group)

    main()

