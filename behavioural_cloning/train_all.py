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


def train(agent, loader_IL, loader_EQUI):
    pbar = tqdm(total=config.epochs)
    best_scores = dict(policy=np.inf, mdn=np.inf, encoder=np.inf)
    optimizer = optim.Adam(agent.parameters(), lr=config.lr) #, weight_decay=config.weight_decay)

    for epoch in range(config.epochs):
        agent.train()
        total_loss, tot_loss_policy, tot_loss_equi, tot_loss_nll = [0.0]*4

        for batch_IL, batch_EQUI in zip(loader_IL, loader_EQUI):
            s, g, a, s1 = batch_IL
            loss_policy, (loss_pos, loss_rot) = agent.training_step_IL((
                s.to(agent.device),
                g.to(agent.device),
                a.to(agent.device),
                s1.to(agent.device)
            ))

            loss_nll = agent.training_step_NLL((
                s.to(agent.device),
                g.to(agent.device),
                a[:,:3].to(agent.device),
                s1.to(agent.device)
            ))

            s, g, a, s1 = batch_EQUI
            loss_equi = agent.training_step_equi((
                s.to(agent.device),
                g.to(agent.device),
                a[:,:3].to(agent.device),
                s1.to(agent.device)
            ))

            loss = loss_policy + loss_equi + loss_nll

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            tot_loss_policy += loss_policy
            tot_loss_equi += loss_equi
            tot_loss_nll += loss_nll

        log_dict = dict(
            total_loss=total_loss / len(loader_IL),
            loss_policy=tot_loss_policy / len(loader_IL),
            loss_equi=tot_loss_equi / len(loader_IL),
            loss_nll=tot_loss_nll / len(loader_IL),
            loss_target_pos=loss_pos,
            loss_target_rot=loss_rot
        )

        wandb.log(log_dict)

        #save best params
        if (epoch + 1) % 10 == 0:

            #if log_dict['loss_policy'] < best_scores['policy']:
            '''best_scores['policy'] = log_dict['total_loss']
            agent.policy.save_model(os.path.join(config.save_dir, f'model_policy.pth'))

            if log_dict['loss_equi'] < best_scores['encoder']:
                best_scores['encoder'] = log_dict['loss_equi']
                agent.encoder.save_model(os.path.join(config.save_dir, f'model_encoder.pth'))

            if log_dict['loss_nll'] < best_scores['mdn']:
                best_scores['mdn'] = log_dict['loss_nll']
                agent.MDN.save_model(os.path.join(config.save_dir, f'model_mdn.pth'))'''

            agent.eval()
            acts = agent.policy(
                states.to(agent.device),
                goals.to(agent.device)
            ).detach().cpu().numpy()

            # Plot
            fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
            ax0.scatter(acts[:, 0], acts[:, 1], s=0.1, zorder=2, color='blue')
            ax1.scatter(acts[:, 0], acts[:, 2], s=0.1, zorder=2, color='blue')
            ax2.scatter(acts[:, 1], acts[:, 2], s=0.1, zorder=2, color='blue')

            ax0.scatter(actions[:, 0], actions[:, 1], s=1, zorder=1, color='red')
            ax1.scatter(actions[:, 0], actions[:, 2], s=1, zorder=1, color='red')
            ax2.scatter(actions[:, 1], actions[:, 2], s=1, zorder=1, color='red')

            plt.savefig(os.path.join(os.getcwd(), 'save/figures', f'{tag}_positions.png'))
            plt.close()

            cur_pos = np.zeros(3)
            cur_pos_ = np.zeros(3)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            for a, a_ in zip(actions[:353], acts[:353]):
                cur_pos += a.numpy()[:3]
                cur_pos_ += a_[:3]
                ax.scatter(*cur_pos, color='red', s=1)
                ax.scatter(*cur_pos_, color='blue', s=0.1)

            plt.savefig(os.path.join(os.getcwd(), 'save/figures', f'{tag}_single_traj.png'))
            plt.close()

            fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
            ax0.scatter(acts[:, 3], acts[:, 4], s=0.1, zorder=2, color='blue')
            ax1.scatter(acts[:, 3], acts[:, 5], s=0.1, zorder=2, color='blue')
            ax2.scatter(acts[:, 3], acts[:, 6], s=0.1, zorder=2, color='blue')

            ax0.scatter(actions[:, 3], actions[:, 4], s=1, zorder=1, color='red')
            ax1.scatter(actions[:, 3], actions[:, 5], s=1, zorder=1, color='red')
            ax2.scatter(actions[:, 3], actions[:, 6], s=1, zorder=1, color='red')

            plt.savefig(os.path.join(os.getcwd(), 'save/figures', f'{tag}_orientations.png'))
            plt.close()

        pbar.update()

    pbar.close()

def main():
    dataloader_IL = DataLoader(TensorDataset(*data_IL), batch_size=config.batch_size, shuffle=True)
    dataloader_EQUI = DataLoader(TensorDataset(*data_EQUI), batch_size=config.batch_size, shuffle=True)

    agent = Model(
        input_size=config.input_size,
        goal_size=2, #config.goal_size,
        action_size=(3, 4),
        N_gaussians=config.N_gaussians,
        device=config.device
    )
    best_score = train(agent, dataloader_IL, dataloader_EQUI)
    #best_score = train(agent, dataloader_IL, data_IL)

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


    '''config.wb_mode = "online" #'offline' #'online'
    config.wb_group = "02/08" '''
    J_only = True
    testing_frequency = 10
    tag = f'J_only_{testing_frequency}hz_{config.task}' if J_only else f'J_J_dot_{testing_frequency}hz_{config.task}'
    config.save_dir = os.path.join(os.getcwd(), config.save_dir, f'grasping_{tag}')
    os.makedirs(config.save_dir, exist_ok=True)

    ## LOAD DATASET
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

    #get the dataset for imitation learning and equivariant representation
    states, goals, actions, next_states, _ = load_data(
        data_path=os.path.join(parent_dir, f'data/{config.task}'),
        j_only=J_only,
        convolving=True,
        single_traj=False,
        frequency=testing_frequency
    )
    goals = goals[:,:2]
    data_IL = (states, goals, actions, next_states)
    data_EQUI = (states, goals, actions[:,:3], next_states)

    print(f'Dataset of {states.shape[0] } samples')


    wb_run = f'{config.task}_{tag}_with_equivariance'
    wandb.init(project="Behavioural_cloning", config=config, name=wb_run,
               mode=config.wb_mode,
               group=config.wb_group)

    main()

