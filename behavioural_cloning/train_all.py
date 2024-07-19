import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from model import Model
from utils import load_data, get_config
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
#import wandb

def train(agent, loader):
    pbar = tqdm(total=config.epochs)
    best_loss = np.inf

    for epoch in range(config.epochs):
        agent.train()
        total_loss = 0.0
        for batch in loader:
            loss = agent.training_step(batch)
            total_loss += loss

        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / len(loader)

            # save checkpoint
            path_cp = os.path.join(os.getcwd(), config.checkpoint_dir, config.task)
            os.makedirs(path_cp, exist_ok=True)
            file_name = os.path.join(path_cp, f'model_ep_{epoch + 1}_score_{avg_loss}.pth')
            agent.save_model(file_name)

            log_dict = {'train_loss': avg_loss}
            #wandb.log(log_dict)

            if avg_loss < best_loss:
                # save best params
                path_save = os.path.join(os.getcwd(), config.save_dir)
                best_loss = avg_loss
                os.makedirs(path_save, exist_ok=True)
                agent.save_model(os.path.join(path_save,f'model_{config.task}.pth'))
        pbar.update()

    pbar.close()
    return best_loss


def main():
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    parent_dir = os.path.dirname(os.getcwd())
    data = load_data(
        data_path=os.path.join(parent_dir,config.load_dir, config.task),
        n_frames=config.n_frames,
        task=config.task
    )
    dataloader = DataLoader(TensorDataset(*data), batch_size=config.batch_size, shuffle=True)

    agent = Model(data[0].shape[-1], data[1].shape[-1], data[2].shape[-1], config.N_gaussians, config.device).to(config.device)

    best_score = train(agent, dataloader)

    print(f'Training finished with best score of {best_score}')


if __name__ == "__main__":
    config = get_config('config/hyperparameters.yaml')

    wb_run = f'{config.task}'
    #wandb.init(project="Behavioural_cloning", config=config, name=wb_run, mode=config.wb_mode, group=config.wb_group)

    main()

