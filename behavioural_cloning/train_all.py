import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from model import Model
from utils_new import load_data, load_hyperparameters
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset


def train(agent, loader, config):
    pbar = tqdm(total=config.epochs)
    best_loss = np.inf

    for epoch in range(config.epochs):
        agent.train()
        total_loss = 0.0
        for batch in loader:
            loss = agent.training_step(batch)
            total_loss += loss

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(loader)

            # save
            if (epoch + 1) % 100 == 0:
                path = os.path.join(config.save_dir, config.task)
                os.makedirs(path, exist_ok=True)
                file_name = os.path.join(path, f'model_ep_{epoch + 1}_score_{avg_loss}.pth')
                agent.save_model(file_name)
                if avg_loss < best_loss:
                    print(best_loss)
                    best_loss = avg_loss
                    file_name = os.path.join(config.save_dir, f'model_{config.task}_stable.pth')
                    agent.save_model(file_name)

        pbar.update()

    return best_loss


def main():

    config = load_hyperparameters('config/hyperparameters_new.yaml')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data = load_data(
        data_path=os.path.join(config.load_dir, config.task),
        n_frames=config.n_frames,
        task=config.task
    )
    dataloader = DataLoader(TensorDataset(*data), batch_size=config.batch_size, shuffle=True)

    agent = Model(data[0].shape[-1], data[1].shape[-1], data[2].shape[-1], config.N_gaussians, device).to(device)

    best_score = train(agent, dataloader, config)

    print(f'Training finished with best score of {best_score}')


if __name__ == "__main__":
    main()

