import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from encoder import EquivariantEncoder
from utils import load_trajectories, create_data_loader,load_hyperparameters, get_last_checkpoint, get_best_model
import wandb
from tqdm import tqdm
import random


def train(model, dataset, config):
    pbar = tqdm(total=config.epochs)
    best_loss = 10000

    for epoch in range(config.epochs):
        model.train() 
        total_loss = 0.0
        batch = sample_batch(dataset, config.batch_size, model.device)
        loss = model.training_step(batch)

        if (epoch+1) % 10 == 0:
            batch = sample_batch(dataset, config.batch_size, model.device)
            valid_loss = evaluate_model(model, batch)
            log_dict = {'Train Loss': loss, 'Val loss': valid_loss}
            wandb.log(log_dict)

            #save
            if (epoch+1) % 100 == 0:
                path = os.path.join(config.save_dir,config.param_tag)
                os.makedirs(path, exist_ok=True)
                file_name = os.path.join(path, f'model_ep_{epoch+1}.pth')
                model.save_model(file_name)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    file_name = os.path.join(config.save_dir, f'model_{config.param_tag}.pth')
                    model.save_model(file_name)

        pbar.update()
    return best_loss

def sample_batch(dataset, bs, device):
    indices = [i for i in range(len(dataset.actions))]
    id_batch = random.sample(indices,bs)
    states = torch.stack([dataset.states[i] for i in id_batch]).float().to(device)
    dpos = torch.stack([dataset.dpos[i] for i in id_batch]).float().to(device)
    #actions = torch.stack([dataset.actions[i] for i in id_batch]).float().to(device)
    next_states = torch.stack([dataset.next_states[i] for i in id_batch]).float().to(device)
    return (states, dpos, next_states)


def evaluate_model(model, batch):
    model.eval()  
    states, actions, next_states = batch 
    with torch.no_grad():  
        loss = model.criterion(model(next_states),model(states) + actions )
    return loss.item()



def main():
    config = load_hyperparameters('config/hyperparameters_encoder.yaml')
    
    wandb.init(project="Canopies Behavioural Cloning", config=config, name="encoder_training")

    data_path = config.load_dir

    dataset = load_trajectories(
        data_path=os.path.join(config.load_dir, config.task),
        n_frames=config.n_frames,
        task=config.task,
        n_freq = int(config.data_freq/config.target_freq)
        )
    
    input_dim = dataset.states[0].shape[-1]
    output_dim = dataset.actions[0].shape[-1]

    print(f'input/output dim: {input_dim}/{output_dim}')

    encoder = EquivariantEncoder(input_dim, output_dim, config.hidden_size,  device=config.device)
    encoder.to(config.device)
    print(f'Encoder created an put in {config.device}')
    
    best_loss = train(model=encoder, dataset=dataset, config=config)
    
    print(f'Training finished with best score of {best_loss}')
    
    

if __name__ == "__main__":
    main()