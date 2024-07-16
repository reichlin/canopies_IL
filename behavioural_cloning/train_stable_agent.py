import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from agent_stable import Agent, KDE
from encoder import EquivariantEncoder
from utils import load_trajectories, load_data, create_data_loader,load_hyperparameters, get_last_checkpoint, get_best_model
from tqdm import tqdm

def train(agent, train_loader, valid_loader, config):
    pbar = tqdm(total=config.epochs)
    best_score = -1

    tot_nll = []

    for epoch in range(config.epochs):
        agent.train() 
        total_loss, avg_nll = 0.0, 0
        for batch in train_loader:
            loss, nll = agent.training_step(batch)
            total_loss += loss
            avg_nll += nll

        tot_nll.append(avg_nll)

        if (epoch+1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            valid_loss = evaluate_model(agent, valid_loader)
            log_dict = {'Avg Loss': avg_loss, 'Val loss': valid_loss}

            #save
            if (epoch+1) % 100 == 0:
                score = round(1/valid_loss*10000)
                path = os.path.join(config.save_dir, config.task)
                os.makedirs(path, exist_ok=True)
                file_name_policy = os.path.join(path, f'policy_ep_{epoch+1}_score_{score}.pth')
                file_name_mdn = os.path.join(path, f'mdn_ep_{epoch + 1}_score_{score}.pth')
                agent.save_model(file_name_policy, file_name_mdn)
                if score > best_score:
                    print(best_score)
                    best_score = score
                    file_name_policy = os.path.join(config.save_dir, f'policy_{config.task}_stable.pth')
                    file_name_mdn = os.path.join(config.save_dir, f'mdn_{config.task}_stable.pth')
                    agent.save_model(file_name_policy, file_name_mdn)

        pbar.update()
    return best_score


def evaluate_model(agent, dataloader):
    agent.eval()  
    total_loss = 0.0

    #with torch.no_grad():  
    for obs,act in dataloader:
        action = agent.select_action(obs.to(agent.device))  
        loss = agent.criterion(action, act.to(agent.device))  
        total_loss += loss.item() * obs.size(0)

    avg_loss = total_loss / len(dataloader.dataset)

    return avg_loss



def main():
    config = load_hyperparameters('config/hyperparameters.yaml')
    config_encoder = load_hyperparameters('config/hyperparameters_encoder.yaml')

    #load the datataset and get the data loader
    data_path = config.load_dir
    X_train, y_train, X_test, y_test = load_data(
        input_ids=config.input_ids,
        output_ids=config.output_ids,
        data_path=os.path.join(config.load_dir, config.task),
        n_frames=config.n_frames,
        task=config.task,
        n_freq=int(config.data_freq/config.target_freq)
        )

    train_loader = create_data_loader(X_train, y_train, batch_size=config.batch_size, shuffle=True)
    valid_loader = create_data_loader(X_test, y_test, batch_size=config.batch_size, shuffle=False)
    input_dim = X_train.shape[-1]
    output_dim = y_train.shape[-1]

    #create the Agent, the Encoder and the density estimator
    agent = Agent(input_dim, config.hidden_size1, config.hidden_size2,config_encoder.hidden_size, output_dim, device=config.device)
    agent.encoder.load_model('./params/model_encode.pth')
    model_to_load = os.path.join(config.save_dir, f'model_{config.task}_stable.pth')
    try:
        agent.load_model(model_to_load)
    except:
        print(f'NO Model ({model_to_load}) to load')

    # # get all trajectories in equivariant representation and create the density estimator
    # dataset = load_trajectories(
    #     data_path=os.path.join(config.load_dir, config.task),
    #     n_frames=config.n_frames,
    #     task=config.task,
    #     n_freq = int(config.data_freq/config.target_freq)
    # )
    # states_tsr = torch.stack(dataset.states).float().to(agent.device)
    # z_states = agent.encoder(states_tsr).detach()
    # density_estimator = KDE(z_states)
    # agent.density_estimator = density_estimator

    

    print(f'Agent created:\n - Input/Output dim: {input_dim}/{output_dim}\n - Device: {config.device}')
    input(f'Press something to continue!')


    #file_load = get_best_model(config.save_dir,config.task)
    #if file_load is not None: agent.load_model(file_load)

    best_score = train(agent=agent,
        train_loader=train_loader,
        valid_loader=valid_loader,
        config=config
        )
    
    print(f'Training finished with best score of {best_score}')
    

if __name__ == "__main__":
    main()