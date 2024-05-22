import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from agent import Agent
from utils import load_data, create_data_loader,load_hyperparameters, get_last_checkpoint, get_best_model
import wandb
from tqdm import tqdm

def train(agent, train_loader, valid_loader, config):
    pbar = tqdm(total=config.epochs)
    best_score = -1

    for epoch in range(config.epochs):
        agent.train() 
        total_loss = 0.0
        for batch in train_loader:
            loss = agent.training_step(batch)
            total_loss += loss

        if (epoch+1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            valid_loss = evaluate_model(agent, valid_loader)
            log_dict = {'Avg Loss': avg_loss, 'Val loss': valid_loss}
            wandb.log(log_dict)

            #save
            if (epoch+1) % 100 == 0:
                score = round(1/valid_loss*10000)
                path = os.path.join(config.save_dir,config.task)
                os.makedirs(path, exist_ok=True)
                file_name = os.path.join(path, f'model_ep_{epoch+1}_score_{score}.pth')
                agent.save_model(file_name)
                if score > best_score:
                    best_score = score
                    file_name = os.path.join(config.save_dir, f'model_{config.task}.pth')
                    agent.save_model(file_name)

        pbar.update()
    return best_score


def evaluate_model(model, dataloader):
    model.eval()  
    total_loss = 0.0

    with torch.no_grad():  
        for inputs, labels in dataloader:
            outputs = model(inputs.to(model.device))  
            loss = model.criterion(outputs, labels.to(model.device))  
            total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(dataloader.dataset)

    return avg_loss



def main():
    config = load_hyperparameters('config/hyperparameters.yaml')
    print(f'Inputs :{config.input_ids} of size unknown for {config.n_frames} frames')
    print(f'Outputs :{config.output_ids} of size {config.output_size}')
    print(f'Hidden: {hidden_size1}, {hidden_size2}')
    print(f'Device :{config.device}')
    print(f'Task: {config.task}')
    input('Continue?')
    wandb.init(project="Canopies Behavioural Cloning", config=config, name=f"{config.task}")

    data_path = config.load_dir
    X_train, y_train, X_test, y_test = load_data(
        input_ids=config.input_ids,
        output_ids=config.output_ids,
        data_path=os.path.join(config.load_dir, config.task),
        n_frames=config.n_frames,
        task=config.task)

    train_loader = create_data_loader(X_train, y_train, batch_size=config.batch_size, shuffle=True)
    valid_loader = create_data_loader(X_test, y_test, batch_size=config.batch_size, shuffle=False)

    agent = Agent(X_train.shape[-1], config.hidden_size1, config.hidden_size2, config.output_size, device=config.device)
    agent.to(config.device)
    print(f'Agent created an put in {config.device}')

    agent.optimizer = optim.Adam(agent.parameters(), lr=config.lr)
    model_to_load = os.path.join(config.save_dir, f'model_{config.task}.pth')
    try:
        agent.load_model(model_to_load)
    except:
        print(f'NO Model ({model_to_load}) to load')
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