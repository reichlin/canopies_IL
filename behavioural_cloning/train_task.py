from utils import get_config
from types import SimpleNamespace
import yaml
import os
from train_all import train
from utils import load_data


def choose_task(path):
    tasks = [x for x in os.listdir(path) if not os.path.isfile(os.path.join(path, x)) ]
    print('Available task-trajectories: ')
    for i, task in enumerate(tasks):    
        task_path = os.path.join(path,task)
        files = [f for f in os.listdir(task_path) if os.path.isfile(os.path.join(task_path, f)) and f.endswith('.npz') ]

        print(f' {i}. {task} ({len(files) } traj.)')
    n=-1
    while n<0 or n>i:
        n = int(input(f'Choose the task number to train on [0-{i}]: '))

    print(f'You chose {tasks[n]}')
    return tasks[n]
    
    
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


if __name__ == "__main__":

    with open('config/hyperparameters.yaml', "r") as f:
        config_dict = yaml.safe_load(f)
    config = SimpleNamespace(**config_dict)
    data_dir = os.path.join(os.path.dirname(os.getcwd()), config.load_dir)
    config.task = choose_task(data_dir)

    main()
