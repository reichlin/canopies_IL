import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
from utils_rl import Agent
import torch


























data2label = {
    'traj_imitation.npz': ('IL trajectory','blue'), 
    #'traj_imitation_stable.npz': 'IL traj stable', #0.001
    #'traj_imitation_stable_1.npz': ('IL traj stable 1','red'),  #0.005
    #'traj_imitation_stable_2.npz': 'IL traj stable 2',  #0.002
}

def plot_trajectories(data_dir):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    agent = Agent(56, 3, 3, 25, stable=True,  device=device).to(device)
    model_file = '/home/alfredo/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/behavioural_cloning/params/model_grasp_new.pth'
    agent.load_model(model_file)

    # agent = Agent(input_size=56, hidden_size1=128, hidden_size2=128, output_size=3, stable=True, device='cuda' if torch.cuda.is_available() else 'cpu')
    # encoder_file = '/home/alfredo/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/behavioural_cloning/params/model_encode.pth'
    # policy_file = '/home/alfredo/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/behavioural_cloning/params/policy_grasp_stable.pth'
    # mdn_file = '/home/alfredo/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/behavioural_cloning/params/mdn_grasp_stable.pth'
    # agent.load_model(policy_file, mdn_file)
    # agent.encoder.load_model(encoder_file)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    folder_path = os.path.join(data_dir, 'grasp_new')
    labelling = True
    # Iterate over all files in the folder
    reached_grapes = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npz') and not file_name == 'traj_imitation.npz':
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)
            points = data['ee_pos']
            grapes = data['obj_poses']
            grape_idx = np.argmin(np.sum(np.abs(points[-1]-grapes[0]), -1))
            grape = grapes[0,grape_idx]
            #ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=file_name)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color='grey', alpha=0.5, label='collected trajectory' if labelling else '')
            ax.scatter(grape[0], grape[1], grape[2], color='tab:blue', alpha=0.5)
            labelling = False

            reached_grapes.append(grapes[0,grape_idx:grape_idx+1])

            # rho = agent.MDN(torch.from_numpy(grape).view(1, -1).float().to(agent.device)).view(1, 25, 3).detach().cpu().numpy()
            # for i in range(rho.shape[1]):
            #     ax.scatter(rho[0,i,0], rho[0,i,1], rho[0,i,2], color='blue', alpha=0.2, s=100)

    reached_grapes = np.concatenate(reached_grapes, 0)

    folder_path = os.path.join(data_dir, 'grasp_results')
    labelling = True
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npz') and not file_name == 'traj_imitation.npz':
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)
            points = data['arr_2']
            #ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=file_name)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color='tab:orange', alpha=1.0, label='test trajectory' if labelling else '')
            labelling = False

    #ax.scatter(0.3508995064453269, -0.34037464597250544, 1.4926518235947732, color='tab:red', alpha=1.0)
    ax.scatter(0.40, -0.32, 1.5, color='tab:red', alpha=1.0, s=150)

    plt.legend()
    plt.tight_layout()
    plt.show()



    folder_path = os.path.join(data_dir, 'grasp_results')
    cmap = plt.cm.get_cmap('hsv',5)

    for i,file_name in enumerate(os.listdir(folder_path)):
        if file_name in list(data2label.keys()):
            traj_imit = os.path.join(folder_path,file_name)
            data = np.load(traj_imit)
            points = data['arr_2'][:-1]
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=data2label[file_name][1], label=data2label[file_name][0], linewidth=2)
    IL_traj = points



    #points = np.stack([points[i] for i in range(points.shape[0]) if not (i<=6 and i>=2)])

    #traj_imit_stable = os.path.join(folder_path,'traj_imitation_stable.npz')
    #data = np.load(traj_imit_stable)
    #points = data['arr_2']
    #points = np.stack([points[i] for i in range(points.shape[0]) if not (i<=2 and i>=1)])
    #ax.plot(points[:, 0], points[:, 1], points[:, 2], color='green', label='stable IL trajectory')

    #traj_imit_stable = os.path.join(folder_path,'traj_imitation_stable_1.npz')
    #data = np.load(traj_imit_stable)
    #points = data['arr_2']
    #points = np.stack([points[i] for i in range(points.shape[0]) if not (i<=2 and i>=1)])
    #ax.plot(points[:, 0], points[:, 1], points[:, 2], color='purple', label='stable IL trajectory 1')

    # Customize the plot
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    #plt.title('Trajectories Plot')
    plt.legend(loc='best')

    # Show the plot
    plt.show()


    folder_path = os.path.join(data_dir, 'heatmaps')
    file_name = os.path.join(folder_path,'heatmap.npz')
    data = np.load(file_name)
    grid_points_prob = data['arr_0']
    grid_points = data['arr_1']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    n = grid_points_prob.shape[0]
    #cmap = plt.cm.get_cmap('viridis')
    cmap = plt.get_cmap('viridis')
    p_min, p_max = np.min(grid_points_prob), np.max(grid_points_prob)

    for p,point,i in zip(grid_points_prob, grid_points, range(n)):
        if i%1 == 0:
            p_new = rescale(p.item(),p_min, p_max)
            ax.scatter(point[0], point[1], point[2], alpha=p_new, color=cmap(p_new), s=100)#, label='probability distribution' if i<1 else '')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.legend(loc='best')

    # Show the plot
    plt.show()

def rescale(p,p_min,p_max):
    return (p-p_min)/(p_max - p_min ) 



if __name__ == "__main__":
    data_dir = '/home/alfredo/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/data'
    plot_trajectories(data_dir)
