import numpy as np
import matplotlib.pyplot as plt
import os

def plot_trajectories(data_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    folder_path = os.path.join(data_dir, 'grasp')
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npz') and not file_name == 'traj_imitation.npz':
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)
            points = data['arr_2']
            #ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=file_name)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color='grey', label=file_name)
    
    folder_path = os.path.join(data_dir, 'grasp_results')

    traj_imit = os.path.join(folder_path,'traj_imitation.npz')
    data = np.load(traj_imit)
    points = data['arr_2']
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color='blue', label='traj_imitation')

    traj_imit_stable = os.path.join(folder_path,'traj_imitation_stable.npz')
    data = np.load(traj_imit_stable)
    points = data['arr_2']
    ax.plot(points[:, 0], points[:, 1], points[:, 2], color='green', label='traj_imitation_stable')

    # Customize the plot
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title('Trajectories Scatter Plot')
    plt.legend(loc='best')

    # Show the plot
    plt.show()

if __name__ == "__main__":
    data_dir = '/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/data' 
    plot_trajectories(data_dir)
