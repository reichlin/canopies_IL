import numpy as np
import matplotlib.pyplot as plt
import os

data2label = {
    'traj_imitation.npz': ('IL trajectory','blue'), 
    #'traj_imitation_stable.npz': 'IL traj stable', #0.001
    #'traj_imitation_stable_1.npz': ('IL traj stable 1','red'),  #0.005
    #'traj_imitation_stable_2.npz': 'IL traj stable 2',  #0.002
}

def plot_trajectories(data_dir):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    folder_path = os.path.join(data_dir, 'grasp')
    labelling = True
    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npz') and not file_name == 'traj_imitation.npz':
            file_path = os.path.join(folder_path, file_name)
            data = np.load(file_path)
            points = data['arr_2']
            #ax.scatter(points[:, 0], points[:, 1], points[:, 2], label=file_name)
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color='grey', alpha=0.5, label='collected trajectory' if labelling else '')
            labelling = False
    
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
        if i%1 ==0:
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
    data_dir = '/home/adriano/Desktop/canopies/code/CanopiesSimulatorROS/workspace/src/imitation_learning/data' 
    plot_trajectories(data_dir)
