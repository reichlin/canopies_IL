import numpy as np
import matplotlib.pyplot as plt
import numpy as np

class AttractivePotentialField:
    def __init__(self, goal, alpha):

        self.goal = np.array(goal)
        self.alpha = alpha

    def get_magnitude(self,dist):
        return  1/(1+np.exp(dist)*0.1)#1 / (1 + np.e ** (x))

    def compute_force(self, position):
        position = np.array(position)
        dist = self.goal - position
        director = dist/np.linalg.norm(dist)
        d = np.linalg.norm(dist, axis=1)
        x = self.alpha * self.get_magnitude(d)
        force = director * np.expand_dims(x, -1) #* position**(-2)
        print(np.min(d, axis=0), np.max(d, axis=0))
        return force

    def plot_potential_field(self, xlim, ylim, resolution=50):
        x = np.linspace(xlim[0], xlim[1], resolution)
        y = np.linspace(ylim[0], ylim[1], resolution)
        X, Y = np.meshgrid(x, y)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(resolution):
            for j in range(resolution):
                position = np.array([X[i, j], Y[i, j]])
                force = self.compute_force(position)
                U[i, j] = force[0]
                V[i, j] = force[1]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(X, Y, U, V, color='r')
        ax.plot(self.goal[0], self.goal[1], 'bo', label='Goal')
        plt.title('Attractive Potential Field')
        plt.legend()
        plt.grid()
        plt.show()


# Example usage
goal = [0,0,0]
potential_field = AttractivePotentialField(goal, alpha=1.0)

# Compute the force at a specific position
t = np.linspace(0.00001,1,30)
#positions = np.stack([np.array([x, y, z]) for x in t  for y in t  for z in t])
positions = np.stack([np.array([x, 0.0, 0.0]) for x in t])

force = potential_field.compute_force(positions)

dist = np.linalg.norm(force, axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], alpha=0.1, c=dist)
fig.show()

plt.scatter(positions[:, 0], positions[:, 1], c=dist)
plt.show()

plt.scatter(positions[:, 0], np.linalg.norm(force, axis=1))
plt.show()

# Plot the potential field
#potential_field.plot_potential_field(xlim=(-10, 10), ylim=(-10, 10), resolution=20)
