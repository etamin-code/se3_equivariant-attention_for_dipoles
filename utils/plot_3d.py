import numpy as np
import matplotlib; matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Fixing random state for reproducibility


def random_walk(num_steps, max_step=0.05):
    """Return a 3D random walk as (num_steps, 3) array."""
    start_pos = np.random.random(3)
    steps = np.random.uniform(-max_step, max_step, size=(num_steps, 3))
    walk = start_pos + np.cumsum(steps, axis=0)
    return walk


def update_lines(num, walks, lines, angles, box_size, charges):
    for line, walk in zip(lines, walks):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(walk[:num, :2].T)
        line.set_3d_properties(walk[:num, 2])
    global dipoles
    dipoles.remove()
    dipoles = ax.quiver(walks[:, num, 0], walks[:, num, 1], walks[:, num, 2],
                        angles[num, 0, :], angles[num, 1, :], angles[num, 2, :], colors=charges, length=box_size / 2)
    return lines


def plot_3d(walks, angles, charges, box_size=1, name="sim.gif", show=False):
    charges = ['r' if el==1 else 'b' for el in charges]
    # Attaching 3D axis to the figure
    fig = plt.figure()
    global ax
    ax = fig.add_subplot(projection="3d")
    # Create lines initially without data
    lines = [ax.plot([], [], [])[0] for _ in walks]
    global dipoles
    dipoles = ax.quiver([], [], [], [], [], [], pivot='middle', normalize=True)

    # Setting the axes properties
    ax.set(xlim3d=(-box_size, box_size), xlabel='X')
    ax.set(ylim3d=(-box_size, box_size), ylabel='Y')
    ax.set(zlim3d=(-box_size, box_size), zlabel='Z')

    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig, update_lines, len(walks[0]), fargs=(walks, lines, angles, box_size, charges), interval=100)
    ani.save(name, fps=30)

    if show:
        plt.show()


if __name__ == "__main__":
    # Data: 40 random walks as (num_steps, 3) arrays
    num_steps = 30
    walks = [random_walk(num_steps) for index in range(5)]

    # Attaching 3D axis to the figure
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    # Create lines initially without data
    lines = [ax.plot([], [], [])[0] for _ in walks]

    # Setting the axes properties
    ax.set(xlim3d=(0, 1), xlabel='X')
    ax.set(ylim3d=(0, 1), ylabel='Y')
    ax.set(zlim3d=(0, 1), zlabel='Z')

    # Creating the Animation object
    ani = animation.FuncAnimation(
        fig, update_lines, num_steps, fargs=(walks, lines), interval=100)
    ani.save('orbita.gif', writer='imagemagick', fps=15)
    plt.show()
