import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

TIME_RESOLUTION = DT = 1e-3  # sec
SPACE_RESOLUTION = 1e-3  # meters

Arena = namedtuple("Arena", ["x_height", "y_width"])


class Arena:
    def __init__(self, x_height, y_width):
        self.x_height = x_height
        self.y_width = y_width

    def does_contain(self, position):
        return (0 < position[0] < self.x_height) and (0 < position[1] < self.y_width)

    def random_position(self):
        return np.random.uniform(0, self.x_height), np.random.uniform(0, self.y_width)


def random_walk(arena, duration, diffusion):
    # Notice equal priors:
    starting_position = arena.random_position()
    current_position = starting_position
    trajectory = [current_position]
    for _ in np.arange(0, duration, DT):
        new_position = get_new_position(arena, current_position, diffusion)
        trajectory.append(new_position)
        current_position = new_position
    return trajectory


def get_new_position(arena, current_position, diffusion):
    next_pos = (-1, -1)
    while not arena.does_contain(next_pos):
        displacement = np.random.multivariate_normal([0, 0], np.eye(2) * 2 * diffusion * DT)
        next_pos = _round(current_position[0] + displacement[0]), _round(current_position[1] + displacement[1])
    return next_pos


def plot_trajectory(arena, trajectory: list[tuple[float, float]]):
    x, y = zip(*trajectory)
    plt.plot(x, y)
    plt.xlim([0, arena.x_height])
    plt.ylim([0, arena.y_width])
    plt.show()


def _round(coordinate):
    return round(coordinate / SPACE_RESOLUTION) * SPACE_RESOLUTION


if __name__ == "__main__":
    arena = Arena(1, 1)
    trajectory = random_walk(arena, 1, 0.5)
    tx = [t[0] for t in trajectory]
    ty = [t[1] for t in trajectory]
    plot_trajectory(arena, trajectory)
