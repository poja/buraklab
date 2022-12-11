import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

from arena import DT, SPACE_RESOLUTION, Arena


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


def sanity_check_random_walk():
    trials = 1000
    moves = 5
    x_dists = []
    y_dists = []
    arena = Arena(1, 1)
    for _ in range(trials):
        trajectory = random_walk(arena, DT * moves, 0.5)
        assert len(trajectory) == moves + 1
        b, e = trajectory[0], trajectory[-1]
        x_dists.append(e[0] - b[0])
        y_dists.append(e[1] - b[1])

    _, p = stats.normaltest(x_dists)
    assert p > 0.01
    _, p = stats.normaltest(y_dists)
    assert p > 0.01


if __name__ == "__main__":
    sanity_check_random_walk()

    arena = Arena(1, 1)
    trajectory = random_walk(arena, 1, 0.5)
    tx = [t[0] for t in trajectory]
    ty = [t[1] for t in trajectory]
    plot_trajectory(arena, trajectory)
