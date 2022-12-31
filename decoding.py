import tqdm
from logging import Logger
import logging
import IPython
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import math
from arena import DT, SPACE_RESOLUTION, Arena
import colormaps
from random_walk import random_walk
from recording import record_cells
from spatial_cells import GridCell, SpatialCell
import numpy as np
from scipy.special import softmax


def decode_with_kernel(
    arena: Arena, cells_spikes: list[(SpatialCell, list[float])], duration: float, time_constant: float
) -> list[np.array]:
    """
    Returns for each timepoint the decoded posterior distribution

    TODO allow different time constants for different modules
    """
    x_dynamic_range = int(arena.x_height / SPACE_RESOLUTION)
    y_dynamic_range = int(arena.y_width / SPACE_RESOLUTION)
    t_dynamic_range = int(duration / DT)

    # This is the value that we update in each iteration.
    # Then to get posterior we will need to "softmax": exp(..) then normalize to sum 1.
    biased_log_posterior = softmax(np.ones([x_dynamic_range, y_dynamic_range]))

    logging.info("Discretizing spikes")
    discretized_spikes = [(c, _discretize_spikes(sp, duration)) for (c, sp) in cells_spikes]

    logging.info("Calculating firing maps")
    firing_maps = []
    for cell, _ in tqdm.tqdm(discretized_spikes):
        firing_maps.append(cell.discrete_firing_map())

    logging.info("Calculating posteriors")
    posteriors = []
    for t in trange(t_dynamic_range):
        biased_log_posterior *= math.exp(-DT / time_constant)
        for cell_i, (_, spikes) in enumerate(discretized_spikes):
            if spikes[t] == 0:
                continue
            assert spikes[t] == 1, f"There are {spikes[t]} spikes in the same DT window"
            biased_log_posterior += np.log(firing_maps[cell_i])
        posteriors.append(softmax(biased_log_posterior))

    return posteriors


def _discretize_spikes(spikes: list[float], duration: float):
    return np.histogram(spikes, bins=np.arange(0, duration + DT, DT))[0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(name)s :: %(message)s")
    arena = Arena(1, 1)
    cells = []
    duration = 15
    for x in np.arange(0, 0.3, 0.05):
        for y in np.arange(0, 0.3, 0.05):
            cells.append(GridCell(arena, (x, y), 0.3, 5, 0.08, 0.3))
    logging.info("Simulating trajectory and spikes")
    trajectory = random_walk(arena, duration, 0.5)
    recording = record_cells(trajectory, cells)
    logging.info("Starting to decode...")
    posteriors = decode_with_kernel(arena, recording.cell_spikes, duration, 0.2)
    print(len(posteriors))
    plt.imshow(posteriors[int(1 / DT)], cmap=colormaps.parula)
    plt.show()
