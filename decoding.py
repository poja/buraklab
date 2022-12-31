from dataclasses import dataclass
import tqdm
from logging import Logger
import logging
import IPython
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import math
from arena import DT, SAMPLES_PER_SEC, SPACE_RESOLUTION, Arena
import colormaps
from random_walk import random_walk
from recording import record_cells
from spatial_cells import GridCell, SpatialCell
import numpy as np
from scipy.special import softmax


@dataclass
class DecodingModule:
    time_constant: float
    cells_spikes: list[(SpatialCell, list[float])]


def decode_with_kernel(arena: Arena, duration: float, modules: list[DecodingModule]) -> list[np.array]:
    """
    Returns for each timepoint the decoded posterior distribution

    cell_info includes the cell, the list of spike times, and an "integration time constant".
    """
    x_dynamic_range = int(arena.x_height / SPACE_RESOLUTION)
    y_dynamic_range = int(arena.y_width / SPACE_RESOLUTION)
    t_dynamic_range = int(duration / DT)

    # This is the value that we update in each iteration.
    # Then to get posterior we will need to "softmax": exp(..) then normalize to sum 1.
    biased_log_posterior = np.ndarray((len(modules), x_dynamic_range, y_dynamic_range))
    for i in range(len(modules)):
        biased_log_posterior[i] = softmax(np.ones([x_dynamic_range, y_dynamic_range]))

    logging.info("Discretizing spikes")
    discretized_spikes: list[SpatialCell, list[float], int] = []  # includes module number
    for mod_i, module in enumerate(modules):
        discretized_spikes.extend([(c, _discretize_spikes(sp, duration), mod_i) for (c, sp) in module.cells_spikes])

    logging.info("Calculating firing maps")
    firing_maps = []
    for cell, _, _ in tqdm.tqdm(discretized_spikes):
        firing_maps.append(cell.discrete_firing_map())

    logging.info("Calculating posteriors")
    posteriors = []
    for t in trange(t_dynamic_range):
        for mod_i, module in enumerate(modules):
            biased_log_posterior[mod_i] *= math.exp(-DT / module.time_constant)
        for cell_i, (_, spikes, mod_i) in enumerate(discretized_spikes):
            if spikes[t] == 0:
                continue
            assert spikes[t] == 1, f"There are {spikes[t]} spikes in the same DT window"
            biased_log_posterior[mod_i] += np.log(firing_maps[cell_i])

        posteriors.append(softmax(np.sum(biased_log_posterior, axis=0)))

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
    decoding_modules = [DecodingModule(0.2, recording.cell_spikes)]
    posteriors = decode_with_kernel(arena, duration, decoding_modules)
    print(len(posteriors))
    plt.imshow(posteriors[1 * SAMPLES_PER_SEC], cmap=colormaps.parula)
    plt.show()
