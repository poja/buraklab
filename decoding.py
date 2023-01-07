import logging
import math
from dataclasses import dataclass
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tqdm
from scipy.special import softmax
from tqdm import trange

import colormaps
from arena import DT, SAMPLES_PER_SEC, SPACE_RESOLUTION, Arena
from random_walk import random_walk
from recording import record_cells
from spatial_cells import GridCell, SpatialCell


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
        biased_log_posterior[i] = _normalize_probs(np.ones([x_dynamic_range, y_dynamic_range]))

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


def decode_with_markov(
    arena: Arena, duration: float, modules: list[DecodingModule], walk_diffusion: float
) -> list[np.array]:
    """
    HMM Filtering formula:
      For each timepoint,
      p(s_newest | o_new) = normalization * p(o_newest | s_newest) *
                            * Sum_{s_old} [ p(s_old | o_old) * p(s_newest | s_old)]
    """
    dt = 1e-3

    x_dynamic_range = int(arena.x_height / SPACE_RESOLUTION)
    y_dynamic_range = int(arena.y_width / SPACE_RESOLUTION)
    t_dynamic_range = int(duration / dt)

    logging.info("Discretizing cells")
    all_cells = list(chain.from_iterable(m.cells_spikes for m in modules))
    discreticized_cells = [(c, c.discrete_firing_map(), _discretize_spikes(s, duration, dt)) for (c, s) in all_cells]

    current_posterior = _normalize_probs(np.ones([x_dynamic_range, y_dynamic_range]))
    all_posteriors = []
    for timepoint_i in tqdm.tqdm(range(t_dynamic_range)):

        # First step - calculate p(o_newest | s_newest)
        logging.debug("Calculating the likelihood of the current observation given every state")
        observation_loglikelihoods = np.ndarray([x_dynamic_range, y_dynamic_range])
        for (i, j) in _iterate_2d(observation_loglikelihoods):
            log_likelihood = 0
            for _, firing_map, spikes in discreticized_cells:
                # Function _biased_log_poisson is biased relative to k, but k doesn't depend on position in space
                firing_rate = firing_map[i][j]
                log_likelihood += _biased_log_poisson(firing_rate, dt, spikes[timepoint_i])
            observation_loglikelihoods[i][j] = log_likelihood

        # It's okay to normalize here because we are going to normalize anyway later
        observation_likelihoods = softmax(observation_loglikelihoods)

        gaussian_kernel_variance = 2 * walk_diffusion * dt
        gaussian_kernel_sd = math.sqrt(gaussian_kernel_variance)
        gaussed_previous = scipy.ndimage.gaussian_filter(current_posterior, gaussian_kernel_sd, mode="constant", cval=0)

        current_posterior = _normalize_probs(observation_likelihoods * gaussed_previous)
        all_posteriors.append(current_posterior)

    return all_posteriors


def _biased_log_poisson(rate, time, value):
    """
    This function is an accurate poisson for k=0 and k=1,
      but for k>1 it is missing an additive element related to k
    """
    overall_rate = rate * time
    if value == 0:  # optimization
        return -overall_rate
    return -overall_rate + math.log(overall_rate) * value


def _iterate_2d(arr: np.array):
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            yield (i, j)


def _discretize_spikes(spikes: list[float], duration: float, dt=DT):
    return np.histogram(spikes, bins=np.arange(0, duration + dt, dt))[0]


def _normalize_probs(arr: np.array):
    assert np.sum(arr) != 0
    return arr / np.sum(arr)


def _time_constant(module_fisher_information_rate, walk_diffusion):
    J = module_fisher_information_rate
    return 1 / math.sqrt(2 * walk_diffusion * J)


def kernel_decoding_example():
    arena = Arena(1, 1)

    module_spacings = [0.25, 0.4, 0.6]
    # module_spacings = [0.4]
    cells = []
    for spacing in module_spacings:
        for x in np.linspace(0, spacing, 8, endpoint=False):
            for y in np.linspace(0, spacing, 8, endpoint=False):
                cells.append(GridCell(arena, (x, y), 0.3, 100, 0.05, spacing))

    logging.info("Simulating trajectory and spikes")
    duration = 15
    walk_diffusion = 0.05
    trajectory = random_walk(arena, duration, walk_diffusion)
    recording = record_cells(trajectory, cells)

    logging.info("Starting to decode...")
    separated_by_module = [[] for _ in range(len(module_spacings))]
    for cell, spikes in recording.cell_spikes:
        for spacing_i, spacing in enumerate(module_spacings):
            if np.isclose(spacing, cell.field_distance):
                separated_by_module[spacing_i].append((cell, spikes))

    decoding_modules = []
    for module_data in separated_by_module:
        fisher_information_rate = len(module_data) * module_data[0][0].fisher_information_rate()
        time_constant = _time_constant(fisher_information_rate, walk_diffusion)
        decoding_modules.append(DecodingModule(time_constant, module_data))

    posteriors = decode_with_kernel(arena, duration, decoding_modules)

    logging.info("Showing posterior...")
    f, axarr = plt.subplots(2, 4)
    for i, timestamp in enumerate([1 * SAMPLES_PER_SEC, 4 * SAMPLES_PER_SEC, 10 * SAMPLES_PER_SEC, -1]):
        axarr[0][i].imshow(posteriors[timestamp], cmap=colormaps.parula)
        axarr[1][i].plot([trajectory[timestamp][1]], [trajectory[timestamp][0]], marker=".")
        axarr[1][i].set_xbound(0, 1)
        axarr[1][i].set_ybound(0, 1)
        axarr[1][i].invert_yaxis()

    plt.show()


def markov_decoding_example():
    arena = Arena(1, 1)

    module_spacings = [0.25, 0.4, 0.6]
    cells = []
    for spacing in module_spacings:
        for x in np.linspace(0, spacing, 5, endpoint=False):
            for y in np.linspace(0, spacing, 5, endpoint=False):
                cells.append(GridCell(arena, (x, y), 0.3, 75, 0.05, spacing))

    logging.info("Simulating trajectory and spikes")
    duration = 0.5
    walk_diffusion = 0.05
    trajectory = random_walk(arena, duration, walk_diffusion)
    recording = record_cells(trajectory, cells)

    logging.info("Starting to decode...")
    decoding_module = DecodingModule(0, recording.cell_spikes)  # this is a bit hacky
    posteriors = decode_with_markov(arena, duration, [decoding_module], walk_diffusion)

    logging.info("Showing posterior...")
    f, axarr = plt.subplots(2, 4)
    for i, timestamp in enumerate([5, 100, 400, -1]):
        axarr[0][i].imshow(posteriors[timestamp], cmap=colormaps.parula)
        axarr[1][i].plot([trajectory[timestamp][1]], [trajectory[timestamp][0]], marker=".")
        axarr[1][i].set_xbound(0, 1)
        axarr[1][i].set_ybound(0, 1)
        axarr[1][i].invert_yaxis()

    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(name)s :: %(message)s")
    markov_decoding_example()
    # cProfile.run("markov_decoding_example()", "decoding.prof")
