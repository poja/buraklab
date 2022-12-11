from dataclasses import dataclass

import matplotlib.pyplot as plt

from arena import DT, Arena
from random_walk import random_walk
from spatial_cells import GridCell, SpatialCell


@dataclass
class Recording:
    trajectory: list[(float, float)]
    cell_spikes: list[(SpatialCell, list[float])]  # for each cell, the times of spikes


def record_cells(trajectory, spatial_cells: list[SpatialCell]) -> Recording:
    spikes = [(cell, []) for cell in spatial_cells]
    for time_i, position in enumerate(trajectory):
        for (cell_i, cell) in enumerate(spatial_cells):
            if cell.did_spike(position, DT):
                spikes[cell_i][1].append(time_i * DT)
    return Recording(trajectory, spikes)


def plot_spike_positions(trajectory, spike_times, marker_size=0.2):
    spike_positions = []
    for spike in spike_times:
        spike_positions.append(trajectory[round(spike / DT)])

    plt.scatter(*zip(*spike_positions), s=marker_size)
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    arena = Arena(1, 1)
    grid_cell = GridCell(arena, (0.1, 0.1), 0.3, 100, 0.08, 0.3)
    trajectory = random_walk(arena, 200, 0.5)
    recording = record_cells(trajectory, [grid_cell])
    spike_times = recording.cell_spikes[0][1]
    plot_spike_positions(trajectory, spike_times)
