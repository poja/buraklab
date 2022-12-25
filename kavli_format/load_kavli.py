import pickle
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np


@dataclass
class Spike:
    timestamp: float
    x_position: float
    y_position: float


@dataclass
class Cell:
    name: int
    spikes: list[Spike]


@dataclass
class Module:
    cells: list[Cell]
    best_binning: int
    best_smoothing: int


@dataclass
class Trajectory:
    timestamps: list[float]
    x: list[float]
    y: list[float]


@dataclass
class ExperimentSession:
    arena_length: float  # cm
    session_time: str
    rat_name: str
    trajectory: Trajectory

    modules: list[Module]
    unsorted_cells: list[Cell]


def load_data(load_unsorted_cells=False):
    # data_folder = "/home/vnormand/Documents/PhD/Data/Phase/Files_v2/Ratbbi/400/0309/"
    data_folder = Path("kavli_data")
    session_time = "2020-03-09 16:20:09"
    with open(data_folder / "data-ratbbi_03-09.pickle", "rb") as f:
        data = pickle.load(f)

    idx_1 = np.load(data_folder / "mod1.npy", allow_pickle=True)
    idx_2 = np.load(data_folder / "mod2.npy", allow_pickle=True)
    idx_3 = np.load(data_folder / "mod3.npy", allow_pickle=True)
    union = reduce(np.union1d, [idx_1, idx_2, idx_3])

    recommended_binnings = [5, 7, 10]
    recommended_smoothings = [2, 2, 3]

    modules: list[Module] = []
    for module_i, module in enumerate([idx_1, idx_2, idx_3]):
        cells: list[Cell] = []
        for cell_i in module:
            spikes = [
                Spike(data["spike_times"][cell_i][s], data["x_pos"][cell_i][s], data["y_pos"][cell_i][s])
                for s in range(len(data["spike_times"][cell_i]))
            ]
            cells.append(Cell(cell_i, spikes))
        modules.append(Module(cells, recommended_binnings[module_i], recommended_smoothings[module_i]))

    if not load_unsorted_cells:
        unsorted_cells = None
    else:
        unsorted_cells = []
        for cell_i in range(len(data["spike_times"])):
            if cell_i in union:
                continue
            spikes = [
                Spike(data["spike_times"][cell_i][s], data["x_pos"][cell_i][s], data["y_pos"][cell_i][s])
                for s in range(len(data["spike_times"][cell_i]))
            ]
            unsorted_cells.append(Cell(cell_i, spikes))

    trajectory = Trajectory(data["time"], data["x"], data["y"])
    return ExperimentSession(data["arena"], session_time, "Ratbbi", trajectory, modules, unsorted_cells)


if __name__ == "__main__":
    session = load_data()
