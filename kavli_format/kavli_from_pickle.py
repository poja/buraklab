import IPython
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import opexebo
import pickle
import colormaps
from pathlib import Path


@dataclass
class Spike:
    timestamp: float
    x_position: float
    y_position: float


@dataclass
class Cell:
    cell_name: int
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


def load_data():
    # data_folder = "/home/vnormand/Documents/PhD/Data/Phase/Files_v2/Ratbbi/400/0309/"
    data_folder = Path("kavli_data")
    session_time = "2020-03-09 16:20:09"
    with open(data_folder / "data-ratbbi_03-09.pickle", "rb") as f:
        data = pickle.load(f)

    idx_1 = np.load(data_folder / "mod1.npy", allow_pickle=True)
    idx_2 = np.load(data_folder / "mod2.npy", allow_pickle=True)
    idx_3 = np.load(data_folder / "mod3.npy", allow_pickle=True)

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

    trajectory = Trajectory(data["time"], data["x"], data["y"])
    return ExperimentSession(data["arena"], session_time, "Ratbbi", trajectory, modules)


def show_module_autocorrelations(session: ExperimentSession):

    autocorrs_per_module = []
    for module in session.modules:
        traj_x, traj_y = session.trajectory.x, session.trajectory.y
        bin_edges = [
            np.linspace(np.min(traj_x), np.max(traj_x), np.int16((session.arena_length / module.best_binning))),
            np.linspace(np.min(traj_y), np.max(traj_y), np.int16((session.arena_length / module.best_binning))),
        ]
        trajectory = session.trajectory
        masked_map, _, bin_edges = opexebo.analysis.spatial_occupancy(
            time=trajectory.timestamps,
            position=np.array([traj_x, traj_y]),
            arena_size=session.arena_length,
            bin_edges=bin_edges,
        )

        cell_autocorrs = []
        for cell in module.cells:
            map = opexebo.analysis.rate_map(
                occupancy_map=masked_map,
                spikes_tracking=np.array(
                    [
                        [s.timestamp for s in cell.spikes],
                        [s.x_position for s in cell.spikes],
                        [s.y_position for s in cell.spikes],
                    ]
                ),
                arena_size=session.arena_length,
                bin_edges=bin_edges,
            )
            map = opexebo.general.smooth(map, module.best_smoothing)
            cell_autocorrs.append(opexebo.analysis.autocorrelation(map))

        autocorrs_per_module.append(cell_autocorrs)

    # Plotting average autocorrelation of the 3 modules.
    _, ax = plt.subplots(1, 3, dpi=100)
    ax[0].imshow(np.mean(autocorrs_per_module[0], axis=0), cmap=colormaps.parula)
    ax[1].imshow(np.mean(autocorrs_per_module[1], axis=0), cmap=colormaps.parula)
    ax[2].imshow(np.mean(autocorrs_per_module[2], axis=0), cmap=colormaps.parula)
    plt.show()


if __name__ == "__main__":
    experiment_session = load_data()
    show_module_autocorrelations(experiment_session)
