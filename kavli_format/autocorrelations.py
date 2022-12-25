import colormaps
import matplotlib.pyplot as plt
import numpy as np
import opexebo

from kavli_format.load_kavli import ExperimentSession, Module, load_data


def calculate_autocorrelations(session: ExperimentSession, module: Module):
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

    return cell_autocorrs


def show_module_autocorrelations(session: ExperimentSession):

    autocorrs_per_module = []
    for module in session.modules:
        cell_autocorrs = calculate_autocorrelations(session, module)
        autocorrs_per_module.append(cell_autocorrs)

    # Plotting average autocorrelation of the 3 modules.
    _, ax = plt.subplots(1, 3, dpi=100)
    ax[0].imshow(np.mean(autocorrs_per_module[0], axis=0), cmap=colormaps.parula)
    ax[1].imshow(np.mean(autocorrs_per_module[1], axis=0), cmap=colormaps.parula)
    ax[2].imshow(np.mean(autocorrs_per_module[2], axis=0), cmap=colormaps.parula)
    plt.show()


def calculate_grid_stats(session):
    """
    Runs grid_score on all the cells that have a module
    Returns a dict from cell name (number) to (autocorr map, grid score, grid stats dict)
    """
    cell_to_grid = dict()
    for module in session.modules:
        autocorrs = calculate_autocorrelations(session, module)
        for cell, autocorr in zip(module.cells, autocorrs):
            grid_score, grid_stats = opexebo.analysis.grid_score(autocorr)
            cell_to_grid[cell.name] = (autocorr, grid_score, grid_stats)

        median_spacing = np.median([cell_to_grid[c.name][2]["grid_spacing"] for c in module.cells])
        print(f"Median spacing: {median_spacing}")

    return cell_to_grid


if __name__ == "__main__":
    session = load_data()
    show_module_autocorrelations(session)
    # cell_to_grid = calculate_grid_stats(session)
    # with open("kavli_data/grid_stats.pickle", "wb") as f:
    #     pickle.dump(cell_to_grid, f)
