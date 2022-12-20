import matplotlib.pyplot as plt
import numpy as np
import opexebo
import pickle
import colormaps
from pathlib import Path


# Loading data.
# data_folder = "/home/vnormand/Documents/PhD/Data/Phase/Files_v2/Ratbbi/400/0309/"
data_folder = Path("data")
session_time = "2020-03-09 16:20:09"
with open(data_folder / "data-ratbbi_03-09.pickle", "rb") as f:
    data = pickle.load(f)

# This load the indexes for the grid cells of module 3.
idx_1 = np.load(data_folder / "mod1.npy", allow_pickle=True)
idx_2 = np.load(data_folder / "mod2.npy", allow_pickle=True)
idx_3 = np.load(data_folder / "mod3.npy", allow_pickle=True)
idx_list = [idx_1, idx_2, idx_3]

# Modules have very different spacing, so we need to use different resolutions.
bin_list = [5, 7, 10]
smooth_list = [2, 2, 3]


"""
Data is a pretty straightforward structure. You have the position data, timestamps for all cells and more.
It is a collection.
"""

corr_list_tot = [None] * len(idx_list)
for mNo in range(len(idx_list)):
    arena_size = data["arena"]
    bin_edges = [
        np.linspace(np.min(data["x"]), np.max(data["x"]), np.int16((data["arena"] / bin_list[mNo]))),
        np.linspace(np.min(data["y"]), np.max(data["y"]), np.int16((data["arena"] / bin_list[mNo]))),
    ]
    masked_map, coverage, bin_edges = opexebo.analysis.spatial_occupancy(
        time=data["time"], position=np.array([data["x"], data["y"]]), arena_size=arena_size, bin_edges=bin_edges
    )

    corr_list = [None] * len(idx_list[mNo])
    for cNo in range(0, len(idx_list[mNo])):
        cell_index = idx_list[mNo][cNo]
        map = opexebo.analysis.rate_map(
            occupancy_map=masked_map,
            spikes_tracking=np.array(
                [
                    data["spike_times"][cell_index],
                    data["x_pos"][cell_index],
                    data["y_pos"][cell_index],
                ]
            ),
            arena_size=arena_size,
            bin_edges=bin_edges,
        )
        map = opexebo.general.smooth(map, smooth_list[mNo])
        corr_list[cNo] = opexebo.analysis.autocorrelation(map)

    corr_list_tot[mNo] = corr_list

# Plotting average autocorrelation of the 3 modules.
fig, ax = plt.subplots(1, 3, dpi=100)
ax[0].imshow(np.mean(corr_list_tot[0], axis=0), cmap=colormaps.parula)
ax[1].imshow(np.mean(corr_list_tot[1], axis=0), cmap=colormaps.parula)
ax[2].imshow(np.mean(corr_list_tot[2], axis=0), cmap=colormaps.parula)
plt.show()
