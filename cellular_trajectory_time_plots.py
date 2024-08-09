# -*- coding: utf-8 -*-
"""
Code to show how persistence of cells at large times is averaged to random motion

Created on Mon Jul 22 17:08:23 2024

@author: Nasser
"""

#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
from matplotlib.gridspec import GridSpec

# Update matplotlib parameters for consistent font usage
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Latin Modern Roman",
    "font.sans-serif": ["Helvetica"]
})
#    "font.family": "Latin Modern Roman",

matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'


# Load the trajectories data
file_path = 'E:/2023_2024_bulk/motility/_thesis/Fig4,0_a_b/trajectories.csv'
trajectories_df = pd.read_csv(file_path)

# Ensure 'frame' column is of integer type
trajectories_df['frame'] = trajectories_df['frame'].astype(int)

# Specific cell IDs to include
specific_cells = [263]  # 0, 28, 47
renamed_cells = list(range(1, len(specific_cells) + 1))

# Map the original cell IDs to the new IDs
cell_id_mapping = dict(zip(specific_cells, renamed_cells))

# Define function to plot trajectories
def plot_trajectories(ax, df, cell_id, time_max, xlim):
    # Filter data based on time_max and selected cell
    df_filtered = df[(df['frame'] <= time_max) & (df['particle'] == cell_id)].copy()

    # Normalize the time column based on the real min and max frame of the filtered data
    df_filtered['norm_time'] = (df_filtered['frame'] - df_filtered['frame'].min()) / (df_filtered['frame'].max() - df_filtered['frame'].min())

    # Center x and y coordinates
    df_filtered['x_um'] = df_filtered['x_um'] - df_filtered['x_um'].mean()
    df_filtered['y_um'] = df_filtered['y_um'] - df_filtered['y_um'].mean()

    # Set the aspect ratio to equal
    ax.set_aspect("equal")

    # Create a color map
    color_map = plt.get_cmap("turbo")

    # Plot the trajectory using LineCollection
    points = np.array([df_filtered["x_um"], df_filtered["y_um"]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=color_map, norm=plt.Normalize(df_filtered["norm_time"].min(), df_filtered["norm_time"].max()))
    lc.set_array(df_filtered["norm_time"])
    ax.add_collection(lc)

    # Set fixed limits to ensure consistency across plots
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-xlim, xlim)

    # Increase the tick label size
    ax.tick_params(axis='both', which='major', labelsize=12)

# Convert times and filter data
dt = 45  # seconds between each frame
time1 = 10  # minutes
time2 = 60  # minutes

# Convert times from minutes to frames
tmin = int((time1 * 60) / dt)
tmax = int((time2 * 60) / dt)

print(f"tmin in frames: {tmin}")
print(f"tmax in frames: {tmax}")

# Filter data based on tmax and specific cell IDs
filtered_particles = trajectories_df.groupby('particle').filter(lambda x: len(x) >= tmax)['particle'].unique()
filtered_df = trajectories_df[(trajectories_df['particle'].isin(filtered_particles)) & (trajectories_df['particle'].isin(specific_cells))]

# Print available cell IDs after filtering
available_cells = filtered_df['particle'].unique()
print("Available cell IDs:", available_cells)
print("Number of cells:", len(available_cells))

#%% Plotting for each valid cell


fig = plt.figure(figsize=(6, 3))
gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.45)

cell_id = available_cells[0]
new_id = cell_id_mapping[cell_id]

# Plot for tmin minutes
ax1 = fig.add_subplot(gs[0, 0])
plot_trajectories(ax1, filtered_df, cell_id, tmin, 25)
ax1.set_title(f"T = {time1} min", fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=10)  # Adjust tick label size

# Plot for tmax minutes
ax2 = fig.add_subplot(gs[0, 1])
plot_trajectories(ax2, filtered_df, cell_id, tmax, 85)
ax2.set_title(f"T = {time2} min", fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=10)  # Adjust tick label size

# Add global labels and colorbar
fig.text(0.5, 0.06, 'X-position (µm)', ha='center', fontsize=14)
fig.text(0.05, 0.5, 'Y-position (µm)', va='center', rotation='vertical', fontsize=14)

# Create a color map for the tmax plot
color_map = plt.get_cmap("turbo")
norm = plt.Normalize(0, 1)  # Explicitly setting the normalization to range from 0 to 1
sm = plt.cm.ScalarMappable(cmap=color_map, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=[ax2], location='right', fraction=0.05, pad=0.04)
cbar.set_label("Norm. time (t/T)", fontsize=14)

# Adjust layout
plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15, wspace=0.35)
plt.show()

# Save the figure
save_path = '/combined_plot.png'
fig.savefig(save_path, dpi=150)


