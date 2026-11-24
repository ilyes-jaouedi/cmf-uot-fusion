import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import json

from cmf_uot.geometry import get_array_from_npy
from cmf_uot.paths import DATA_DIR, RESULTS_DIR

# --- IEEE TSP Plot Style ---
mpl.rcParams.update({
    'font.family':      'serif',
    'font.size':        8,
    'axes.labelsize':   8,
    'axes.titlesize':   8,
    'xtick.labelsize':  7,
    'ytick.labelsize':  7,
    'legend.fontsize':  7,
    'lines.linewidth':  1.0,
})
# Included at natural size in LaTeX, so the point sizes above are the printed sizes.

OUTPUT_DIR = str(RESULTS_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Real geometry (units: m) ---
CAMERA_POS_CM = (7, -300, 90)
cam_x = CAMERA_POS_CM[0] / 100
cam_y = CAMERA_POS_CM[1] / 100
cam_z = CAMERA_POS_CM[2] / 100

SOURCES_CM = [
    {'x':  93,    'y':  65,    'z': 112,  'label': 'S1'},
    {'x': -22,    'y':   0,    'z': 131,  'label': 'S2'},
    {'x':  31,    'y': -44.5,  'z': 158,  'label': 'S3'},
    {'x':  -5,    'y': -99.5,  'z': 123,  'label': 'S4'},
]
sources = [{'x': s['x']/100, 'y': s['y']/100, 'z': s['z']/100, 'label': s['label']}
           for s in SOURCES_CM]


def plot_config():
    Array = get_array_from_npy()

    fig = plt.figure(figsize=(3.2, 2.7))
    ax = fig.add_subplot(111, projection='3d')

    # Microphone array
    ax.scatter(Array[:, 0], Array[:, 1], Array[:, 2],
               c='black', marker='o', s=8, label='Microphone Array',
               depthshade=True, linewidths=0)

    # Camera
    ax.scatter(cam_x, cam_y, cam_z,
               color='deepskyblue', marker='s', s=50,
               label='Camera Position', depthshade=False,
               edgecolor='black', linewidths=0.6, zorder=10)
    ax.plot([cam_x, cam_x], [cam_y, cam_y], [0, cam_z],
            color='deepskyblue', linestyle='--', lw=0.7, alpha=0.5)

    # Sources and lines of sight
    for i, s in enumerate(sources):
        ax.scatter(s['x'], s['y'], s['z'],
                   c='crimson', marker='s', s=40,
                   label='Ground Truth Sources' if i == 0 else '',
                   depthshade=False, edgecolor='black', linewidths=0.6, zorder=10)
        ax.plot([s['x'], s['x']], [s['y'], s['y']], [0, s['z']],
                color='crimson', linestyle='--', lw=0.7, alpha=0.5)
        # No background: a crimson box next to a crimson marker merged into one
        # blob. Plain crimson text also matches the scenario figures (mark_gt_3d).
        ax.text(s['x'], s['y'], s['z'] + 0.13, s['label'],
                color='crimson', fontsize=7, fontweight='bold',
                ha='center', zorder=11)
        ax.plot([cam_x, s['x']], [cam_y, s['y']], [cam_z, s['z']],
                color='forestgreen', linestyle=':', lw=0.8, alpha=0.7,
                label='Line of Sight' if i == 0 else '')

    ax.set_xlabel('X (m)', labelpad=4)
    ax.set_ylabel('Y (m)', labelpad=4)
    ax.set_zlabel('Z (m)', labelpad=4)

    ax.tick_params(axis='both', labelsize=7, pad=2)
    ax.tick_params(axis='z', labelsize=7, pad=2)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(),
              loc='upper left', bbox_to_anchor=(0.01, 0.995), ncol=2, columnspacing=0.8,
              framealpha=0.85, fontsize=7, markerscale=0.8,
              borderpad=0.3, labelspacing=0.25, handletextpad=0.3)

    ax.view_init(elev=25, azim=-75)
    # 3-D axis labels are lost by bbox_inches='tight'; fix the frame by hand instead
    ax.set_box_aspect(None, zoom=0.88)
    ax.zaxis.set_major_locator(mpl.ticker.MultipleLocator(0.5))
    fig.subplots_adjust(left=0.0, right=0.94, bottom=0.0, top=0.95)

    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'_config.{ext}'), dpi=600)
    plt.close(fig)
    print("Saved _config.pdf / _config.png to results/")


if __name__ == "__main__":
    plot_config()
