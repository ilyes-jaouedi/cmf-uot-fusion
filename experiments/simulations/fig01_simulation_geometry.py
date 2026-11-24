import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import matplotlib as mpl
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
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'grid.linewidth':   0.5,
    'grid.linestyle':   '--',
})
# Saved at its printed size (65 mm square, included at natural size in LaTeX),
# so the point sizes above are the printed sizes (Elsevier: >= 7 pt).
FIG_W = 65 / 25.4

OUTPUT_DIR = str(RESULTS_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Geometry ---
M_MIC = 8
FREQ = 4000.0
C_SPEED = 343.0
D_SPACING = (C_SPEED / FREQ) / 2.0

ARRAY_CENTER = np.array([-1.0, 0.0])
indices = np.arange(M_MIC) - (M_MIC - 1) / 2.0
y_mics = ARRAY_CENTER[1] + indices * D_SPACING
x_mics = np.full(M_MIC, ARRAY_CENTER[0])

CAM_POS = np.array([0.0, -1.0])
X0, Y0 = 0.5, 0.5

grid_x = [X0 - 0.2, X0 + 0.2]
grid_y = [Y0 - 0.2, Y0 + 0.2]


def plot_setup():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_W))

    # Microphone array
    ax.scatter(x_mics, y_mics, color='royalblue', marker='o', s=10,
               zorder=5, label='Microphones')
    ax.plot(x_mics, y_mics, color='royalblue', lw=0.8, alpha=0.5)

    # Camera
    ax.scatter(CAM_POS[0], CAM_POS[1], color='darkorchid', marker='D',
               s=40, zorder=10, label='Camera')

    # Source
    ax.scatter(X0, Y0, color='crimson', marker='*', s=80,
               zorder=6, label='Source')
    ax.annotate(f'$S$ ({X0}, {Y0})', (X0, Y0),
                xytext=(0, -17), textcoords='offset points', ha='center', va='top',
                color='crimson', fontsize=7, fontweight='bold')

    # Search grid
    grid_rect_x = [grid_x[0], grid_x[1], grid_x[1], grid_x[0], grid_x[0]]
    grid_rect_y = [grid_y[0], grid_y[0], grid_y[1], grid_y[1], grid_y[0]]
    ax.plot(grid_rect_x, grid_rect_y, color='forestgreen', lw=1.0,
            linestyle='--', label='Search grid')
    ax.fill(grid_rect_x, grid_rect_y, color='forestgreen', alpha=0.08)

    # Lines of sight
    ax.plot([ARRAY_CENTER[0], X0], [ARRAY_CENTER[1], Y0],
            color='royalblue', linestyle=':', lw=0.9, alpha=0.6)
    ax.plot([CAM_POS[0], X0], [CAM_POS[1], Y0],
            color='darkorchid', linestyle=':', lw=0.9, alpha=0.6)

    # Coordinate arrows
    arrow_kw = dict(head_width=0.04, head_length=0.06,
                    fc='k', ec='k', lw=0.8)
    ax.arrow(-1.35, -1.35, 0.22, 0, **arrow_kw)
    ax.arrow(-1.35, -1.35, 0, 0.22, **arrow_kw)
    ax.text(-1.10, -1.38, 'X', fontsize=7, va='top')
    ax.text(-1.40, -1.10, 'Y', fontsize=7, ha='right')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal')
    ax.set_xlim(-1.55, 1.1)
    ax.set_ylim(-1.55, 1.1)
    ax.legend(loc='upper left', framealpha=0.9, borderpad=0.3, labelspacing=0.25,
              handletextpad=0.3, markerscale=0.8)

    fig.tight_layout(pad=0.2)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'simulation_setup_2d.{ext}'),
                    dpi=600)
    plt.close(fig)
    print("Geometry plot saved.")


if __name__ == "__main__":
    plot_setup()
