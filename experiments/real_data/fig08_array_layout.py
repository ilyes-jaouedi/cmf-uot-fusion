import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib as mpl

from cmf_uot.geometry import get_array_from_npy
from cmf_uot.paths import DATA_DIR, RESULTS_DIR

# --- IEEE TSP Plot Style ---
mpl.rcParams.update({
    'font.family':      'serif',
    'font.size':        9,
    'axes.labelsize':   9,
    'axes.titlesize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  7,
    'lines.linewidth':  1.2,
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'grid.linewidth':   0.5,
    'grid.linestyle':   '--',
})
IEEE_W1 = 3.5

OUTPUT_DIR = str(RESULTS_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def visualize_array():
    print("--- Visualizing Microphone Array Layout ---")

    try:
        Array = get_array_from_npy()
    except Exception as e:
        print(f"Error loading array: {e}. Using ULA-32 fallback.")
        M = 32
        indices = np.arange(M) - (M - 1) / 2.0
        Array = np.column_stack((np.zeros(M), indices * 0.042, np.zeros(M)))

    Y = Array[:, 1]
    Z = Array[:, 2]

    fig, ax = plt.subplots(figsize=(IEEE_W1, 2.4))

    ax.scatter(Y, Z, c='k', marker='o', s=14, zorder=5, label='Microphone')

    # Label every 4th mic to avoid clutter
    for i in range(len(Array)):
        if (i + 1) % 4 == 0 or i == 0:
            ax.annotate(str(i + 1), (Y[i], Z[i]),
                        xytext=(3, 2), textcoords='offset points',
                        fontsize=7, color='dimgray')

    ax.axvline(x=0, color='gray', lw=0.6, linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='gray', lw=0.6, linestyle='--', alpha=0.5)

    margin_y = 0.08
    margin_z = max(0.08, (Z.max() - Z.min()) * 0.3 + 0.05)
    ax.set_xlim(Y.min() - margin_y, Y.max() + margin_y)
    ax.set_ylim(Z.min() - margin_z, Z.max() + margin_z)

    ax.set_xlabel('Y (m)')
    ax.set_ylabel('Z (m)')
    ax.set_aspect('equal')
    ax.legend(loc='upper right', framealpha=0.9)

    fig.tight_layout(pad=0.8)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(OUTPUT_DIR, f'array_layout.{ext}'),
                    dpi=600, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print("Array layout saved.")


if __name__ == "__main__":
    visualize_array()
