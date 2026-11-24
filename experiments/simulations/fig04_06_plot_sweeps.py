"""
Plot the three Monte Carlo simulation figures from the saved CSVs, one file per panel.

Elsevier asks that two different graphs are not combined in one file, so each former
two-panel figure becomes two files, assembled as (a)/(b) subfigures in LaTeX:

  sim_results_snr.csv        -> sim_comparison_pos.pdf,       sim_comparison_pow.pdf
  sim_results_snapshots.csv  -> robustness_snapshots_pos.pdf, robustness_snapshots_pow.pdf
  sim_results_camera.csv     -> robustness_camera_pos.pdf,    robustness_camera_pow.pdf

No simulation is run: curves, styles and labels are those of run_methods_comparison.py,
run_robustness_snapshots.py and run_robustness_camera.py (paper naming: CSV uot2_* is
CMF-UOT1, uot1_* is CMF-UOT2). Panels are saved at their printed size (80 mm wide,
half of the full text width), so the point sizes below are the printed sizes.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cmf_uot.paths import DATA_DIR, RESULTS_DIR

mpl.rcParams.update({
    'font.family':     'serif',
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
PANEL_W = 80 / 25.4
PANEL_H = 58 / 25.4
PAPER_DIR = str(RESULTS_DIR)
ALPHA = 300.0  # focal length (px/rad), as in run_robustness_camera.py

# (column prefix, label, matplotlib format, extra kwargs)
CURVES = [
    ('crb',  'Fusion CRB', 'k-',   dict(lw=1.3)),
    ('cmf',  'CMF',        'k--v', dict(ms=3.5)),
    ('mle',  'MLE',        'b-^',  dict(ms=3.5)),
    ('uot2', 'CMF-UOT1',   'r-s',  dict(ms=3.5)),
    ('uot1', 'CMF-UOT2',   'g--d', dict(ms=3.5)),
]


def panel(x, df, kind, xlabel, ylabel, fname, loglog, show_crb):
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    plot = ax.loglog if loglog else ax.semilogy
    for key, label, fmt, kw in CURVES:
        if key == 'crb' and not show_crb:
            continue
        plot(x, df[f'{key}_{kind}'].values, fmt, label=label, **kw)
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel(ylabel, labelpad=1)
    ax.grid(True, which='both')
    ax.legend(framealpha=0.85, borderpad=0.3, labelspacing=0.25, handletextpad=0.4)
    fig.tight_layout(pad=0.25)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(PAPER_DIR, f'{fname}.{ext}'), dpi=600)
    plt.close(fig)
    print(f'Saved {fname}.pdf / .png')


def figure(csv, xcol, xlabel, stem, loglog, crb_in_power, xfun=lambda v: v):
    df = pd.read_csv(os.path.join(PAPER_DIR, csv))
    x = xfun(df[xcol].values)
    panel(x, df, 'pos', xlabel, 'Position RMSE (m)', f'{stem}_pos', loglog, True)
    panel(x, df, 'pow', xlabel, 'Power RMSE', f'{stem}_pow', loglog, crb_in_power)


if __name__ == '__main__':
    # SNR sweep: the CRB is drawn in both panels (as in run_methods_comparison.py)
    figure('sim_results_snr.csv', 'SNR', 'SNR (dB)', 'sim_comparison',
           loglog=False, crb_in_power=True)
    # snapshot and camera sweeps: no CRB in the power panel (biased power estimator)
    figure('sim_results_snapshots.csv', 'Snapshots', 'Number of Snapshots ($L$)',
           'robustness_snapshots', loglog=True, crb_in_power=False)
    figure('sim_results_camera.csv', 'Sigma_C', r'Camera Uncertainty $\sigma_c$ ($^\circ$)',
           'robustness_camera', loglog=True, crb_in_power=False,
           xfun=lambda s: np.asarray(s) * 180.0 / (np.pi * ALPHA))
