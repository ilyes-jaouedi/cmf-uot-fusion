"""
Regenerate the 6 sensitivity figures from existing CSV data (no simulation).

Rename mapping (CSV column → paper label):
  uot1_* → CMF-UOT2   (b=P·1 coupling, formerly UOT1)
  uot2_* → CMF-UOT1   (decoupled b,    formerly UOT2)
  uot_kl_* → dropped (column ignored if present)

Outputs (in results/):
  sensitivity_lambda_averaged.pdf/png
  sensitivity_mu1_averaged.pdf/png
  sensitivity_mu2_averaged.pdf/png
  sensitivity_lambda_power_averaged.pdf/png
  sensitivity_mu1_power_averaged.pdf/png
  sensitivity_mu2_power_averaged.pdf/png
"""

import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cmf_uot.paths import DATA_DIR, RESULTS_DIR

# Sized for the printed panel: 3 panels per row of a 190 mm full-width figure
# (Elsevier artwork sizes) -> 0.31 x 190 mm = 59 mm per panel. The file is saved at
# exactly that size (no tight bbox), so the point sizes below are the printed sizes.
mpl.rcParams.update({
    'font.family':     'serif',
    'font.size':        8,
    'axes.labelsize':   8,
    'axes.titlesize':   8,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  7,
    'lines.linewidth':  1.0,
    'axes.grid':        True,
    'grid.alpha':       0.35,
    'grid.linewidth':   0.5,
    'grid.linestyle':   '--',
})
PANEL_W = 59 / 25.4   # inches: 0.31 of a 190 mm full-width figure
PANEL_H = 1.6
PAPER_DIR = str(RESULTS_DIR)
def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(PAPER_DIR, f'{name}.{ext}'),
                    dpi=600)
    print(f'Saved {name}.pdf / .png')


def single_panel(x, curves, xlabel, ylabel, vline_x, vline_label, fname, legend_loc='best'):
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    for y, color, ls, marker, label in curves:
        kw = dict(color=color, linestyle=ls, label=label)
        if marker:
            kw.update(marker=marker, markersize=3)
        ax.loglog(x, y, **kw)
    ax.axvline(x=vline_x, color='k', linestyle='--', lw=1.0, label=vline_label)
    # plain decimals on the y axis (0.05, 0.06, ...) instead of 5x10^-2: shorter,
    # and no exponents that would print below 6 pt in a 59 mm panel
    plain = mticker.FuncFormatter(lambda v, _: f'{v:g}')
    ax.yaxis.set_major_formatter(plain)
    y0, y1 = ax.get_ylim()
    # label minor ticks only when the axis spans less than a decade (else they collide)
    ax.yaxis.set_minor_formatter(plain if y1 / y0 < 10 else mticker.NullFormatter())
    ax.set_xlabel(xlabel, labelpad=1)
    ax.set_ylabel(ylabel, labelpad=1)
    ax.legend(loc=legend_loc, handlelength=1.3, borderpad=0.3, labelspacing=0.25, handletextpad=0.4)
    fig.tight_layout(pad=0.2)
    save(fig, fname)
    plt.close(fig)


# ── Lambda sweep ──────────────────────────────────────────────────────────────
df_lam = pd.read_csv(os.path.join(PAPER_DIR, 'sensitivity_lambda.csv'))
# CSV was saved with 1/factors (old convention); invert and sort to get λ/λ* increasing
x_lam_raw = df_lam['lambda_ratio'].values
sort_idx = x_lam_raw.argsort()[::-1]          # ascending λ/λ*
x_lam = 1.0 / x_lam_raw[sort_idx]
df_lam = df_lam.iloc[sort_idx].reset_index(drop=True)

single_panel(
    x_lam,
    [
        (df_lam['uot2_pos'], 'r',  '-',  's', 'CMF-UOT1'),
        (df_lam['uot1_pos'], 'g', '--', 'd', 'CMF-UOT2'),
    ],
    r'$\lambda / \lambda^*$', 'Position RMSE (m)',
    1.0, r'Heuristic $\lambda^*$',
    'sensitivity_lambda_averaged',
)
single_panel(
    x_lam,
    [
        (df_lam['uot2_pow'], 'r',  '-',  's', 'CMF-UOT1'),
        (df_lam['uot1_pow'], 'g', '--', 'd', 'CMF-UOT2'),
    ],
    r'$\lambda / \lambda^*$', 'Power RMSE',
    1.0, r'Heuristic $\lambda^*$',
    'sensitivity_lambda_power_averaged',
)

# ── Beta (mu1) sweep ──────────────────────────────────────────────────────────
df_mu1 = pd.read_csv(os.path.join(PAPER_DIR, 'sensitivity_mu1.csv'))
x_mu1 = df_mu1['beta_ratio'].values

single_panel(
    x_mu1,
    [
        (df_mu1['uot2_pos'], 'r',  '-',  's', 'CMF-UOT1'),
        (df_mu1['uot1_pos'], 'g', '--', 'd', 'CMF-UOT2'),
    ],
    r'$\beta / \beta^*$', 'Position RMSE (m)',
    1.0, r'Heuristic $\beta^*$',
    'sensitivity_mu1_averaged',
)
single_panel(
    x_mu1,
    [
        (df_mu1['uot2_pow'], 'r',  '-',  's', 'CMF-UOT1'),
        (df_mu1['uot1_pow'], 'g', '--', 'd', 'CMF-UOT2'),
    ],
    r'$\beta / \beta^*$', 'Power RMSE',
    1.0, r'Heuristic $\beta^*$',
    'sensitivity_mu1_power_averaged',
)

# ── Alpha (mu2) sweep — CMF-UOT1 only + CMF-UOT2 baseline ───────────────────
df_mu2 = pd.read_csv(os.path.join(PAPER_DIR, 'sensitivity_mu2.csv'))
x_mu2 = df_mu2['alpha_ratio'].values

single_panel(
    x_mu2,
    [
        (df_mu2['uot2_pos'],          'r',  '-',  's',  'CMF-UOT1'),
        (df_mu2['uot1_baseline_pos'], 'g', '--', None, 'CMF-UOT2 (baseline)'),
    ],
    r'$\alpha / \alpha^*$', 'Position RMSE (m)',
    1.0, r'Heuristic $\alpha^*$',
    'sensitivity_mu2_averaged', legend_loc='center right',
)
single_panel(
    x_mu2,
    [
        (df_mu2['uot2_pow'],          'r',  '-',  's',  'CMF-UOT1'),
        (df_mu2['uot1_baseline_pow'], 'g', '--', None, 'CMF-UOT2 (baseline)'),
    ],
    r'$\alpha / \alpha^*$', 'Power RMSE',
    1.0, r'Heuristic $\alpha^*$',
    'sensitivity_mu2_power_averaged', legend_loc='center right',
)
