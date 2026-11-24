import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import os
import sys


from cmf_uot.geometry import get_grid_from_config, get_array_from_npy, freefield
from cmf_uot.covariance import calculate_sigma
from cmf_uot.data import load_camera_detections, load_sigs
from cmf_uot.camera import compute_C
from cmf_uot.solvers.cmf import CMFSolver
from cmf_uot.solvers.cmf_uot2 import TransportOptimizer_numpy
from cmf_uot.paths import DATA_DIR, RESULTS_DIR

# ── IEEE / TSP publication style ──────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':      'serif',
    'font.size':        9,
    'axes.labelsize':   9,
    'axes.titlesize':   9,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'legend.fontsize':  8,
    'lines.linewidth':  1.5,
    'axes.grid':        False,
})

IEEE_W2  = 7.16   # double-column width (inches)
IEEE_W1  = 3.5    # single-column width (inches)
CMAP     = 'viridis'
THRESH   = 0.02   # hide grid points below 2 % of peak

ROOT_DIR = str(DATA_DIR)
OUT_DIR = str(RESULTS_DIR)
os.makedirs(OUT_DIR, exist_ok=True)


# ── Utilities ──────────────────────────────────────────────────────────────────

def threshold_power(power, frac=THRESH):
    out = power.copy()
    out[out < frac * out.max()] = 0.0
    return out


def project_2d(power, xg, yg, zg, proj):
    """
    Collapse a (Ny*Nx*Nz,) power vector to a 2-D plane.

    Grid was built with np.meshgrid(xg, yg, zg)  →  shape (Ny, Nx, Nz).

    proj='XY' : sum over Z  → returns xg, yg,  p(Ny,Nx)
    proj='XZ' : sum over Y  → returns xg, zg,  p(Nz,Nx)
    proj='YZ' : sum over X  → returns yg, zg,  p(Nz,Ny)
    """
    Ny, Nx, Nz = len(yg), len(xg), len(zg)
    p3 = power.reshape(Ny, Nx, Nz)
    if proj == 'XY':
        return xg, yg, p3.sum(axis=2)          # (Ny, Nx)
    if proj == 'XZ':
        return xg, zg, p3.sum(axis=0).T        # (Nz, Nx)
    if proj == 'YZ':
        return yg, zg, p3.sum(axis=1).T        # (Nz, Ny)


def gt_from_detections(detections):
    return [(d['x'] / 100, d['y'] / 100, d['z'] / 100) for d in detections]


def mark_gt_2d(ax, gt_pos, i_h, i_v, fs=9):
    for k, g in enumerate(gt_pos):
        ax.scatter(g[i_h], g[i_v], marker='o', facecolors='none',
                   edgecolors='red', s=180, linewidths=1.8,
                   zorder=10,
                   label='Ground truth' if k == 0 else '')
        ax.annotate(f'$S_{{{k+1}}}$', (g[i_h], g[i_v]),
                    xytext=(4, 12), textcoords='offset points',
                    color='red', fontsize=fs, fontweight='bold')


def mark_gt_3d(ax, gt_pos, fs=10, z_floor=None):
    # the drop lines stop at the bottom of the z axis, not at z = 0
    if z_floor is None:
        z_floor = ax.get_zlim()[0]
    for k, g in enumerate(gt_pos):
        ax.plot([g[0], g[0]], [g[1], g[1]], [z_floor, g[2]],
                color='red', lw=0.9, linestyle='--', alpha=0.6, zorder=9)
        ax.scatter(g[0], g[1], z_floor,
                   marker='o', c='red', s=12,
                   depthshade=False, alpha=0.5, zorder=9)
        # X marker at true position
        ax.scatter(g[0], g[1], g[2], marker='x', c='red',
                   s=80, linewidths=2, depthshade=False, zorder=10,
                   label='Ground truth' if k == 0 else '')
        ax.text(g[0], g[1], g[2] + 0.08, f'$S_{{{k+1}}}$',
                color='red', fontsize=fs, fontweight='bold')


# ── Sensor indicators ─────────────────────────────────────────────────────────
# The array (x = -3 m) and the camera (y = -3 m) both lie outside the
# reconstruction volume. On the 2-D projections each is drawn as an arrow at the
# border of the axes, but only on the views where the axis along which it lies
# outside is actually plotted: the array on every view (x is always shown), the
# camera on the top view only, since its offset is along y.
ARRAY_POS_3D = (-3.0, -0.07, 1.20)     # centre of the 32-microphone array
CAMERA_POS_3D = (0.07, -3.0, 0.90)     # camera optical centre
_SENSORS = ((ARRAY_POS_3D, 'Array', '#1f4e79', 0),      # informative axis: x
            (CAMERA_POS_3D, 'Camera', '#7f3f00', 1))    # informative axis: y


def mark_sensors_edge_2d(ax, i_h, i_v, fs=7):
    """Indicate the sensors at the border of a 2-D projection."""
    h0, h1 = ax.get_xlim()
    v0, v1 = ax.get_ylim()
    for pos, name, colour, axis in _SENSORS:
        if axis not in (i_h, i_v):
            continue                                   # not informative in this view
        h, v = pos[i_h], pos[i_v]
        style = dict(xycoords='axes fraction', textcoords='axes fraction',
                     fontsize=fs, color=colour, ha='center', va='center', zorder=20,
                     bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.5),
                     arrowprops=dict(arrowstyle='-|>', color=colour, lw=0.9,
                                     shrinkA=1.5, shrinkB=0))
        if h < h0 or h > h1:                           # off the left or right side
            xt = 0.0 if h < h0 else 1.0
            frac_v = (min(max(v, v0), v1) - v0) / (v1 - v0)
            ax.annotate(name, xy=(xt, frac_v),
                        xytext=(xt + (0.16 if h < h0 else -0.16), frac_v), **style)
        elif v < v0 or v > v1:                         # off the bottom or top
            yt = 0.0 if v < v0 else 1.0
            frac_h = (min(max(h, h0), h1) - h0) / (h1 - h0)
            off_h = 0.22 if frac_h < 0.78 else -0.22
            ax.annotate(name, xy=(frac_h, yt),
                        xytext=(frac_h + off_h, yt + (0.09 if v < v0 else -0.09)), **style)
        else:                                          # inside the frame
            ax.plot(h, v, marker='v' if name == 'Camera' else 's',
                    color=colour, ms=4, zorder=20)
            ax.annotate(name, (h, v), xytext=(3, 3), textcoords='offset points',
                        fontsize=fs, color=colour, zorder=20)


def draw_3d(ax, Grid, power, gt_pos, vmax, label=''):
    mask = power > 0
    if mask.any():
        p = power[mask]
        s = 5 + 80 * np.clip(p / vmax, 0, 1)
        sc = ax.scatter(Grid[mask, 0], Grid[mask, 1], Grid[mask, 2],
                        s=s, c=np.clip(p / vmax, 0, 1),
                        cmap=CMAP, vmin=0, vmax=1,
                        alpha=0.75, linewidths=0)
    mark_gt_3d(ax, gt_pos, z_floor=Grid[:, 2].min())
    ax.set_xlabel('X (m)', labelpad=1, fontsize=8)
    ax.set_ylabel('Y (m)', labelpad=1, fontsize=8)
    ax.set_zlabel('Z (m)', labelpad=1, fontsize=8)
    ax.tick_params(labelsize=7.5, pad=0)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(4))
    # Clean background panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')
    ax.view_init(elev=20, azim=-60)
    # identical extent on every panel: the full reconstruction volume
    ax.set_xlim(Grid[:, 0].min(), Grid[:, 0].max())
    ax.set_ylim(Grid[:, 1].min(), Grid[:, 1].max())
    ax.set_zlim(Grid[:, 2].min(), Grid[:, 2].max())
    if label:
        ax.set_title(label, loc='left', fontsize=9, fontweight='bold', pad=9)
    return sc


LOG_VMIN = 1e-3   # floor at 0.1 % of peak — reveals sidelobes at 1–10 %

def draw_proj(ax, h_ax, v_ax, p2d, gt_pos, i_h, i_v,
              xlabel, ylabel, vmax, label='', show_cbar=False, fig=None):
    normed = np.clip(p2d / (vmax + 1e-12), LOG_VMIN, 1.0)
    im = ax.pcolormesh(h_ax, v_ax, normed, cmap=CMAP,
                       norm=LogNorm(vmin=LOG_VMIN, vmax=1.0), shading='auto')
    mark_gt_2d(ax, gt_pos, i_h, i_v)
    mark_sensors_edge_2d(ax, i_h, i_v)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
    if show_cbar and fig is not None:
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03,
                          ticks=[1e-3, 1e-2, 1e-1, 1.0])
        cb.ax.set_yticklabels(['0.001', '0.01', '0.1', '1'], fontsize=7)
        cb.set_label('Norm. power (log)', fontsize=7)
    if label:
        ax.set_title(label, loc='left', fontsize=9, fontweight='bold', pad=9)
    return im


# ── Main ───────────────────────────────────────────────────────────────────────

def run_scenario():
    print("--- Running Standard Scenario (Manip 1) ---")

    manip = 1
    Grid, xg, yg, zg = get_grid_from_config()
    Array = get_array_from_npy()

    camera_detections = load_camera_detections(
        os.path.join(ROOT_DIR, f'camera_detections/mesure_manip_{manip}.json'))
    A    = np.load(os.path.join(ROOT_DIR, 'transformation_matrix_A.npy'))
    Sigs = load_sigs(manip=manip)

    n_src = len(camera_detections)
    Lseg  = 1024
    freq, c = 4000, 343.0
    k = 2 * np.pi * freq / c

    Sigma = calculate_sigma(Sigs, Lseg, freq)
    G     = freefield(Array, Grid, k, normalized=False)

    L         = Sigs.shape[0] // Lseg
    eigvals   = np.linalg.eigvalsh(Sigma)
    sigma_a2  = np.mean(eigvals[:-n_src])
    sig_trace = np.trace(Sigma).real
    lam       = (L / sigma_a2) / (sig_trace ** 2)

    depths = np.maximum(np.array([d['z'] / 100 for d in camera_detections]), 0.1)
    sig_c  = np.mean((300.0 * 0.04) / depths) / 2.0
    mu     = 0.01 / sig_c**2
    print(f"  lam = {lam:.2e},  mu = {mu:.2e}")

    # ── Acoustic-only CMF ──────────────────────────────────────────────────────
    print("  Running CMF...")
    cmf = CMFSolver(G, Sigma)
    cmf.GCD(200, inside=False)
    b_cmf = cmf.p

    # ── CMF-UOT fusion ─────────────────────────────────────────────────────────
    print("  Running CMF-UOT...")
    a   = np.ones(n_src) / n_src
    C   = compute_C(Grid, camera_detections, A)
    opt = TransportOptimizer_numpy(C, a, Sigma, G, lambda_=lam, mu=mu)
    opt.optimize_greedy_coordinate_descent(num_iterations=200, verbose=False)
    P_fus  = opt.P
    b_fus  = P_fus.sum(axis=1)

    # ── Normalisation:
    #   CMF    → no threshold (keep all sidelobes / artefacts intentionally)
    #   CMF-UOT → threshold applied for clean display
    #   Each method is normalised to its own peak so colour scale is relative.
    b_fus_t  = threshold_power(b_fus)
    vmax_cmf = b_cmf.max()
    vmax_fus = b_fus_t.max() if b_fus_t.max() > 0 else 1.0

    gt_pos = gt_from_detections(camera_detections)

    # ── Residual CMF (commented out) ──────────────────────────────────────────
    # Sigma_residual = Sigma - (G * b_fus) @ G.conj().T
    # cmf_res = CMFSolver(G, Sigma_residual)
    # cmf_res.GCD(200, inside=False)
    # b_residual = cmf_res.p
    # vmax_res = b_residual.max() if b_residual.max() > 0 else 1.0

    # ── Print results summary ─────────────────────────────────────────────────
    print("\n=== SCENARIO 1 — LOCALIZATION SUMMARY ===")
    print(f"  L={L} snapshots,  lam={lam:.2e},  mu={mu:.2e}")
    print(f"  Grid: {Grid.shape[0]} pts  "
          f"x=[{xg.min():.2f},{xg.max():.2f}]  "
          f"y=[{yg.min():.2f},{yg.max():.2f}]  "
          f"z=[{zg.min():.2f},{zg.max():.2f}] m")
    print("\n  Ground-truth positions (m):")
    for k, g in enumerate(gt_pos):
        print(f"    S{k+1}: x={g[0]:.3f}  y={g[1]:.3f}  z={g[2]:.3f}")
    # CMF global peak
    cmf_peak = Grid[np.argmax(b_cmf)]
    cmf_errs = [np.linalg.norm(np.array([cmf_peak[0], cmf_peak[1]]) -
                                np.array([g[0], g[1]])) for g in gt_pos]
    print(f"\n  CMF peak: ({cmf_peak[0]:.3f}, {cmf_peak[1]:.3f}, {cmf_peak[2]:.3f}) m"
          f"  -> nearest S{np.argmin(cmf_errs)+1}, err={min(cmf_errs)*100:.1f} cm")
    # CMF-UOT per column
    print("\n  CMF-UOT transport plan columns (per camera detection):")
    for j in range(P_fus.shape[1]):
        col   = P_fus[:, j]
        peak  = Grid[np.argmax(col)]
        mass  = col.sum()
        errs  = [np.linalg.norm(np.array([peak[0], peak[1]]) -
                                 np.array([g[0], g[1]])) for g in gt_pos]
        nn    = np.argmin(errs)
        print(f"    Col {j+1}: peak=({peak[0]:.3f},{peak[1]:.3f},{peak[2]:.3f}) m  "
              f"mass={mass:.4f}  nearest=S{nn+1}  err={min(errs)*100:.1f} cm")
    # res_peak = Grid[np.argmax(b_residual)]
    print("==========================================\n")

    # ══ Figure 1 — 3-D point clouds ══════════════════════════════════════════
    #   (a) CMF  |  (b) CMF-UOT  — single column, stacked vertically
    # ══════════════════════════════════════════════════════════════════════════
    fig_3d, axes_3d = plt.subplots(1, 2, figsize=(IEEE_W1, 3.0),
                                   subplot_kw={'projection': '3d'})
    draw_3d(axes_3d[0], Grid, b_cmf,   gt_pos, vmax_cmf, label='(a) CMF')
    draw_3d(axes_3d[1], Grid, b_fus_t, gt_pos, vmax_fus, label='(b) CMF--UOT')
    fig_3d.tight_layout(pad=0.8, w_pad=2.0)
    for ext in ('pdf', 'png'):
        fig_3d.savefig(os.path.join(OUT_DIR, f'scenario1_3d.{ext}'),
                       dpi=600, bbox_inches='tight')
    print("  Saved scenario1_3d.{pdf,png}")
    plt.close(fig_3d)

    # Each panel as its own file, 80 x 56 mm (Elsevier: no two graphs in one file);
    # LaTeX sets the (a)/(b) labels. Fixed margins: bbox 'tight' clips 3-D axis labels.
    for power, vmax, tag in ((b_cmf, vmax_cmf, 'cmf'), (b_fus_t, vmax_fus, 'uot')):
        fig_p = plt.figure(figsize=(80 / 25.4, 56 / 25.4))
        ax_p = fig_p.add_subplot(111, projection='3d')
        draw_3d(ax_p, Grid, power, gt_pos, vmax)
        ax_p.set_box_aspect(None, zoom=0.86)
        fig_p.subplots_adjust(left=0.0, right=0.97, bottom=0.03, top=1.0)
        for ext in ('pdf', 'png'):
            fig_p.savefig(os.path.join(OUT_DIR, f'scenario1_3d_{tag}.{ext}'), dpi=600)
        plt.close(fig_p)
    print("  Saved scenario1_3d_cmf / _3d_uot")

    # ══ Figure 2 — 2-D projections ════════════════════════════════════════════
    #   Row 1 : XY top-down    (a) CMF   |  (b) CMF-UOT
    #   Row 2 : XZ front view  (c) CMF   |  (d) CMF-UOT
    # ══════════════════════════════════════════════════════════════════════════
    xh,  xv,  p_xy_cmf = project_2d(b_cmf,   xg, yg, zg, 'XY')
    _,   _,   p_xy_fus = project_2d(b_fus_t, xg, yg, zg, 'XY')
    xh2, xv2, p_xz_cmf = project_2d(b_cmf,   xg, yg, zg, 'XZ')
    _,   _,   p_xz_fus = project_2d(b_fus_t, xg, yg, zg, 'XZ')

    fig_2d, axs2d = plt.subplots(2, 2, figsize=(IEEE_W2, 5.0))
    draw_proj(axs2d[0, 0], xh,  xv,  p_xy_cmf, gt_pos, 0, 1,
              'X (m)', 'Y (m)', vmax_cmf, label='(a) CMF — top view (X-Y)')
    draw_proj(axs2d[0, 1], xh,  xv,  p_xy_fus, gt_pos, 0, 1,
              'X (m)', 'Y (m)', vmax_fus, label='(b) CMF--UOT — top view (X-Y)',
              show_cbar=True, fig=fig_2d)
    draw_proj(axs2d[1, 0], xh2, xv2, p_xz_cmf, gt_pos, 0, 2,
              'X (m)', 'Z (m)', vmax_cmf, label='(c) CMF — front view (X-Z)')
    draw_proj(axs2d[1, 1], xh2, xv2, p_xz_fus, gt_pos, 0, 2,
              'X (m)', 'Z (m)', vmax_fus, label='(d) CMF--UOT — front view (X-Z)',
              show_cbar=True, fig=fig_2d)
    handles, labels_ = axs2d[0, 1].get_legend_handles_labels()
    if handles:
        fig_2d.legend(handles[:1], labels_[:1],
                      loc='lower center', ncol=1, fontsize=8, frameon=True,
                      bbox_to_anchor=(0.5, -0.01))
    fig_2d.tight_layout(pad=1.2, h_pad=1.5, w_pad=1.2)
    for ext in ('pdf', 'png'):
        fig_2d.savefig(os.path.join(OUT_DIR, f'scenario1_2d.{ext}'),
                       dpi=600, bbox_inches='tight')
    print("  Saved scenario1_2d.{pdf,png}")
    plt.close(fig_2d)

    # ══ Figure 2b — front view (X-Z) — inferno/LogNorm, estimate circles ══════
    FV_LOG_VMIN = 1e-3
    fig_fv, axs_fv = plt.subplots(1, 2, figsize=(IEEE_W2, 1.55))

    for ax_fv, p2d_fv, vmax_fv, lbl_fv in [
        (axs_fv[0], p_xz_cmf, vmax_cmf, '(a) CMF — front view (X-Z)'),
        (axs_fv[1], p_xz_fus, vmax_fus, '(b) CMF--UOT — front view (X-Z)'),
    ]:
        normed_fv = np.clip(p2d_fv / (vmax_fv + 1e-12), FV_LOG_VMIN, 1.0)
        im_fv = ax_fv.pcolormesh(xh2, xv2, normed_fv, cmap='inferno',
                                 norm=LogNorm(vmin=FV_LOG_VMIN, vmax=1.0),
                                 shading='auto')
        mark_gt_2d(ax_fv, gt_pos, 0, 2)
        mark_sensors_edge_2d(ax_fv, 0, 2)
        ax_fv.set_xlabel('X (m)', fontsize=8)
        ax_fv.set_ylabel('Z (m)', fontsize=8)
        ax_fv.tick_params(labelsize=7)
        ax_fv.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax_fv.yaxis.set_major_locator(ticker.MaxNLocator(5))
        ax_fv.set_aspect('equal')
        ax_fv.set_title(lbl_fv, loc='left', fontsize=9, fontweight='bold', pad=9)

    fig_fv.tight_layout(pad=0.6, w_pad=0.8)
    # Horizontal colorbar spanning both panels (ground truth noted in caption)
    cb_fv = fig_fv.colorbar(im_fv, ax=axs_fv, orientation='vertical',
                             fraction=0.02, pad=0.012, aspect=15,
                             ticks=[1e-3, 1e-2, 1e-1, 1.0])
    cb_fv.ax.set_yticklabels(['0.001', '0.01', '0.1', '1'], fontsize=7)
    cb_fv.set_label('Norm. power (log)', fontsize=7)
    for ext in ('pdf', 'png'):
        fig_fv.savefig(os.path.join(OUT_DIR, f'scenario1_frontview.{ext}'),
                       dpi=600, bbox_inches='tight')
    print("  Saved scenario1_frontview.{pdf,png}")
    plt.close(fig_fv)

    # ══ Figure 3 — 1×4 source decomposition (transport plan slices) ══════════
    #   Each panel: XY top-down projection for one source's column of P
    # ══════════════════════════════════════════════════════════════════════════
    n_panels = min(4, P_fus.shape[1])
    panel_labels = ['(a)', '(b)', '(c)', '(d)']
    DECOMP_LOG_VMIN = 1e-3

    fig3, axs = plt.subplots(1, 4, figsize=(IEEE_W2, 2.05))
    last_im = None
    for idx in range(n_panels):
        ax  = axs[idx]
        col = P_fus[:, idx]
        col_t  = threshold_power(col, frac=0.001)
        vmax_c = col_t.max() if col_t.max() > 0 else 1.0

        xh3, xv3, p2d = project_2d(col_t, xg, yg, zg, 'XY')
        p2d_n = np.clip(p2d / (vmax_c + 1e-12), DECOMP_LOG_VMIN, 1.0)

        last_im = ax.pcolormesh(xh3, xv3, p2d_n, cmap='inferno',
                                norm=LogNorm(vmin=DECOMP_LOG_VMIN, vmax=1.0),
                                shading='auto')
        mark_gt_2d(ax, gt_pos, 0, 1)
        mark_sensors_edge_2d(ax, 0, 1)


        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontsize=7)
        ax.set_ylabel('Y (m)' if idx == 0 else '', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.set_title(f'{panel_labels[idx]} Source {idx + 1}',
                     loc='left', fontsize=8, fontweight='bold', pad=9)

    fig3.tight_layout(pad=0.6, w_pad=0.6)
    # Horizontal colorbar below the row (ground truth noted in caption)
    if last_im is not None:
        cb3 = fig3.colorbar(last_im, ax=axs, orientation='vertical',
                            fraction=0.015, pad=0.012, aspect=15,
                            ticks=[1e-3, 1e-2, 1e-1, 1.0])
        cb3.ax.set_yticklabels(['0.001', '0.01', '0.1', '1'], fontsize=7)
        cb3.set_label('Norm. power (log)', fontsize=7)
        cb3.ax.tick_params(labelsize=7)

    for ext in ('pdf', 'png'):
        fig3.savefig(os.path.join(OUT_DIR, f'scenario1_source_decomp.{ext}'),
                     dpi=600, bbox_inches='tight')
    print("  Saved scenario1_source_decomp.{pdf,png}")
    plt.close(fig3)


if __name__ == "__main__":
    run_scenario()
