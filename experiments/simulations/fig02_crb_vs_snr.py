import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib as mpl

from cmf_uot.bounds import camera_fim
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
IEEE_W2 = 7.16
IEEE_W1 = 3.5

OUTPUT_DIR = str(RESULTS_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Helper Functions ---

def Sigma_a(g, p, sigma_a2):
    return p * np.outer(g, g.conj()) + sigma_a2 * np.eye(len(g))

def dSigma_a_dx(g, gx, p):
    return p * (np.outer(gx, g.conj()) + np.outer(g, gx.conj()))

def dSigma_a_dy(g, gy, p):
    return p * (np.outer(gy, g.conj()) + np.outer(g, gy.conj()))

def dSigma_a_dp(g):
    return np.outer(g, g.conj())

def numerical_gradient(func, x, y, k, Array, h=1e-6):
    gx = (func(x + h, y, k, Array) - func(x - h, y, k, Array)) / (2 * h)
    gy = (func(x, y + h, k, Array) - func(x, y - h, k, Array)) / (2 * h)
    return gx, gy

def g_func_notebook(x, y, k, mic_pos):
    z0 = 0.0
    source = np.array([x, y, z0])
    diff = mic_pos - source
    d = np.linalg.norm(diff, axis=1)
    return np.exp(-1j * k * d) / d

def compute_fim_acoustic(x, y, p, k, Array, g_func, sigma_a2, nb_snaps):
    g = g_func(x, y, k, Array)
    gx, gy = numerical_gradient(g_func, x, y, k, Array)
    Sa = Sigma_a(g, p, sigma_a2)

    try:
        iSa = np.linalg.inv(Sa)
    except np.linalg.LinAlgError:
        return np.zeros((3, 3))

    dSa_x = dSigma_a_dx(g, gx, p)
    dSa_y = dSigma_a_dy(g, gy, p)
    dSa_p = dSigma_a_dp(g)

    F = np.zeros((3, 3))
    d_list = [dSa_x, dSa_y, dSa_p]

    # Slepian-Bangs for circular complex Gaussian snapshots: no factor 1/2
    # (the 1/2 belongs to the real-valued Gaussian formula)
    for i in range(3):
        for j in range(3):
            term = iSa @ d_list[i] @ iSa @ d_list[j]
            F[i, j] = np.trace(term).real

    return nb_snaps * F

def compute_fim_camera(alpha, beta, x, y, cam_pos, sigma_c_deg):
    """Camera FIM; sigma_c_deg is angular noise in degrees.

    Converts to pixel-equivalent variance via sigma_c_pix = alpha * sigma_c_rad,
    matching the standard pinhole model (alpha = focal length in px/rad).
    """
    sigma_c_rad = sigma_c_deg * np.pi / 180.0
    sigma_c2 = (alpha * sigma_c_rad) ** 2  # pixel-equivalent variance

    # beta is no longer used: the simulated camera measures one perspective pixel
    return camera_fim(x, y, cam_pos, alpha, sigma_c2)

def create_ula(M, d=0.1, center=(-1.0, 0.0)):
    indices = np.arange(M) - (M - 1) / 2.0
    y_pos = center[1] + indices * d
    return np.column_stack((np.full(M, center[0]), y_pos, np.zeros(M)))

# --- Main Plotting Function ---
def generate_theoretical_plots():
    print("--- Generating Balanced Theoretical Plots (Decoupled Geometry) ---")

    # Standard parameters
    M = 8
    freq = 4000.0
    c = 343.0
    k = 2 * np.pi * freq / c
    d_spacing = (c / freq) / 2.0

    array_center = (-1.0, 0.0)
    cam_pos = (0.0, -1.0)
    Array_Theory = create_ula(M, d_spacing, center=array_center)

    x0 = 0.5
    y0 = 0.5
    p0 = 10.0
    nb_snapshots = 500

    alpha_theory = 300.0   # focal length (px/rad)
    beta_theory  = 300.0

    # Three camera noise levels (degrees) for the fusion CRB family
    SIGMA_C_VARIANTS = [0.5, 2.0, 8.0]
    SIGMA_C_DEFAULT  = 2.0            # used for non-SNR plots
    FUSION_STYLES    = ['r-s', 'g-^', 'm-D']

    # Fixed noise level for distance / M sweeps (SNR = 10 dB)
    sigma_a2_fixed = p0 / (10 ** (10 / 10.0))

    # -----------------------------------------------------------------------
    # Plot 1: CRB vs SNR — position (a) and power (b), vertically stacked
    # -----------------------------------------------------------------------
    snr_db_vals = np.linspace(-10, 30, 15)

    rmse_pos_ac  = []
    rmse_pow_ac  = []
    rmse_pos_fus = [[] for _ in SIGMA_C_VARIANTS]
    rmse_pow_fus = [[] for _ in SIGMA_C_VARIANTS]

    for snr in snr_db_vals:
        sigma_a2 = p0 / (10 ** (snr / 10.0))
        Fa = compute_fim_acoustic(x0, y0, p0, k, Array_Theory, g_func_notebook,
                                  sigma_a2, nb_snapshots)
        try:
            Ca = np.linalg.inv(Fa)
            rmse_pos_ac.append(np.sqrt(Ca[0, 0] + Ca[1, 1]))
            rmse_pow_ac.append(np.sqrt(Ca[2, 2]))
        except np.linalg.LinAlgError:
            rmse_pos_ac.append(np.nan)
            rmse_pow_ac.append(np.nan)

        for i, sc in enumerate(SIGMA_C_VARIANTS):
            Fc    = compute_fim_camera(alpha_theory, beta_theory, x0, y0, cam_pos, sc)
            F_fus = Fa + Fc
            try:
                Cf = np.linalg.inv(F_fus)
                rmse_pos_fus[i].append(np.sqrt(Cf[0, 0] + Cf[1, 1]))
                rmse_pow_fus[i].append(np.sqrt(Cf[2, 2]))
            except np.linalg.LinAlgError:
                rmse_pos_fus[i].append(np.nan)
                rmse_pow_fus[i].append(np.nan)

    fig_snr, (ax_pos, ax_pow) = plt.subplots(1, 2, figsize=(IEEE_W2, 2.8))

    ax_pos.semilogy(snr_db_vals, rmse_pos_ac, 'b--o', label='Acoustic-only', ms=4)
    for i, sc in enumerate(SIGMA_C_VARIANTS):
        ax_pos.semilogy(snr_db_vals, rmse_pos_fus[i], FUSION_STYLES[i],
                        label=fr'Fusion ($\sigma_c={sc}^\circ$)', ms=4)
    ax_pos.set_xlabel('SNR (dB)')
    ax_pos.set_ylabel('Position RMSE (m)')
    ax_pos.legend()
    ax_pos.grid(True, which='both')
    ax_pos.set_title('(a)', loc='left', fontsize=9, fontweight='bold')

    ax_pow.semilogy(snr_db_vals, rmse_pow_ac, 'b--o', label='Acoustic-only', ms=4)
    for i, sc in enumerate(SIGMA_C_VARIANTS):
        ax_pow.semilogy(snr_db_vals, rmse_pow_fus[i], FUSION_STYLES[i],
                        label=fr'Fusion ($\sigma_c={sc}^\circ$)', ms=4)
    ax_pow.set_xlabel('SNR (dB)')
    ax_pow.set_ylabel('Power RMSE')
    ax_pow.legend()
    ax_pow.grid(True, which='both')
    ax_pow.set_title('(b)', loc='left', fontsize=9, fontweight='bold')

    fig_snr.tight_layout(pad=0.8)
    for ext in ('pdf', 'png'):
        fig_snr.savefig(os.path.join(OUTPUT_DIR, f'crb_vs_snr.{ext}'),
                        dpi=300, bbox_inches='tight')
    plt.close(fig_snr)

    # Same two panels as separate files (Elsevier: no two graphs in one file),
    # saved at their printed size, 80 x 58 mm, for (a)/(b) subfigures in LaTeX.
    panel_rc = {'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                'ytick.labelsize': 7, 'legend.fontsize': 7, 'lines.linewidth': 1.0}
    for kind, ac, fus, ylabel in (('pos', rmse_pos_ac, rmse_pos_fus, 'Position RMSE (m)'),
                                  ('pow', rmse_pow_ac, rmse_pow_fus, 'Power RMSE')):
        with mpl.rc_context(panel_rc):
            fig_p, ax_p = plt.subplots(figsize=(80 / 25.4, 58 / 25.4))
            ax_p.semilogy(snr_db_vals, ac, 'b--o', label='Acoustic-only', ms=3.5)
            for i, sc in enumerate(SIGMA_C_VARIANTS):
                ax_p.semilogy(snr_db_vals, fus[i], FUSION_STYLES[i],
                              label=fr'Fusion ($\sigma_c={sc}^\circ$)', ms=3.5)
            ax_p.set_xlabel('SNR (dB)', labelpad=1)
            ax_p.set_ylabel(ylabel, labelpad=1)
            ax_p.legend(framealpha=0.85, borderpad=0.3, labelspacing=0.25, handletextpad=0.4)
            ax_p.grid(True, which='both')
            fig_p.tight_layout(pad=0.25)
            for ext in ('pdf', 'png'):
                fig_p.savefig(os.path.join(OUTPUT_DIR, f'crb_vs_snr_{kind}.{ext}'), dpi=600)
            plt.close(fig_p)

    # -----------------------------------------------------------------------
    # Plot 2: CRB vs Distance (depth y)
    # -----------------------------------------------------------------------
    dists = np.linspace(0.5, 8.0, 20)
    rmse_dist_ac  = []
    rmse_dist_fus = []

    for d_val in dists:
        curr_y = d_val
        Fa    = compute_fim_acoustic(x0, curr_y, p0, k, Array_Theory, g_func_notebook,
                                     sigma_a2_fixed, nb_snapshots)
        Fc    = compute_fim_camera(alpha_theory, beta_theory, x0, curr_y, cam_pos,
                                   SIGMA_C_DEFAULT)
        F_fus = Fa + Fc
        try:
            Ca = np.linalg.inv(Fa)
            Cf = np.linalg.inv(F_fus)
            rmse_dist_ac.append(np.sqrt(Ca[0, 0] + Ca[1, 1]))
            rmse_dist_fus.append(np.sqrt(Cf[0, 0] + Cf[1, 1]))
        except np.linalg.LinAlgError:
            rmse_dist_ac.append(np.nan)
            rmse_dist_fus.append(np.nan)

    fig_dist, ax_dist = plt.subplots(figsize=(3.5, 2.8))
    ax_dist.plot(dists, rmse_dist_ac,  'b--o', label='Acoustic-only', ms=4)
    ax_dist.plot(dists, rmse_dist_fus, 'r-s',
                 label=fr'Fusion ($\sigma_c={SIGMA_C_DEFAULT}^\circ$)', ms=4)
    ax_dist.set_xlabel('Distance $y$ (m)')
    ax_dist.set_ylabel('Position RMSE (m)')
    ax_dist.legend()
    ax_dist.grid(True, which='both')
    fig_dist.tight_layout(pad=0.8)
    for ext in ('pdf', 'png'):
        fig_dist.savefig(os.path.join(OUTPUT_DIR, f'crb_vs_distance.{ext}'),
                         dpi=300, bbox_inches='tight')
    plt.close(fig_dist)

    # -----------------------------------------------------------------------
    # Plot 3+4: CRB vs camera noise (degrees) and vs M (side-by-side)
    # -----------------------------------------------------------------------
    sigma_c_deg_vals = np.linspace(0.5, 15.0, 20)
    rmse_noise_fus   = []

    Fa_base  = compute_fim_acoustic(x0, y0, p0, k, Array_Theory, g_func_notebook,
                                    sigma_a2_fixed, nb_snapshots)
    Ca_base  = np.linalg.inv(Fa_base)
    ac_base_rmse = np.sqrt(Ca_base[0, 0] + Ca_base[1, 1])

    for sc in sigma_c_deg_vals:
        Fc    = compute_fim_camera(alpha_theory, beta_theory, x0, y0, cam_pos, sc)
        F_fus = Fa_base + Fc
        try:
            Cf = np.linalg.inv(F_fus)
            rmse_noise_fus.append(np.sqrt(Cf[0, 0] + Cf[1, 1]))
        except np.linalg.LinAlgError:
            rmse_noise_fus.append(np.nan)

    m_vals    = np.arange(4, 24, 4)
    rmse_m_ac  = []
    rmse_m_fus = []

    for m in m_vals:
        Arr   = create_ula(m, d_spacing, center=array_center)
        Fa    = compute_fim_acoustic(x0, y0, p0, k, Arr, g_func_notebook,
                                     sigma_a2_fixed, nb_snapshots)
        Fc    = compute_fim_camera(alpha_theory, beta_theory, x0, y0, cam_pos,
                                   SIGMA_C_DEFAULT)
        F_fus = Fa + Fc
        try:
            Ca = np.linalg.inv(Fa)
            Cf = np.linalg.inv(F_fus)
            rmse_m_ac.append(np.sqrt(Ca[0, 0] + Ca[1, 1]))
            rmse_m_fus.append(np.sqrt(Cf[0, 0] + Cf[1, 1]))
        except np.linalg.LinAlgError:
            rmse_m_ac.append(np.nan)
            rmse_m_fus.append(np.nan)

    fig_sens, (ax_c, ax_m) = plt.subplots(1, 2, figsize=(IEEE_W2, 2.8))

    ax_c.plot(sigma_c_deg_vals, rmse_noise_fus, 'r-s', label='Fusion', ms=4)
    ax_c.axhline(y=ac_base_rmse, color='k', linestyle='--', lw=1.2, label='Acoustic limit')
    ax_c.set_xlabel(r'Camera noise $\sigma_c$ (deg)')
    ax_c.set_ylabel('Position RMSE (m)')
    ax_c.legend()
    ax_c.grid(True, which='both')
    ax_c.set_title('(a)', loc='left', fontsize=9, fontweight='bold')

    ax_m.plot(m_vals, rmse_m_ac,  'b--o', label='Acoustic-only', ms=4)
    ax_m.plot(m_vals, rmse_m_fus, 'r-s',
              label=fr'Fusion ($\sigma_c={SIGMA_C_DEFAULT}^\circ$)', ms=4)
    ax_m.set_xlabel('Number of sensors $M$')
    ax_m.set_ylabel('Position RMSE (m)')
    ax_m.legend()
    ax_m.grid(True, which='both')
    ax_m.set_title('(b)', loc='left', fontsize=9, fontweight='bold')

    fig_sens.tight_layout(pad=0.8)
    for ext in ('pdf', 'png'):
        fig_sens.savefig(os.path.join(OUTPUT_DIR, f'crb_vs_sensors.{ext}'),
                         dpi=300, bbox_inches='tight')
    plt.close(fig_sens)

    # Stand-alone version of panel (b), used as crb_vs_m in the thesis
    fig_m, ax = plt.subplots(figsize=(IEEE_W1 * 2, 4.0))
    ax.plot(m_vals, rmse_m_ac,  'b--o', label='Acoustic-only', ms=6, lw=2)
    ax.plot(m_vals, rmse_m_fus, 'r-s',  label='Fusion',        ms=6, lw=2)
    ax.set_xlabel('Number of Sensors S', fontsize=12)
    ax.set_ylabel('Position RMSE Bound (m)', fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=12)
    ax.grid(True, which='both')
    fig_m.tight_layout(pad=0.8)
    for ext in ('pdf', 'png'):
        fig_m.savefig(os.path.join(OUTPUT_DIR, f'crb_vs_m.{ext}'),
                      dpi=300, bbox_inches='tight')
    plt.close(fig_m)

    print("CRB plots saved to results/")

if __name__ == "__main__":
    generate_theoretical_plots()
