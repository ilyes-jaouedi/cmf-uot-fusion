import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib as mpl
import pandas as pd

from cmf_uot.solvers import cmf_uot2 as gcd_uot1       # b=P·1 coupling  → paper: CMF-UOT2
from cmf_uot.solvers import cmf_uot1 as gcd_uot2  # decoupled b     → paper: CMF-UOT1
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

# --- Parameters ---
M_MIC = 8
FREQ = 4000.0
C_SPEED = 343.0
K_WAVE = 2 * np.pi * FREQ / C_SPEED
D_SPACING = (C_SPEED / FREQ) / 2.0
L_SNAPSHOTS = 500

# Geometry
ARRAY_CENTER = np.array([-1.0, 0.0, 0.0])
CAM_POS = np.array([0.0, -1.0, 0.0])

ALPHA = 300.0 
SIGMA_C = 10.0 
SIGMA_C2 = SIGMA_C**2

X0, Y0, Z0 = 0.5, 0.5, 0.0
P0 = 1.0

SNR_DB = 0.0

N_TRIALS_SENSITIVITY = 100  # Averaging over 10 trials for smoothness

# --- Helper: Array & Grid ---
def create_ula(M, d, center):
    indices = np.arange(M) - (M - 1) / 2.0
    y_pos = center[1] + indices * d
    return np.column_stack((np.full(M, center[0]), y_pos, np.full(M, center[2])))

def create_search_grid():
    step = 0.002 # 2mm
    x = np.arange(X0 - 0.2, X0 + 0.2 + step, step) 
    y = np.arange(Y0 - 0.2, Y0 + 0.2 + step, step) 
    z = np.array([0.0])
    Xg, Yg, Zg = np.meshgrid(x, y, z)
    Grid = np.vstack([Xg.flatten(), Yg.flatten(), Zg.flatten()]).T
    return Grid, x, y, z

def run_sensitivity_averaged():
    print(f"--- Running Sensitivity Analysis (SNR={SNR_DB}dB, Trials={N_TRIALS_SENSITIVITY}, Decoupled Geometry) ---")
    
    Array = create_ula(M_MIC, D_SPACING, ARRAY_CENTER)
    dists_true = np.linalg.norm(Array - np.array([X0, Y0, Z0]), axis=1)
    g_true_norms2 = np.sum(1.0 / dists_true**2)
    sigma_a2 = P0 / (10**(SNR_DB/10.0) * g_true_norms2)

    Grid, x_ax, y_ax, z_ax = create_search_grid()
    dist = np.linalg.norm(Grid[:, np.newaxis, :] - Array[np.newaxis, :, :], axis=2)
    G = (np.exp(-1j * K_WAVE * dist) / dist).T

    g_norm_sq = np.sum(np.abs(G)**2, axis=0)
    mean_g4 = np.mean(g_norm_sq**2)

    g_true = np.exp(-1j * K_WAVE * dists_true) / dists_true
    R_true = P0 * np.outer(g_true, g_true.conj()) + sigma_a2 * np.eye(M_MIC)
    trace_expected = np.trace(R_true).real
    lambda_heur = (L_SNAPSHOTS / sigma_a2) / (trace_expected ** 2)

    print(f"Heuristic Lambda: {lambda_heur:.2e}")

    # Camera model relative to CAM_POS
    dy_grid = Grid[:, 1] - CAM_POS[1]
    U_grid = ALPHA * (Grid[:,0] - CAM_POS[0]) / dy_grid

    # Generate N_TRIALS realizations of data
    datasets = []
    np.random.seed(42)
    for _ in range(N_TRIALS_SENSITIVITY):
        X = np.linalg.cholesky(R_true) @ ((np.random.randn(M_MIC, L_SNAPSHOTS) + 1j * np.random.randn(M_MIC, L_SNAPSHOTS)) / np.sqrt(2))
        Sigma_hat = (X @ X.conj().T) / L_SNAPSHOTS
        u_meas = (ALPHA * (X0 - CAM_POS[0]) / (Y0 - CAM_POS[1])) + np.random.randn() * np.sqrt(SIGMA_C2 / (Y0 - CAM_POS[1]))
        C_mat = (dy_grid / (2.0 * SIGMA_C2) * (U_grid - u_meas)**2).reshape(-1, 1)
        datasets.append((Sigma_hat, C_mat))
    
    a_vec = np.array([1.0])
    
    # --- 1. Sweep Lambda (Normalized) ---
    print("Sweeping Lambda...")
    factors = np.logspace(-6, 6, 15) # 0.01x to 100x heuristic
    res_lam = {'uot1':[], 'uot2':[]}
    res_lam_pow = {'uot1':[], 'uot2':[]}

    for f in factors:
        lam = lambda_heur * f
        # Scale Mus with Lambda (Testing Global Regularization Strength)
        # m1_l2 = lam * 0.0005 * (1.0/SIGMA_C2)
        # m1_kl = lam * 0.0005 * (1.0/SIGMA_C2)
        # m2 = lam * mean_g4
        m1_l2 = 1.0 / SIGMA_C2
        m2 = mean_g4

        errs = {'uot1':[], 'uot2':[]}
        pow_errs = {'uot1':[], 'uot2':[]}
        for Sigma_hat, C_mat in datasets:
            # UOT 1
            opt1 = gcd_uot1.TransportOptimizer_numpy(C_mat, a_vec, Sigma_hat, G, lambda_=lam, mu=m1_l2)
            opt1.optimize_greedy_coordinate_descent(num_iterations=100, verbose=False)
            p1 = Grid[np.argmax(opt1.P[:, 0])]
            errs['uot1'].append((p1[0]-X0)**2 + (p1[1]-Y0)**2)
            pow_errs['uot1'].append((np.max(opt1.P[:, 0]) - P0)**2)

            # UOT 2
            opt2 = gcd_uot2.TransportOptimizerUnifiedGCD(C_mat, a_vec, Sigma_hat, G, lambda_=lam, mu1=m1_l2, mu2=m2)
            opt2.optimize_unified_gcd(num_iterations=100, verbose=False)
            p2 = Grid[np.argmax(opt2.b)]
            errs['uot2'].append((p2[0]-X0)**2 + (p2[1]-Y0)**2)
            pow_errs['uot2'].append((np.max(opt2.b) - P0)**2)

        res_lam['uot1'].append(np.sqrt(np.mean(errs['uot1'])))
        res_lam['uot2'].append(np.sqrt(np.mean(errs['uot2'])))
        res_lam_pow['uot1'].append(np.sqrt(np.mean(pow_errs['uot1'])))
        res_lam_pow['uot2'].append(np.sqrt(np.mean(pow_errs['uot2'])))

    # --- 2. Sweep Mu1 (Marginal Constraint) ---
    print("Sweeping Mu1...")
    res_mu1 = {'uot1':[], 'uot2':[]}
    res_mu1_pow = {'uot1':[], 'uot2':[]}

    # Fix Lambda at optimal
    lam_opt = lambda_heur
    m2_opt = mean_g4

    # Base heuristics for Mu1
    base_m1_l2 = 1.0 / SIGMA_C2
    mu1_factors = np.logspace(-6, 6, 15)
    for f in mu1_factors:
        curr_m1_l2 = base_m1_l2 * f

        errs = {'uot1':[], 'uot2':[]}
        pow_errs = {'uot1':[], 'uot2':[]}
        for Sigma_hat, C_mat in datasets:
            # UOT 1
            opt1 = gcd_uot1.TransportOptimizer_numpy(C_mat, a_vec, Sigma_hat, G, lambda_=lam_opt, mu=curr_m1_l2)
            opt1.optimize_greedy_coordinate_descent(num_iterations=100, verbose=False)
            p1 = Grid[np.argmax(opt1.P[:, 0])]
            errs['uot1'].append((p1[0]-X0)**2 + (p1[1]-Y0)**2)
            pow_errs['uot1'].append((np.max(opt1.P[:, 0]) - P0)**2)

            # UOT 2
            opt2 = gcd_uot2.TransportOptimizerUnifiedGCD(C_mat, a_vec, Sigma_hat, G, lambda_=lam_opt, mu1=curr_m1_l2, mu2=m2_opt)
            opt2.optimize_unified_gcd(num_iterations=100, verbose=False)
            p2 = Grid[np.argmax(opt2.b)]
            errs['uot2'].append((p2[0]-X0)**2 + (p2[1]-Y0)**2)
            pow_errs['uot2'].append((np.max(opt2.b) - P0)**2)

        res_mu1['uot1'].append(np.sqrt(np.mean(errs['uot1'])))
        res_mu1['uot2'].append(np.sqrt(np.mean(errs['uot2'])))
        res_mu1_pow['uot1'].append(np.sqrt(np.mean(pow_errs['uot1'])))
        res_mu1_pow['uot2'].append(np.sqrt(np.mean(pow_errs['uot2'])))

    # --- 3. Sweep Mu2 (Coupling - UOT2 Only) ---
    print("Sweeping Mu2...")
    res_mu2 = []
    res_mu2_uot1_baseline = []
    res_mu2_pow = []
    res_mu2_pow_uot1_baseline = []

    # Fix Mu1 at optimal
    m1_opt = base_m1_l2
    base_m2 = m2_opt

    # Calculate UOT1 baseline once (since it doesn't depend on Mu2)
    errs_uot1_base = []
    pow_errs_uot1_base = []
    for Sigma_hat, C_mat in datasets:
        opt1 = gcd_uot1.TransportOptimizer_numpy(C_mat, a_vec, Sigma_hat, G, lambda_=lam_opt, mu=m1_opt)
        opt1.optimize_greedy_coordinate_descent(num_iterations=100, verbose=False)
        p1 = Grid[np.argmax(opt1.P[:, 0])]
        errs_uot1_base.append((p1[0]-X0)**2 + (p1[1]-Y0)**2)
        pow_errs_uot1_base.append((np.max(opt1.P[:, 0]) - P0)**2)
    uot1_baseline_rmse = np.sqrt(np.mean(errs_uot1_base))
    uot1_baseline_pow_rmse = np.sqrt(np.mean(pow_errs_uot1_base))

    mu2_factors = np.logspace(-6, 6, 15)
    for f in mu2_factors:
        curr_m2 = base_m2 * f
        errs_mu2 = []
        pow_errs_mu2 = []
        for Sigma_hat, C_mat in datasets:
            opt2 = gcd_uot2.TransportOptimizerUnifiedGCD(C_mat, a_vec, Sigma_hat, G, lambda_=lam_opt, mu1=m1_opt, mu2=curr_m2)
            opt2.optimize_unified_gcd(num_iterations=100, verbose=False)
            p2 = Grid[np.argmax(opt2.b)]
            errs_mu2.append((p2[0]-X0)**2 + (p2[1]-Y0)**2)
            pow_errs_mu2.append((np.max(opt2.b) - P0)**2)
        res_mu2.append(np.sqrt(np.mean(errs_mu2)))
        res_mu2_uot1_baseline.append(uot1_baseline_rmse)
        res_mu2_pow.append(np.sqrt(np.mean(pow_errs_mu2)))
        res_mu2_pow_uot1_baseline.append(uot1_baseline_pow_rmse)

    # --- Plotting ---
    # Plot 1: Lambda
    fig1, ax1 = plt.subplots(figsize=(IEEE_W1, 2.5))
    ax1.loglog(factors, res_lam['uot2'], 'r-s',  label='CMF-UOT1', ms=4)
    ax1.loglog(factors, res_lam['uot1'], 'g--d', label='CMF-UOT2', ms=4)
    ax1.axvline(x=1.0, color='k', linestyle='--', lw=1.2, label=r'Heuristic $\lambda^*$')
    ax1.set_xlabel(r'$\lambda / \lambda^*$')
    ax1.set_ylabel('Position RMSE (m)')
    ax1.legend()
    ax1.grid(True, which='both')
    fig1.tight_layout(pad=0.8)
    for ext in ('pdf', 'png'):
        fig1.savefig(os.path.join(OUTPUT_DIR, f'sensitivity_lambda_averaged.{ext}'),
                     dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # Plot 2: Mu1
    fig2, ax2 = plt.subplots(figsize=(IEEE_W1, 2.5))
    ax2.loglog(mu1_factors, res_mu1['uot2'], 'r-s',  label='CMF-UOT1', ms=4)
    ax2.loglog(mu1_factors, res_mu1['uot1'], 'g--d', label='CMF-UOT2', ms=4)
    ax2.axvline(x=1.0, color='k', linestyle='--', lw=1.2, label=r'Heuristic $\beta^*$')
    ax2.set_xlabel(r'$\beta / \beta^*$')
    ax2.set_ylabel('Position RMSE (m)')
    ax2.legend()
    ax2.grid(True, which='both')
    fig2.tight_layout(pad=0.8)
    for ext in ('pdf', 'png'):
        fig2.savefig(os.path.join(OUTPUT_DIR, f'sensitivity_mu1_averaged.{ext}'),
                     dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # Plot 3: Mu2
    fig3, ax3 = plt.subplots(figsize=(IEEE_W1, 2.5))
    ax3.loglog(mu2_factors, res_mu2,               'r-s',  label='CMF-UOT1',           ms=4)
    ax3.loglog(mu2_factors, res_mu2_uot1_baseline, 'g--',  label='CMF-UOT2 (baseline)', lw=1.2)
    ax3.axvline(x=1.0, color='k', linestyle='--', lw=1.2, label=r'Heuristic $\alpha^*$')
    ax3.set_xlabel(r'$\alpha / \alpha^*$')
    ax3.set_ylabel('Position RMSE (m)')
    ax3.legend()
    ax3.grid(True, which='both')
    fig3.tight_layout(pad=0.8)
    for ext in ('pdf', 'png'):
        fig3.savefig(os.path.join(OUTPUT_DIR, f'sensitivity_mu2_averaged.{ext}'),
                     dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # Plot 4: Lambda — Power RMSE
    fig4, ax4 = plt.subplots(figsize=(IEEE_W1, 2.5))
    ax4.loglog(factors, res_lam_pow['uot2'], 'r-s',  label='CMF-UOT1', ms=4)
    ax4.loglog(factors, res_lam_pow['uot1'], 'g--d', label='CMF-UOT2', ms=4)
    ax4.axvline(x=1.0, color='k', linestyle='--', lw=1.2, label=r'Heuristic $\lambda^*$')
    ax4.set_xlabel(r'$\lambda / \lambda^*$')
    ax4.set_ylabel('Power RMSE')
    ax4.legend()
    ax4.grid(True, which='both')
    fig4.tight_layout(pad=0.8)
    for ext in ('pdf', 'png'):
        fig4.savefig(os.path.join(OUTPUT_DIR, f'sensitivity_lambda_power_averaged.{ext}'),
                     dpi=300, bbox_inches='tight')
    plt.close(fig4)

    # Plot 5: Mu1 — Power RMSE
    fig5, ax5 = plt.subplots(figsize=(IEEE_W1, 2.5))
    ax5.loglog(mu1_factors, res_mu1_pow['uot2'], 'r-s',  label='CMF-UOT1', ms=4)
    ax5.loglog(mu1_factors, res_mu1_pow['uot1'], 'g--d', label='CMF-UOT2', ms=4)
    ax5.axvline(x=1.0, color='k', linestyle='--', lw=1.2, label=r'Heuristic $\beta^*$')
    ax5.set_xlabel(r'$\beta / \beta^*$')
    ax5.set_ylabel('Power RMSE')
    ax5.legend()
    ax5.grid(True, which='both')
    fig5.tight_layout(pad=0.8)
    for ext in ('pdf', 'png'):
        fig5.savefig(os.path.join(OUTPUT_DIR, f'sensitivity_mu1_power_averaged.{ext}'),
                     dpi=300, bbox_inches='tight')
    plt.close(fig5)

    # Plot 6: Mu2 — Power RMSE
    fig6, ax6 = plt.subplots(figsize=(IEEE_W1, 2.5))
    ax6.loglog(mu2_factors, res_mu2_pow,               'r-s',  label='CMF-UOT1',           ms=4)
    ax6.loglog(mu2_factors, res_mu2_pow_uot1_baseline, 'g--',  label='CMF-UOT2 (baseline)', lw=1.2)
    ax6.axvline(x=1.0, color='k', linestyle='--', lw=1.2, label=r'Heuristic $\alpha^*$')
    ax6.set_xlabel(r'$\alpha / \alpha^*$')
    ax6.set_ylabel('Power RMSE')
    ax6.legend()
    ax6.grid(True, which='both')
    fig6.tight_layout(pad=0.8)
    for ext in ('pdf', 'png'):
        fig6.savefig(os.path.join(OUTPUT_DIR, f'sensitivity_mu2_power_averaged.{ext}'),
                     dpi=300, bbox_inches='tight')
    plt.close(fig6)

    print("Averaged Sensitivity plots saved.")

    # --- Save CSVs ---
    df_lambda = pd.DataFrame({
        'lambda_ratio': factors,
        'uot1_pos':     res_lam['uot1'],
        'uot2_pos':     res_lam['uot2'],
        'uot1_pow':     res_lam_pow['uot1'],
        'uot2_pow':     res_lam_pow['uot2'],
    })
    csv_path = os.path.join(OUTPUT_DIR, 'sensitivity_lambda.csv')
    df_lambda.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    df_mu1 = pd.DataFrame({
        'beta_ratio': mu1_factors,
        'uot1_pos':   res_mu1['uot1'],
        'uot2_pos':   res_mu1['uot2'],
        'uot1_pow':   res_mu1_pow['uot1'],
        'uot2_pow':   res_mu1_pow['uot2'],
    })
    csv_path = os.path.join(OUTPUT_DIR, 'sensitivity_mu1.csv')
    df_mu1.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    df_mu2 = pd.DataFrame({
        'alpha_ratio':        mu2_factors,
        'uot2_pos':           res_mu2,
        'uot1_baseline_pos':  res_mu2_uot1_baseline,
        'uot2_pow':           res_mu2_pow,
        'uot1_baseline_pow':  res_mu2_pow_uot1_baseline,
    })
    csv_path = os.path.join(OUTPUT_DIR, 'sensitivity_mu2.csv')
    df_mu2.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

if __name__ == "__main__":
    run_sensitivity_averaged()
