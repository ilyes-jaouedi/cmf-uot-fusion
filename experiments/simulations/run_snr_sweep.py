import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib as mpl
import pandas as pd
from tqdm import tqdm

from cmf_uot.solvers import cmf_uot2 as gcd_uot1       # b = P 1 coupling  (thesis CMF-UOT1, paper CMF-UOT2)
from cmf_uot.solvers import cmf_uot1 as gcd_uot2  # decoupled b       (thesis CMF-UOT2, paper CMF-UOT1)
from cmf_uot.solvers import cmf as cmf_module
from cmf_uot.bounds import camera_fim
from fig_naming import LABELS, NAMINGS, out_dir
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

OUTPUT_DIR = str(RESULTS_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Parameters ---
M_MIC = 8
FREQ = 4000.0
C_SPEED = 343.0
K_WAVE = 2 * np.pi * FREQ / C_SPEED
D_SPACING = (C_SPEED / FREQ) / 2.0
L_SNAPSHOTS = 500 # Restored to 500

# Geometry
ARRAY_CENTER = np.array([-1.0, 0.0, 0.0])
CAM_POS = np.array([0.0, -1.0, 0.0])

# Camera
ALPHA = 300.0 
BETA = 300.0
SIGMA_C = 10.0 
SIGMA_C2 = SIGMA_C**2

# Source
X0, Y0, Z0 = 0.5, 0.5, 0.0
P0 = 1.0

# Simulation Settings
N_TRIALS = 100
GRID_STEP = 0.001  # m
SEED = 0
# Quick timing runs: --trials N (outputs get a _test suffix)
if '--trials' in sys.argv:
    N_TRIALS = int(sys.argv[sys.argv.index('--trials') + 1])
OUT_SUFFIX = '_test' if '--trials' in sys.argv else ''
SNR_LEVELS = np.linspace(-15, 30, 10) # dB

# --- Helper: Array & Grid ---
def create_ula(M, d, center):
    indices = np.arange(M) - (M - 1) / 2.0
    y_pos = center[1] + indices * d
    return np.column_stack((np.full(M, center[0]), y_pos, np.full(M, center[2])))

def create_search_grid():
    # Fine Grid (1mm step) in local box
    step = GRID_STEP
    x = np.arange(X0 - 0.2, X0 + 0.2 + step, step) 
    y = np.arange(Y0 - 0.2, Y0 + 0.2 + step, step) 
    z = np.array([0.0])
    Xg, Yg, Zg = np.meshgrid(x, y, z)
    Grid = np.vstack([Xg.flatten(), Yg.flatten(), Zg.flatten()]).T
    return Grid, x, y, z

# --- Helper: CRB Calculation ---
def compute_crb_fusion(snr_db):
    Array = create_ula(M_MIC, D_SPACING, ARRAY_CENTER)
    dists_true = np.linalg.norm(Array - np.array([X0, Y0, Z0]), axis=1)
    g_true_norms2 = np.sum(1.0 / dists_true**2)
    sigma_a2 = P0 / (10**(snr_db/10.0) * g_true_norms2)
    
    def g_func(x, y):
        source = np.array([x, y, Z0])
        diff = Array - source
        d = np.linalg.norm(diff, axis=1)
        return np.exp(-1j * K_WAVE * d) / d
    
    g = g_func(X0, Y0)
    h = 1e-6
    gx = (g_func(X0+h, Y0) - g_func(X0-h, Y0)) / (2*h)
    gy = (g_func(X0, Y0+h) - g_func(X0, Y0-h)) / (2*h)
    
    Sa = P0 * np.outer(g, g.conj()) + sigma_a2 * np.eye(M_MIC)
    iSa = np.linalg.inv(Sa)
    
    dSa_x = P0 * (np.outer(gx, g.conj()) + np.outer(g, gx.conj()))
    dSa_y = P0 * (np.outer(gy, g.conj()) + np.outer(g, gy.conj()))
    dSa_p = np.outer(g, g.conj())
    
    F = np.zeros((3,3))
    derivs = [dSa_x, dSa_y, dSa_p]
    for i in range(3):
        for j in range(3):
            val = np.trace(iSa @ derivs[i] @ iSa @ derivs[j])
            F[i,j] = np.real(val)
    
    # Perspective camera actually simulated (see camera_fim.py)
    F_cam = camera_fim(X0, Y0, CAM_POS, ALPHA, SIGMA_C2)
    
    F_fus = L_SNAPSHOTS * F + F_cam
    CRB = np.linalg.inv(F_fus)
    return np.sqrt(CRB[0,0] + CRB[1,1]), np.sqrt(CRB[2,2])

# --- Simulation Core ---
def run_comparison():
    print("--- Running CMF-UOT1 vs UOT2 vs UOT-KL vs MLE vs CRB (Decoupled Geometry) ---")
    
    Array = create_ula(M_MIC, D_SPACING, ARRAY_CENTER)
    dists_true = np.linalg.norm(Array - np.array([X0, Y0, Z0]), axis=1)
    g_true_norms2 = np.sum(1.0 / dists_true**2)
    Grid, x_ax, y_ax, z_ax = create_search_grid()

    print(f"Grid: {Grid.shape[0]} points")
    dist = np.linalg.norm(Grid[:, np.newaxis, :] - Array[np.newaxis, :, :], axis=2)
    G = (np.exp(-1j * K_WAVE * dist) / dist).T
    norm_g_sq = np.sum(np.abs(G)**2, axis=0)

    # Camera model relative to CAM_POS
    dy_grid = Grid[:, 1] - CAM_POS[1]                       # depth of each grid point from camera
    U_grid = ALPHA * (Grid[:,0] - CAM_POS[0]) / dy_grid

    # Pre-compute Hessian scaling factor for mu2
    g_norm_sq = np.sum(np.abs(G)**2, axis=0)
    mean_g4 = np.mean(g_norm_sq**2)
    print(f"Mean |g|^4 factor: {mean_g4:.4e}")

        
    results = {
        'crb_pos': [], 'crb_pow': [],
        'cmf_pos': [], 'cmf_pow': [],
        'mle_pos': [], 'mle_pow': [],
        'uot1_pos': [], 'uot1_pow': [],
        'uot2_pos': [], 'uot2_pow': [],
    }
    
    np.random.seed(SEED)
    for snr in tqdm(SNR_LEVELS, desc="Simulating SNR"):
        # print(f"Simulating SNR = {snr:.1f} dB...")
        sigma_a2 = P0 / (10**(snr/10.0) * g_true_norms2)

        c_pos, c_pow = compute_crb_fusion(snr)
        results['crb_pos'].append(c_pos)
        results['crb_pow'].append(c_pow)
        
        trials = {'cmf':[], 'mle':[], 'uot1':[], 'uot2':[]}

        for k in range(N_TRIALS):
            # 1. Signal
            # True source drawn uniformly inside the grid cell centred on (X0, Y0):
            # with the source on a grid point a grid search can beat the CRB.
            x0 = X0 + np.random.uniform(-GRID_STEP / 2, GRID_STEP / 2)
            y0 = Y0 + np.random.uniform(-GRID_STEP / 2, GRID_STEP / 2)
            d_true = np.linalg.norm(Array - np.array([x0, y0, Z0]), axis=1)
            g_true = np.exp(-1j * K_WAVE * d_true) / d_true
            R_true = P0 * np.outer(g_true, g_true.conj()) + sigma_a2 * np.eye(M_MIC)
            X = np.linalg.cholesky(R_true) @ ((np.random.randn(M_MIC, L_SNAPSHOTS) + 1j * np.random.randn(M_MIC, L_SNAPSHOTS)) / np.sqrt(2))
            Sigma_hat = (X @ X.conj().T) / L_SNAPSHOTS
            
            # 2. Camera relative to CAM_POS
            u_meas = (ALPHA * (x0 - CAM_POS[0]) / (y0 - CAM_POS[1])) + np.random.randn() * np.sqrt(SIGMA_C2 / (y0 - CAM_POS[1]))
            # MLE-consistent normalised camera cost: (dy/(2σ_c²)) * (u - u_meas)²
            C_mat = (dy_grid / (2.0 * SIGMA_C2) * (U_grid - u_meas)**2).reshape(-1, 1)

            # Common Params (trace-normalised scaling, consistent with real-data scripts)
            sig_trace = np.trace(Sigma_hat).real
            lambda_val = (L_SNAPSHOTS / sigma_a2) / (sig_trace ** 2)

            mu_val =  1.0 / SIGMA_C2    # original: camera-noise-scaled marginal weight
            mu2_val = mean_g4           # acoustic b-coupling in UOT2 (SNR-independent)
            mu_kl =   1.0 / SIGMA_C2
            
            a_vec = np.array([1.0])
            n_iter = 100

            # Acoustic MLE power formula reused by all UOT methods
            gH_Sigma_g = np.sum(G.conj() * (Sigma_hat @ G), axis=0).real

            # --- UOT 1 (Original) ---
            opt1 = gcd_uot1.TransportOptimizer_numpy(C_mat, a_vec, Sigma_hat, G, lambda_=lambda_val, mu=mu_val)
            opt1.optimize_greedy_coordinate_descent(num_iterations=n_iter, verbose=False)
            idx1 = np.argmax(opt1.P[:, 0])
            pos1 = Grid[idx1]
            pow1 = np.maximum((gH_Sigma_g[idx1] / norm_g_sq[idx1] - sigma_a2) / norm_g_sq[idx1], 0.0)
            trials['uot1'].append(((pos1[0]-x0)**2 + (pos1[1]-y0)**2, (pow1-P0)**2))

            # --- UOT 2 (Unified Block) ---
            opt2 = gcd_uot2.TransportOptimizerUnifiedGCD(C_mat, a_vec, Sigma_hat, G, lambda_=lambda_val, mu1=mu_val, mu2=mu2_val)
            opt2.optimize_unified_gcd(num_iterations=n_iter, verbose=False)
            idx2 = np.argmax(opt2.b)
            pos2 = Grid[idx2]
            pow2 = np.maximum((gH_Sigma_g[idx2] / norm_g_sq[idx2] - sigma_a2) / norm_g_sq[idx2], 0.0)
            trials['uot2'].append(((pos2[0]-x0)**2 + (pos2[1]-y0)**2, (pow2-P0)**2))

            # --- CMF (acoustic only, covariance matrix fitting) ---
            cmf_obj = cmf_module.CMFSolver(G, Sigma_hat)
            cmf_obj.GCD(50)
            b_cmf = cmf_obj.p
            cmf_idx = np.argmax(b_cmf)
            cmf_pos = Grid[cmf_idx]
            cmf_pow = b_cmf[cmf_idx]
            trials['cmf'].append(((cmf_pos[0]-x0)**2 + (cmf_pos[1]-y0)**2, (cmf_pow-P0)**2))

            # --- MLE (fusion) ---
            dy_grid = np.maximum(Grid[:, 1] - CAM_POS[1], 1e-3)
            J = L_SNAPSHOTS * (np.log(np.maximum(gH_Sigma_g / norm_g_sq, 1e-12)) - (gH_Sigma_g / (sigma_a2 * norm_g_sq))) - 0.5 * np.log(dy_grid) + C_mat.flatten()
            mle_idx = np.argmin(J)
            mle_pos = Grid[mle_idx]
            mle_pow = np.maximum((gH_Sigma_g[mle_idx] / norm_g_sq[mle_idx] - sigma_a2) / norm_g_sq[mle_idx], 0.0)
            trials['mle'].append(((mle_pos[0]-x0)**2 + (mle_pos[1]-y0)**2, (mle_pow-P0)**2))
            
        # Avg
        for key in ['cmf', 'mle', 'uot1', 'uot2']:
            pos_errs = [x[0] for x in trials[key]]
            pow_errs = [x[1] for x in trials[key]]
            results[f'{key}_pos'].append(np.sqrt(np.mean(pos_errs)))
            results[f'{key}_pow'].append(np.sqrt(np.mean(pow_errs)))

    # --- Save Results to CSV ---
    try:
        df = pd.DataFrame(results)
        df.insert(0, 'SNR', SNR_LEVELS)
        csv_path = os.path.join(OUTPUT_DIR, f'sim_results_snr{OUT_SUFFIX}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")
    except Exception as e:
        print(f"Error saving CSV: {e}")
            
    for naming in NAMINGS:
        # Plotting — single double-column figure with two panels
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(IEEE_W2, 2.8))

        ax1.semilogy(SNR_LEVELS, results['crb_pos'],  'k-',   label='Fusion CRB', lw=1.5)
        ax1.semilogy(SNR_LEVELS, results['cmf_pos'],  'k--v', label='CMF',        ms=4)
        ax1.semilogy(SNR_LEVELS, results['mle_pos'],  'b-^',  label='MLE',        ms=4)
        ax1.semilogy(SNR_LEVELS, results['uot2_pos'], 'r-s',  label=LABELS[naming]['uot2'],   ms=4)
        ax1.semilogy(SNR_LEVELS, results['uot1_pos'], 'g--d', label=LABELS[naming]['uot1'],   ms=4)
        # The grid floor (RMSE of snapping a source drawn uniformly in a cell to
        # the cell centre) is not drawn: no curve comes near it, so it only
        # stretched the y axis down by a decade and flattened the differences.
        ax1.set_xlabel('SNR (dB)')
        ax1.set_ylabel('Position RMSE (m)')
        ax1.legend()
        ax1.grid(True, which='both')
        ax1.set_title('(a)', loc='left', fontsize=9, fontweight='bold')

        ax2.semilogy(SNR_LEVELS, results['crb_pow'],  'k-',   label='Fusion CRB', lw=1.5)
        ax2.semilogy(SNR_LEVELS, results['cmf_pow'],  'k--v', label='CMF',        ms=4)
        ax2.semilogy(SNR_LEVELS, results['mle_pow'],  'b-^',  label='MLE',        ms=4)
        ax2.semilogy(SNR_LEVELS, results['uot2_pow'], 'r-s',  label=LABELS[naming]['uot2'],   ms=4)
        ax2.semilogy(SNR_LEVELS, results['uot1_pow'], 'g--d', label=LABELS[naming]['uot1'],   ms=4)
        ax2.set_xlabel('SNR (dB)')
        ax2.set_ylabel('Power RMSE')
        ax2.legend()
        ax2.grid(True, which='both')
        ax2.set_title('(b)', loc='left', fontsize=9, fontweight='bold')

        fig.tight_layout(pad=0.8)
        for ext in ('pdf', 'png'):
            fig.savefig(os.path.join(out_dir(OUTPUT_DIR, naming), f'sim_comparison{OUT_SUFFIX}.{ext}'),
                        dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Comparison plots saved to extension/plots/journal_ready/")

if __name__ == "__main__":
    run_comparison()
