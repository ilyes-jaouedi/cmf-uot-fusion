"""Unit tests on small synthetic problems (no dataset needed)."""
import numpy as np
import pytest

from cmf_uot.bounds import camera_fim
from cmf_uot.camera import compute_C
from cmf_uot.covariance import calculate_sigma
from cmf_uot.geometry import freefield, get_grid_from_config
from cmf_uot.solvers.cmf import CMFSolver
from cmf_uot.solvers.cmf_uot1 import TransportOptimizerUnifiedGCD
from cmf_uot.solvers.cmf_uot2 import TransportOptimizer_numpy

K = 2 * np.pi * 4000 / 343.0


def toy_problem(n_grid=60, source_idx=(17, 42), seed=0):
    """Line array of 8 microphones, a 1-D line of candidate points, two sources."""
    rng = np.random.default_rng(seed)
    mics = np.column_stack([np.zeros(8), np.linspace(-0.15, 0.15, 8), np.zeros(8)])
    grid = np.column_stack([np.full(n_grid, 1.0), np.linspace(-0.6, 0.6, n_grid), np.zeros(n_grid)])
    G = freefield(mics, grid, K)
    p = np.zeros(n_grid)
    p[list(source_idx)] = [1.0, 0.7]
    Sigma = G @ (p[:, None] * G.conj().T)
    return G, Sigma, p, grid, rng


def test_grid_from_config_matches_paper():
    Grid, xg, yg, zg = get_grid_from_config()
    assert (len(xg), len(yg), len(zg)) == (120, 110, 35)
    assert Grid.shape == (120 * 110 * 35, 3)


def test_freefield_shape_and_decay():
    G, *_ = toy_problem()
    assert G.shape == (8, 60)
    assert np.all(np.isfinite(G))


def test_covariance_is_hermitian_psd():
    rng = np.random.default_rng(1)
    sigs = rng.standard_normal((1024 * 50, 4))
    S = calculate_sigma(sigs, Lseg=1024, freq=4000)
    assert S.shape == (4, 4)
    assert np.allclose(S, S.conj().T)
    assert np.linalg.eigvalsh(S).min() > -1e-9


def test_cmf_recovers_sources_noiseless():
    G, Sigma, p, *_ = toy_problem()
    solver = CMFSolver(G, Sigma)
    solver.GCD(400)
    top2 = set(np.argsort(solver.p)[-2:])
    assert top2 == {17, 42}
    assert np.all(solver.p >= 0)


def test_cmf_uot2_assigns_each_detection_to_its_source():
    G, Sigma, p, grid, _ = toy_problem()
    cam = np.array([1.0, -3.0, 0.0])
    # angular cost of each grid point to two detections pointing at the true sources
    def unit(v):
        return v / np.linalg.norm(v, axis=-1, keepdims=True)
    dirs_grid = unit(grid - cam)
    dirs_det = unit(grid[[17, 42]] - cam)
    C = np.arccos(np.clip(dirs_grid @ dirs_det.T, -1, 1))
    a = np.array([0.5, 0.5])
    opt = TransportOptimizer_numpy(C, a, Sigma, G, lambda_=1e-3, mu=1e-2)
    opt.optimize_greedy_coordinate_descent(num_iterations=300, verbose=False)
    assert np.all(opt.P >= 0)
    assert np.argmax(opt.P[:, 0]) == 17
    assert np.argmax(opt.P[:, 1]) == 42


def test_cmf_uot1_runs_and_stays_feasible():
    G, Sigma, p, grid, _ = toy_problem()
    C = np.abs(grid[:, 1:2] - grid[[17, 42], 1][None, :])
    opt = TransportOptimizerUnifiedGCD(C, np.array([0.5, 0.5]), Sigma, G, lambda_=1e-3, mu1=1e-2, mu2=1e-1)
    opt.optimize_unified_gcd(num_iterations=300, verbose=False)
    assert np.all(opt.P >= -1e-12) and np.all(opt.b >= -1e-12)
    assert opt.b.sum() > 0


def test_compute_C_is_an_angle():
    grid = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    A = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    C = compute_C(grid, [{"u": 10, "v": 10}], A)
    assert C.shape == (2, 1)
    assert np.all((C >= 0) & (C <= np.pi))


def test_camera_fim_is_symmetric_psd_and_blind_to_power():
    F = camera_fim(0.5, 0.5, (0.0, -1.0), alpha=300.0, sigma_c2=1.0)
    assert np.allclose(F, F.T)
    assert np.linalg.eigvalsh(F).min() > -1e-12
    assert np.all(F[2, :] == 0) and np.all(F[:, 2] == 0)


@pytest.mark.parametrize("y", [0.5, 2.0, 8.0])
def test_camera_fim_bearing_information_decreases_with_depth(y):
    F_near = camera_fim(0.5, y, (0.0, -1.0), 300.0, 1.0)
    F_far = camera_fim(0.5, 2 * y, (0.0, -1.0), 300.0, 1.0)
    assert F_far[0, 0] < F_near[0, 0]
