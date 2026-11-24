"""Fisher information of the simulated camera measurement.

All simulation scripts model the camera as one perspective pixel coordinate

    u = alpha * (x - cx) / (y - cy)  +  noise,   noise ~ N(0, sigma_c2 / (y - cy)),

where (cx, cy) is the camera position, alpha the focal length in pixels and
sigma_c2 the pixel noise variance at unit depth. The parameters are
theta = (x, y, p); the camera carries no information on the power p.

For a scalar Gaussian observation whose mean mu and variance v both depend on
theta, the Fisher information is

    F_ab = (dmu/da)(dmu/db) / v + 0.5 (dv/da)(dv/db) / v^2.

This replaces the former F_xx = alpha^2 dy / sigma_c2 and
F_yy = beta^2 dy / sigma_c2 + 1/dy^2, which described a linear two-coordinate
camera (alpha x, beta y) that also measured depth, not the camera simulated.
"""
import numpy as np


def camera_fim(x, y, cam_pos, alpha, sigma_c2):
    """3x3 Fisher information on (x, y, p) of the perspective pixel u."""
    dx = x - cam_pos[0]
    dy = y - cam_pos[1]
    v = sigma_c2 / dy
    du_dx = alpha / dy
    du_dy = -alpha * dx / dy**2
    dv_dy = -sigma_c2 / dy**2

    F = np.zeros((3, 3))
    F[0, 0] = du_dx**2 / v
    F[0, 1] = F[1, 0] = du_dx * du_dy / v
    F[1, 1] = du_dy**2 / v + 0.5 * dv_dy**2 / v**2
    return F
