import numpy as np
from GIFT import geometry
from GIFT_3D import (
    interaction_height,
    induced_metric_from_height,
    build_demo,
)

def crop(a, n=5):
    return a[n:-n, n:-n]

def test_induced_plane_is_flat():
    n = 61
    x = np.linspace(-2.0, 2.0, n)
    H = np.zeros((n, n))
    g11, g12, g22, _, _ = induced_metric_from_height(
        H, x[1]-x[0], x[1]-x[0]
    )
    _, _, R, det = geometry(g11, g12, g22, x[1]-x[0], x[1]-x[0])
    assert np.max(np.abs(R)) < 1e-10
    assert np.min(det) > 0.0

def test_gaussian_graph_curvature_matches_analytic_formula():
    n = 161
    A = 2.2
    sigma = 1.6
    x = np.linspace(-4.0, 4.0, n)
    X, Y = np.meshgrid(x, x)
    H = interaction_height(X, Y, amplitude=A, spread=sigma)
    dx = x[1] - x[0]

    g11, g12, g22, _, _ = induced_metric_from_height(H, dx, dx)
    _, _, R, _ = geometry(g11, g12, g22, dx, dx)

    hx = -(X / sigma**2) * H
    hy = -(Y / sigma**2) * H
    hxx = (X**2 / sigma**4 - 1.0 / sigma**2) * H
    hyy = (Y**2 / sigma**4 - 1.0 / sigma**2) * H
    hxy = (X * Y / sigma**4) * H
    K_exact = (hxx*hyy - hxy*hxy) / (1.0 + hx*hx + hy*hy)**2
    R_exact = 2.0 * K_exact

    rmse = np.sqrt(np.mean((crop(R) - crop(R_exact))**2))
    assert rmse < 1.2e-3

def test_pure_geodesic_nearly_conserves_energy():
    data = build_demo(grid_size=201, dt=0.005, steps=1000)
    assert data["relative_energy_drift"] < 2.0e-3
    assert np.min(data["det_g"]) > 0.0
