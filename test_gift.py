import warnings
import numpy as np
from GIFT import geometry, metric_from_source, gaussian_source


def crop(a, n=4):
    return a[n:-n, n:-n]


def as_matrix(parts):
    a, b, c = parts
    return np.array([[a, b], [b, c]], dtype=float)


def test_flat_metric_zero_curvature():
    n = 41
    x = np.linspace(-2, 2, n)
    z = np.zeros((n, n))
    o = np.ones((n, n))
    _, Ric, R, det = geometry(o, z, o, x[1]-x[0], x[1]-x[0])
    assert np.max(np.abs(R)) < 1e-10
    assert np.max(np.abs(Ric)) < 1e-10
    assert np.min(det) > 0


def test_metric_positive_and_zero_source_recovers_background():
    x = np.linspace(-3, 3, 51)
    X, Y = np.meshgrid(x, x)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g = metric_from_source(*gaussian_source(X, Y, amp=3.0))

    det = g[0] * g[2] - g[1]**2
    assert np.min(g[0]) > 0
    assert np.min(det) > 0

    g0 = (1.5, 0.2, 1.0)
    recovered = as_matrix(
        metric_from_source(0.0, 0.0, 0.0, background=g0)
    )
    assert np.allclose(recovered, as_matrix(g0), atol=1e-13, rtol=1e-13)


def test_source_to_metric_is_coordinate_covariant():
    # x = J x'.  Covariant tensors transform as T'=J^T T J.
    T = np.array([[1.2, 0.3], [0.3, -0.2]])
    g0 = np.array([[1.5, 0.2], [0.2, 1.0]])
    J = np.array([[2.0, 0.4], [0.1, 1.3]])

    g = as_matrix(metric_from_source(
        T[0, 0], T[0, 1], T[1, 1],
        background=(g0[0, 0], g0[0, 1], g0[1, 1]),
        isotropic_weight=0.8,
        deviatoric_weight=1.3,
    ))

    Tp = J.T @ T @ J
    g0p = J.T @ g0 @ J
    gp = as_matrix(metric_from_source(
        Tp[0, 0], Tp[0, 1], Tp[1, 1],
        background=(g0p[0, 0], g0p[0, 1], g0p[1, 1]),
        isotropic_weight=0.8,
        deviatoric_weight=1.3,
    ))

    assert np.allclose(gp, J.T @ g @ J, atol=5e-13, rtol=5e-13)


def test_constant_conformal_metric_flat():
    n = 31
    x = np.linspace(-1, 1, n)
    o = np.ones((n, n)) * 2.5
    z = np.zeros((n, n))
    _, _, R, _ = geometry(o, z, o, x[1]-x[0], x[1]-x[0])
    assert np.max(np.abs(R)) < 1e-10


def test_flat_offdiagonal_coordinate_metric():
    n = 81
    u = np.linspace(-2, 2, n)
    U, V = np.meshgrid(u, u)
    a = .3
    g00 = 1 + 4*a*a*U**2
    g01 = 2*a*U
    g11 = np.ones_like(U)
    _, Ric, R, _ = geometry(
        g00, g01, g11, u[1]-u[0], u[1]-u[0]
    )
    assert np.max(np.abs(crop(R))) < 1e-10
    assert np.max(np.abs(crop(Ric[0, 1]-Ric[1, 0]))) < 1e-10


def sphere_error(n):
    x = np.linspace(-1.5, 1.5, n)
    X, Y = np.meshgrid(x, x)
    lam2 = 4/(1+X**2+Y**2)**2
    z = np.zeros_like(lam2)
    _, Ric, R, _ = geometry(
        lam2, z, lam2, x[1]-x[0], x[1]-x[0]
    )
    e = crop(R)-2.0
    return (
        np.sqrt(np.mean(e*e)),
        np.max(np.abs(crop(Ric[0, 1]-Ric[1, 0]))),
    )


def test_unit_sphere_scalar_curvature_and_convergence():
    e81, a81 = sphere_error(81)
    e161, a161 = sphere_error(161)
    assert e81 < 8e-3 and e161 < 2e-3
    assert e161 < 0.30*e81
    assert a161 < a81


def test_poincare_disk_negative_curvature():
    n = 161
    x = np.linspace(-.6, .6, n)
    X, Y = np.meshgrid(x, x)
    lam2 = 4/(1-X**2-Y**2)**2
    z = np.zeros_like(lam2)
    _, _, R, _ = geometry(
        lam2, z, lam2, x[1]-x[0], x[1]-x[0]
    )
    rmse = np.sqrt(np.mean((crop(R)+2.0)**2))
    assert rmse < 3e-3


def test_variable_conformal_curvature():
    n = 161
    a = .12
    x = np.linspace(-1.2, 1.2, n)
    X, Y = np.meshgrid(x, x)
    u = a*(X**2+Y**2)
    lam2 = np.exp(2*u)
    z = np.zeros_like(lam2)
    _, _, R, _ = geometry(
        lam2, z, lam2, x[1]-x[0], x[1]-x[0]
    )
    exact = -8*a*np.exp(-2*u)
    rmse = np.sqrt(np.mean((crop(R)-crop(exact))**2))
    assert rmse < 6e-5
