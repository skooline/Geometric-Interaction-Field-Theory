"""GIFT 2D reference solver.

The source-to-metric map is coordinate covariant.  A symmetric covariant
source T is converted to a deformation tensor H using the background metric
g0, then the mixed tensor A=g0^{-1}H is exponentiated:

    g(u,v) = g0(exp(kappa A) u, v).

Equivalently, with B=g0^{-1/2} H g0^{-1/2},
    g = g0^{1/2} exp(kappa B) g0^{1/2},
which is manifestly symmetric positive definite.
"""
import numpy as np


def gaussian_source(X, Y, amp=1.0, spread=1.2, center=(0.0, 0.0)):
    """Local-chart source components used for demonstrations."""
    px, py = center
    r2 = (X - px) ** 2 + (Y - py) ** 2
    phi = amp * np.exp(-r2 / (2.0 * spread ** 2))
    return (
        phi,
        phi * (-(Y - py) / spread),
        phi * (1.0 - r2 / (2.0 * spread ** 2)),
    )


def _stack_symmetric(a00, a01, a11):
    a00, a01, a11 = np.broadcast_arrays(
        np.asarray(a00, dtype=float),
        np.asarray(a01, dtype=float),
        np.asarray(a11, dtype=float),
    )
    M = np.empty(a00.shape + (2, 2), dtype=float)
    M[..., 0, 0] = a00
    M[..., 0, 1] = a01
    M[..., 1, 0] = a01
    M[..., 1, 1] = a11
    return M


def _unstack_symmetric(M):
    return M[..., 0, 0], M[..., 0, 1], M[..., 1, 1]


def metric_from_source(
    T00,
    T01,
    T11,
    kappa=0.35,
    background=(1.0, 0.0, 1.0),
    isotropic_weight=1.0,
    deviatoric_weight=1.0,
):
    """Construct a coordinate-covariant positive-definite metric.

    Parameters
    ----------
    T00, T01, T11:
        Components of a symmetric covariant source tensor T in the active chart.
    kappa:
        Constitutive coupling.
    background:
        Components (g0_00, g0_01, g0_11) of a positive-definite background
        metric in the same chart.  The default is Euclidean in this chart.
    isotropic_weight, deviatoric_weight:
        Scalar weights for the coordinate-invariant trace/deviatoric
        decomposition of T relative to g0.

    Notes
    -----
    Let tr0(T)=g0^{ij}T_ij and n=2.  Define
        H = a * (tr0(T)/n) g0
            + b * [T - (tr0(T)/n) g0].
    Then A=g0^{-1}H is a (1,1) tensor.  The constitutive law is
        g(u,v)=g0(exp(kappa A)u,v).
    """
    T = _stack_symmetric(T00, T01, T11)
    G0 = _stack_symmetric(*background)

    shape = np.broadcast_shapes(T.shape[:-2], G0.shape[:-2])
    T = np.broadcast_to(T, shape + (2, 2))
    G0 = np.broadcast_to(G0, shape + (2, 2))

    eig0, Q0 = np.linalg.eigh(G0)
    if np.any(~np.isfinite(eig0)) or np.any(eig0 <= 0.0):
        raise ValueError("Background metric must be finite and positive definite")

    G0inv = np.linalg.inv(G0)
    trace0 = np.einsum("...ij,...ji->...", G0inv, T)

    isotropic = (trace0 / 2.0)[..., None, None] * G0
    deviatoric = T - isotropic
    H = isotropic_weight * isotropic + deviatoric_weight * deviatoric

    # Coordinate-invariant construction evaluated through the self-adjoint
    # representative B=g0^{-1/2} H g0^{-1/2}.
    G0sqrt = (
        Q0 * np.sqrt(eig0)[..., None, :]
    ) @ np.swapaxes(Q0, -1, -2)
    G0invsqrt = (
        Q0 * (1.0 / np.sqrt(eig0))[..., None, :]
    ) @ np.swapaxes(Q0, -1, -2)

    B = G0invsqrt @ H @ G0invsqrt
    eigB, QB = np.linalg.eigh(B)
    expB = (
        QB * np.exp(kappa * eigB)[..., None, :]
    ) @ np.swapaxes(QB, -1, -2)

    G = G0sqrt @ expB @ G0sqrt
    return _unstack_symmetric(G)


def inverse_metric(g00, g01, g11):
    det = g00 * g11 - g01 * g01
    if np.any(~np.isfinite(det)) or np.any(det <= 0):
        raise ValueError("Metric must be finite and positive definite")
    return g11 / det, -g01 / det, g00 / det, det


def geometry(g00, g01, g11, dx, dy):
    """Return Levi-Civita symbols, Ricci tensor, scalar curvature, determinant."""
    gi00, gi01, gi11, det = inverse_metric(g00, g01, g11)

    def deriv(f):
        fy, fx = np.gradient(f, dy, dx, edge_order=2)
        return fx, fy

    g00x, g00y = deriv(g00)
    g01x, g01y = deriv(g01)
    g11x, g11y = deriv(g11)

    G = np.zeros((2, 2, 2) + g00.shape)
    G[0, 0, 0] = .5 * (gi00 * g00x + gi01 * (2 * g01x - g00y))
    G[0, 0, 1] = G[0, 1, 0] = .5 * (gi00 * g00y + gi01 * g11x)
    G[0, 1, 1] = .5 * (gi00 * (2 * g01y - g11x) + gi01 * g11y)
    G[1, 0, 0] = .5 * (gi01 * g00x + gi11 * (2 * g01x - g00y))
    G[1, 0, 1] = G[1, 1, 0] = .5 * (gi01 * g00y + gi11 * g11x)
    G[1, 1, 1] = .5 * (gi01 * (2 * g01y - g11x) + gi11 * g11y)

    dG = np.zeros((2,) + G.shape)
    for k in range(2):
        for i in range(2):
            for j in range(2):
                gx, gy = deriv(G[k, i, j])
                dG[0, k, i, j] = gx
                dG[1, k, i, j] = gy

    Ric = np.zeros((2, 2) + g00.shape)
    for i in range(2):
        for j in range(2):
            r = np.zeros_like(g00, dtype=float)
            for k in range(2):
                r += dG[k, k, i, j] - dG[j, k, i, k]
                for l in range(2):
                    r += (
                        G[k, i, j] * G[l, k, l]
                        - G[l, i, k] * G[k, j, l]
                    )
            Ric[i, j] = r

    R = (
        gi00 * Ric[0, 0]
        + gi01 * (Ric[0, 1] + Ric[1, 0])
        + gi11 * Ric[1, 1]
    )
    return G, Ric, R, det


def bilinear(field, x, y, xgrid, ygrid):
    x = np.clip(x, xgrid[0], xgrid[-1])
    y = np.clip(y, ygrid[0], ygrid[-1])
    ix = max(0, min(np.searchsorted(xgrid, x) - 1, len(xgrid) - 2))
    iy = max(0, min(np.searchsorted(ygrid, y) - 1, len(ygrid) - 2))
    tx = (x - xgrid[ix]) / (xgrid[ix + 1] - xgrid[ix])
    ty = (y - ygrid[iy]) / (ygrid[iy + 1] - ygrid[iy])
    return (
        (1 - tx) * (1 - ty) * field[iy, ix]
        + tx * (1 - ty) * field[iy, ix + 1]
        + (1 - tx) * ty * field[iy + 1, ix]
        + tx * ty * field[iy + 1, ix + 1]
    )


def covariant_acceleration(pos, vel, G, xgrid, ygrid, force=None, damping=0.0):
    """Coordinate acceleration for geodesic or forced covariant motion."""
    Gam = np.empty((2, 2, 2))
    for k in range(2):
        for i in range(2):
            for j in range(2):
                Gam[k, i, j] = bilinear(
                    G[k, i, j], *pos, xgrid, ygrid
                )
    a = -np.einsum("kij,i,j->k", Gam, vel, vel)
    if force is not None:
        a = a + np.asarray(force, dtype=float)
    return a - damping * vel


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-5, 5, 101)
    y = np.linspace(-5, 5, 101)
    X, Y = np.meshgrid(x, y)
    g = metric_from_source(*gaussian_source(X, Y, amp=1.5))
    _, _, R, _ = geometry(*g, x[1] - x[0], y[1] - y[0])

    fig, ax = plt.subplots()
    im = ax.contourf(X, Y, R, 30)
    fig.colorbar(im, ax=ax, label="Scalar curvature R")
    ax.set(
        xlabel="Interaction coordinate I1",
        ylabel="Interaction coordinate I2",
        title="GIFT: scalar curvature of the interaction metric",
    )
    plt.show()
