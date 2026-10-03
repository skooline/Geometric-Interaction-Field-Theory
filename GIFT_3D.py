"""3D embedded-manifold geodesic demo for GIFT.

This module visualizes a genuine 2D Riemannian manifold embedded in R^3:
    X(I1,I2) = (I1, I2, h(I1,I2)).
The metric is induced by the embedding, g_ij = d_i X . d_j X.
A pure geodesic is integrated intrinsically in (I1,I2) coordinates and then
mapped back to R^3, so the animated trajectory lies on the plotted surface.

Important: scalar curvature R is used as surface COLOR, not surface HEIGHT.
An arbitrary abstract GIFT metric cannot in general be represented globally
as a single graph z=h(I1,I2); this file is the geometrically consistent
embedded-surface mode.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, cm, colors

from GIFT import geometry, bilinear, covariant_acceleration


def interaction_height(X, Y, amplitude=2.2, spread=1.6, center=(0.0, 0.0)):
    """Reference embedding h(I1,I2): a smooth localized interaction deformation."""
    cx, cy = center
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    return amplitude * np.exp(-r2 / (2.0 * spread ** 2))


def induced_metric_from_height(H, dx, dy):
    """Metric induced by X=(I1,I2,H): g = J^T J."""
    Hy, Hx = np.gradient(H, dy, dx, edge_order=2)
    g11 = 1.0 + Hx * Hx
    g12 = Hx * Hy
    g22 = 1.0 + Hy * Hy
    return g11, g12, g22, Hx, Hy


def sample_height(H, pos, xgrid, ygrid):
    return bilinear(H, float(pos[0]), float(pos[1]), xgrid, ygrid)


def rk4_step(state, dt, rhs):
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def integrate_geodesic(Gamma, xgrid, ygrid, pos0, vel0, dt=0.005, steps=1800):
    """Integrate Dv/dt=0 with RK4 in intrinsic coordinates."""
    state = np.array([pos0[0], pos0[1], vel0[0], vel0[1]], dtype=float)
    states = [state.copy()]

    def rhs(s):
        pos = s[:2]
        vel = s[2:]
        acc = covariant_acceleration(
            pos, vel, Gamma, xgrid, ygrid, force=None, damping=0.0
        )
        return np.r_[vel, acc]

    for _ in range(steps):
        candidate = rk4_step(state, dt, rhs)
        if not (
            xgrid[1] < candidate[0] < xgrid[-2]
            and ygrid[1] < candidate[1] < ygrid[-2]
        ):
            break
        state = candidate
        states.append(state.copy())

    return np.asarray(states)


def geodesic_energy(states, g11, g12, g22, xgrid, ygrid):
    """Return g_ij v^i v^j along the numerical path."""
    E = np.empty(len(states))
    for n, s in enumerate(states):
        x, y, vx, vy = s
        a = bilinear(g11, x, y, xgrid, ygrid)
        b = bilinear(g12, x, y, xgrid, ygrid)
        c = bilinear(g22, x, y, xgrid, ygrid)
        E[n] = a*vx*vx + 2.0*b*vx*vy + c*vy*vy
    return E


def build_demo(
    grid_size=201,
    amplitude=2.2,
    spread=1.6,
    pos0=(-3.4, -1.0),
    vel0=(1.0, 0.25),
    dt=0.005,
    steps=1800,
):
    x = np.linspace(-5.0, 5.0, grid_size)
    y = np.linspace(-5.0, 5.0, grid_size)
    X, Y = np.meshgrid(x, y)
    dx, dy = x[1] - x[0], y[1] - y[0]

    H = interaction_height(X, Y, amplitude=amplitude, spread=spread)
    g11, g12, g22, _, _ = induced_metric_from_height(H, dx, dy)
    Gamma, Ric, R, det_g = geometry(g11, g12, g22, dx, dy)

    states = integrate_geodesic(Gamma, x, y, pos0, vel0, dt=dt, steps=steps)
    traj_x = states[:, 0]
    traj_y = states[:, 1]
    traj_z = np.array([sample_height(H, p, x, y) for p in states[:, :2]])

    E = geodesic_energy(states, g11, g12, g22, x, y)
    relative_energy_drift = (E.max() - E.min()) / E.mean()

    return {
        "x": x, "y": y, "X": X, "Y": Y, "H": H,
        "g11": g11, "g12": g12, "g22": g22,
        "Gamma": Gamma, "Ricci": Ric, "R": R, "det_g": det_g,
        "states": states,
        "traj_x": traj_x, "traj_y": traj_y, "traj_z": traj_z,
        "energy": E, "relative_energy_drift": relative_energy_drift,
    }


def animate_demo(data, frame_stride=8, interval_ms=30):
    X, Y, H, R = data["X"], data["Y"], data["H"], data["R"]
    tx, ty, tz = data["traj_x"], data["traj_y"], data["traj_z"]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    finite_R = R[np.isfinite(R)]
    vmax = max(abs(finite_R.min()), abs(finite_R.max()))
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    facecolors = cm.coolwarm(norm(R))

    stride = max(1, len(X) // 90)
    ax.plot_surface(
        X, Y, H,
        facecolors=facecolors,
        rstride=stride,
        cstride=stride,
        linewidth=0,
        antialiased=True,
        alpha=0.86,
        shade=False,
    )

    mappable = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)
    mappable.set_array(R)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.68, pad=0.08)
    cbar.set_label("Intrinsic scalar curvature R")

    path_line, = ax.plot([], [], [], linewidth=2.4, label="Pure geodesic")
    point, = ax.plot([], [], [], marker="o", markersize=7)

    ax.set_xlabel("Interaction coordinate I1")
    ax.set_ylabel("Interaction coordinate I2")
    ax.set_zlabel("Embedding height h(I1,I2)")
    ax.set_title(
        "GIFT embedded interaction manifold\n"
        f"geodesic energy drift = {data['relative_energy_drift']:.3e}"
    )
    ax.legend(loc="upper left")
    ax.view_init(elev=29, azim=-58)

    frame_ids = list(range(1, len(tx), frame_stride))
    if frame_ids[-1] != len(tx) - 1:
        frame_ids.append(len(tx) - 1)

    def update(frame_index):
        n = frame_ids[frame_index]
        path_line.set_data(tx[:n+1], ty[:n+1])
        path_line.set_3d_properties(tz[:n+1])
        point.set_data([tx[n]], [ty[n]])
        point.set_3d_properties([tz[n]])
        return path_line, point

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    return fig, ani


if __name__ == "__main__":
    data = build_demo()
    print(f"Trajectory points: {len(data['states'])}")
    print(f"min(det g): {data['det_g'].min():.6g}")
    print(f"relative geodesic-energy drift: {data['relative_energy_drift']:.6g}")
    fig, ani = animate_demo(data)
    plt.show()
