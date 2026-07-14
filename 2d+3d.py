import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter

# =====================================================================
# 1. Simulation parameters
# =====================================================================
N = 48
x_vals = np.linspace(-3.0, 3.0, N)
y_vals = np.linspace(-3.0, 3.0, N)
X, Y = np.meshgrid(x_vals, y_vals)
dx = x_vals[1] - x_vals[0]
dy = y_vals[1] - y_vals[0]
dA = dx * dy

# Interaction field equation constants:
#   R_mu_nu - 1/2 R g_mu_nu + Lambda g_mu_nu = kappa T_mu_nu
kappa = 0.55
Lambda = 0.02
dt = 0.012

# Reinforcement/Fokker-Planck parameters from the source-tensor paper:
#   d_t P = -partial_i(R^i P) + 1/2 partial_i partial_j(D^ij P) + Sigma
positive_reinforcement_gain = 0.45
negative_reinforcement_gain = 0.30
acquisition_rate = 0.018
extinction_rate = 0.010
max_total_response_rate = 2.25
max_local_density = 1.80

schedule_type = "variable_interval"  # "fixed_ratio", "fixed_interval", "variable_ratio", "variable_interval"
schedule_diffusion = {
    "fixed_ratio": 0.035,
    "fixed_interval": 0.045,
    "variable_ratio": 0.085,
    "variable_interval": 0.075,
}[schedule_type]

initial_response_rate = 1.0
display_surface = "geometry"  # "geometry" shows R_space; "residual" shows R_space - source

# Reinforced response density P(t, I_S).  It is a frequency density, not
# merely a probability density, so its integral M0 is allowed to evolve.
initial_P = np.exp(-((X - 1.25) ** 2 + (Y - 1.15) ** 2) / 1.5)
initial_P *= initial_response_rate / (np.sum(initial_P) * dA)
P = initial_P.copy()

traj_x, traj_y, traj_z = [], [], []
moment_history = []


def reset_simulation_state():
    """Return global simulation variables to the initial condition."""
    global P, traj_x, traj_y, traj_z, moment_history
    P = initial_P.copy()
    traj_x = []
    traj_y = []
    traj_z = []
    moment_history = []


# =====================================================================
# 2. Numerical differential geometry helpers
# =====================================================================
def gradient_xy(field):
    """Return partial_x and partial_y for an array indexed as field[y, x]."""
    df_dy, df_dx = np.gradient(field, dy, dx, edge_order=2)
    return df_dx, df_dy


def divergence_xy(vx, vy):
    dvx_dx, _ = gradient_xy(vx)
    _, dvy_dy = gradient_xy(vy)
    return dvx_dx + dvy_dy


def interaction_gamma(t_val):
    """State-dependent gamma(t, I_S) in the constraint I_P = gamma I_S."""
    r2 = X**2 + Y**2
    decay = np.exp(-r2 / 6.0)

    gamma = np.empty((2, 2, N, N))
    gamma[0, 0] = 0.30 * decay * np.cos(t_val) + 0.10 * np.tanh(Y)
    gamma[0, 1] = 0.18 * decay * np.sin(t_val) + 0.08 * np.tanh(X * Y / 3.0)
    gamma[1, 0] = -0.16 * decay * np.sin(t_val) + 0.08 * np.tanh(X - Y)
    gamma[1, 1] = 0.28 * decay * np.cos(t_val + 0.7) + 0.10 * np.tanh(X)
    return gamma


def induced_metric(t_val):
    """Compute G_ij = delta_ij + J^T J, J_ai = gamma_ai + (grad gamma_ai).I_S."""
    gamma = interaction_gamma(t_val)
    J = np.empty_like(gamma)

    for a in range(2):
        for i in range(2):
            dgamma_dx, dgamma_dy = gradient_xy(gamma[a, i])
            J[a, i] = gamma[a, i] + X * dgamma_dx + Y * dgamma_dy

    g_xx = 1.0 + J[0, 0] ** 2 + J[1, 0] ** 2
    g_xy = J[0, 0] * J[0, 1] + J[1, 0] * J[1, 1]
    g_yy = 1.0 + J[0, 1] ** 2 + J[1, 1] ** 2
    return g_xx, g_xy, g_yy


def spatial_ricci_scalar(g_xx, g_xy, g_yy):
    """Finite-difference Ricci scalar of the 2D spatial metric G_ij."""
    det_g = np.clip(g_xx * g_yy - g_xy**2, 1e-8, None)
    inv = np.empty((2, 2, N, N))
    inv[0, 0] = g_yy / det_g
    inv[0, 1] = -g_xy / det_g
    inv[1, 0] = -g_xy / det_g
    inv[1, 1] = g_xx / det_g

    g = np.empty((2, 2, N, N))
    g[0, 0], g[0, 1] = g_xx, g_xy
    g[1, 0], g[1, 1] = g_xy, g_yy

    dg = np.empty((2, 2, 2, N, N))
    for i in range(2):
        for j in range(2):
            dg[0, i, j], dg[1, i, j] = gradient_xy(g[i, j])

    christoffel = np.zeros((2, 2, 2, N, N))
    for upper in range(2):
        for lower_a in range(2):
            for lower_b in range(2):
                value = 0.0
                for ell in range(2):
                    value += inv[upper, ell] * (
                        dg[lower_a, lower_b, ell]
                        + dg[lower_b, lower_a, ell]
                        - dg[ell, lower_a, lower_b]
                    )
                christoffel[upper, lower_a, lower_b] = 0.5 * value

    d_christoffel = np.empty((2, 2, 2, 2, N, N))
    for upper in range(2):
        for lower_a in range(2):
            for lower_b in range(2):
                d_christoffel[0, upper, lower_a, lower_b], d_christoffel[1, upper, lower_a, lower_b] = (
                    gradient_xy(christoffel[upper, lower_a, lower_b])
                )

    ricci = np.zeros((2, 2, N, N))
    for i in range(2):
        for j in range(2):
            term = 0.0
            for k in range(2):
                term += d_christoffel[k, k, i, j]
                term -= d_christoffel[j, k, i, k]
                for ell in range(2):
                    term += christoffel[k, i, j] * christoffel[ell, k, ell]
                    term -= christoffel[ell, i, k] * christoffel[k, j, ell]
            ricci[i, j] = term

    scalar = inv[0, 0] * ricci[0, 0] + 2.0 * inv[0, 1] * ricci[0, 1] + inv[1, 1] * ricci[1, 1]
    return gaussian_filter(scalar, sigma=0.65)


def bounded_surface(field, limit=2.5):
    """Compress extreme visualization spikes without changing the sign/order near zero."""
    return limit * np.tanh(field / limit)


def compute_manifold_geometry(t_val, P_current):
    """Return the selected display surface and supporting geometric/source fields."""
    g_xx, g_xy, g_yy = induced_metric(t_val)
    R_space = spatial_ricci_scalar(g_xx, g_xy, g_yy)

    # Local T00 integrand.  The full moment T00 is integral(P dA), but the
    # numerical field plot needs the pointwise source density that integrates to it.
    T00_density = P_current
    raw_residual = R_space - (kappa * T00_density + Lambda)
    field_residual = bounded_surface(raw_residual)
    geometry_surface = bounded_surface(R_space, limit=0.75)
    display_Z = geometry_surface if display_surface == "geometry" else field_residual

    geometry = {
        "g_xx": g_xx,
        "g_xy": g_xy,
        "g_yy": g_yy,
        "R_space": R_space,
        "T00_density": T00_density,
        "raw_residual": raw_residual,
        "field_residual": field_residual,
        "geometry_surface": geometry_surface,
    }
    return display_Z, geometry


# =====================================================================
# 3. Reinforcement source tensor and Fokker-Planck dynamics
# =====================================================================
def reinforcement_fields(t_val, P_current=None):
    """Positive and negative reinforcement drift, diffusion tensor, and Sigma."""
    appetitive_target = np.array([0.0, 0.0])
    aversive_state = np.array([-1.75, -1.50])

    R_plus_x = positive_reinforcement_gain * (appetitive_target[0] - X)
    R_plus_y = positive_reinforcement_gain * (appetitive_target[1] - Y)

    aversive_r2 = (X - aversive_state[0]) ** 2 + (Y - aversive_state[1]) ** 2
    aversive_weight = np.exp(-aversive_r2 / 1.2)
    R_minus_x = negative_reinforcement_gain * (X - aversive_state[0]) * aversive_weight
    R_minus_y = negative_reinforcement_gain * (Y - aversive_state[1]) * aversive_weight

    R_x = R_plus_x + R_minus_x
    R_y = R_plus_y + R_minus_y

    # Symmetric positive semi-definite diffusion tensor D^ij.
    D_xx = schedule_diffusion * (1.0 + 0.20 * np.tanh(Y) ** 2)
    D_yy = schedule_diffusion * (1.0 + 0.20 * np.tanh(X) ** 2)
    D_xy = 0.04 * schedule_diffusion * np.tanh(X * Y / 4.0)

    target_r2 = (X - appetitive_target[0]) ** 2 + (Y - appetitive_target[1]) ** 2
    density = P if P_current is None else P_current
    acquisition_capacity = np.clip(1.0 - density / max_local_density, 0.0, 1.0)
    acquisition = acquisition_rate * np.exp(-target_r2 / 0.9) * acquisition_capacity
    extinction = extinction_rate * (1.0 + 0.35 * np.tanh(np.sqrt(X**2 + Y**2))) * density
    Sigma = acquisition - extinction

    return R_x, R_y, D_xx, D_xy, D_yy, Sigma, R_plus_x, R_plus_y, R_minus_x, R_minus_y


def source_tensor_moments(P_current, R_x, R_y, Sigma):
    """Moment-derived T_mu_nu from Definition 6.1 of the source-tensor paper."""
    T00 = np.sum(P_current) * dA
    Mx = np.sum(X * P_current) * dA
    My = np.sum(Y * P_current) * dA

    T01 = (np.sum(R_x * P_current) + np.sum(X * Sigma)) * dA
    T02 = (np.sum(R_y * P_current) + np.sum(Y * Sigma)) * dA

    T11 = np.sum(X * X * P_current) * dA
    T12 = np.sum(X * Y * P_current) * dA
    T22 = np.sum(Y * Y * P_current) * dA

    tensor = np.array(
        [
            [T00, T01, T02],
            [T01, T11, T12],
            [T02, T12, T22],
        ]
    )
    mean_x = Mx / T00 if T00 > 1e-12 else 0.0
    mean_y = My / T00 if T00 > 1e-12 else 0.0
    return tensor, mean_x, mean_y


def finite_volume_advection(P_current, R_x, R_y):
    """Stable no-flux upwind discretisation of -div(R P)."""
    flux_x = np.zeros((N, N + 1))
    velocity_x = 0.5 * (R_x[:, :-1] + R_x[:, 1:])
    flux_x[:, 1:-1] = np.where(
        velocity_x >= 0.0,
        velocity_x * P_current[:, :-1],
        velocity_x * P_current[:, 1:],
    )

    flux_y = np.zeros((N + 1, N))
    velocity_y = 0.5 * (R_y[:-1, :] + R_y[1:, :])
    flux_y[1:-1, :] = np.where(
        velocity_y >= 0.0,
        velocity_y * P_current[:-1, :],
        velocity_y * P_current[1:, :],
    )

    return -((flux_x[:, 1:] - flux_x[:, :-1]) / dx + (flux_y[1:, :] - flux_y[:-1, :]) / dy)


def enforce_response_bounds(P_candidate, previous_mass):
    """Keep the frequency density non-negative and physiologically bounded."""
    P_candidate = np.nan_to_num(P_candidate, nan=0.0, posinf=max_local_density, neginf=0.0)
    P_candidate = np.clip(P_candidate, 0.0, max_local_density)

    current_mass = np.sum(P_candidate) * dA
    if current_mass > max_total_response_rate:
        P_candidate *= max_total_response_rate / current_mass
    elif current_mass < 1e-10 and previous_mass > 1e-10:
        P_candidate *= previous_mass / max(current_mass, 1e-10)

    return gaussian_filter(P_candidate, sigma=0.25)


def update_fokker_planck(P_current, t_val):
    R_x, R_y, D_xx, D_xy, D_yy, Sigma, *_ = reinforcement_fields(t_val, P_current)

    drift_term = finite_volume_advection(P_current, R_x, R_y)

    DxxP = D_xx * P_current
    DxyP = D_xy * P_current
    DyyP = D_yy * P_current

    d_DxxP_dx, _ = gradient_xy(DxxP)
    d2_DxxP_dx2, _ = gradient_xy(d_DxxP_dx)

    d_DxyP_dx, _ = gradient_xy(DxyP)
    _, d2_DxyP_dxdy = gradient_xy(d_DxyP_dx)

    _, d_DyyP_dy = gradient_xy(DyyP)
    _, d2_DyyP_dy2 = gradient_xy(d_DyyP_dy)

    diffusion_term = 0.5 * (d2_DxxP_dx2 + 2.0 * d2_DxyP_dxdy + d2_DyyP_dy2)

    previous_mass = np.sum(P_current) * dA
    P_new = P_current + dt * (drift_term + diffusion_term + Sigma)
    return enforce_response_bounds(P_new, previous_mass)


# =====================================================================
# 4. Dual 2D/3D visualisation
# =====================================================================
fig = plt.figure(figsize=(16, 7))
ax_2d = fig.add_subplot(121)
ax_3d = fig.add_subplot(122, projection="3d")


def animate(frame):
    global P, traj_x, traj_y, traj_z, moment_history

    if frame == 0 and moment_history:
        reset_simulation_state()

    ax_2d.clear()
    ax_3d.clear()

    t_current = frame * dt
    Z, geometry = compute_manifold_geometry(t_current, P)

    R_x, R_y, _, _, _, Sigma, *_ = reinforcement_fields(t_current, P)
    T_tensor, current_agent_x, current_agent_y = source_tensor_moments(P, R_x, R_y, Sigma)
    moment_history.append(T_tensor)

    idx_x = np.abs(x_vals - current_agent_x).argmin()
    idx_y = np.abs(y_vals - current_agent_y).argmin()
    current_agent_z = Z[idx_y, idx_x]

    traj_x.append(current_agent_x)
    traj_y.append(current_agent_y)
    traj_z.append(current_agent_z)

    if len(traj_x) > 80:
        traj_x.pop(0)
        traj_y.pop(0)
        traj_z.pop(0)

    P = update_fokker_planck(P, t_current)

    contour_bg = ax_2d.contourf(X, Y, geometry["field_residual"], levels=24, cmap="viridis", alpha=0.88)
    ax_2d.contour(X, Y, P, levels=6, colors="white", alpha=0.45, linewidths=1.0)
    ax_2d.quiver(X[::4, ::4], Y[::4, ::4], R_x[::4, ::4], R_y[::4, ::4], color="black", alpha=0.35)

    if len(traj_x) > 1:
        ax_2d.plot(traj_x, traj_y, color="cyan", linewidth=2.4, label="Moment trajectory")
    ax_2d.scatter(
        [current_agent_x],
        [current_agent_y],
        color="red",
        s=80,
        edgecolor="black",
        zorder=5,
        label="E[I_S]",
    )

    T00 = T_tensor[0, 0]
    T01 = T_tensor[0, 1]
    T02 = T_tensor[0, 2]
    T11 = T_tensor[1, 1]
    T12 = T_tensor[1, 2]
    T22 = T_tensor[2, 2]

    ax_2d.set_title(
        "2D Field Residual and Reinforced Response Density\n"
        f"T00={T00:.3f}, T0i=({T01:.3f}, {T02:.3f}), "
        f"Tij=({T11:.3f}, {T12:.3f}; {T12:.3f}, {T22:.3f})"
    )
    ax_2d.set_xlabel("Agent 1 State ($I_{S1}$)")
    ax_2d.set_ylabel("Agent 2 State ($I_{S2}$)")
    ax_2d.set_xlim(-3.0, 3.0)
    ax_2d.set_ylim(-3.0, 3.0)
    ax_2d.legend(loc="upper right")
    ax_2d.grid(True, linestyle="--", alpha=0.35)

    surf = ax_3d.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.78, zorder=1)
    if len(traj_x) > 1:
        ax_3d.plot(traj_x, traj_y, traj_z, color="cyan", linewidth=3.2, zorder=5, label="3D moment trajectory")
    ax_3d.scatter(
        [current_agent_x],
        [current_agent_y],
        [current_agent_z],
        color="red",
        s=95,
        depthshade=False,
        zorder=6,
        label="E[I_S]",
    )

    z_low, z_high = np.percentile(Z, [2, 98])
    z_pad = max(0.03, 0.15 * (z_high - z_low))
    ax_3d.set_zlim(z_low - z_pad, z_high + z_pad)
    raw_min = np.percentile(geometry["raw_residual"], 2)
    raw_max = np.percentile(geometry["raw_residual"], 98)
    surface_label = r"$R^{space}$" if display_surface == "geometry" else r"$R^{space} - (\kappa T_{00}^{density} + \Lambda)$"
    ax_3d.set_title(
        "3D Interaction Geometry\n"
        f"{surface_label}, t={t_current:.2f}, residual p2/p98=({raw_min:.2f}, {raw_max:.2f})"
    )
    ax_3d.set_xlabel("$I_{S1}$")
    ax_3d.set_ylabel("$I_{S2}$")
    ax_3d.set_zlabel("Spatial curvature" if display_surface == "geometry" else "Field residual")
    ax_3d.legend(loc="upper right")

    return surf, contour_bg


ani = FuncAnimation(fig, animate, frames=250, interval=40, blit=False)

if __name__ == "__main__":
    plt.tight_layout()
    plt.show()
