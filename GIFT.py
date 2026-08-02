import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import matplotlib.animation as animation

# ==============================================================================
# Geometric Interaction Field Theory (GIFT) - Rigorous Field Engine
# ==============================================================================

# 1. Base Interaction Flow Generator (M Matrix)
M = np.array([
    [-0.1,  0.8],
    [-0.8, -0.1]
])

# Grid setup for Manifold calculations
grid_size = 50
x_range = np.linspace(-5, 5, grid_size)
y_range = np.linspace(-5, 5, grid_size)
X, Y = np.meshgrid(x_range, y_range)
dx = x_range[1] - x_range[0]
dy = y_range[1] - y_range[0]

# Initial State & Velocity of Agent
agent_pos = np.array([1.0, 1.0])
agent_vel = np.dot(M, agent_pos)

# Trajectory memory
path_x, path_y = [], []

# ==============================================================================
# 2. Metric & Source Tensor Engine
# ==============================================================================

def compute_source_tensor_components(X, Y, amp, spread, px, py, active_labels):
    r2 = (X - px)**2 + (Y - py)**2
    phi = amp * np.exp(-r2 / (2.0 * spread**2))
    
    T00 = np.zeros_like(X)
    T01 = np.zeros_like(X)
    T11 = np.zeros_like(X)
    T12 = np.zeros_like(X)
    
    for label in active_labels:
        if 'T_00' in label:
            T00 += phi
        if 'T_01' in label:
            T01 += phi * (-(Y - py) / spread)
        if 'T_11' in label:
            T11 += phi * (1.0 - r2 / (2.0 * spread**2))
        if 'T_12' in label:
            T12 += phi * ((X - px) * (Y - py) / (spread**2))
            
    return T00, T01, T11, T12

def compute_metric_and_christoffel(X, Y, T00, T01, T11, T12, amp):
    kappa = 0.5  # Coupling constant
    
    g11 = 1.0 + kappa * (T00 + T11)
    g12 = kappa * (T01 + T12)
    g21 = g12
    g22 = 1.0 + kappa * (T00 - T11)
    
    det_g = g11 * g22 - g12 * g21
    det_g = np.maximum(det_g, 1e-5)
    
    g11_inv =  g22 / det_g
    g12_inv = -g12 / det_g
    g21_inv = -g21 / det_g
    g22_inv =  g11 / det_g
    
    dg11_dx, dg11_dy = np.gradient(g11, dx, dy, axis=(1, 0))
    dg12_dx, dg12_dy = np.gradient(g12, dx, dy, axis=(1, 0))
    dg22_dx, dg22_dy = np.gradient(g22, dx, dy, axis=(1, 0))
    
    Gamma_0_00 = 0.5 * (g11_inv * dg11_dx + g12_inv * (2*dg12_dx - dg11_dy))
    Gamma_0_01 = 0.5 * (g11_inv * dg11_dy + g12_inv * dg22_dx)
    Gamma_0_11 = 0.5 * (g11_inv * (2*dg12_dy - dg22_dx) + g12_inv * dg22_dy)
    
    Gamma_1_00 = 0.5 * (g21_inv * dg11_dx + g22_inv * (2*dg12_dx - dg11_dy))
    Gamma_1_01 = 0.5 * (g21_inv * dg11_dy + g22_inv * dg22_dx)
    Gamma_1_11 = 0.5 * (g21_inv * (2*dg12_dy - dg22_dx) + g22_inv * dg22_dy)
    
    h11 = g11 - 1.0
    h12 = g12
    h22 = g22 - 1.0
    sign = np.sign(amp) if amp != 0 else 1.0
    Z = np.sqrt(h11**2 + 2.0 * h12**2 + h22**2) * sign
    
    return Z, (g11, g12, g22), (g11_inv, g12_inv, g22_inv), \
           (Gamma_0_00, Gamma_0_01, Gamma_0_11), (Gamma_1_00, Gamma_1_01, Gamma_1_11)

def sample_grid_field(field_grid, pos):
    """Bilinear interpolation with NaN and Bound safety guards."""
    # Safety Check: ป้องกัน NaN / Inf หลุดเข้ามา
    if np.any(np.isnan(pos)) or np.any(np.isinf(pos)):
        pos = np.array([0.0, 0.0])
        
    px = np.clip(pos[0], x_range[0], x_range[-1])
    py = np.clip(pos[1], y_range[0], y_range[-1])
    
    ix = int((px - x_range[0]) / dx)
    iy = int((py - y_range[0]) / dy)
    
    ix = np.clip(ix, 0, grid_size - 2)
    iy = np.clip(iy, 0, grid_size - 2)
    
    rx = (px - x_range[ix]) / dx
    ry = (py - y_range[iy]) / dy
    
    f00 = field_grid[iy, ix]
    f10 = field_grid[iy, ix+1]
    f01 = field_grid[iy+1, ix]
    f11 = field_grid[iy+1, ix+1]
    
    return (1-rx)*(1-ry)*f00 + rx*(1-ry)*f10 + (1-rx)*ry*f01 + rx*ry*f11

# ==============================================================================
# 3. Setup UI Interface
# ==============================================================================

fig = plt.figure(figsize=(14, 7.5))
plt.subplots_adjust(left=0.05, bottom=0.38, right=0.95, top=0.92, wspace=0.2)

ax_2d = fig.add_subplot(1, 2, 1)
ax_2d.set_title("2D Interaction Manifold (Geodesic Streamlines)", fontsize=12, fontweight='bold')
ax_2d.set_xlabel("Agent 1 Expressive State (I_S1)")
ax_2d.set_ylabel("Agent 2 Expressive State (I_S2)")
ax_2d.set_xlim(-5, 5)
ax_2d.set_ylim(-5, 5)
ax_2d.grid(True, linestyle=':', alpha=0.6)

ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.set_title("3D Metric Curvature Surface Z(I_S)", fontsize=12, fontweight='bold')
ax_3d.set_xlabel("I_S1")
ax_3d.set_ylabel("I_S2")
ax_3d.set_zlabel("Curvature Z")

axcolor = '#f0f4f8'

ax_check = plt.axes([0.03, 0.05, 0.24, 0.26], facecolor=axcolor)
labels = [
    'T_00 (Emotional Scalar Energy)', 
    'T_01 / T_10 (Momentum Swirl Flow)', 
    'T_11 / T_22 (Isotropic Pressure)', 
    'T_12 / T_21 (Shear Interaction)'
]
visibility = [True, True, True, True]
check_tij = CheckButtons(ax_check, labels, visibility)

ax_amp = plt.axes([0.38, 0.22, 0.55, 0.03], facecolor=axcolor)
ax_spread = plt.axes([0.38, 0.17, 0.55, 0.03], facecolor=axcolor)
ax_pos_x = plt.axes([0.38, 0.12, 0.55, 0.03], facecolor=axcolor)
ax_pos_y = plt.axes([0.38, 0.07, 0.55, 0.03], facecolor=axcolor)

s_amp = Slider(ax_amp, 'Tensor Amplitude', -3.0, 3.0, valinit=1.5)
s_spread = Slider(ax_spread, 'Tensor Spread', 0.5, 3.0, valinit=1.2)
s_pos_x = Slider(ax_pos_x, 'Tensor Pos X', -4.0, 4.0, valinit=0.0)
s_pos_y = Slider(ax_pos_y, 'Tensor Pos Y', -4.0, 4.0, valinit=0.0)

surf_plot = None
contour_plot = None

trail_line, = ax_2d.plot([], [], 'w--', linewidth=1.5, alpha=0.8, label="Geodesic Path")
agent_marker_2d, = ax_2d.plot([], [], 'ro', markersize=9, markeredgecolor='white', markeredgewidth=1.5)
agent_marker_3d, = ax_3d.plot([], [], [], 'ro', markersize=9, markeredgecolor='white', markeredgewidth=1.5)

# ==============================================================================
# 4. Rendering & Dynamic Geodesic Physics Engine
# ==============================================================================

def update(frame):
    global agent_pos, agent_vel, surf_plot, contour_plot
    
    amp = s_amp.val
    spread = s_spread.val
    px = s_pos_x.val
    py = s_pos_y.val
    
    # 1. อ่านค่า CheckBox และอัปเดต Active Tensors ทันที
    status = check_tij.get_status()
    active_labels = [labels[i] for i, checked in enumerate(status) if checked]
    
    # 2. คำนวณ Field ใหม่ทั้งหมดของทั้ง Grid
    T00, T01, T11, T12 = compute_source_tensor_components(X, Y, amp, spread, px, py, active_labels)
    Z, g_comp, g_inv_comp, Gamma0, Gamma1 = compute_metric_and_christoffel(X, Y, T00, T01, T11, T12, amp)
    
    # Render 2D / 3D Graph
    if contour_plot is not None:
        contour_plot.remove()
    contour_plot = ax_2d.contourf(X, Y, Z, levels=20, cmap='plasma', alpha=0.6)
    
    if surf_plot is not None:
        surf_plot.remove()
    surf_plot = ax_3d.plot_surface(X, Y, Z, cmap='plasma', alpha=0.85, edgecolor='none')
    
    # --------------------------------------------------------------------------
    # 3. Dynamic Physics Calculation (ให้ Trajectory สนใจแรงจาก T_mu_nu ที่เปิดอยู่)
    # --------------------------------------------------------------------------
    dt = 0.03
    
    # สุ่มดึงค่า Christoffel Symbols ณ จุดที่ Agent อยู่ ณ เฟรมปัจจุบัน
    G0_00 = sample_grid_field(Gamma0[0], agent_pos)
    G0_01 = sample_grid_field(Gamma0[1], agent_pos)
    G0_11 = sample_grid_field(Gamma0[2], agent_pos)
    
    G1_00 = sample_grid_field(Gamma1[0], agent_pos)
    G1_01 = sample_grid_field(Gamma1[1], agent_pos)
    G1_11 = sample_grid_field(Gamma1[2], agent_pos)
    
    v0, v1 = agent_vel[0], agent_vel[1]
    
    # Geodesic Force (- Gamma * v * v)
    acc_geo_x = - (G0_00 * v0**2 + 2.0 * G0_01 * v0 * v1 + G0_11 * v1**2)
    acc_geo_y = - (G1_00 * v0**2 + 2.0 * G1_01 * v0 * v1 + G1_11 * v1**2)
    acc_geo = np.clip(np.array([acc_geo_x, acc_geo_y]), -15.0, 15.0)
    
    # Gradient Potential Force (- grad Z) คำนวณตรงจากสนาม Z ปัจจุบัน
    delta = 0.05
    z_center = sample_grid_field(Z, agent_pos)
    z_dx = sample_grid_field(Z, agent_pos + np.array([delta, 0]))
    z_dy = sample_grid_field(Z, agent_pos + np.array([0, delta]))
    grad_z = np.array([(z_dx - z_center) / delta, (z_dy - z_center) / delta])
    
    # ปรับสเกลแรงดึงดูด/ผลัก ตาม Amplitude
    # หาก Amp > 0 -> ดึงลงบ่อ (Attractive)
    # หาก Amp < 0 -> ผลักออกจากยอดเขา (Repulsive)
    potential_force = -1.5 * grad_z 
    
    # Damping ซับโมเมนตัมส่วนเกินเพื่อให้เห็นทิศทางการไหลเข้าหา Curvature ชัดเจน
    damping = -0.4 * agent_vel
    
    # รวมแรงทั้งหมด
    acc_total = acc_geo + potential_force + damping
    #acc_total = acc_geo
    
    # Update Velocity & Position
    agent_vel = agent_vel + acc_total * dt
    
    # Limit max speed กันหลุดขอบ
    speed = np.linalg.norm(agent_vel)
    if speed > 4.0:
        agent_vel = (agent_vel / speed) * 4.0
        
    agent_pos = agent_pos + agent_vel * dt
    
    # Reflection boundary
    for i in range(2):
        if abs(agent_pos[i]) > 4.8:
            agent_pos[i] = np.sign(agent_pos[i]) * 4.8
            agent_vel[i] *= -0.5
            
    path_x.append(agent_pos[0])
    path_y.append(agent_pos[1])
    if len(path_x) > 120:
        path_x.pop(0)
        path_y.pop(0)
        
    trail_line.set_data(path_x, path_y)
    agent_marker_2d.set_data([agent_pos[0]], [agent_pos[1]])
    
    agent_z = sample_grid_field(Z, agent_pos)
    agent_marker_3d.set_data([agent_pos[0]], [agent_pos[1]])
    agent_marker_3d.set_3d_properties([agent_z])
    
    return trail_line, agent_marker_2d, agent_marker_3d

def on_ui_change(val):
    fig.canvas.draw_idle()

s_amp.on_changed(on_ui_change)
s_spread.on_changed(on_ui_change)
s_pos_x.on_changed(on_ui_change)
s_pos_y.on_changed(on_ui_change)
check_tij.on_clicked(on_ui_change)

ani = animation.FuncAnimation(fig, update, frames=200, interval=40, blit=False)

if __name__ == '__main__':
    plt.show()