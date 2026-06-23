import numpy as np
import sympy as sp
from sympy import symbols, Matrix, diff, lambdify
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =====================================================================
# STAGE 1: คำนวณเทนเซอร์เชิงสัญลักษณ์เพียง "ครั้งเดียว" (Pre-computation)
# =====================================================================
print("กำลังคำนวณสัญลักษณ์เทนเซอร์ขั้นสูงด้วย SymPy (ขั้นตอนนี้ทำครั้งเดียว)...")

t, x, y = symbols('t x y')
coords = [t, x, y]
dim = len(coords)
a_val = 1.0

# นิยาม \gamma(t, I_S) ตามทฤษฎีในเปเปอร์
gamma_core = sp.exp(-(x**2 + y**2)/4)
gamma_mat = Matrix([
    [gamma_core, 0],
    [0, gamma_core]
])
I_S = Matrix([x, y])

# คำนวณ Jacobian: J = \gamma + (\nabla_{I_S}\gamma) * I_S
dgamma_dx = diff(gamma_mat, x)
dgamma_dy = diff(gamma_mat, y)
grad_gamma_times_IS = Matrix([
    [(dgamma_dx * I_S)[0], (dgamma_dy * I_S)[0]],
    [(dgamma_dx * I_S)[1], (dgamma_dy * I_S)[1]]
])
J = gamma_mat + grad_gamma_times_IS

# สร้าง Spatial Metric: G = I + J^T * J
G_spatial = sp.eye(2) + J.T * J

# สร้าง Spacetime Metric รวม 3x3
g_cov = Matrix([
    [a_val**2, 0, 0],
    [0, G_spatial[0,0], G_spatial[0,1]],
    [0, G_spatial[1,0], G_spatial[1,1]]
])
g_inv = g_cov.inv()

# คำนวณ Christoffel Symbols
Christoffel = {}
for lam in range(dim):
    for mu in range(dim):
        for nu in range(dim):
            tmp = 0
            for sigma in range(dim):
                term1 = diff(g_cov[nu, sigma], coords[mu])
                term2 = diff(g_cov[mu, sigma], coords[nu])
                term3 = diff(g_cov[mu, nu], coords[sigma])
                tmp += g_inv[lam, sigma] * (term1 + term2 - term3)
            Christoffel[(lam, mu, nu)] = 0.5 * tmp

# คำนวณ Riemann -> Ricci Tensor
Riemann = {}
for rho in range(dim):
    for sigma in range(dim):
        for mu in range(dim):
            for nu in range(dim):
                term1 = diff(Christoffel[(rho, sigma, nu)], coords[mu])
                term2 = diff(Christoffel[(rho, sigma, mu)], coords[nu])
                term3 = sum(Christoffel[(rho, mu, lam)] * Christoffel[(lam, sigma, nu)] for lam in range(dim))
                term4 = sum(Christoffel[(rho, nu, lam)] * Christoffel[(lam, sigma, mu)] for lam in range(dim))
                Riemann[(rho, sigma, mu, nu)] = term1 - term2 + term3 - term4

R_ricci = Matrix.zeros(dim, dim)
for mu in range(dim):
    for nu in range(dim):
        R_ricci[mu, nu] = sum(Riemann[(lam, mu, lam, nu)] for lam in range(dim))

R_scalar = sum(g_inv[mu, nu] * R_ricci[mu, nu] for mu in range(dim) for nu in range(dim))

# สกัดโครงสร้างความโค้งของ Einstein Tensor
Einstein_Tensor = R_ricci - 0.5 * R_scalar * g_cov
R_space_expr = Einstein_Tensor[1, 1] + Einstein_Tensor[2, 2]

# หาความชัน Geodesic Force เชิงสัญลักษณ์ล่วงหน้า
force_x_expr = -diff(R_space_expr, x)
force_y_expr = -diff(R_space_expr, y)

# แปลงสัญลักษณ์ทั้งหมดให้กลายเป็นฟังก์ชัน NumPy แบบ Vectorized ความเร็วสูง
# การระบุโมดูล 'numpy' จะช่วยให้มันคำนวณอาเรย์ของกริดทั้งหมดได้ในคำสั่งเดียว
print("แปลงสมการเป็นฟังก์ชันความเร็วสูง...")
fast_landscape = lambdify((x, y), R_space_expr, 'numpy')
fast_force_x = lambdify((x, y), force_x_expr, 'numpy')
fast_force_y = lambdify((x, y), force_y_expr, 'numpy')

print("เสร็จสิ้นขั้นตอนสัญลักษณ์! ระบบพร้อมรันพลศาสตร์ความเร็วสูง (Real-time)")

# =====================================================================
# STAGE 2: จำลองพลศาสตร์ความเร็วสูงเชิงตัวเลข (NumPy Only inside Loop)
# =====================================================================
fig = plt.figure(figsize=(14, 6))
ax3d = fig.add_subplot(121, projection='3d')
ax2d = fig.add_subplot(122)

x_grid = np.linspace(-5, 5, 40)
y_grid = np.linspace(-5, 5, 40)
X, Y = np.meshgrid(x_grid, y_grid)

num_agents = 6
noise_strength = 0.12
kappa = 0.6
Lambda = 0.0

trajectories_x = [[np.random.uniform(-3, 3)] for _ in range(num_agents)]
trajectories_y = [[np.random.uniform(-3, 3)] for _ in range(num_agents)]
colors = plt.cm.plasma(np.linspace(0.1, 0.9, num_agents))

def update(frame):
    ax3d.clear()
    ax2d.clear()
    
    # 1. ดึงตำแหน่งล่าสุดของกลุ่มเอเจนต์ (ทำงานเร็วมากด้วย NumPy บรรทัดเดียว)
    current_x = np.array([tx[-1] for tx in trajectories_x])
    current_y = np.array([ty[-1] for ty in trajectories_y])
    
    # 2. คำนวณ Interaction Source Tensor (T_μν) 
    T_source = np.zeros_like(X)
    for i in range(num_agents):
        T_source += 0.7 * np.exp(-((X - current_x[i])**2 + (Y - current_y[i])**2) / 1.5)
        
    # คำนวณระนาบความโค้งผลเฉลยรวม
    Z = Lambda - (kappa * T_source)
    
    # ค้นหาจุดยุบตัวลึกที่สุด (ศูนย์กลางหลุมความโค้ง)
    min_idx = np.unravel_index(np.argmin(Z, axis=None), Z.shape)
    deepest_x, deepest_y = X[min_idx], Y[min_idx]
    max_distortion = abs(Z[min_idx] - Lambda)
    
    # 3. อัปเดตและพลอตข้อมูลเอเจนต์แต่ละตัวด้วยแรงเรขาคณิตสัมบูรณ์
    for i in range(num_agents):
        px = trajectories_x[i][-1]
        py = trajectories_y[i][-1]
        
        # ค้นหาเวกเตอร์แรงดึงดูดเชิงเรขาคณิต ณ พิกัดสัมพัทธ์ที่เอเจนต์ยืนอยู่
        fx_geodesic = fast_force_x(px - deepest_x, py - deepest_y)
        fy_geodesic = fast_force_y(px - deepest_x, py - deepest_y)
        
        # เคลื่อนที่ตามกฎ Geodesic Flow (แรงเรขาคณิตบีบอัด) + Noise สุ่มทางพฤติกรรม
        # (คุณสามารถปรับตัวคูณด้านหน้าแรง เช่น 0.5, 1.0 หรือ 1.5 เพื่อดูความเร็วการไหลได้ครับ)
        next_x = px + 1.2 * fx_geodesic + np.random.normal(0, noise_strength)
        next_y = py + 1.2 * fy_geodesic + np.random.normal(0, noise_strength)
        
        next_x = np.clip(next_x, -4.9, 4.9)
        next_y = np.clip(next_y, -4.9, 4.9)
        
        trajectories_x[i].append(next_x)
        trajectories_y[i].append(next_y)
        
        # จำกัดความยาวประวัติเส้นทาง (Memory Management) เพื่อไม่ให้หน่วงเครื่อง
        if len(trajectories_x[i]) > 25:
            trajectories_x[i].pop(0)
            trajectories_y[i].pop(0)
            
        hist_x = np.array(trajectories_x[i])
        hist_y = np.array(trajectories_y[i])
        
        # คำนวณประวัติระดับความลึกแกน Z ด้วย NumPy ความเร็วสูง
        hist_z = np.zeros_like(hist_x)
        for j in range(len(hist_x)):
            t_val = np.sum(0.7 * np.exp(-((hist_x[j] - current_x)**2 + (hist_y[j] - current_y)**2) / 1.5))
            hist_z[j] = Lambda - (kappa * t_val)
            
        # พลอตสายธารเส้นทาง
        ax3d.plot(hist_x, hist_y, hist_z, color=colors[i], lw=2)
        ax3d.scatter(hist_x[-1], hist_y[-1], hist_z[-1], color=colors[i], s=40, zorder=10)
        
        ax2d.plot(hist_x, hist_y, color=colors[i], lw=1.5)
        ax2d.scatter(hist_x[-1], hist_y[-1], color=colors[i], s=40, edgecolors='black', zorder=5)
        
    # วาดพื้นผิวและคอนทัวร์อย่างลื่นไหล
    ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.45, edgecolor='none')
    ax2d.contourf(X, Y, Z, cmap='viridis', alpha=0.35)
    
    status_text = f"Max Tensor Field Distortion: {max_distortion:.4f}\n"
    status_text += "Phase: Field Distortion & Trapping!" if max_distortion > 0.3 else "Phase: Flat Geometry"
    ax2d.text(-4.5, 4.2, status_text, color="black", fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.85))
    
    ax3d.set_title("3D Interaction Field Tensor Geometric Hole ($G_{\\mu\\nu}$)", fontsize=10)
    ax3d.set_zlim(-3.0, 0.2)
    ax3d.view_init(elev=30, azim=-60)
    
    ax2d.set_title("2D Agent Paths under Pre-computed $\\gamma$ Chain", fontsize=10)
    ax2d.set_xlim(-5, 5)
    ax2d.set_ylim(-5, 5)
    ax2d.grid(True, linestyle=':', alpha=0.5)

# สั่งรันแอนิเมชันลื่นๆ สไตล์เดียวกับโค้ดต้นแบบของคุณ
ani = FuncAnimation(fig, update, frames=200, interval=40, repeat=False)
plt.tight_layout()
plt.show()