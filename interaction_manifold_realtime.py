import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.animation as animation

# ==========================================
# 1. Mathematical Framework & Parameters
# ==========================================
# Interaction Flow Matrix (M)
# กำหนดให้ระบบมี decay เล็กน้อย และมีการแลกเปลี่ยนปฏิสัมพันธ์ระหว่าง Agent 1 และ 2
M = np.array([
    [-0.1,  0.8],
    [-0.8, -0.1]
])

# จุดวัตถุเริ่มต้น (Agent 1, Agent 2)
agent_state = np.array([2.0, 2.0])

# ขอบเขตของ Interaction Space (I_S1, I_S2)
grid_size = 40
x = np.linspace(-5, 5, grid_size)
y = np.linspace(-5, 5, grid_size)
X, Y = np.meshgrid(x, y)

# ==========================================
# 2. Setup Figure & Interface
# ==========================================
fig = plt.figure(figsize=(12, 6))
plt.subplots_adjust(left=0.05, bottom=0.35, right=0.95, top=0.95, wspace=0.2)

# Subplot 2D (I_S1, I_S2)
ax_2d = fig.add_subplot(1, 2, 1)
ax_2d.set_title('2D Interaction Manifold')
ax_2d.set_xlabel('Agent 1 (I_S1)')
ax_2d.set_ylabel('Agent 2 (I_S2)')
ax_2d.set_xlim(-5, 5)
ax_2d.set_ylim(-5, 5)
ax_2d.grid(True, linestyle='--', alpha=0.5)

# Subplot 3D (I_S1, I_S2, Curvature)
ax_3d = fig.add_subplot(1, 2, 2, projection='3d')
ax_3d.set_title('3D Interaction Manifold (z = Curvature)')
ax_3d.set_xlabel('Agent 1 (I_S1)')
ax_3d.set_ylabel('Agent 2 (I_S2)')
ax_3d.set_zlabel('Curvature (T_00)')
ax_3d.set_zlim(0, 3)

# ==========================================
# 3. Sliders for Reinforcement Source Tensor
# ==========================================
# Reinforcement Tensor กำหนดค่าแอมพลิจูดและการกระจายตัว
axcolor = 'lightgoldenrodyellow'
ax_amp = plt.axes([0.15, 0.20, 0.7, 0.03], facecolor=axcolor)
ax_spread = plt.axes([0.15, 0.15, 0.7, 0.03], facecolor=axcolor)
ax_pos_x = plt.axes([0.15, 0.10, 0.7, 0.03], facecolor=axcolor)
ax_pos_y = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor=axcolor)

s_amp = Slider(ax_amp, 'Tensor Amplitude', -3.0, 3.0, valinit=1.5)
s_spread = Slider(ax_spread, 'Tensor Spread', 0.5, 3.0, valinit=1.5)
s_pos_x = Slider(ax_pos_x, 'Tensor Pos X', -4.0, 4.0, valinit=0.0)
s_pos_y = Slider(ax_pos_y, 'Tensor Pos Y', -4.0, 4.0, valinit=0.0)

# ==========================================
# 4. Rendering & Physics Engine
# ==========================================
# กราฟิกวัตถุ 
contour_plot = None
surf_plot = None
agent_dot_2d, = ax_2d.plot([], [], 'ro', markersize=8, label='Agent State')
agent_path_2d, = ax_2d.plot([], [], 'r--', alpha=0.5)
agent_dot_3d, = ax_3d.plot([], [], [], 'ro', markersize=8)

path_x, path_y = [], []

def calculate_tensor_field(X, Y, amp, spread, px, py):
    """คำนวณ Reinforcement Source Tensor (Gaussian Ansatz)[cite: 4]"""
    return amp * np.exp(-((X - px)**2 + (Y - py)**2) / (2 * spread**2))

def calculate_tensor_gradient(state, amp, spread, px, py):
    """คำนวณแรงดึงดูดที่เกิดจาก Gradient ของ Curvature"""
    dx = state[0] - px
    dy = state[1] - py
    grad_factor = -(amp / (spread**2)) * np.exp(-(dx**2 + dy**2) / (2 * spread**2))
    return np.array([grad_factor * dx, grad_factor * dy])

def update(frame):
    global agent_state, contour_plot, surf_plot
    
    # อ่านค่าจาก Slider ปัจจุบันแบบ Realtime
    amp = s_amp.val
    spread = s_spread.val
    px = s_pos_x.val
    py = s_pos_y.val
    
    # คำนวณ Curvature Z สำหรับแสดงผล 3D[cite: 3]
    Z = calculate_tensor_field(X, Y, amp, spread, px, py)
    
    # อัปเดต Surface 3D
    if surf_plot is not None:
        surf_plot.remove()
    surf_plot = ax_3d.plot_surface(X, Y, Z, cmap='plasma', alpha=0.8, edgecolor='none')
    
    # อัปเดต Contour 2D
    ax_2d.clear()
    ax_2d.set_title('2D Interaction Manifold')
    ax_2d.set_xlim(-5, 5)
    ax_2d.set_ylim(-5, 5)
    ax_2d.contourf(X, Y, Z, levels=20, cmap='plasma', alpha=0.5)
    
    # ----------------------------------------------------
    # คำนวณการเคลื่อนที่ (Kinetic Formulation)[cite: 4]
    # dI/dt = M * I_S - Gradient(T_mu_nu)
    # ----------------------------------------------------
    dt = 0.05
    # 1. Base continuous flow จาก M matrix
    flow_velocity = np.dot(M, agent_state)
    # 2. Reinforcement Drift ดึงดูด/ผลัก จาก Manifold Curvature[cite: 4]
    reinforcement_drift = calculate_tensor_gradient(agent_state, amp, spread, px, py)
    
    # อัปเดตตำแหน่งจุดวัตถุ
    d_state = flow_velocity - reinforcement_drift # เครื่องหมายลบแสดงการกลิ้งลงสู่หลุมหรือการดึงดูด
    agent_state = agent_state + d_state * dt
    
    # บันทึกเส้นทาง
    path_x.append(agent_state[0])
    path_y.append(agent_state[1])
    if len(path_x) > 100: # จำกัดความยาวเส้นทาง
        path_x.pop(0)
        path_y.pop(0)
        
    # วาดจุดและเส้นทางใน 2D
    ax_2d.plot(path_x, path_y, 'w--', alpha=0.6)
    ax_2d.plot(agent_state[0], agent_state[1], 'wo', markersize=8, markeredgecolor='black')
    
    # วาดจุดใน 3D (หาค่า Z ของจุดวัตถุปัจจุบัน)
    agent_z = calculate_tensor_field(agent_state[0], agent_state[1], amp, spread, px, py)
    agent_dot_3d.set_data(np.array([agent_state[0]]), np.array([agent_state[1]]))
    agent_dot_3d.set_3d_properties(np.array([agent_z]))
    
    return agent_dot_2d, agent_dot_3d

# เริ่มการ Animation
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=False)

plt.show()