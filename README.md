# Geometric Interaction Field Theory (GIFT) 

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An implementation of the unified geometric framework for multi-agent interaction dynamics, tracking the mathematical evolution from **Time-Dependent** to **State-Dependent Manifolds** using symbolic differential geometry and real-time physical simulation.

Author: **Chewin Pinmook** (chewinp@gmail.com)  
Date: May 2026

---

## 📌 Overview

This repository contains the core framework and numerical simulation for the research paper: **"Riemannian Geometry and Geometric Field Equations of Two-Agent Interaction Dynamics: From Time-Dependent to State-Dependent Manifolds"**.

Traditional multi-agent systems rely on phenomenological force laws or heuristic potential fields. This research reformulates behavioral interactions as an emergent property of space-time geometry itself. By treating interaction constraints as metric deformations on a Riemannian manifold $\mathcal{M}=\mathbb{R}_{t}\times\mathbb{R}_{I_{S}}^{2}$, we derive agent trajectories directly from **Geodesic Flow** dictated by the **Interaction Field Equation**:

$$R_{\mu\nu} - \frac{1}{2}R g_{\mu\nu} + \Lambda g_{\mu\nu} = \kappa T_{\mu\nu}$$

Where $T_{\mu\nu}$ represents the behavioral energy-momentum source tensor (perceptual forcing and reinforcement density), and $g_{\mu\nu}$ is the state-dependent spacetime metric tensor.

---

## 🧬 Theoretical Workflow & Geometry Chain

The mathematical architecture of the model computes the space-time curvature through a rigorous symbolic geometry pipeline using `SymPy`:

[Interaction Rule Matrix] \gamma(t, I_S)
│
▼
[Interaction Jacobian] J = \gamma + (\nabla_{I_S}\gamma)I_S
│
▼
[Spatial Metric]     G(t, I_S) = I + J^T * J
│
▼
[Spacetime Metric]    g_{\mu\nu} (3x3 Block-Diagonal)
│
▼
[Christoffel Symbols]    \Gamma^\lambda_{\mu\nu}
│
▼
[Einstein Tensor]      G_{\mu\nu} = R_{\mu\nu} - 1/2 * R * g_{\mu\nu}

---

## ⚡ Features

- **Pre-computed Symbolic Geometry:** Utilizes `SymPy` to derive exact analytical solutions for Christoffel symbols, Ricci scalars, and Geodesic force gradients outside the main loop.
- **Vectorized Performance:** Compiles symbolic expressions into ultra-fast `NumPy` executables via `lambdify`, boosting rendering speed up to 1000x for real-time visualization.
- **Dual-Pane Visualization:** Displays a simultaneous 3D Einstein Tensor Geometric Hole mesh and a 2D Contour Field trajectory map mapping state-dependent trapping.

---

## 🛠️ Installation & Requirements

Ensure you have Python 3.10 or higher installed. Clone the repository and install the dependencies:

```bash
git clone [https://github.com/yourusername/geometric-ift.git](https://github.com/yourusername/geometric-ift.git)
cd geometric-ift
pip install numpy sympy matplotlib

```

---

## 🚀 Usage

To execute the vectorized real-time simulation model:

```bash
python testGIFT_Precompile.py

```

### Simulation Parameters:

* `num_agents = 6` : Number of active interacting agents.
* `kappa = 0.6` : Coupling constant dictating how strongly agent density warps the geometry.
* `noise_strength = 0.12` : Stochastic behavioral variance (Stochastic Evolution).

---

## 📊 Expected Output

The script initiates a dual-axis Matplotlib animation:

1. **Left Panel (3D Surface):** Real-time metric deformation showing space collapsing into a smooth **Geometric Well (Metric Hole)** centered at the group's collective behavioral focus.
2. **Right Panel (2D Contour):** Streamlines of trajectories where agents converge towards the center of maximum curvature driven solely by Geodesic acceleration gradients.

---

## 📖 Citation

If you use this framework or mathematical methodology in your research, please cite the foundational paper:

```bibtex
@article{pinmook2026geometric,
  title={Riemannian Geometry and Geometric Field Equations of Two-Agent Interaction Dynamics: From Time-Dependent to State-Dependent Manifolds},
  author={Pinmook, Chewin},
  journal={ResearchGate / Submissions},
  year={2026},
  month={May}
}

```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
