# GIFT validation and proof obligations

## Mathematical checks

**Proposition 1 (positive definiteness).** If H is real symmetric and g=exp(kappa H), then g is symmetric positive definite. Diagonalize H=Q Lambda Q^T; then g=Q exp(kappa Lambda) Q^T and every eigenvalue is exp(kappa lambda_a)>0.

**Proposition 2 (connection).** For a C2 positive-definite metric, the Levi-Civita theorem gives the unique torsion-free metric-compatible connection
Gamma^k_ij = 1/2 g^kl(d_i g_jl+d_j g_il-d_l g_ij).
GIFT.py implements these components directly.

**Proposition 3 (curvature).** The solver computes
R_ij=d_k Gamma^k_ij-d_j Gamma^k_ik+Gamma^k_ij Gamma^l_kl-Gamma^l_ik Gamma^k_jl
and contracts R=g^ij R_ij. In the discrete implementation both off-diagonal Ricci terms are contracted explicitly because finite differences need not preserve exact symmetry pointwise.

**Proposition 4 (flat consistency).** Constant Euclidean and constant conformal metrics give R=0. A nonconstant off-diagonal metric obtained by a nonlinear coordinate change of the Euclidean plane also gives R=0.

**Proposition 5 (constant-curvature benchmarks).** For the unit sphere in stereographic coordinates,
g=4(1+x^2+y^2)^(-2)(dx^2+dy^2), exact scalar curvature is R=2.
The numerical RMSE is about 7.01e-3 at 81x81 and 1.78e-3 at 161x161, showing approximately second-order convergence.
For the Poincare disk metric g=4(1-r^2)^(-2)(dx^2+dy^2), exact R=-2; the 161x161 interior RMSE is about 2.04e-3.

**Proposition 6 (variable-curvature benchmark).** For g=e^(2u)(dx^2+dy^2), u=a(x^2+y^2), exact R=-8a e^(-2u). With a=0.12 on a 161x161 grid, interior RMSE is about 5.06e-5.

**Proposition 7 (motion semantics).** a^k=-Gamma^k_ij v^i v^j is the coordinate form of an affinely parameterized geodesic. Adding force or damping produces forced covariant motion and must not be labelled a geodesic.

## Numerical limitations
Errors are largest near grid boundaries where finite differences use one-sided information. Validation therefore reports interior errors. Very large source amplitudes can overflow the matrix exponential and should be bounded or nondimensionalized before inference.

## What mathematics cannot prove
No theorem establishes that emotion, reward, MBTI, or human interaction *is* curvature. Those links are empirical hypotheses. Measurement reliability, parameter identifiability, held-out prediction, baseline comparison and replication are required.

## Empirical falsification plan
Predefine observable coordinates and source variables; estimate parameters on training participants only; compare with linear/state-space/flexible nonlinear baselines; evaluate held-out proper scoring metrics; ablate source components; report uncertainty and failures; replicate externally. Reject or revise a psychological mapping when it does not replicate out of sample.
