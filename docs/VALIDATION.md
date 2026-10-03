# GIFT validation and proof obligations

## Mathematical checks

**Proposition 1 (coordinate-covariant source-to-metric map).** Let g0 be a positive-definite background metric and H a symmetric covariant source deformation. Define A=g0^{-1}H and g(u,v)=g0(exp(kappa A)u,v). Since A transforms by similarity, exp(kappa A) does also, and therefore g transforms by congruence as a covariant (0,2) tensor.

**Proposition 2 (positive definiteness).** A is self-adjoint relative to g0, so it has real eigenvalues lambda_a and a g0-orthonormal eigenbasis. Hence g(v,v)=sum_a exp(kappa lambda_a) v_a^2>0 for v nonzero.

**Proposition 3 (connection).** For a C2 positive-definite metric, the Levi-Civita theorem gives the unique torsion-free metric-compatible connection
Gamma^k_ij = 1/2 g^kl(d_i g_jl+d_j g_il-d_l g_ij).
GIFT.py implements these components directly.

**Proposition 4 (curvature).** The solver computes
R_ij=d_k Gamma^k_ij-d_j Gamma^k_ik+Gamma^k_ij Gamma^l_kl-Gamma^l_ik Gamma^k_jl
and contracts R=g^ij R_ij. In the discrete implementation both off-diagonal Ricci terms are contracted explicitly because finite differences need not preserve exact symmetry pointwise.

**Proposition 5 (flat consistency).** Constant Euclidean and constant conformal metrics give R=0. A nonconstant off-diagonal metric obtained by a nonlinear coordinate change of the Euclidean plane also gives R=0.

**Proposition 6 (constant-curvature benchmarks).** For the unit sphere in stereographic coordinates,
g=4(1+x^2+y^2)^(-2)(dx^2+dy^2), exact scalar curvature is R=2.
The numerical RMSE is about 7.01e-3 at 81x81 and 1.78e-3 at 161x161, showing approximately second-order convergence.
For the Poincare disk metric g=4(1-r^2)^(-2)(dx^2+dy^2), exact R=-2; the 161x161 interior RMSE is about 2.04e-3.

**Proposition 7 (variable-curvature benchmark).** For g=e^(2u)(dx^2+dy^2), u=a(x^2+y^2), exact R=-8a e^(-2u). With a=0.12 on a 161x161 grid, interior RMSE is about 5.06e-5.

**Proposition 8 (motion semantics).** a^k=-Gamma^k_ij v^i v^j is the coordinate form of an affinely parameterized geodesic. Adding force or damping produces forced covariant motion and must not be labelled a geodesic.

**Proposition 9 (coordinate-covariance numerical test).** A non-orthogonal change of coordinates was applied to both a source tensor and a non-Euclidean background metric. The directly recomputed transformed metric agreed with J^T g J to about 2.2e-15, i.e. floating-point precision.

## Numerical limitations
Errors are largest near grid boundaries where finite differences use one-sided information. Validation therefore reports interior errors. Very large source amplitudes can overflow the matrix exponential and should be bounded or nondimensionalized before inference.

## What mathematics cannot prove
No theorem establishes that emotion, reward, MBTI, or human interaction *is* curvature. Those links are empirical hypotheses. Measurement reliability, parameter identifiability, held-out prediction, baseline comparison and replication are required.

## Empirical falsification plan
Predefine observable coordinates and source variables; estimate parameters on training participants only; compare with linear/state-space/flexible nonlinear baselines; evaluate held-out proper scoring metrics; ablate source components; report uncertainty and failures; replicate externally. Reject or revise a psychological mapping when it does not replicate out of sample.
