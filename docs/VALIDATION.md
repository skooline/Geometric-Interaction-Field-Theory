# GIFT validation and proof obligations

## Mathematical checks
**Proposition 1 (positive definiteness).** If H is real symmetric and g=exp(kappa H), then g is symmetric positive definite. Diagonalize H=Q Lambda Q^T; g=Q exp(kappa Lambda) Q^T has eigenvalues exp(kappa lambda_a)>0.

**Proposition 2 (connection).** For a C2 positive-definite metric, the Levi-Civita theorem gives the unique torsion-free metric-compatible connection Gamma^k_ij = 1/2 g^kl(d_i g_jl+d_j g_il-d_l g_ij). GIFT.py implements this componentwise.

**Proposition 3 (curvature).** The solver computes R_ij=d_k Gamma^k_ij-d_j Gamma^k_ik+Gamma^k_ij Gamma^l_kl-Gamma^l_ik Gamma^k_jl and R=g^ij R_ij. The displayed field is therefore scalar curvature, not the Frobenius norm of g-I.

**Proposition 4 (flat consistency).** Constant Euclidean g=I has zero metric derivatives, hence Gamma=Ric=R=0. Automated tests also check a constant positive multiple of I.

**Proposition 5 (motion semantics).** a^k=-Gamma^k_ij v^i v^j is the coordinate form of an affinely parameterized geodesic. Adding force or damping gives forced covariant motion and must not be labelled a geodesic.

## What mathematics cannot prove
No theorem establishes that emotion, reward, MBTI, or human interaction *is* curvature from these definitions. Those links are empirical hypotheses. Measurement reliability, parameter identifiability, held-out prediction, baseline comparison and replication are required.

## Empirical falsification plan
Predefine observable coordinates and source variables; estimate parameters on training participants only; compare with linear/state-space/flexible nonlinear baselines; evaluate held-out proper scoring metrics; ablate source components; report uncertainty and failures; replicate externally. Reject or revise a psychological mapping when it does not replicate out of sample.
