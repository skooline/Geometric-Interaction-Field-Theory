# Geometric Interaction Field Theory (GIFT)

GIFT is an exploratory mathematical framework for representing interaction-state dynamics on a differentiable manifold. This revision separates **definitions**, **derived geometry**, and **empirical hypotheses**. Psychological constructs are not treated as physical spacetime quantities, and empirical validation is not claimed without data.

## Mathematical pipeline
1. Represent specified interaction observations locally by coordinates `I=(I1,I2)` on a 2D manifold.
2. Define a symmetric source field `T_ij(I)` from measured or modelled variables.
3. Choose and report a constitutive map `g=C(T)`. The reference solver uses `g=exp(kappa H(T))`, guaranteeing a positive-definite Riemannian metric. This is a modelling assumption, not a fundamental field equation.
4. Derive the Levi-Civita connection and compute the Ricci tensor and genuine scalar curvature `R=g^ij R_ij`.
5. Distinguish pure geodesic motion `Dv/dt=0` from forced/damped covariant motion `Dv/dt=F-eta v`.
6. Fit parameters on training data and test preregistered out-of-sample predictions against simpler baselines.

## Revised files
- `GIFT.py` - reference 2D geometry solver.
- `test_gift.py` - invariant/numerical tests.
- `docs/Geometric_IFT_Revised.tex` - formal model and proofs.
- `docs/Reinforcement_Source_Tensor_Revised.tex` - tensor definition and transformation requirements.
- `docs/Emotion_Dynamics_Revised.tex` - affective claims recast as falsifiable hypotheses.
- `docs/VALIDATION.md` - proof obligations and empirical falsification plan.

Legacy PDFs on `main` are retained for provenance; the revised sources on this branch supersede their mathematical claims.

## Run
```bash
pip install numpy matplotlib pytest
pytest -q
python GIFT.py
```

## Established vs hypothesized
**Mathematical consequences conditional on a smooth positive-definite metric:** the unique Levi-Civita connection, coordinate-covariant curvature tensors, the geodesic equation, and positivity of the reference exponential metric.

**Model assumptions:** interaction coordinates, source parameterization, the constitutive source-to-metric map, and psychological interpretation of curvature.

**Empirical hypotheses:** whether curvature or geometric trajectories improve held-out prediction relative to appropriate baselines. Differential geometry alone cannot prove these claims.

Status: active exploratory research, rigorous revision (2026).
