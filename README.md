# Geometric Interaction Field Theory (GIFT)

GIFT is an exploratory mathematical framework for representing interaction-state dynamics on a differentiable manifold. This revision separates **definitions**, **derived geometry**, and **empirical hypotheses**. Psychological constructs are not treated as physical spacetime quantities, and empirical validation is not claimed without data.

## Mathematical pipeline
1. Represent specified interaction observations locally by coordinates `I=(I1,I2)` on a 2D manifold.
2. Define a symmetric source field `T_ij(I)` from measured or modelled variables.
3. Choose and report a positive-definite background metric `g0` and a constitutive source deformation `H(T)`. Raise one index, `A=g0^{-1}H`, and construct `g(u,v)=g0(exp(kappa A)u,v)`. This map is coordinate covariant and positive definite by construction. It remains a modelling assumption, not a fundamental field equation.
4. Derive the Levi-Civita connection and compute the Ricci tensor and genuine scalar curvature `R=g^ij R_ij`.
5. Distinguish pure geodesic motion `Dv/dt=0` from forced/damped covariant motion `Dv/dt=F-eta v`.
6. Fit parameters on training data and test preregistered out-of-sample predictions against simpler baselines.

## Revised files
- `GIFT.py` - reference 2D geometry solver.
- `GIFT_3D.py` - embedded 3D interaction manifold with intrinsic geodesic integration and animation; scalar curvature is shown as color, not height.
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
python GIFT_3D.py
```

## Established vs hypothesized
**Mathematical consequences conditional on a smooth positive-definite background metric and tensorial source:** coordinate covariance and positive definiteness of the reference exponential deformation, the unique Levi-Civita connection, coordinate-covariant curvature tensors, and the geodesic equation.

**Model assumptions:** interaction coordinates, source parameterization, the constitutive source-to-metric map, and psychological interpretation of curvature.

**Empirical hypotheses:** whether curvature or geometric trajectories improve held-out prediction relative to appropriate baselines. Differential geometry alone cannot prove these claims.

Status: active exploratory research, rigorous revision (2026).
