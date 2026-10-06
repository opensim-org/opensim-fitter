# Supporting an SQP solver in osimfit

Findings from investigating whether the problems osimfit builds could be handed to a
QP-based solver, and what it would take to support IPOPT and SQP side by side.

Status: investigation and prototype only. No osimfit code has been changed for this.
Line references are against `opensim-fitter/src/osimfit` as of 2026-10-06.

## Headline

**The axis that matters is scalar-vs-residual, not IPOPT-vs-SQP.** Adding an SQP
backend on its own makes things *worse*. All of the gain comes from exposing the
least-squares residual vector so a Gauss-Newton Hessian can be formed — and once that
exists, IPOPT benefits from it too.

Measured on a prototype that mimics the osimfit structure (an opaque residual source
backed by a CasADi `Callback` with an analytic Jacobian, bound constraints only):

| configuration | iterations |
| --- | --- |
| IPOPT, L-BFGS (what osimfit does today) | 18 |
| **SQP (qrqp), L-BFGS, scalar objective** | **50, did not converge** |
| IPOPT + Gauss-Newton Hessian | 5 |
| SQP (qrqp) + Gauss-Newton | 3 |
| SQP (osqp) + Gauss-Newton | 3 |
| SQP (qpoases) + Gauss-Newton | 3 |
| feasiblesqpmethod + Gauss-Newton | 4 |

Every row used the same residual callback. The second row is the one to note: a
quasi-Newton SQP on a scalar objective has no advantage over IPOPT and gives up
IPOPT's maturity.

Note that the problem is a **nonlinear** least-squares, not a QP — the marker
residuals are nonlinear in the coordinates and in the body scales. The realistic
target is Gauss-Newton, or an SQP whose *subproblem* is a QP.

## What is available in this build

CasADi 3.7.2:

    nlpsol: ipopt, sqpmethod, feasiblesqpmethod, blocksqp, scpgen
    conic:  qrqp, osqp, qpoases, proxqp, highs, clp, daqp
    absent: worhp, snopt, cplex, gurobi, hpipm

## How the problems are currently shaped

All three solvers build the same thing (`solvers.py:473`, `:973`, `:1163`):

```python
nlp = {'x': x, 'f': f}          # no 'g'
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)
```

Bound-constrained scalar minimizations with no constraints at all. `lbg`/`ubg` never
appear. That part is fine for a QP-based method; the objective is the problem.

### The blocker: the callback boundary collapses the residuals

`CallbackCostRep` hard-codes a scalar output (`costs.py:347` and `:364`):

```python
def _get_num_outputs(self):
    return 1

def _get_output_size(self, i):
    if i == 0:
        return 1
```

and the tracking terms compute the residual vector, then immediately discard it
(`costs.py:1052`, `:1148`):

```python
residuals = positions - arrays.reference_positions
return float(np.sum(arrays.weights * np.sum(residuals * residuals, axis=1)))
```

`calc_jacobian` correspondingly returns a 1xN row — the *gradient of the scalar* —
assembled with `multiplyByStationJacobianTranspose`, which contracts the station
Jacobian against the residual before CasADi ever sees it. The design intent is stated
at `solvers.py:263`: *"We only support callback functions with first-order
derivatives."*

This is exactly the information Gauss-Newton needs and cannot recover: it needs `J`
(m x n) to form `J'J`, and `J'r` does not determine `J`. The structure is computed and
thrown away one layer below the solver interface.

### What splits cleanly

| cost | kind | exact Hessian available today? |
| --- | --- | --- |
| `BodyScaleIsotropyCost`, `OffsetRegularizationCost`, `BodyScaleRegularizationCost`, `EllipsoidRadiiScaleRegularizationCost`, `BeamLengthScaleRegularizationCost` | `SymbolicCost`, pure CasADi MX | **Yes**, exactly — all are `sumsqr`-type and literally quadratic |
| `CoordinateStiffnessCost` | symbolic MX built in `CostRep.__call__` | **Yes** — `stiffness * (q - target)**2` |
| both damping terms (`solvers.py:472`, `:955`) | symbolic MX | **Yes** — both are L2 |
| `TrackingCost`, `BilevelCost`, `AnthropometricRegularizationCost` | `CallbackCostRep` | **No** — opaque scalar plus gradient |

So roughly half the objective is already a genuine QP in the decision variables and
CasADi could differentiate it to second order today. The tracking term — the dominant
cost, and the one driving the conditioning problems — is the opaque part.

**Every cost in the package is a sum of squares.** That includes the ones that are not
obviously so:

- `AnthropometricRegularizationCost` is a Mahalanobis distance
  `0.5 (m - mu)' Sigma^-1 (m - mu)`, so with a Cholesky `Sigma = L L'` the residual is
  `r = L^-1 (m - mu)`.
- The frame orientation term is `w * (3 - trace(R_ref' R)) / 4`, which the code notes
  equals `sin(theta/2)**2` — already a perfect square.

Both damping terms are L2 as currently written, so they fold in directly. (The IK
damping was briefly an L1 `sum1(fabs(...))`; as an L1 it would *not* have been a sum
of squares, and its kink sits exactly where the optimizer wants to sit.)

## What a dual-backend design looks like

### 1. `Function` needs almost nothing

The m x n case is already structurally supported. `Function.get_sparsity_out` is
`dense(get_output_size(i), 1)`, and the nested `JacobianFunction` already declares
`dense(output_size(iout), input_size(iin))`. Only `CallbackCostRep._get_output_size`
hard-codes the 1:

```python
def _get_output_size(self, i):
    if i == 0:
        return self.num_residuals     # was: return 1
```

### 2. `CostRep` grows an optional residual interface

Default to the scalar path so nothing existing breaks:

```python
class CostRep(ABC):
    def __call__(self, input: CostInput) -> ca.MX:   # unchanged, scalar
        ...

    def residuals(self, input: CostInput) -> ca.MX | None:
        """m x 1 residual vector r with this cost equal to sumsqr(r), or None."""
        return None
```

Weights fold in as `sqrt(w)`, safe because every weight is validated non-negative.
The trapezoidal time average in `compute_average_trapezoidal_error` folds in the same
way, as `sqrt(wt_i / duration)` per timestep.

### 3. The tracking terms return residuals and a full Jacobian

`calc_error` returns `(sqrt(w)[:, None] * residuals).ravel()`, length `3 * num_tasks`.
`calc_jacobian` is the real work: instead of one
`multiplyByStationJacobianTranspose` contracting against the residual, it needs the
full block — cheapest via `multiplyByStationJacobian` column-wise, roughly one call
per coordinate (~49) rather than one per residual row (~189).

The orientation term should become a **3-component** residual (quaternion vector
part), not a scalar `sin(theta/2)`. As a scalar it has an absolute-value kink at
perfect alignment; as a 3-vector no square root is ever taken and it stays smooth.

### 4. Solvers get a backend strategy

`get_ipopt_options` becomes backend-aware:

```python
def get_solver_options(self) -> tuple[str, dict]:
    """Return (nlpsol plugin, options) for the configured backend."""
```

with a shared Gauss-Newton helper. One wrinkle the prototype hit: **IPOPT wants
`triu(J'J)` and sqpmethod wants the full symmetric matrix** — passing `triu` to
sqpmethod fails with `Hessian must be symmetric`.

```python
GN = ca.triu(Jr.T @ Jr) if plugin == 'ipopt' else Jr.T @ Jr
hess_lag = ca.Function('nlp_hess_l',
                       {'x': x, 'p': ca.MX.sym('p', 0), 'lam_f': sigma,
                        'hess_gamma_x_x': sigma * GN},
                       ['x', 'p', 'lam_f', 'lam_g'], ['hess_gamma_x_x'])
```

Tolerances need translating too: IPOPT uses `tol` / `dual_inf_tol`, sqpmethod uses
`tol_pr` / `tol_du` plus `qpsol` and `qpsol_options`. That mapping belongs in the same
place so `convergence_tolerance` stays the single user-facing knob.

## Why this stays tractable at 25k variables

`J'J` for n = 25,158 would be 5 GB dense. It is not dense, because of where the
existing architecture puts the split: the callbacks take `coordinates` (49) as input,
and `q = B @ coeffs` is applied **symbolically in CasADi outside the callback**. Each
per-timestep Jacobian block is therefore a small dense 189 x 49 plus parameter blocks,
and CasADi composes it with the banded B-spline basis to assemble a sparse global
Jacobian on its own. The resulting `J'J` has block-arrow structure: banded within each
trial's control points, dense only in the 217 shared parameters.

## Cost and risk

Step 3 makes each Jacobian evaluation roughly 20-40x more expensive, against about a
5x iteration reduction in the prototype. On paper that is a wash. The reason to expect
a net win is the *function* evaluation count rather than the gradient count: on the
real problem IPOPT spent 2479 function evaluations against 444 gradients, at roughly
8 ms and 23 ms each, and Gauss-Newton collapses the former. That is an argument, not a
measurement — it should not be promised without measuring on the real problem.

For reference, the baseline to beat: with `mu_strategy='monotone'` and
`mu_init=1e-6`, the full six-trial problem converges in 444 iterations and 336 s at
1.21 cm marker RMS. Under the stock settings it does not converge in 400 iterations
and sits at 4.20 cm.

## Suggested sequencing

1. **`mu_strategy` / `mu_init`.** Already measured; turns non-convergence into 444
   iterations. Two lines, no structural change.
2. **Residual interface plus a Gauss-Newton Hessian for IPOPT only.** No new backend,
   no new failure modes, and it is the load-bearing step. Measure here before going
   further.
3. **SQP backend**, reusing the identical residuals. At that point it is a small
   addition, because the hard part is done.

Doing 3 before 2 would add a solver and make the pipeline slower.

## Reproducing the prototype

The benchmark table came from a standalone script, not from osimfit: a `ca.Callback`
with a vector output and an analytic `get_jacobian`, wrapped as `f = 0.5*sumsqr(r)`,
solved under each backend with and without a `hess_lag` override. The pieces worth
recreating are the `hess_lag` construction above, the `triu`-vs-full distinction, and
driving the same `nlp` dict through `ca.nlpsol` with each plugin name.
