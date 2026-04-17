import numpy as np
import casadi as ca
import opensim as osim
import matplotlib.pyplot as plt

# Load the IK solution and extract the first column.
table = osim.TimeSeriesTable('jump_1_ik_solution.sto')
times = table.getIndependentColumn()
column = table.getDependentColumnAtIndex(0).to_numpy()
n = table.getNumRows()

class MyQFunc(ca.Callback):
    """Maps joint angles q -> observations y at a single time step.

    Extend get_sparsity_in/out, eval, and get_jacobian for multi-DOF FK:
      - get_sparsity_in:  ca.Sparsity.dense(n_dof, 1)
      - get_sparsity_out: ca.Sparsity.dense(n_obs, 1)
      - eval: call OpenSim FK with np.array(arg[0]).flatten()
      - get_jacobian: return d(FK)/dq as a ca.Function
    """
    def __init__(self):
        ca.Callback.__init__(self)
        self.construct("MyQFunc", {})

    def get_n_in(self):  return 1
    def get_n_out(self): return 1

    def get_sparsity_in(self, i):  return ca.Sparsity.dense(1, 1)
    def get_sparsity_out(self, i): return ca.Sparsity.dense(1, 1)

    def eval(self, arg):
        q = float(arg[0])
        y = q          # replace with FK evaluation
        return [ca.DM(y)]

    def has_jacobian(self): return True

    def get_jacobian(self, name, inames, onames, opts):
        # Inputs: primal input q and primal output y (available if J depends on them).
        # Output: J = dy/dq, shape (n_obs, n_dof).
        q   = ca.MX.sym("q",   1, 1)
        out = ca.MX.sym("out", 1, 1)
        J = ca.DM([[1.0]])   # d(q)/dq; replace with analytic dFK/dq
        return ca.Function(name, [q, out], [J], inames, onames)

my_q_func = MyQFunc()

# Fitting parameters.
t_data = np.linspace(0, 1, n)
degree = 3
nc = 80

# Clamped knot vector: (degree+1) repeats at each end, (nc-degree-1) interior.
# Total knots = nc + degree + 1.
n_interior = nc - degree - 1
knots = np.concatenate([
    np.repeat(0.0, degree + 1),
    np.linspace(0, 1, n_interior + 2)[1:-1],
    np.repeat(1.0, degree + 1),
])

# Build basis matrix B[i,j] = N_j(t_i) numerically.
# ca.bspline does not support symbolic differentiation w.r.t. coefficients, so
# we evaluate each basis function by passing a unit coefficient vector.
# Symbolic spline — used both to build B and for evaluation after fitting.
t_sym = ca.MX.sym("t")
c = ca.MX.sym("c", nc)
spline_expr = ca.bspline(t_sym, c, [knots], [degree], 1)
spline_fn = ca.Function("spline", [t_sym, c], [spline_expr])

# Build basis matrix B[i,j] = N_j(t_i) by evaluating with unit coefficient vectors.
B = np.zeros((n, nc))
for j in range(nc):
    e_j = np.zeros(nc); e_j[j] = 1.0
    B[:, j] = [float(spline_fn(ti, e_j)) for ti in t_data]

# Optimization: min_c ||B @ c - y||^2
B_dm = ca.DM(B)
q_pred = B_dm @ c

y_pred = my_q_func.map(n)(q_pred.T).T

residuals = y_pred - ca.DM(column)
cost = ca.sumsqr(residuals)

solver = ca.nlpsol("solver", "ipopt", {"x": c, "f": cost},
                   {"ipopt.print_level": 5, "ipopt.sb": "yes"})
sol = solver(x0=np.zeros(nc))
ctrl_pts = np.array(sol["x"]).flatten()

# Evaluate the fitted spline on a dense grid.
f = ca.Function("f", [t_sym, c], [spline_expr])
t_dense = np.linspace(0, 1, 500)
y_fit = np.array([float(f(ti, ctrl_pts)) for ti in t_dense])

# Plot
plt.figure(figsize=(10, 6))
plt.plot(t_data, column, 'o', markersize=3, label='Original data', alpha=0.6)
plt.plot(t_dense, y_fit, '-', linewidth=2, label=f'B-spline fit (deg={degree}, nc={nc})')
plt.xlabel('Normalized time')
plt.ylabel('Joint angle')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
