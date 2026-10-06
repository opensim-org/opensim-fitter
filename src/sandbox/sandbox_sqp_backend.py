"""
End-to-end comparison of the IPOPT and SQP backends on a problem with ground truth.

Synthesizes double-pendulum marker data from known body lengths, then solves the
same bilevel problem with each backend and reports wall time, iteration count,
objective, and how well the ground-truth lengths are recovered.

The SQP backend is CasADi's `sqpmethod` with its QP subproblems solved by the
plugin named in `--qp` (default qpOASES). Its Hessian is L-BFGS, because the cost
callbacks expose first derivatives only; an exact Hessian needs the residual
interface that does not exist yet, so this measures the quasi-Newton SQP, which is
the configuration available today.

Run from the repository root:

    python src/sandbox/sandbox_sqp_backend.py
    python src/sandbox/sandbox_sqp_backend.py --qp qrqp --qp osqp
"""

import argparse
import sys
import time
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import opensim as osim

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tests'))
from test_double_pendulum import (create_double_pendulum,            # noqa: E402
                                  create_synthetic_markers_file)

from osimfit.bounds import Bounds                                     # noqa: E402
from osimfit.costs import BodyScaleRegularizationCost                 # noqa: E402
from osimfit.data_sources import MarkerSource, Trial                  # noqa: E402
from osimfit.model import BodyScale                                   # noqa: E402
from osimfit.solvers import SplinedKinematicsSolver                   # noqa: E402

TRUE_B0_LENGTH = 1.25
TRUE_B1_LENGTH = 0.75


def build_solver(trc_path, label_map, nlp_solver, qp_solver, tolerance,
                 gauss_newton=False):
    """Build the same bilevel problem the double-pendulum test solves."""
    unscaled_model = create_double_pendulum(1.0, 1.0)
    unscaled_model.initSystem()
    solver = SplinedKinematicsSolver(unscaled_model,
                                     convergence_tolerance=tolerance,
                                     knot_interval=0.07,
                                     position_weight=5.0)
    solver.add_trial(Trial('pendulum', [MarkerSource('markers', trc_path,
                                                     label_map=label_map)]))
    solver.add_cost(BodyScaleRegularizationCost(1e-2))
    solver.add_parameter(BodyScale('/bodyset/b0', Bounds(0.5, 2.0), np.ones(3)))
    solver.add_parameter(BodyScale('/bodyset/b1', Bounds(0.5, 2.0), np.ones(3)))
    solver.nlp_solver = nlp_solver
    solver.qp_solver = qp_solver
    solver.use_gauss_newton = gauss_newton
    return solver


def solve_and_report(label, solver):
    """Solve, timing it and capturing any non-convergence warning."""
    tracemalloc.start()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        start = time.perf_counter()
        try:
            solution = solver.solve()
            elapsed = time.perf_counter() - start
        except Exception as exc:              # a backend may fail outright
            elapsed = time.perf_counter() - start
            peak = tracemalloc.get_traced_memory()[1] / 1e6
            tracemalloc.stop()
            return {'label': label, 'ok': False, 'seconds': elapsed, 'peak_mb': peak,
                    'detail': f'{type(exc).__name__}: {str(exc).splitlines()[0][:70]}'}
        peak = tracemalloc.get_traced_memory()[1] / 1e6
        tracemalloc.stop()
        warned = [str(w.message).split(':')[-1].strip() for w in caught
                  if w.category is RuntimeWarning]

    scales = [p for p in solution.parameters if isinstance(p, BodyScale)]
    b0, b1 = scales[0].value[0], scales[1].value[0]
    return {'label': label, 'ok': True, 'seconds': elapsed, 'peak_mb': peak,
            'b0': b0, 'b1': b1,
            'err': max(abs(b0 - TRUE_B0_LENGTH), abs(b1 - TRUE_B1_LENGTH)),
            'rows': solution.states_tables['pendulum'].getNumRows(),
            'detail': '; '.join(warned) if warned else 'converged'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qp', action='append', default=None,
                        help='QP plugin for sqpmethod; repeatable. Default: qpoases.')
    parser.add_argument('--tolerance', type=float, default=1e-5)
    parser.add_argument('--skip-ipopt', action='store_true')
    args = parser.parse_args()
    qp_solvers = args.qp or ['qpoases']

    out_dir = Path(__file__).resolve().parent
    trc_path = str(out_dir / 'sqp_backend_markers.trc')
    create_synthetic_markers_file(trc_path, TRUE_B0_LENGTH, TRUE_B1_LENGTH)
    label_map = {lab: lab.replace('|location', '')
                 for lab in osim.TimeSeriesTableVec3(trc_path).getColumnLabels()}

    configs = []
    if not args.skip_ipopt:
        configs.append(('ipopt, L-BFGS', 'ipopt', '-', False))
        configs.append(('ipopt, Gauss-Newton', 'ipopt', '-', True))
    for qp in qp_solvers:
        configs.append((f'sqp({qp}), L-BFGS', 'sqpmethod', qp, False))
        configs.append((f'sqp({qp}), Gauss-Newton', 'sqpmethod', qp, True))

    results = []
    for label, nlp, qp, gn in configs:
        print(f'\n=== {label} ===', flush=True)
        solver = build_solver(trc_path, label_map, nlp, qp, args.tolerance, gn)
        results.append(solve_and_report(label, solver))

    print(f'\n\n{"=" * 100}')
    print(f'Ground truth: b0 = {TRUE_B0_LENGTH}, b1 = {TRUE_B1_LENGTH}   '
          f'(the test\'s pass threshold is max error < 0.02)')
    print(f'{"backend":26} {"time (s)":>9} {"peak MB":>8} {"b0":>8} {"b1":>8} '
          f'{"max err":>9}  status')
    print('-' * 100)
    for r in results:
        if not r['ok']:
            print(f'{r["label"]:26} {r["seconds"]:>9.1f} {r["peak_mb"]:>8.1f} '
                  f'{"-":>8} {"-":>8} {"-":>9}  FAILED {r["detail"]}')
            continue
        flag = 'PASS' if r['err'] < 0.02 else 'OFF '
        print(f'{r["label"]:26} {r["seconds"]:>9.1f} {r["peak_mb"]:>8.1f} '
              f'{r["b0"]:>8.4f} {r["b1"]:>8.4f} {r["err"]:>9.4f}  {flag} '
              f'({r["detail"]})')


if __name__ == '__main__':
    main()
