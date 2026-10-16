import os
import json
import numpy as np
import casadi as ca
import opensim as osim
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from .costs import CostInput
from .model import Parameter


############
# SOLUTION #
############

@dataclass
class Solution:
    """
    The result of a solve, returned by every solver.

    Every solver optimizes over one or more `Trial` objects, so the kinematic result is
    always a states table per trial, keyed by trial name. Parameters coupled across
    trials (e.g., body scales, placement offsets) are shared and so live in a single
    flat list. Solvers can also provide additional solution information via `outputs`.
    Use the `print()` method to write out all results contained in this `Solution` to a
    directory.

    Attributes
    ----------
    states_tables: dict[str, osim.TimeSeriesTable]
        The table of optimized model states for each trial, keyed by trial name.
    parameters: list[Parameter], optional
        The optimized parameters (e.g., body scales, marker and frame offsets), each
        carrying its optimal ``value``, shared across every trial.
    outputs: dict[str, Any]
        Solver-specific results that are neither states tables nor parameters, keyed by
        output name. Per-trial results nest a dict keyed by trial name under the output
        name, e.g. ``outputs['spline_nodes']['walk_1']``. Every solver also reports
        ``outputs['cost_terms']``: the breakdown of the objective at the optimum into
        its individual terms, summed over trials.
    """

    SUPPORTED_OUTPUTS = frozenset({'cost_terms', 'spline_nodes'})

    states_tables: dict[str, osim.TimeSeriesTable] = field(default_factory=dict)
    parameters: list[Parameter] = None
    outputs: dict[str, Any] = field(default_factory=dict)

    def get_coordinates(self, trial_name: str) -> osim.TimeSeriesTable:
        """
        Return an `osim.TimeSeriesTable` containing the coordinate values (e.g., joint
        angles) for a particular trial.

        Parameters
        ----------
        trial_name: str
            The name of the trial.

        Raises
        ------
        ValueError
            If a Trial with 'trial_name' does not exist in this Solution.
        """

        if trial_name not in self.states_tables:
            raise ValueError(f"Trial with name '{trial_name}' not found in solution.")

        states = self.states_tables[trial_name]
        coordinates = osim.TimeSeriesTable(states.getIndependentColumn())
        coordinate_labels = [label for label in states.getColumnLabels()
                             if '/value' in label]
        for label in coordinate_labels:
            coordinates.appendColumn(label, states.getDependentColumn(label))

        return coordinates

    def get_parameter(self, path: str, cls: type = Parameter) -> Parameter:
        """
        Return the optimized parameter of type `cls` whose group contains `path`.

        Parameters
        ----------
        path: str
            Absolute model path of a component in the target parameter's group.
        cls: type, optional
            Restrict the search to parameters of this `Parameter` subtype (e.g.,
            `BodyScale`, `MarkerOffset`, or `FrameOffset`). Defaults to `Parameter`
            (any type).

        Raises
        ------
        KeyError
            If not exactly one parameter of type `cls` has `path` in its group.
        """
        matches = [p for p in (self.parameters or [])
                   if isinstance(p, cls) and path in p.paths]
        if len(matches) != 1:
            raise KeyError(
                f'Expected exactly one {cls.__name__} whose group contains {path}, '
                f'but found {len(matches)}.')
        return matches[0]

    def print(self, directory: str = '.', prefix: str = '') -> list[str]:
        """
        Write every result this solution carries to `directory`, and echo a summary
        of the solve to stdout.

        One file is written per result: each trial's states table as a `.sto`, the
        optimized parameters and the objective breakdown as JSON, and any per-trial
        array output (e.g. a splined solver's spline nodes) as its own `.sto`. Any
        remaining output goes to a single JSON file. This replaces writing each table
        by hand with `osim.STOFileAdapter`.

        Parameters
        ----------
        directory: str, optional
            Directory to write into, created if it does not exist. Defaults to the
            working directory.
        prefix: str, optional
            Prepended to every filename, so several solutions can share a directory.
            Omitted by default.

        Returns
        -------
        list[str]
            The paths written, in the order written.

        Raises
        ------
        ValueError
            If an output's name is not one of `SUPPORTED_OUTPUTS`. Checked before
            anything is written, so a solution carrying an unsupported output writes
            no files at all.
        TypeError
            If a supported output holds a value that cannot be written as JSON.
        """
        unsupported = sorted(set(self.outputs) - self.SUPPORTED_OUTPUTS)
        if unsupported:
            raise ValueError(
                f'Solution carries output(s) {unsupported} that print() does not '
                f'support (supported: {sorted(self.SUPPORTED_OUTPUTS)}); add the name '
                f'to Solution.SUPPORTED_OUTPUTS and teach print() to write it.')

        os.makedirs(directory, exist_ok=True)
        sto = osim.STOFileAdapter()
        written = []

        def path_for(*parts: str) -> str:
            stem = '_'.join(([prefix] if prefix else []) + list(parts))
            return os.path.join(directory, stem)

        for trial_name, table in self.states_tables.items():
            fpath = path_for(trial_name, 'states.sto')
            sto.write(table, fpath)
            written.append(fpath)

        if self.parameters:
            fpath = path_for('parameters.json')
            with open(fpath, 'w') as f:
                json.dump([self._describe_parameter(p) for p in self.parameters],
                          f, indent=2)
                f.write('\n')
            written.append(fpath)

        for name, value in self.outputs.items():
            if name == 'cost_terms':
                fpath = path_for('cost_terms.json')
                with open(fpath, 'w') as f:
                    json.dump(value, f, indent=2)
                    f.write('\n')
                written.append(fpath)
            elif self._is_per_trial_array(value):
                for trial_name, array in value.items():
                    fpath = path_for(trial_name, f'{name}.sto')
                    sto.write(self._array_as_table(np.asarray(array), trial_name),
                              fpath)
                    written.append(fpath)
            else:
                # Serialized up front so an unwritable value raises before the file
                # is created rather than truncating it part-written.
                text = json.dumps(value, indent=2,
                                  default=partial(self._as_json, name))
                fpath = path_for(f'{name}.json')
                with open(fpath, 'w') as f:
                    f.write(text + '\n')
                written.append(fpath)

        self._print_summary(written)
        return written

    @staticmethod
    def create_states_table(model, state, coordinate_q_indexes, coordinate_u_indexes,
                            times, q_opt, qdot_opt=None) -> osim.TimeSeriesTable:
        """
        Build an OpenSim StatesTrajectory and export it to a TimeSeriesTable.

        Parameters
        ----------
        model: osim.Model
        state: osim.State
            An initialized state that will be mutated in place during construction.
        coordinate_q_indexes: list[int]
            q-indexes of the independent coordinates in the full state vector.
        coordinate_u_indexes: list[int]
            u-indexes of the independent coordinates in the full state vector.
        times: sequence of float
        q_opt: np.ndarray, shape (num_times, num_coords)
        qdot_opt: np.ndarray, shape (num_times, num_coords), optional
        """
        statesTraj = osim.StatesTrajectory()
        for i, time in enumerate(times):
            state.setTime(time)
            q = np.zeros(state.getNQ())
            q[coordinate_q_indexes] = q_opt[i, :]
            state.setQ(osim.Vector.createFromMat(q))
            if qdot_opt is not None:
                u = np.zeros(state.getNU())
                u[coordinate_u_indexes] = qdot_opt[i, :]
                state.setU(osim.Vector.createFromMat(u))
            statesTraj.append(state)
        return statesTraj.exportToTable(model)

    @staticmethod
    def _as_json(name: str, value):
        """
        Convert a value json does not handle natively, raising if there is no
        meaningful conversion rather than writing a repr nobody can read back.
        """
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(
            f"Solution output '{name}' holds a {type(value).__name__}, which "
            f'print() cannot write as JSON.')

    @staticmethod
    def _is_per_trial_array(value) -> bool:
        """
        Whether an output is a per-trial mapping of arrays, which is written as one
        table per trial rather than as JSON.
        """
        return (isinstance(value, dict) and bool(value)
                and all(isinstance(v, np.ndarray) and v.ndim == 2
                        for v in value.values()))

    def _array_as_table(self, array: np.ndarray,
                        trial_name: str) -> osim.TimeSeriesTable:
        """
        Wrap a 2-D array as a table indexed by row number, labelling its columns with
        the trial's coordinate paths when the widths agree and generically otherwise.
        """
        coordinates = []
        table = self.states_tables.get(trial_name)
        if table is not None:
            coordinates = [label[:-len('/value')]
                           for label in table.getColumnLabels()
                           if label.endswith('/value')]
        labels = (coordinates if len(coordinates) == array.shape[1]
                  else [f'col_{i}' for i in range(array.shape[1])])

        out = osim.TimeSeriesTable()
        out.setColumnLabels(labels)
        for row in range(array.shape[0]):
            out.appendRow(float(row), osim.RowVector(array[row, :].tolist()))
        return out

    @staticmethod
    def _describe_parameter(parameter: Parameter) -> dict:
        """
        A JSON-ready description of one optimized parameter.
        """
        value = parameter.value
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif value is not None:
            value = float(value)
        described = {'type': type(parameter).__name__,
                     'paths': list(getattr(parameter, 'paths', [])),
                     'value': value}
        bounds = getattr(parameter, 'bounds', None)
        if bounds is not None:
            described['bounds'] = {'lower': bounds.lower_bound,
                                   'upper': bounds.upper_bound}
        return described

    def _print_summary(self, written: list[str]) -> None:
        """
        Echo the solve's trials, parameters and objective breakdown to stdout.
        """
        print()
        if self.states_tables:
            print('trials')
            for trial_name, table in self.states_tables.items():
                times = table.getIndependentColumn()
                span = (f'{times[0]:.4g} to {times[-1]:.4g} s' if len(times)
                        else 'empty')
                print(f'  {trial_name:<28} {table.getNumRows():>6} rows   {span}')

        if self.parameters:
            print('\nparameters')
            for parameter in self.parameters:
                described = self._describe_parameter(parameter)
                value = described['value']
                if isinstance(value, list):
                    shown = '[' + ', '.join(f'{v:.6g}' for v in value) + ']'
                else:
                    shown = 'unset' if value is None else f'{value:.6g}'
                label = f"{described['type']}({', '.join(described['paths'])})"
                print(f'  {label:<48} {shown}')

        cost_terms = self.outputs.get('cost_terms')
        if cost_terms:
            print('\ncost terms')
            print(f"  {'term':<40} {'value':>12} {'fraction':>10}")
            for name, term in cost_terms['terms'].items():
                print(f"  {name:<40} {term['value']:>12.4e} "
                      f"{term['fraction']:>9.1%}")
            print(f"  {'total':<40} {cost_terms['total']:>12.4e} "
                  f"{1.0:>9.1%}")

        print(f'\nwrote {len(written)} file(s):')
        for fpath in written:
            print(f'  {fpath}')
        print()


###################
# OBJECTIVE TERMS #
###################

class ObjectiveTerms:
    """
    The labelled scalar terms a solver's objective is built from.

    A solver names each contribution rather than accumulating one opaque expression,
    so a solution can report how much of the optimum each contribution accounts for.
    The terms sum to the objective by construction, which is what makes the reported
    fractions meaningful.

    Solvers that solve once build their terms here and call `summarize`. Solvers that
    solve repeatedly, once per time sample, instead evaluate each sample's terms and
    hand the values to a second instance via `accumulate`, reporting the mean with
    `summarize_mean`.
    """

    def __init__(self):
        self.terms: dict[str, Any] = {}
        self._totals: dict[str, float] = {}
        self._num_accumulated = 0

    def add(self, label: str, expr) -> None:
        """
        Add `expr` to the term labelled `label`, starting that term at zero if it is
        new. Adding rather than assigning is what lets a solver accumulate a term
        across trials without seeding the labels up front.
        """
        self.terms[label] = self.terms.get(label, 0) + expr

    def add_costs(self, named_reps: list, cost_input: CostInput) -> None:
        """
        Add one term per registered cost, evaluated against `cost_input`.

        Parameters
        ----------
        named_reps: list[tuple[str, CostRep]]
            Each cost's label paired with its rep; a cost is labelled by its type.
        cost_input: CostInput
            The input to evaluate each rep against.
        """
        for label, cost_rep in named_reps:
            self.add(label, cost_rep(cost_input))

    def scale(self, factor: float) -> None:
        """
        Multiply every term accumulated so far by `factor`, for solvers that average
        their objective over trials. Terms added afterwards are left alone, so a cost
        evaluated once for the whole problem can bypass the averaging.
        """
        for label in self.terms:
            self.terms[label] = factor * self.terms[label]

    @property
    def objective(self):
        """
        The objective to hand the NLP solver: the sum of every term.
        """
        return sum(self.terms.values())

    def evaluate(self, x, x_opt) -> dict[str, float]:
        """
        Each term's value at `x_opt`, keyed by label.
        """
        function = ca.Function('cost_terms', [x], list(self.terms.values()))
        values = function(x_opt)
        # CasADi returns a bare DM, rather than a list of them, for a single-output
        # function.
        if len(self.terms) == 1:
            values = [values]
        return {name: float(value) for name, value in zip(self.terms, values)}

    def summarize(self, x, x_opt) -> dict[str, Any]:
        """
        The breakdown to report under ``Solution.outputs['cost_terms']``, evaluated
        at `x_opt`; see `Solution`.
        """
        return self._summarize_values(self.evaluate(x, x_opt))

    def accumulate(self, values: dict[str, float]) -> None:
        """
        Add one solve's term values into a running total, for a solver that reports
        the mean over many solves. Pair with `summarize_mean`.
        """
        for label, value in values.items():
            self._totals[label] = self._totals.get(label, 0.0) + value
        self._num_accumulated += 1

    def summarize_mean(self) -> dict[str, Any]:
        """
        The breakdown of the mean of every accumulated solve; see `accumulate`.
        """
        if not self._num_accumulated:
            return self._summarize_values({})
        return self._summarize_values(
            {label: total / self._num_accumulated
             for label, total in self._totals.items()})

    @staticmethod
    def _summarize_values(values: dict[str, float]) -> dict[str, Any]:
        """
        Package per-term values into the breakdown solvers report under
        ``Solution.outputs['cost_terms']``; see `Solution`.
        """
        total = sum(values.values())
        return {
            'total': total,
            'terms': {name: {'value': value,
                             'fraction': value / total if total != 0.0 else 0.0}
                      for name, value in values.items()},
        }
