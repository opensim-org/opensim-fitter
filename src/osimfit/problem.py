"""
Per-solve problem representations shared across a solver's cost evaluations.

A `ProblemRep` is the counterpart to OpenSim Moco's `MocoProblemRep`: a solver builds
one after every trial, cost, and parameter has been registered, and the rep owns every
model-bound object that solve needs. Tracking reps are built once per *trial* rather
than once per time sample, with each sample's reference data supplied through the rep's
reference input, so a single CasADi callback can be evaluated across all of a trial's
samples via `ca.Function.map`.
"""

import queue
from contextlib import contextmanager

import numpy as np
import casadi as ca
import opensim as osim

from .costs import CallbackCostRep, CostInput, Function
from .data_sources import Trial
from .model import ModelCache


##################
# MODEL REPLICAS #
##################

def replicate_model_cache(model_cache: ModelCache) -> ModelCache:
    """
    Build an independent `ModelCache` wrapping a copy of `model_cache`'s model, with
    the same parameter groups registered.

    Parameter groups hold only model-independent descriptors, so the very same group
    objects are registered on the copy; the per-model `Joint`s and baselines they imply
    are re-derived by `add_parameter_group` against the copy's own model.

    Parameters
    ----------
    model_cache: ModelCache
        The cache to replicate.

    Returns
    -------
    ModelCache
        A cache sharing no `osim.Model` or `SimTK::State` with `model_cache`.
    """
    copy = ModelCache(osim.Model(model_cache.model))
    for group in (model_cache.body_scale_groups +
                  model_cache.marker_offset_groups +
                  model_cache.frame_offset_groups +
                  model_cache.ellipsoid_radii_scale_groups +
                  model_cache.beam_length_scale_groups):
        copy.add_parameter_group(group)
    copy.cache_body_scale_group_joints()
    return copy


#######
# JAR #
#######

class WorkerJar:
    """
    A thread-safe pool of interchangeable evaluators, mirroring Moco's `ThreadsafeJar`.

    Evaluating a cost callback from several threads concurrently requires each thread to
    own a distinct `osim.Model` and `SimTK::State`: realizing a state mutates its cache,
    and the model-bound objects a tracking term holds (markers, frames, station arrays)
    belong to one specific model. Sharing them across threads yields silently wrong
    values rather than an error, so each worker is a complete, independent evaluator and
    a thread borrows one for the duration of a single evaluation.

    Parameters
    ----------
    workers: list
        The pool members. Must be non-empty. ``workers[0]`` is the `primary`, used for
        any query that only reads problem structure rather than evaluating.
    """
    def __init__(self, workers: list):
        if not workers:
            raise ValueError('A WorkerJar requires at least one worker.')
        self.workers = list(workers)
        self._pool = queue.LifoQueue()
        for worker in self.workers:
            self._pool.put(worker)

    @property
    def num_workers(self) -> int:
        return len(self.workers)

    @property
    def primary(self):
        """
        The first worker. Safe to read for structural queries (declared input sizes,
        registered tasks), but never for evaluation, which must go through `borrow`.
        """
        return self.workers[0]

    @contextmanager
    def borrow(self):
        """
        Check out a worker for the duration of the block, returning it afterwards even
        if the body raises. Blocks while every worker is checked out, so a pool smaller
        than the number of calling threads throttles rather than deadlocks.
        """
        worker = self._pool.get()
        try:
            yield worker
        finally:
            self._pool.put(worker)


class PooledTrackingRep(CallbackCostRep):
    """
    A single CasADi callback that evaluates a tracking cost by borrowing a worker from a
    `WorkerJar`.

    Each worker is a complete tracking rep with its own `ModelCache` and its own
    marker/frame terms, so concurrent evaluations share no mutable model state. The
    workers are used purely as evaluators and are never registered with CasADi; this
    wrapper is the only CasADi function, which is what lets `ca.Function.map` vectorise
    one callback across a trial's time samples.

    Parameters
    ----------
    name: str
        The name of the callback function.
    jar: WorkerJar
        The pool of tracking reps to delegate evaluations to.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences this callback instead of using the
        workers' analytic Jacobians. Default is ``False``.
    """
    def __init__(self, name: str, jar: WorkerJar, enable_fd: bool = False):
        self.jar = jar
        Function.__init__(self, name, jar.primary.mc, enable_fd=enable_fd,
                          defer_construction=True)

    # Structural queries delegate to the primary worker; every worker is registered
    # with identical tasks, so they all report the same sizes.
    @property
    def reference_size(self) -> int:
        return self.jar.primary.reference_size

    @property
    def default_reference(self) -> np.ndarray:
        return self.jar.primary.default_reference

    @property
    def marker_term(self):
        return self.jar.primary.marker_term

    @property
    def frame_term(self):
        return self.jar.primary.frame_term

    def _eval(self, arg):
        with self.jar.borrow() as worker:
            return worker._eval(arg)

    def _jac_eval(self, arg):
        with self.jar.borrow() as worker:
            return worker._jac_eval(arg)


###############
# PROBLEM REP #
###############

class ProblemRep:
    """
    A solver's per-solve representation of its whole fitting problem.

    Built once, after every trial, cost, and parameter has been registered, and held
    for the lifetime of the solve: CasADi keeps only borrowed references to callbacks,
    so the reps must outlive the NLP that uses them.

    Parameters
    ----------
    model_cache: ModelCache
        The solver's `ModelCache`, with all parameter groups already registered. It
        becomes the primary worker of every trial's jar, so a single-threaded problem
        creates no model copies at all.
    trials: list[Trial]
        The trials to track, each contributing one tracking rep.
    costs: list[Cost]
        The solver's registered costs. Split into those evaluated per time sample
        (those reading ``'coordinates'``) and those evaluated once on the shared
        parameters.
    tracking_cost: TrackingCostBase
        The tracking cost description used to build each trial's rep.
    num_threads: int, optional
        Number of workers per trial, i.e. the number of threads a mapped evaluation may
        use. Default is 1.
    enable_fd: bool, optional
        If ``True``, CasADi finite-differences the tracking callbacks. Default is
        ``False``.

    Attributes
    ----------
    tracking_reps: dict[str, PooledTrackingRep]
        One tracking callback per trial, keyed by trial name.
    reference: dict[str, ca.DM]
        Per trial, a ``(reference_size, num_times)`` matrix whose column ``i`` is the
        reference data for time sample ``i``.
    coordinate_cost_reps: list[CostRep]
        Reps of registered costs evaluated at every time sample.
    parameter_cost_reps: list[CostRep]
        Reps of registered costs evaluated once on the shared parameters.
    jars: dict[str, WorkerJar]
        Per trial, the pool backing that trial's tracking callback.
    """
    def __init__(self, model_cache: ModelCache, trials: list[Trial], costs: list,
                 tracking_cost, num_threads: int = 1, enable_fd: bool = False):
        if num_threads < 1:
            raise ValueError(
                f'Expected num_threads to be at least 1, but got {num_threads}.')

        self.mc = model_cache
        self.num_threads = num_threads
        self.tracking_cost = tracking_cost

        self.coordinate_cost_reps = [cost.create_rep(model_cache) for cost in costs
                                     if 'coordinates' in cost.required_inputs]
        self.parameter_cost_reps = [cost.create_rep(model_cache) for cost in costs
                                    if 'coordinates' not in cost.required_inputs]

        self.jars: dict[str, WorkerJar] = {}
        self.tracking_reps: dict[str, PooledTrackingRep] = {}
        self.reference: dict[str, ca.DM] = {}
        for trial in trials:
            workers = []
            for iworker in range(num_threads):
                # The first worker reuses the solver's own cache so a single-threaded
                # problem adds no model copies; the rest get independent replicas.
                mc = model_cache if iworker == 0 else replicate_model_cache(model_cache)
                workers.append(tracking_cost.create_trial_rep(
                    f'tracking_worker_{trial.name}_{iworker}', mc, trial))
            jar = WorkerJar(workers)
            self.jars[trial.name] = jar
            self.tracking_reps[trial.name] = PooledTrackingRep(
                f'tracking_cost_{trial.name}', jar, enable_fd=enable_fd)
            self.reference[trial.name] = ca.DM(tracking_cost.build_reference(trial))

        self._maps: dict[tuple, ca.Function] = {}

    def tracking_map(self, trial_name: str, num_samples: int) -> ca.Function:
        """
        Return this trial's tracking callback mapped across `num_samples` time samples.

        Maps are cached per ``(trial_name, num_samples)``, since constructing one
        forces construction of the underlying callback.

        Notes
        -----
        This deliberately uses the ``map(n, parallelization, max_num_threads)``
        overload rather than the ``reduce_in`` overload that can broadcast the
        sample-invariant parameter blocks. The ``reduce_in`` overload does not honour a
        thread cap: it spawns one OS thread per map column regardless of any
        ``max_num_threads`` option (which is silently accepted and ignored), so a
        1000-sample trial would spawn 1000 threads. The capped overload has no
        ``reduce_in``, so `evaluate_trial` broadcasts the parameter blocks explicitly
        with `ca.repmat`; those blocks are small (three entries per group), so
        materializing them costs far less than the thread explosion.

        Parameters
        ----------
        trial_name: str
            Name of the trial whose callback to map.
        num_samples: int
            Number of time samples to evaluate, i.e. the map's width.
        """
        key = (trial_name, num_samples)
        if key not in self._maps:
            rep = self.tracking_reps[trial_name]
            rep.construct_callback()
            self._maps[key] = (
                rep.map(num_samples, 'thread', self.num_threads)
                if self.num_threads > 1 else rep.map(num_samples))
        return self._maps[key]

    def evaluate_trial(self, trial_name: str, coordinates: ca.MX,
                       input: CostInput) -> ca.MX:
        """
        Evaluate a trial's tracking cost at all of its time samples at once, returning
        a ``(1, num_samples)`` row of per-sample errors.

        Parameters
        ----------
        trial_name: str
            Name of the trial to evaluate.
        coordinates: ca.MX
            An ``(nq, num_samples)`` expression whose column ``i`` holds the
            coordinates at time sample ``i``.
        input: CostInput
            Supplies the parameter blocks, which are shared across every sample and are
            broadcast to the map's width here. Its `coordinates` field is ignored in
            favour of the `coordinates` argument.
        """
        num_samples = coordinates.size2()
        mapped = self.tracking_map(trial_name, num_samples)
        parameters = [ca.repmat(getattr(input, name), 1, num_samples)
                      for name in CostInput.INPUT_ORDER[1:]]
        return mapped(coordinates, *parameters, self.reference[trial_name])

    def assert_offset_groups_used(self) -> None:
        """
        Verify that every registered offset group is tracked by at least one task in at
        least one trial, so that no offset is left unconstrained.

        Raises
        ------
        ValueError
            If any registered marker or frame offset group is tracked by no task.
        """
        def assert_used(used, offset_groups, label):
            for i, group in enumerate(offset_groups):
                if i not in used:
                    raise ValueError(
                        f'{label.capitalize()} offset group {group.component_paths} is '
                        f'not tracked by any registered {label} in any trial; its '
                        f'offset would be unconstrained.')

        reps = self.tracking_reps.values()
        used_markers = {g for rep in reps
                        for g in rep.marker_term.offset_group_indexes
                        if g is not None}
        used_frames = {g for rep in reps
                       for g in rep.frame_term.offset_group_indexes
                       if g is not None}
        assert_used(used_markers, self.mc.marker_offset_groups, 'marker')
        assert_used(used_frames, self.mc.frame_offset_groups, 'frame')
