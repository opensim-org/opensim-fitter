# Replacing `CostRep` with a memoizing `SolveCache`

Status: **implemented** on branch `solve_cache_refactor`, 2026-09-23. Line references
below are against `src/osimfit/costs.py` and `src/osimfit/solvers.py` as of
2026-09-22 (pre-refactor), and describe the design as proposed. See "Measured
outcome" at the end for what the implementation actually delivered, including one
prediction in this document that turned out to be wrong.

## What a `CostRep` is today

Each rep fuses three separable concerns:

1. **Resolution** — turning model-independent descriptors (paths, labels) into
   model-bound handles: `StationCache`, `osim.Marker`/`PhysicalFrame`, mobod
   indexes, base stations. Depends only on *(model, label set)*.
2. **Reference-data binding** — pinning the rep to one `(trial, itime)` sample.
   This is the *only* thing that varies per rep.
3. **CasADi plumbing** — being a `ca.Callback` with declared input sizes
   (derived from `mc.*_groups`) and `eval`/`jac_eval`.

Concern 1 is invariant across time, concern 3 is invariant across everything,
and yet all three are rebuilt per sample. In `SplinedKinematicsSolver.solve`
(solvers.py:867-884) you build `J × N` `BilevelCostRep`s, each of which runs
`StationCache.from_station` + `hasComponent` + `safeDownCast` +
`realizePosition` per marker. For a 5 s trial at 100 Hz with 40 markers that's
~20,000 redundant `StationCache` constructions and 500
`ca.Callback`+`JacobianFunction` pairs, all encoding the same geometry.

So the rep-per-cost hierarchy isn't just class bloat — it's what forces the
redundant work.

## Proposal: a memoizing `SolveCache`

The key question for "can the reps go away" is: *is there any per-solve state
that depends on the cost itself, rather than on the model/trials?* Going
through the three concrete reps:

- `TrackingCostRep`/`BilevelCostRep` (costs.py:1019, costs.py:1129) — hold
  tasks keyed by *(model, trial)*, plus weights. Weights are the only
  cost-specific part, and they don't need to be baked into the tasks; apply
  them at evaluate time. Then **one task set per trial serves both costs**.
- `AnthropometricRegularizationCostRep` (costs.py:1267) — holds `default_q` (a
  property of the model) and a `StationCache` per measurement path. If the
  cache memoizes `station_cache(path)`, the cost needs no stored state: it
  already has `self.measurements`.
- `SymbolicCostRep` (costs.py:204) — holds nothing at all. Pure boilerplate.

So for everything that exists today, **zero cost-specific per-solve state is
required**, provided the cache exposes memoized services keyed by the things
costs actually vary over (path, trial). Sketch:

```python
class SolveCache:
    """
    Everything a solver resolves once per solve.
    """
    def __init__(self, mc: ModelCache, trials: list[Trial]):
        self.mc = mc
        self.state = mc.state
        self.default_q = ...                    # model default pose
        self._station_caches: dict[str, StationCache] = {}
        self._task_sets: dict[str, TaskSet] = {}     # keyed by trial name
        self._callbacks: list[ca.Callback] = []      # keeps CasADi refs alive

    def station_cache(self, path: str) -> StationCache: ...   # memoized
    def task_set(self, trial: Trial) -> TaskSet: ...          # memoized, geometry only
    def callback(self, key, factory) -> Function: ...         # memoized + retained
```

`TaskSet` is the per-trial geometry (`station_caches`, `mobod_indexes`,
`base_frames`, `base_stations`, `offset_group_indexes`) built once, plus the
trial's reference data as dense arrays — `positions[itime, itask, :]`,
`orientations[itime, itask, :]` — instead of scalars scattered across N
objects. It's essentially today's `MarkerTasks`/`FrameTasks` with the time axis
added and the weights removed.

Costs then collapse to a single method:

```python
class Cost(ABC):
    required_inputs = frozenset()

    @abstractmethod
    def __call__(self, cache: SolveCache, input: CostInput) -> ca.MX: ...


class TrackingCostBase(ABC):
    @abstractmethod
    def __call__(self, cache, trial, itime, input: CostInput) -> ca.MX: ...
```

`SymbolicCost.__call__` returns an expression directly (no callback at all).
`CallbackCost.__call__` asks the cache for its callback and delegates to
`self.evaluate(cache, ...)` / `self.jacobian(cache, ...)`. One generic
`CostCallback(Function)` class replaces `TrackingCostRep`, `BilevelCostRep`,
`AnthropometricRegularizationCostRep`, and `SymbolicCostRep`.

## Two things that fall out of this

**The cache becomes the owner of CasADi lifetimes.** Right now solvers keep
three ad-hoc lists alive for exactly this reason (`tracking_reps` at
solvers.py:867, `cost_reps` at solvers.py:899, `placement_reps` at
solvers.py:1081), each with a comment explaining why. That concern moves into
`SolveCache._callbacks` and disappears from the solvers.

**`InverseKinematicsSolver` could build one `ca.nlpsol` per trial instead of
per timestep.** Today `create_tracking_solver` (solvers.py:431) is called
inside the timestep loop (solvers.py:507) and reconstructs the whole NLP every
sample, purely because the reference data is baked into the rep. If the
reference data is addressed by index at eval time, the NLP is identical across
timesteps — build it once per trial, and advance the sample index between
`solver(...)` calls. I'd guess that's the largest wall-clock win here, larger
than the resolution savings. The catch is it needs a mutable "current sample"
on the callback (fixed for the duration of any one IPOPT solve, so it's safe,
but it is mutable state). The non-mutating alternative is to feed reference
data through CasADi's `p` (parameter) channel, which is cleaner but needs the
callback to declare a zero Jacobian block for `p`.

## What to decide before writing anything

- **Scope.** Full collapse (costs become stateless, one generic callback), or
  just the cheap half — hoist task resolution to a per-trial cache and leave
  the `CostRep` hierarchy alone? The cheap half gets most of the
  construction-time win for a fraction of the churn.
- **The IK `nlpsol` change.** That's a behavioral/performance change riding
  along on a structural refactor. Better as a separate commit after the
  refactor lands, unless it's wanted together.
- **Merging `TrackingCost` into `BilevelCost`.** With no parameter groups
  registered, `BilevelCost` (costs.py:1078) is mathematically identical to
  `TrackingCost` (costs.py:977) — the scale/offset loops become no-ops.
  `SplinedKinematicsSolver` already picks between them with a one-line
  ternary. They could merge into one cost with a fast path, removing a fair
  amount of near-duplicate code in the `*Term` classes. Separate question from
  the rep refactor, but adjacent; needs an in/out call.
- **Test migration.** `tests/test_costs.py` (1135 lines, 52 test functions)
  has 39 rep touch points across 30 of those tests:
  - 10 direct `TrackingCostRep(...)` constructions, lines 137-216
  - 1 direct `BilevelCostRep(...)`, inside the `build_bilevel_rep` helper
    (defined line 267), which has 22 call sites spanning lines 283-718
  - 6 `cost.create_rep(...)` calls (lines 88, 97, 945, 1043, 1063, 1081)

  Most sites route through the helper or `create_rep`, so the helper absorbs a
  large share of the churn: retarget `build_bilevel_rep` at a `TaskSet` and 22
  of the 39 sites need no edit. The 10 `TrackingCostRep` constructions have no
  such shim and must each be rewritten. It's mechanical work — the natural
  seam is constructing a `TaskSet` directly, since `add_marker`/`add_frame`
  already exist — but it's where the risk of silently breaking a Jacobian
  check lives.

## Teeing up `ca.Function.map` in the spline solver

The refactor should be shaped so `SplinedKinematicsSolver` can evaluate its
tracking cost through one mapped `Function` instead of `J x N` callbacks. The
findings below are measured against CasADi 3.7.2 and OpenSim 4.6 in the
`opensim_fitter` env, not assumed.

### Target signature

`map` requires a single function with a uniform signature across samples.
CasADi 3.7.2 provides the reduction form

```python
fm = f.map(name, parallelization, n, reduce_in, reduce_out, opts)
```

where `reduce_in` names the inputs that are *not* repeated. That maps onto the
tracking cost directly:

| input | role | mapped shape |
|---|---|---|
| `coordinates` | repeated per sample | `(nq, N)` |
| `sample_index` | repeated per sample, constant `DM` | `(1, N)` |
| `body_scales` | shared (`reduce_in`) | `(3*ngroups, 1)` |
| `marker_offsets` | shared (`reduce_in`) | `(3*ngroups, 1)` |
| `frame_offsets` | shared (`reduce_in`) | `(3*ngroups, 1)` |
| output | per-sample error | `(1, N)` |

Two deliberate choices:

- **Do not use `reduce_out`.** It sums the outputs, but the solver needs
  trapezoidal weights (`compute_average_trapezoidal_error`, solvers.py:324).
  Keeping the output `(1, N)` leaves that weighting symbolic and outside the
  map, so the numerics are unchanged from today.
- **Reference data does not flow through CasADi.** It stays as dense numpy in
  the `TaskSet` (`positions[itime, itask, :]`, `orientations[itime, itask, :]`)
  and is addressed by `sample_index` inside `eval`. The Jacobian block for
  `sample_index` is declared structurally empty (`ca.Sparsity(1, 1)`).

This is what actually removes the per-sample rep: after it, the only
per-sample quantity crossing the CasADi boundary is one integer.

Verified end to end on a Python `ca.Callback`: the mapped signature comes back
as `[(3,5), (1,5), (2,1)] -> [(1,5)]`, the mapped value matches a numpy
reference exactly, and the analytic Jacobian with the structurally-zero index
block agrees with central finite differences to `1.1e-8`.

### The same trick simplifies the IK solver

The `nlpsol`-per-timestep issue above listed two options: a mutable "current
sample" on the callback, or pushing reference data through `nlpsol`'s `p`
channel. With an index input there is a third, better one: push the *index*
through `p`. One scalar parameter, no mutable state, no large reference block
in the NLP.

### What `map` will and will not buy

**Structural wins, available immediately with `'serial'`:**

- One callback per trial instead of one per sample. For a 5 s trial at 100 Hz
  that is 1 instead of 500, and `JacobianFunction` construction drops with it.
- A much smaller MX graph, so `nlpsol` construction time and memory drop.
- CasADi assembles the block-diagonal Jacobian from the per-instance blocks.

**Threaded parallelism is GIL-blocked.** Measured, 64 samples, one pre-built
model/state replica per instance so there is zero pool contention:

| backend | wall | speedup |
|---|---|---|
| `'serial'` | 0.073 s | 1.00x |
| `'thread'`, 8 threads | 0.086 s | 0.85x |
| `'thread'`, numpy/BLAS control | - | 2.47x |

`'thread'` does invoke the Python `eval` from N real OS threads without
crashing, and results stay correct. It is simply GIL-bound: OpenSim's SWIG
bindings do not release the GIL, so the samples serialize. The numpy control
in the same harness reaches 2.47x, which confirms the measurement would detect
real parallelism if it were there.

Per-thread model and state replicas do **not** fix this. They are necessary
for correctness, since `eval` mutates its `State`, but the GIL is per
interpreter, not per object. The 0.85x above already used one replica per
instance.

Two further constraints found along the way:

- `'openmp'` is unavailable: the CasADi wheel is built `WITH_OPENMP=OFF` and
  silently falls back to serial with a warning.
- `'thread'` spawns one OS thread per map instance, not a bounded pool.
  `{'max_num_threads': 8}` still produced 64 distinct thread ids.

### Process-backed `map`: real parallelism today

Separate processes have separate interpreters, so the GIL does not apply. The
per-sample work is ~0.096 ms while the data crossing the boundary is a few
hundred floats, so IPC is not the bottleneck. Two measured variants, 500
samples (~5 s at 100 Hz), `nq=39`, 41 markers:

| approach | wall | speedup | notes |
|---|---|---|---|
| serial baseline | 0.0478 s | 1.00x | 0.096 ms/sample |
| `ProcessPoolExecutor`, 8 workers | 0.0127 s | 3.77x | `ex.map` over 8 chunks |
| **`map('thread')`, 10 chunks, one worker process each** | **0.0063 s** | **8.06x** | results bit-identical |

The third row is the one to build toward, and it keeps `map` central. The
construction is:

- Chunk the trial's samples into `nchunk` blocks of `K` samples.
- The mapped function takes `(q_chunk, chunk_index)` repeated and the shared
  parameters via `reduce_in`, returning that chunk's `K` errors.
- `eval` uses `chunk_index` to select its own worker pipe, sends the block,
  and blocks on the reply. **Blocking on IPC releases the GIL**, so the
  `'thread'` instances genuinely overlap.

That is why the index input in the target signature matters beyond removing
the per-sample rep: it is also what lets an instance identify which worker it
owns, since CasADi does not tell a mapped function which slot it is.

Measured IPC floor was 0.9 ms per dispatch for 8 chunks, i.e. about 7% of the
parallel wall time. Worker spawn plus model load is a one-time ~0.4 s per
solve on macOS `spawn`.

### Caveats on the 8x

- **The Jacobian path is unverified.** The probe set `has_jacobian` to
  `False`. `_jac_eval` is the more expensive path and would need the same
  worker round trip. It should be symmetric, but it has not been measured.
- **Shared parameters must be shipped each call.** Workers need to apply body
  scales and offsets themselves (`set_scaled_mobilizer_frame_positions` and
  the term `apply_state` calls). That is a few dozen floats, so IPC stays
  cheap, but the worker has to own a full `ModelCache`.
- **This is an eval-path speedup, not a wall-clock solve speedup.** Amdahl
  applies: IPOPT's own linear algebra is untouched. An end-to-end measurement
  on `examples/example_walk` should come before committing to the complexity.
- **Real operational cost.** Worker lifecycle, crash and hang handling,
  cleanup on exception, and interrupt behavior during a long IPOPT solve are
  all new failure modes that do not exist today.

### Design implications for `SolveCache`

1. **The cache owns the worker pool and hands out replicas, not a state.**
   Replace a bare `cache.state` with an accessor over pre-built
   `(model, state, handles)` replicas, pool size defaulting to 1 so today's
   behavior is unchanged. In-process replicas are required for correctness
   regardless of backend; the process pool is the same abstraction with the
   replicas living in workers.
2. **Pool checkout must not spin.** An early probe used a spin-lock and cost
   9x versus serial, because a spinning Python loop holds the GIL. Use a
   blocking handoff.
3. **Keep the backend a single option on the cache** (`serial`, `process`),
   so the choice is one argument rather than a structural change.

### Route to a compiled cost

A compiled cost linked against OpenSim and loaded via `ca.external` remains
the endgame: it would make `'thread'` work natively and the whole NLP
code-generable. But it is no longer the *only* route to parallelism, so it
should be costed on its own merits rather than treated as a prerequisite.

Not worth pursuing: `'openmp'` (not compiled in), and rebuilding OpenSim's
SWIG bindings with `-threads` purely for this, since the process-backed route
gets the same win without touching opensim-core.

### Open questions on the map design

- **One map per trial, or one across all trials?** Per-trial is simplest and
  handles differing `N` and differing marker sets. If all trials track the
  same task set, a single map over `sum(N_j)` samples with a global sample
  index would be maximal, with the output row sliced per trial for
  trapezoidal weighting. Needs a call on whether cross-trial task sets are
  guaranteed uniform.
- **Does `map('serial')` speed up IPOPT iterations, or only construction?**
  The measurements above are construction- and eval-side. Worth one end-to-end
  benchmark on `examples/example_walk` before committing to the design.
- **`assert_offset_groups_used` (solvers.py:686) introspects the per-sample
  reps** via `rep.marker_term.offset_group_indexes`. With one mapped function
  that check has to read offset-group usage off the per-trial `TaskSet`
  instead. Cheap, but it is a real call site that the refactor must move.

## Open uncertainty

Whether a future cost will need per-solve state that *isn't* keyed by path or
trial, which would reintroduce something rep-shaped. It isn't present in the
current three, and the memoized-service design absorbs new cases by adding a
service rather than a class. But the abstraction is not provably closed.


## Measured outcome

Implemented as proposed, with the two follow-ups still deferred (the IK
`nlpsol`-per-trial change, and merging `TrackingCost` into `BilevelCost`). All
122 tests pass, and the three solvers produce output **bit-identical** to `main`
(max absolute difference 0.000e+00 across IK coordinates, splined coordinates and
spline nodes, bilevel coordinates and body scales, and marker-placer offsets and
poses).

### Object counts: as predicted

One splined solve, full-body model, 500 samples, 41 markers, one body-scale group:

| | main | refactor |
|---|---|---|
| `StationCache` constructions | 20,500 | **41** |
| CasADi cost callbacks | 500 | **1** |

That is exactly the 20,000-redundant-constructions figure this document opened
with, and the per-sample rep is gone.

### Construction wall-clock: real, but smaller than implied

Time to build the NLP (everything up to `nlpsol`), same model and trial:

| samples | main | refactor | speedup |
|---|---|---|---|
| 50 | 0.040 s | 0.012 s | 3.3x |
| 200 | 0.207 s | 0.085 s | 2.4x |
| 500 | 0.805 s | 0.479 s | 1.7x |
| 1000 | 2.893 s | 2.251 s | 1.3x |

The speedup *shrinks* with trial length, which is the opposite of what the
object-count table suggests. The reason is the correction below.

### Correction: the MX graph does not shrink

This document claimed a "much smaller MX graph, so `nlpsol` construction time and
memory drop." That is wrong. The refactor collapses the number of `Function`
*objects* from one per sample to one per trial, but the objective still contains
one call node per sample, so the graph has the same number of nodes as before.
What was eliminated is the repeated OpenSim resolution behind those nodes, plus
499 `ca.Callback` + `JacobianFunction` constructions.

At short trials resolution dominates and the win is large; at long trials graph
assembly dominates and the win tapers. Shrinking the graph itself requires the
`map` work described above, which replaces N call nodes with one.

### Design notes from the implementation

- **Weights split in two.** A `TaskSet` keeps a per-task weight, and the cost
  applies its own `position_weight`/`orientation_weight` on top at evaluation
  time; the effective weight is the product. This is what lets one task set serve
  a cost regardless of its weights, and it preserves the per-task weights the
  tests rely on.
- **`CostCallback` takes a `TaskSet`, not a `Trial`.** Resolving the task set in
  the cost rather than the callback keeps `Trial` out of the callback entirely,
  and lets a test build a task set by hand.
- **Inputs are declared from `required_inputs`.** A `TrackingCost` callback now
  declares one input (`coordinates`) rather than four, three of which were empty.
- **`num_times` bounds reference loading.** `MarkerPlacer` reads only the first
  sample of each trial, so it passes `num_times=1` and the cache loads one row
  rather than the whole table.
- **Tests use an explicit `CostHarness`** that assembles the cache, task set, and
  callback the way a solver does, so a test can exercise one cost at one pose
  without constructing a `Trial`.
