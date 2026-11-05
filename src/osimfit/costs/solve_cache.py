import opensim as osim

from ..model import ModelCache, StationCache
from .cost import CostBase, CostCallback


class SolveCache:
    """
    Everything a solver resolves once per solve, memoized. Costs are stateless
    descriptions; any model-bound quantity they need is requested from the cache and
    keyed by the thing it actually varies over (e.g., a `StationCache` by component
    path), so it is resolved at most once per solve no matter how many time samples or
    costs request it.

    The cache also owns every `CostCallback` created during the solve, which is what
    keeps CasADi's references to them valid.

    Parameters
    ----------
    mc: ModelCache
        The solver's `ModelCache`. All parameter groups must already be registered on
        it, since they set the input sizes the callbacks declare to CasADi.

    Attributes
    ----------
    mc: ModelCache
        The wrapped model cache.
    state: osim.State
        The state every cost evaluation mutates and realizes.
    task_sets: list
        Every tracking task set created during the solve, so a solver can validate
        coverage across trials (e.g., that every registered offset group is tracked).
    """

    def __init__(self, mc: ModelCache):
        self.mc = mc
        self.state = mc.state
        self.task_sets = []
        self._callbacks: list[CostCallback] = []
        self._station_caches: dict[str, StationCache] = {}
        self._memoized: dict[tuple, tuple] = {}

    def station_cache(self, path: str) -> StationCache:
        """
        Return the memoized `StationCache` for the station or frame at `path`.

        Parameters
        ----------
        path: str
            The model path to an `osim.Station` (e.g., an `osim.Marker`) or an
            `osim.PhysicalFrame`.

        Returns
        -------
        StationCache
            The cached station, resolved on first request.
        """
        if path not in self._station_caches:
            component = self.mc.model.getComponent(path)
            station = osim.Station.safeDownCast(component)
            if station is not None:
                self.mc.model.realizePosition(self.state)
                cache = StationCache.from_station(self.mc, station)
            else:
                cache = StationCache.from_frame(
                    self.mc, osim.PhysicalFrame.safeDownCast(component))
            self._station_caches[path] = cache
        return self._station_caches[path]

    def memoize(self, owner: CostBase, key: str, factory):
        """
        Return the value memoized for `owner` under `key`, building it via `factory`
        on first request. A cost uses this to resolve its model-bound data once per
        solve rather than on every evaluation.

        Parameters
        ----------
        owner: CostBase
            The cost the value belongs to. The cache holds a reference to it, so its
            identity stays unique for the lifetime of the solve.
        key: str
            Distinguishes multiple values memoized for the same owner.
        factory: Callable[[], Any]
            Builds the value.
        """
        memo_key = (id(owner), key)
        if memo_key not in self._memoized:
            self._memoized[memo_key] = (owner, factory())
        return self._memoized[memo_key][1]

    def add_callback(self, name: str, cost: CostBase, context=None,
                     enable_fd: bool = False) -> CostCallback:
        """
        Create a `CostCallback` for `cost` and retain it for the lifetime of this
        cache.

        Parameters
        ----------
        name: str
            The name of the callback function.
        cost: CostBase
            The cost the callback evaluates.
        context: optional
            The cost's per-callback data, passed back to the cost's `calc_value` and
            `calc_jacobian`. Default is ``None``.
        enable_fd: bool, optional
            If ``True``, CasADi finite-differences the callback instead of using its
            analytic Jacobian. Default is ``False``.
        """
        callback = CostCallback(name, cost, self, context, enable_fd=enable_fd)
        self._callbacks.append(callback)
        return callback

    def keep_alive(self, obj) -> None:
        """
        Retain `obj` for the lifetime of this cache. Used while costs that still build
        their own CasADi callbacks are migrated to `CostCallback`.
        """
        self._callbacks.append(obj)
